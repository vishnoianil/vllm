"""
Benchmark: CuTile kernel vs CUTLASS C++ kernel for blockwise W8A8 FP8
scaled GEMM.

Compares:
  1. CuTile   - vllm/kernels/cutile/cutile_w8a8.py (cutile_blockwise_mm)
  2. CUTLASS  - called via vllm._custom_ops.cutlass_scaled_mm
               (csrc/libtorch_stable/quantization/w8a8/cutlass/
                scaled_mm_c3x_sm100.cu)

Both kernels compute blockwise-scaled FP8 GEMM:
  C = sum_k (A_tile @ B_tile) * (A_scale * B_scale)

  - A: (M, K) FP8 e4m3, row-major
  - B: (K, N) FP8 e4m3, col-major  (stored as (N, K).T for CUTLASS)
  - A_scale: (M, K//128) per-token-group FP32 scales
  - B_scale: (N//128, K//128) per-block FP32 scales
  - C: (M, N) BF16 or FP16

Usage:
    python benchmarks/cutlass_benchmarks/bench_blockwise_fp8_cutile_vs_cutlass.py

    # Custom sizes
    python benchmarks/cutlass_benchmarks/bench_blockwise_fp8_cutile_vs_cutlass.py \
        --m 128,256,512,1024 --k 4096 --n 4096

    # Specify model weight shapes
    python benchmarks/cutlass_benchmarks/bench_blockwise_fp8_cutile_vs_cutlass.py \
        --models meta-llama/Llama-2-7b-hf --tp-sizes 1
"""

import argparse
import os
import sys
from typing import Callable, List, Tuple

# Ensure repo root is on sys.path so 'tests' package is importable.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

# ---------------------------------------------------------------------------
# CuTile imports
# ---------------------------------------------------------------------------
_cutile_available = False
try:
    import cuda.tile as ct
    from vllm.kernels.cutile.cutile_w8a8 import get_tile_config

    _cutile_available = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# CUTLASS via vLLM custom ops
# ---------------------------------------------------------------------------
_cutlass_ops_available = False
try:
    from vllm import _custom_ops as ops

    _cutlass_ops_available = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Weight shapes for model-aware benchmarking
# ---------------------------------------------------------------------------
try:
    from weight_shapes import WEIGHT_SHAPES
except ImportError:
    WEIGHT_SHAPES = {}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BLOCK_SIZE = 128

OUT_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


# ---------------------------------------------------------------------------
# Tensor creation for blockwise FP8 GEMM
# ---------------------------------------------------------------------------
def make_tensors(
    m: int, n: int, k: int, out_dtype: torch.dtype
) -> Tuple[torch.Tensor, ...]:
    """Create FP8 A/B and blockwise FP32 scales.

    Returns:
      a:       (M, K) fp8 e4m3, row-major
      b:       (N, K) fp8 e4m3, row-major (will be transposed per-kernel)
      a_scale: (M, K//128) fp32, column-major
      b_scale: (N//128, K//128) fp32
    """
    a = torch.randn((m, k), device="cuda", dtype=torch.float32)
    a = a.clamp(-448, 448).to(torch.float8_e4m3fn)

    b = torch.randn((n, k), device="cuda", dtype=torch.float32)
    b = b.clamp(-448, 448).to(torch.float8_e4m3fn)

    k_tiles = (k + BLOCK_SIZE - 1) // BLOCK_SIZE
    n_tiles = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Column-major a_scale to match CUTLASS/CuTile convention
    a_scale = torch.rand((m, k_tiles), device="cuda", dtype=torch.float32)
    b_scale = torch.rand((n_tiles, k_tiles), device="cuda", dtype=torch.float32)

    return a, b, a_scale, b_scale


# ---------------------------------------------------------------------------
# CUDA-event based timing
# ---------------------------------------------------------------------------
def bench_cuda_events(
    fn: Callable, warmup: int = 10, iters: int = 100
) -> Tuple[float, float, float]:
    """Time *fn* with CUDA events. Returns (median_ms, mean_ms, min_ms)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    median = times[len(times) // 2]
    mean = sum(times) / len(times)
    minimum = times[0]
    return median, mean, minimum


# ---------------------------------------------------------------------------
# Kernel runner helpers
# ---------------------------------------------------------------------------
def run_cutlass(a, b, a_scale, b_scale, out_dtype):
    """CUTLASS path: ensure VLLM_SWITCH_CUTILE is off."""
    os.environ["VLLM_SWITCH_CUTILE"] = "0"
    return ops.cutlass_scaled_mm(
        a,
        b.t(),
        scale_a=a_scale,
        scale_b=b_scale.t(),
        out_dtype=out_dtype,
    )


def run_cutile(a, b, a_scale, b_scale, out_dtype):
    """CuTile path: route through ops.cutlass_scaled_mm with VLLM_SWITCH_CUTILE=1."""
    os.environ["VLLM_SWITCH_CUTILE"] = "1"
    return ops.cutlass_scaled_mm(
        a,
        b.t(),
        scale_a=a_scale,
        scale_b=b_scale.t(),
        out_dtype=out_dtype,
    )


# ---------------------------------------------------------------------------
# Model weight shape extraction
# ---------------------------------------------------------------------------
def get_weight_shapes(
    model_names: List[str], tp_sizes: List[int]
) -> List[Tuple[int, int]]:
    unique_shapes = set()
    for model in model_names:
        if model not in WEIGHT_SHAPES:
            print(f"Warning: {model} not found in WEIGHT_SHAPES. Skipping.")
            continue
        for shape, tp_split_dim in WEIGHT_SHAPES[model]:
            k_raw, n_raw = shape
            for tp_size in tp_sizes:
                if tp_split_dim == 0:
                    unique_shapes.add((n_raw, k_raw // tp_size))
                else:
                    unique_shapes.add((n_raw // tp_size, k_raw))
    return sorted(unique_shapes)


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------
def run_benchmarks(
    m_sizes: List[int],
    n_sizes: List[int],
    k_sizes: List[int],
    out_dtype_str: str,
    warmup: int,
    iters: int,
):
    out_dtype = OUT_DTYPE_MAP[out_dtype_str]

    header = (
        f"{'M':>6}  {'N':>6}  {'K':>6}  "
        f"{'Tile (MxNxK,G)':>20}  "
        f"{'Kernel':<40}  "
        f"{'Median(ms)':>11}  {'Mean(ms)':>11}  {'Min(ms)':>10}  {'TFLOPS':>8}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for m in m_sizes:
        for n in n_sizes:
            for k in k_sizes:
                flops = 2.0 * m * n * k
                results: List[Tuple[str, float, float, float, float]] = []

                # Get tile config for this shape
                if _cutile_available:
                    tile_m, tile_n, tile_k, group_m = get_tile_config(
                        m, n, k
                    )
                else:
                    tile_m, tile_n, tile_k, group_m = (
                        BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, 1
                    )
                tile_str = f"{tile_m}x{tile_n}x{tile_k},{group_m}"

                a, b, a_scale, b_scale = make_tensors(m, n, k, out_dtype)

                # ---- CUTLASS C++ (via vLLM ops) ----
                if _cutlass_ops_available:
                    def cutlass_fn(
                        _a=a, _b=b, _as=a_scale, _bs=b_scale, _dt=out_dtype
                    ):
                        run_cutlass(_a, _b, _as, _bs, _dt)

                    try:
                        med, mean, mn = bench_cuda_events(
                            cutlass_fn, warmup, iters
                        )
                        tflops = flops / (med * 1e-3) / 1e12
                        results.append(
                            ("CUTLASS C++ (vllm ops)", med, mean, mn, tflops)
                        )
                    except Exception as e:
                        results.append(
                            (f"CUTLASS C++ [ERR: {e}]", -1, -1, -1, 0)
                        )
                else:
                    results.append(
                        ("CUTLASS C++ [NOT AVAILABLE]", -1, -1, -1, 0)
                    )

                # ---- CuTile ----
                if _cutile_available:
                    def cutile_fn(
                        _a=a, _b=b, _as=a_scale, _bs=b_scale, _dt=out_dtype
                    ):
                        run_cutile(_a, _b, _as, _bs, _dt)

                    try:
                        med, mean, mn = bench_cuda_events(
                            cutile_fn, warmup, iters
                        )
                        tflops = flops / (med * 1e-3) / 1e12
                        results.append(
                            ("CuTile (cutile_blockwise_mm)", med, mean, mn,
                             tflops)
                        )
                    except Exception as e:
                        results.append(
                            (f"CuTile [ERR: {e}]", -1, -1, -1, 0)
                        )
                else:
                    results.append(
                        ("CuTile [NOT AVAILABLE]", -1, -1, -1, 0)
                    )

                # ---- PyTorch torch._scaled_mm (per-tensor, perf baseline) --
                try:
                    # Use tile_m x tile_n aligned input for fair comparison
                    m_aligned = ((m + tile_m - 1) // tile_m) * tile_m
                    k_aligned = ((k + tile_k - 1) // tile_k) * tile_k

                    a_pt = torch.randn(
                        (m_aligned, k_aligned), device="cuda",
                        dtype=torch.float32,
                    ).clamp(-448, 448).to(torch.float8_e4m3fn)[:m, :k] \
                     .contiguous()
                    # (N, K) contiguous then .t() gives col-major (K, N)
                    b_pt = b.contiguous().t()
                    sa_pt = torch.tensor(
                        1.0, device="cuda", dtype=torch.float32
                    )
                    sb_pt = torch.tensor(
                        1.0, device="cuda", dtype=torch.float32
                    )

                    def pytorch_fn(
                        _a=a_pt, _b=b_pt, _sa=sa_pt, _sb=sb_pt,
                        _dt=out_dtype,
                    ):
                        torch._scaled_mm(
                            _a, _b, _sa, _sb,
                            out_dtype=_dt, use_fast_accum=True,
                        )

                    med, mean, mn = bench_cuda_events(
                        pytorch_fn, warmup, iters
                    )
                    tflops = flops / (med * 1e-3) / 1e12
                    results.append(
                        ("PyTorch _scaled_mm (per-tensor)", med, mean, mn,
                         tflops)
                    )
                except Exception as e:
                    results.append(
                        (f"PyTorch _scaled_mm [ERR: {e}]", -1, -1, -1, 0)
                    )

                # Print results for this (M, N, K)
                for name, med, mean, mn, tflops in results:
                    if med < 0:
                        print(
                            f"{m:>6}  {n:>6}  {k:>6}  "
                            f"{tile_str:>20}  {name:<40}  "
                            f"{'N/A':>11}  {'N/A':>11}  {'N/A':>10}  "
                            f"{'N/A':>8}"
                        )
                    else:
                        print(
                            f"{m:>6}  {n:>6}  {k:>6}  "
                            f"{tile_str:>20}  {name:<40}  "
                            f"{med:>11.4f}  {mean:>11.4f}  {mn:>10.4f}  "
                            f"{tflops:>8.2f}"
                        )

                # Speedup summary
                cutlass_med = next(
                    (
                        r[1]
                        for r in results
                        if "CUTLASS C++" in r[0] and r[1] > 0
                    ),
                    None,
                )
                cutile_med = next(
                    (
                        r[1]
                        for r in results
                        if "CuTile" in r[0] and r[1] > 0
                    ),
                    None,
                )
                if cutlass_med and cutile_med:
                    speedup = cutlass_med / cutile_med
                    faster = "CuTile" if speedup > 1 else "CUTLASS C++"
                    ratio = speedup if speedup > 1 else 1.0 / speedup
                    print(
                        f"{'':>6}  {'':>6}  {'':>6}  "
                        f"{'>> ' + faster + f' is {ratio:.2f}x faster':<40}"
                    )
                print()

    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark CuTile vs CUTLASS C++ blockwise W8A8 FP8 scaled GEMM"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--m",
        type=parse_int_list,
        default=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        help="Comma-separated M dimensions",
    )
    parser.add_argument(
        "--n",
        type=parse_int_list,
        default=[4096],
        help="Comma-separated N dimensions",
    )
    parser.add_argument(
        "--k",
        type=parse_int_list,
        default=[4096],
        help="Comma-separated K dimensions",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model names to extract (N, K) weight shapes from "
        "(overrides --n/--k)",
    )
    parser.add_argument(
        "--tp-sizes",
        nargs="+",
        type=int,
        default=[1],
        help="TP sizes for model weight shape splitting",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Override M with batch sizes (alias for --m)",
    )
    parser.add_argument(
        "--out-dtype",
        choices=["bf16", "fp16"],
        default="bf16",
        help="Output dtype (default: bf16)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=500,
        help="Number of timed iterations",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")

    cap = torch.cuda.get_device_capability()

    # Determine M sizes
    m_sizes = args.m
    if args.batch_sizes is not None:
        m_sizes = args.batch_sizes

    # Determine N/K sizes (from models or CLI)
    if args.models is not None:
        shapes = get_weight_shapes(args.models, args.tp_sizes)
        if not shapes:
            raise ValueError("No shapes found for specified models.")
        n_sizes = sorted({s[0] for s in shapes})
        k_sizes = sorted({s[1] for s in shapes})
    else:
        n_sizes = args.n
        k_sizes = args.k

    print()
    print("=" * 70)
    print("  Blockwise W8A8 FP8 GEMM Benchmark: CuTile vs CUTLASS C++")
    print("=" * 70)
    print(f"  GPU             : {torch.cuda.get_device_name()}")
    print(f"  Compute cap     : {cap[0]}.{cap[1]}")
    print(f"  Block size      : {BLOCK_SIZE}")
    print(f"  M sizes         : {m_sizes}")
    print(f"  N sizes         : {n_sizes}")
    print(f"  K sizes         : {k_sizes}")
    print(f"  Output dtype    : {args.out_dtype}")
    print(f"  Warmup iters    : {args.warmup}")
    print(f"  Timed iters     : {args.iters}")
    print(f"  CuTile avail    : {_cutile_available}")
    print(f"  CUTLASS ops avail: {_cutlass_ops_available}")
    if args.models:
        print(f"  Models          : {args.models}")
        print(f"  TP sizes        : {args.tp_sizes}")
    print()

    run_benchmarks(
        m_sizes=m_sizes,
        n_sizes=n_sizes,
        k_sizes=k_sizes,
        out_dtype_str=args.out_dtype,
        warmup=args.warmup,
        iters=args.iters,
    )


if __name__ == "__main__":
    main()
