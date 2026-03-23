"""
Benchmark: CuTe DSL kernel vs CUTLASS kernel for blockwise FP8 scaled GEMM on SM100.

Compares:
  1. CuTe DSL  - vllm/kernels/cutedsl/scaled_mm_blockwise_sm100_fp8.py
  2. CUTLASS  - called via vllm._custom_ops.cutlass_scaled_mm (blockwise path)

Usage:
    python benchmarks/cutlass_benchmarks/bench_blockwise_fp8_cutedsl_vs_cutlass.py \
        [--m 128,256,512] [--k 4096] [--n 4096] \
        [--out-dtype bf16] [--warmup 5] [--iters 20]
"""

import argparse
import math
import time
from typing import Callable, List, Tuple

import torch

# ---------------------------------------------------------------------------
# CuTe DSL imports (lazy – may not be installed everywhere)
# ---------------------------------------------------------------------------
_cute_dsl_available = False
try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    _cute_dsl_available = True
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
# Helpers
# ---------------------------------------------------------------------------
def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


OUT_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

CUTE_C_DTYPE_MAP = {}
if _cute_dsl_available:
    CUTE_C_DTYPE_MAP = {
        "bf16": cutlass.BFloat16,
        "fp16": cutlass.Float16,
    }


# ---------------------------------------------------------------------------
# Tensor creation helpers
# ---------------------------------------------------------------------------
def make_fp8_tensors_cutlass(m: int, n: int, k: int, out_dtype: torch.dtype):
    """Create FP8 A (m, k), B (n, k).T (Transpose) and blockwise scales for CUTLASS path.

    CUTLASS expects:
      a: (m, k) fp8, contiguous
      b: (k, n) fp8 - stored as (n, k).T  (column-major)
      scale_a: (m, ceil(k/128)) fp32, M-major (contiguous along M)
      scale_b: (ceil(k/128), ceil(n/128)) fp32, K-major (contiguous along K)
    """
    a = torch.randn((m, k), device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn((n, k), device="cuda").t().to(torch.float8_e4m3fn)  # (k, n) col-major

    scale_a = torch.rand((m, cdiv(k, 128)), device="cuda", dtype=torch.float32) + 0.5
    scale_b = torch.rand((cdiv(k, 128), cdiv(n, 128)), device="cuda", dtype=torch.float32) + 0.5

    # Ensure correct memory layout (M-major / K-major)
    scale_a = scale_a.t().contiguous().t()  # M-major
    scale_b = scale_b.t().contiguous().t()  # K-major

    out = torch.empty((m, n), device="cuda", dtype=out_dtype)
    return a, b, scale_a, scale_b, out


def make_fp8_tensors_cutedsl(m: int, n: int, k: int, c_dtype_str: str):
    """Create CuTe-style tensors for the CuTe DSL kernel.

    The CuTe DSL kernel operates on 3-D tensors (M, K, L) with L=1 (batch=1).
    A is (M, K, 1) row-major ("K"), B is (N, K, 1) col-major ("K"),
    C is (M, N, 1) row-major ("N").
    """
    ab_dtype = cutlass.Float8E4M3FN
    c_dtype = CUTE_C_DTYPE_MAP[c_dtype_str]
    scale_dtype = cutlass.Float32
    l = 1

    a_major, b_major, cd_major = "k", "k", "n"

    torch.manual_seed(42)
    a_torch_cpu = cutlass_torch.matrix(l, m, k, a_major == "m", ab_dtype)
    b_torch_cpu = cutlass_torch.matrix(l, n, k, b_major == "n", ab_dtype)
    c_torch_cpu = cutlass_torch.matrix(l, m, n, cd_major == "m", c_dtype)
    sfa_torch_cpu = cutlass_torch.matrix(l, m, cdiv(k, 128), True, scale_dtype)
    sfb_torch_cpu = cutlass_torch.matrix(l, cdiv(n, 128), cdiv(k, 128), False, scale_dtype)

    a_tensor, _ = cutlass_torch.cute_tensor_like(
        a_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16)
    b_tensor, _ = cutlass_torch.cute_tensor_like(
        b_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16)
    c_tensor, c_torch_gpu = cutlass_torch.cute_tensor_like(
        c_torch_cpu, c_dtype, is_dynamic_layout=True, assumed_align=16)
    sfa_tensor, _ = cutlass_torch.cute_tensor_like(
        sfa_torch_cpu, scale_dtype, is_dynamic_layout=True, assumed_align=16)
    sfb_tensor, _ = cutlass_torch.cute_tensor_like(
        sfb_torch_cpu, scale_dtype, is_dynamic_layout=True, assumed_align=16)

    return a_tensor, b_tensor, c_tensor, sfa_tensor, sfb_tensor


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------
def bench_fn_cuda_events(
    fn: Callable, warmup: int = 5, iters: int = 20
) -> Tuple[float, float]:
    """Time *fn* using CUDA events. Returns (median_ms, mean_ms)."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    median = times[len(times) // 2]
    mean = sum(times) / len(times)
    return median, mean


# ---------------------------------------------------------------------------
# CuTe DSL kernel wrapper
# ---------------------------------------------------------------------------
class CuteDslBenchWrapper:
    """Compile the CuTe DSL kernel once, then benchmark repeated launches."""

    def __init__(self, m: int, n: int, k: int, c_dtype_str: str,
                 mma_tiler_mn: Tuple[int, int] = (128, 128),
                 cluster_shape_mn: Tuple[int, int] = (1, 1),
                 use_2cta_instrs: bool = False):
        import sys, os
        # Make sure the kernel module is importable
        dsl_dir = os.path.join(os.path.dirname(__file__), "..", "..",
                               "vllm", "kernels", "cutedsl")
        dsl_dir = os.path.abspath(dsl_dir)
        if dsl_dir not in sys.path:
            sys.path.insert(0, dsl_dir)
        from scaled_mm_blockwise_sm120_fp8 import BlockwiseGemmKernel

        acc_dtype = cutlass.Float32
        c_dtype = CUTE_C_DTYPE_MAP[c_dtype_str]

        self.tensors = make_fp8_tensors_cutedsl(m, n, k, c_dtype_str)
        a_t, b_t, c_t, sfa_t, sfb_t = self.tensors

        self.gemm = BlockwiseGemmKernel(
            acc_dtype, use_2cta_instrs, mma_tiler_mn, cluster_shape_mn)

        hardware_info = cutlass.utils.HardwareInfo()
        self.max_active_clusters = hardware_info.get_max_active_clusters(
            cluster_shape_mn[0] * cluster_shape_mn[1])

        torch_stream = torch.cuda.current_stream()
        self.stream = cuda.CUstream(torch_stream.cuda_stream)

        # Determine opt level
        try:
            from cutlass import CUDA_VERSION
            opt_level = (3 if CUDA_VERSION.major < 13
                         or (CUDA_VERSION.major == 13 and CUDA_VERSION.minor < 1)
                         else 2)
        except ImportError:
            opt_level = 3

        # Compile once
        self.compiled = cute.compile(
            self.gemm,
            a_t, b_t, c_t, sfa_t, sfb_t,
            self.max_active_clusters, self.stream,
            options=f"--opt-level {opt_level}",
        )

    def __call__(self):
        a_t, b_t, c_t, sfa_t, sfb_t = self.tensors
        self.compiled(a_t, b_t, c_t, sfa_t, sfb_t, self.stream)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_benchmarks(
    m_sizes: List[int],
    k_sizes: List[int],
    n_sizes: List[int],
    out_dtype_str: str,
    warmup: int,
    iters: int,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
):
    out_dtype = OUT_DTYPE_MAP[out_dtype_str]

    header = (f"{'M':>6}  {'N':>6}  {'K':>6}  {'Kernel':<40}  "
              f"{'Median(ms)':>11}  {'Mean(ms)':>11}  {'TFLOPS':>8}")
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for m in m_sizes:
        for n in n_sizes:
            for k in k_sizes:
                flops = 2.0 * m * n * k  # GEMM FLOPs
                results = []

                # --- CUTLASS (via vLLM ops) ---
                if _cutlass_ops_available:
                    a, b, sa, sb, out = make_fp8_tensors_cutlass(m, n, k, out_dtype)
                    def cutlass_fn(a=a, b=b, sa=sa, sb=sb, out_dtype=out_dtype):
                        ops.cutlass_scaled_mm(a, b, sa, sb, out_dtype)
                    try:
                        med, mean = bench_fn_cuda_events(cutlass_fn, warmup, iters)
                        tflops = flops / (med * 1e-3) / 1e12
                        results.append(("CUTLASS (vllm ops)", med, mean, tflops))
                    except Exception as e:
                        results.append((f"CUTLASS (vllm ops) [ERR: {e}]", -1, -1, 0))
                else:
                    results.append(("CUTLASS (vllm ops) [NOT AVAILABLE]", -1, -1, 0))

                # --- CuTe DSL ---
                if _cute_dsl_available:
                    try:
                        wrapper = CuteDslBenchWrapper(
                            m, n, k, out_dtype_str,
                            mma_tiler_mn=mma_tiler_mn,
                            cluster_shape_mn=cluster_shape_mn)
                        med, mean = bench_fn_cuda_events(wrapper, warmup, iters)
                        tflops = flops / (med * 1e-3) / 1e12
                        results.append(("CuTe DSL", med, mean, tflops))
                    except Exception as e:
                        results.append((f"CuTe DSL [ERR: {e}]", -1, -1, 0))
                else:
                    results.append(("CuTe DSL [NOT AVAILABLE]", -1, -1, 0))

                # Print results
                for name, med, mean, tflops in results:
                    if med < 0:
                        print(f"{m:>6}  {n:>6}  {k:>6}  {name:<40}  "
                              f"{'N/A':>11}  {'N/A':>11}  {'N/A':>8}")
                    else:
                        print(f"{m:>6}  {n:>6}  {k:>6}  {name:<40}  "
                              f"{med:>11.4f}  {mean:>11.4f}  {tflops:>8.2f}")
                print()

    print(sep)


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",")]


def parse_int_tuple(s: str) -> Tuple[int, int]:
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected two comma-separated ints")
    return tuple(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark CuTe DSL vs CUTLASS blockwise FP8 scaled GEMM")
    parser.add_argument("--m", type=parse_int_list, default=[128, 256, 512, 1024, 2048, 4096],
                        help="Comma-separated M dimensions (default: 128,256,512,1024,2048,4096)")
    parser.add_argument("--n", type=parse_int_list, default=[4096],
                        help="Comma-separated N dimensions (default: 4096)")
    parser.add_argument("--k", type=parse_int_list, default=[4096],
                        help="Comma-separated K dimensions (default: 4096)")
    parser.add_argument("--out-dtype", choices=["bf16", "fp16"], default="bf16",
                        help="Output dtype (default: bf16)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of warmup iterations (default: 5)")
    parser.add_argument("--iters", type=int, default=20,
                        help="Number of timed iterations (default: 20)")
    parser.add_argument("--mma-tiler-mn", type=parse_int_tuple, default=(128, 128),
                        help="MMA tiler (M,N) for CuTe DSL kernel (default: 128,128)")
    parser.add_argument("--cluster-shape-mn", type=parse_int_tuple, default=(1, 1),
                        help="Cluster shape (M,N) for CuTe DSL kernel (default: 1,1)")
    args = parser.parse_args()

    print(f"\nBenchmark config:")
    print(f"  M sizes       : {args.m}")
    print(f"  N sizes       : {args.n}")
    print(f"  K sizes       : {args.k}")
    print(f"  Output dtype  : {args.out_dtype}")
    print(f"  Warmup iters  : {args.warmup}")
    print(f"  Timed iters   : {args.iters}")
    print(f"  MMA tiler MN  : {args.mma_tiler_mn}")
    print(f"  Cluster shape : {args.cluster_shape_mn}")
    if torch.cuda.is_available():
        print(f"  GPU           : {torch.cuda.get_device_name()}")
        cap = torch.cuda.get_device_capability()
        print(f"  Compute cap   : {cap[0]}.{cap[1]}")
    print()

    run_benchmarks(
        m_sizes=args.m,
        k_sizes=args.k,
        n_sizes=args.n,
        out_dtype_str=args.out_dtype,
        warmup=args.warmup,
        iters=args.iters,
        mma_tiler_mn=args.mma_tiler_mn,
        cluster_shape_mn=args.cluster_shape_mn,
    )


if __name__ == "__main__":
    main()
