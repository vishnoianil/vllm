# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CuTe DSL dispatch layer for scaled GEMM (FP8 / INT8) on SM90.

Provides ``cutedsl_scaled_mm`` as a drop-in replacement for the CUTLASS C++
path used by ``torch.ops._C.cutlass_scaled_mm``. The function is only called
when ``VLLM_SWITCH_TO_CUTEDSL=1`` is set and the inputs are FP8 or INT8 on a
Hopper GPU.

Supports per-tensor, per-token (row-wise), and per-channel (column-wise)
scales.  The dispatch layer pre-combines ``scale_a * scale_b`` into an
``(M, N)`` tensor which is fused into the kernel epilogue via partition_C,
eliminating any post-kernel scaling passes.

Uses fake_tensor and fake_stream to JIT-compile kernels with dynamic layouts to
pass torch tensors directly to CuTe DSL without any data copies. The compiled 
kernels are cached for reuse across multiple calls with the same
tile/cluster/dtype configuration. Kernels are JIT-compiled on first use and 
cached for subsequent calls.
"""

import logging
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

from vllm.kernels.quantization.cutedsl.scaled_mm_sm90_fp8 import ScaledMmSm90Fp8Kernel
from vllm.kernels.quantization.cutedsl.scaled_mm_sm90_int8 import ScaledMmSm90Int8Kernel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compiled-kernel cache
# ---------------------------------------------------------------------------
class _KernelCache:
    """Cache for compiled CuTe DSL kernels.

    Cache key: ``(tile_mn, cluster_mn, input_dtype, output_dtype)``
    Each unique tile/cluster/dtype combination is compiled once and reused for
    all matrix sizes (the kernels are compiled with ``is_dynamic_layout=True``).
    """

    def __init__(self):
        self._cache: dict[tuple, object] = {}

    def get_or_compile(self, key: tuple, compile_fn):
        if key not in self._cache:
            self._cache[key] = compile_fn()
        return self._cache[key]


_kernel_cache = _KernelCache()


# PyTorch <-> CUTLASS dtype helpers
_TORCH_TO_CUTLASS: dict[torch.dtype, object] = {}


def _cutlass_dtype(torch_dtype: torch.dtype):
    """Map a PyTorch dtype to its CUTLASS equivalent (lazy init)."""
    global _TORCH_TO_CUTLASS
    if not _TORCH_TO_CUTLASS:
        import cutlass
        _TORCH_TO_CUTLASS = {
            torch.float8_e4m3fn: cutlass.Float8E4M3FN,
            torch.int8: cutlass.Int8,
            torch.bfloat16: cutlass.BFloat16,
            torch.float16: cutlass.Float16,
            torch.float32: cutlass.Float32,
        }
    return _TORCH_TO_CUTLASS[torch_dtype]


def _opt_level() -> int:
    """Select nvcc optimization level for CuTe DSL compilation."""
    try:
        from cutlass import CUDA_VERSION
        if (CUDA_VERSION.major < 13
                or (CUDA_VERSION.major == 13 and CUDA_VERSION.minor < 1)):
            return 3
        return 2
    except ImportError:
        return 3


# Kernel compilation helpers
def _compile_kernel(
    is_fp8: bool,
    tile_mn: tuple[int, int],
    cluster_mn: tuple[int, int],
    template_m: int,
    template_n: int,
    template_k: int,
    out_dtype_torch: torch.dtype,
):
    """JIT-compile a CuTe DSL scaled MM kernel and return the callable."""
    if is_fp8:
        ab_dtype = cutlass.Float8E4M3FN
        acc_dtype = cutlass.Float32
        KernelClass = ScaledMmSm90Fp8Kernel
    else:
        ab_dtype = cutlass.Int8
        acc_dtype = cutlass.Int32
        KernelClass = ScaledMmSm90Int8Kernel

    c_dtype = _cutlass_dtype(out_dtype_torch)
    scale_dtype = cutlass.Float32

    # Create fake template tensors for compilation with is_dynamic_layout=True
    # so the compiled kernel works for any M, N, K at runtime.
    # m = max(template_m, tile_mn[0])
    # n = max(template_n, tile_mn[1])
    # k = max(template_k, 128)
    # l = 1

    m = cute.sym_int()
    n = cute.sym_int(divisibility=16)
    k = cute.sym_int(divisibility=16)
    l = cute.sym_int()

    # Contiguous on K
    fake_a = make_fake_compact_tensor(
        ab_dtype, (l, m, k), stride_order=(2, 1, 0), assumed_align=16
    )
    # Contiguous on N
    fake_b = make_fake_compact_tensor(
        ab_dtype, (l, k, n), stride_order=(2, 1, 0), assumed_align=16
    )
    # Contiguous on N
    fake_c = make_fake_compact_tensor(
        c_dtype, (l, m, n), stride_order=(2, 1, 0), assumed_align=16
    )

    fake_sa = make_fake_compact_tensor(
        scale_dtype, (1, m, n), stride_order=(2, 1, 0), assumed_align=16
    )
    fake_sb = make_fake_compact_tensor(
        scale_dtype, (1, 1, 1), stride_order=(2, 1, 0), assumed_align=16
    )

    fake_stream = make_fake_stream(use_tvm_ffi_env_stream=True)

    scaled_mm = KernelClass(
        acc_dtype=acc_dtype,
        tile_shape_mn=tile_mn,
        cluster_shape_mn=cluster_mn,
    )
    compiled_fn = cute.compile(
        scaled_mm, fake_a, fake_b, fake_c, fake_sa, fake_sb, fake_stream,
        options=f"--opt-level {_opt_level()} --enable-tvm-ffi --ptxas-options -maxrregcount=64"
    )
    return compiled_fn


# PyTorch custom op - raw GEMM only, opaque to torch.compile
@torch.library.custom_op(
    "vllm::cutedsl_scaled_mm",
    mutates_args=[],
    device_types="cuda",
)
def _cutedsl_scaled_mm_op(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Custom op: CuTe DSL Scaled MM with fused scaling.

    Computes ``scale_a * scale_b * (A @ B)``. All scale types (per-tensor,
    per-token, per-channel) are pre-combined into an (M, N) scale tensor
    and fused into the kernel epilogue via partition_C.

    """

    M, K = a.shape
    N = b.shape[1]
    is_fp8 = a.dtype == torch.float8_e4m3fn

    # select tile / cluster config
    if is_fp8:
        from vllm.kernels.quantization.cutedsl.scaled_mm_sm90_fp8 import (
            select_tile_config,
        )
    else:
        from vllm.kernels.quantization.cutedsl.scaled_mm_sm90_int8 import (
            select_tile_config,
        )
    tile_mn, cluster_mn = select_tile_config(M, N, K)

    # get or compile kernel
    cache_key = (tile_mn, cluster_mn, a.dtype, out_dtype)
    compiled = _kernel_cache.get_or_compile(
        cache_key,
        lambda: _compile_kernel(
            is_fp8, tile_mn, cluster_mn, M, N, K, out_dtype,
        ),
    )

    # pre-combine scales into (M, N) for fused epilogue
    # Pad to tile boundaries so partial-tile element reads don't go OOB.
    tile_m, tile_n = tile_mn
    padded_M = ((M + tile_m - 1) // tile_m) * tile_m
    padded_N = ((N + tile_n - 1) // tile_n) * tile_n
    if padded_M == M and padded_N == N:
        scale_combined = (
            scale_a.float() * scale_b.float()
        ).expand(M, N).contiguous()
    else:
        scale_combined = torch.ones(
            padded_M, padded_N, device=a.device, dtype=torch.float32)
        scale_combined[:M, :N] = (
            scale_a.float() * scale_b.float()
        ).expand(M, N)
    scale_combined_3d = scale_combined.unsqueeze(-1)
    dummy_sb = torch.ones(1, 1, 1, device=a.device, dtype=torch.float32)

    # from_dlpack requires the current CUDA device to match the tensor's
    # device, so set it explicitly for multi-GPU correctness.
    with torch.cuda.device(a.device):
        # zero-copy wrap inputs as dynamic-layout CuTe tensors
        # Kernel expects 3D: A (M,K,1), B (N,K,1), C (M,N,1), scale (M,N,1).
        # b is (K,N) column-major from weight loading; b.t() yields a
        # contiguous (N,K) view — no data copy.  unsqueeze is also a view.
        a_3d = a.contiguous().unsqueeze(-1)
        b_3d = b.t().unsqueeze(-1)
        c_3d = torch.empty(M, N, 1, dtype=out_dtype, device=a.device)

        # launch kernel — use CUDA events for true GPU execution time
        torch_stream = torch.cuda.current_stream()
        with torch.cuda.stream(torch_stream):
            compiled(a_3d, b_3d, c_3d, scale_combined_3d, dummy_sb, torch_stream)

    return c_3d.squeeze(-1)


@torch.library.register_fake("vllm::cutedsl_scaled_mm")
def _cutedsl_scaled_mm_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Fake impl for torch.compile shape inference."""
    return torch.empty(
        a.shape[0], b.shape[1],
        dtype=out_dtype,
        device=a.device,
    )


def cutedsl_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Run scaled GEMM using the CuTe DSL kernel.

    This is a drop-in replacement for the ``torch.ops._C.cutlass_scaled_mm``
    path inside ``vllm._custom_ops.cutlass_scaled_mm``.

    Supports per-tensor, per-token (row-wise), and per-channel (column-wise)
    scales.  The dispatch layer pre-combines ``scale_a * scale_b`` into an
    ``(M, N)`` tensor fused into the kernel epilogue via partition_C.

    Returns ``None`` if the inputs are not supported (unsupported dtype,
    non-Hopper GPU, missing dependencies).

    Args:
        a: (M, K) FP8-e4m3 or INT8 input, contiguous, on CUDA.
        b: (K, N) FP8-e4m3 or INT8 weight, column-major, on CUDA.
        scale_a: FP32 scale for *a* -- scalar (per-tensor) or (M, 1)
            (per-token).
        scale_b: FP32 scale for *b* -- scalar (per-tensor) or (N, 1) / (1, N)
            (per-channel).
        out_dtype: ``torch.bfloat16`` or ``torch.float16``.
        bias: optional per-channel bias tensor of shape ``(N,)`` in
            ``out_dtype``.

    Returns:
        ``(M, N)`` output tensor, or ``None`` on unsupported config.
    """

    # dtype must be FP8 or INT8
    is_fp8 = a.dtype == torch.float8_e4m3fn
    is_int8 = a.dtype == torch.int8
    if not (is_fp8 or is_int8):
        return None

    # raw GEMM via custom op (opaque to torch.compile)
    # All scale types (scalar, per-token, per-channel) are fused into
    # the kernel epilogue.
    raw_output = torch.ops.vllm.cutedsl_scaled_mm(
        a, b, scale_a, scale_b, out_dtype)

    result = raw_output
    if bias is not None:
        result = result + bias

    return result
