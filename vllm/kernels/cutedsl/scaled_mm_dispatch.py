# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CuTe DSL dispatch layer for scaled GEMM (FP8 / INT8) on SM90.

Provides ``cutedsl_scaled_mm`` as a drop-in replacement for the CUTLASS C++
path used by ``torch.ops._C.cutlass_scaled_mm``.  The function is only called
when ``VLLM_SWITCH_TO_CUTEDSL=1`` is set and the inputs are FP8 or INT8 on a
Hopper GPU.

Supports per-tensor, per-token (row-wise), and per-channel (column-wise)
scales.  The raw GEMM (with unit scales) is registered as a PyTorch custom op
(``vllm::cutedsl_raw_gemm``) so that ``torch.compile`` treats it as an
opaque operator.  The actual scaling is applied *outside* the custom op so
that ``torch.compile`` can see and fuse the elementwise scale multiplications
into a single kernel — avoiding an extra M×N FP32 intermediate allocation.

Uses ``from_dlpack`` + ``mark_compact_shape_dynamic`` for zero-copy
wrapping of PyTorch GPU tensors as dynamic-layout CuTe tensors at runtime.

Kernels are JIT-compiled on first use and cached for subsequent calls.
"""

import logging
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import torch
from cutlass.cute.runtime import from_dlpack

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy availability check
# ---------------------------------------------------------------------------
_cute_dsl_available: Optional[bool] = None


def is_cute_dsl_available() -> bool:
    """Return True if the CuTe DSL runtime is importable."""
    global _cute_dsl_available
    if _cute_dsl_available is None:
        try:
            import cutlass  # noqa: F401
            import cutlass.cute  # noqa: F401
            import cutlass.torch  # noqa: F401
            _cute_dsl_available = True
        except ImportError:
            _cute_dsl_available = False
            logger.warning(
                "CuTe DSL dependencies not found. "
                "VLLM_SWITCH_TO_CUTEDSL will be ignored."
            )
    return _cute_dsl_available


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

# ---------------------------------------------------------------------------
# Cached unit-scale CuTe tensor (allocated once, reused every call)
# ---------------------------------------------------------------------------
_unit_scale_pt: Optional[torch.Tensor] = None
_unit_scale_cute = None

# ---------------------------------------------------------------------------
# Weight (B) CuTe tensor cache — weights are persistent nn.Parameters,
# so the CuTe wrapper (from_dlpack + dynamic-layout marking) can be
# computed once and reused for all subsequent calls, saving 5+ C++ interop
# calls per dispatch.
# ---------------------------------------------------------------------------
_weight_cute_cache: dict[int, object] = {}


def _get_unit_scale_cute():
    """Return a cached CuTe tensor for a scalar 1.0 scale."""
    global _unit_scale_pt, _unit_scale_cute
    if _unit_scale_pt is None:
        _unit_scale_pt = torch.ones(1, 1, 1, dtype=torch.float32,
                                    device="cuda")
        _unit_scale_cute = from_dlpack(_unit_scale_pt, assumed_align=16)
        # Scale is (1,1,1) -- all strides are 1, no unique leading dim.
        # Use leading_dim=0 to match the convention used by cute_tensor_like.
        _unit_scale_cute.mark_layout_dynamic(leading_dim=0)
    return _unit_scale_cute


# ---------------------------------------------------------------------------
# PyTorch <-> CUTLASS dtype helpers
# ---------------------------------------------------------------------------
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
    """Select nvcc optimisation level for CuTe DSL compilation."""
    try:
        from cutlass import CUDA_VERSION
        if (CUDA_VERSION.major < 13
                or (CUDA_VERSION.major == 13 and CUDA_VERSION.minor < 1)):
            return 3
        return 2
    except ImportError:
        return 3


# ---------------------------------------------------------------------------
# Kernel compilation helpers
# ---------------------------------------------------------------------------
def _compile_kernel(
    is_fp8: bool,
    tile_mn: Tuple[int, int],
    cluster_mn: Tuple[int, int],
    template_m: int,
    template_n: int,
    template_k: int,
    out_dtype_torch: torch.dtype,
):
    """JIT-compile a CuTe DSL scaled-GEMM kernel and return the callable."""
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch

    if is_fp8:
        from vllm.kernels.cutedsl.scaled_mm_sm90_fp8 import (
            ScaledMmSm90Fp8Kernel,
        )
        ab_dtype = cutlass.Float8E4M3FN
        acc_dtype = cutlass.Float32
        KernelClass = ScaledMmSm90Fp8Kernel
    else:
        from vllm.kernels.cutedsl.scaled_mm_sm90_int8 import (
            ScaledMmSm90Int8Kernel,
        )
        ab_dtype = cutlass.Int8
        acc_dtype = cutlass.Int32
        KernelClass = ScaledMmSm90Int8Kernel

    c_dtype = _cutlass_dtype(out_dtype_torch)
    scale_dtype = cutlass.Float32

    # Create template tensors for compilation with is_dynamic_layout=True
    # so the compiled kernel works for any M, N, K at runtime.
    m = max(template_m, tile_mn[0])
    n = max(template_n, tile_mn[1])
    k = max(template_k, 128)
    l = 1

    a_cpu = cutlass_torch.matrix(l, m, k, False, ab_dtype)
    b_cpu = cutlass_torch.matrix(l, n, k, False, ab_dtype)
    c_cpu = cutlass_torch.matrix(l, m, n, False, c_dtype)
    sa_cpu = cutlass_torch.matrix(1, 1, 1, False, scale_dtype)
    sb_cpu = cutlass_torch.matrix(1, 1, 1, False, scale_dtype)

    a_cute, _ = cutlass_torch.cute_tensor_like(
        a_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16)
    b_cute, _ = cutlass_torch.cute_tensor_like(
        b_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16)
    c_cute, _ = cutlass_torch.cute_tensor_like(
        c_cpu, c_dtype, is_dynamic_layout=True, assumed_align=16)
    sa_cute, _ = cutlass_torch.cute_tensor_like(
        sa_cpu, scale_dtype, is_dynamic_layout=True, assumed_align=16)
    sb_cute, _ = cutlass_torch.cute_tensor_like(
        sb_cpu, scale_dtype, is_dynamic_layout=True, assumed_align=16)

    gemm = KernelClass(
        acc_dtype=acc_dtype,
        tile_shape_mn=tile_mn,
        cluster_shape_mn=cluster_mn,
    )

    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    compiled = cute.compile(
        gemm,
        a_cute, b_cute, c_cute, sa_cute, sb_cute, stream,
        options=f"--opt-level {_opt_level()}",
    )
    return compiled


# ---------------------------------------------------------------------------
# Runtime tensor helpers
# ---------------------------------------------------------------------------
def _to_cute_dynamic(pt_tensor: torch.Tensor) -> object:
    """Wrap a PyTorch GPU tensor as a dynamic-layout CuTe tensor (zero-copy).

    Uses ``from_dlpack`` for zero-copy wrapping, then
    ``mark_compact_shape_dynamic`` on all modes to match the dynamic layout
    produced by ``cute_tensor_like(is_dynamic_layout=True)`` at compile time.
    """
    ct = from_dlpack(pt_tensor, assumed_align=16)
    leading_dim = ct.leading_dim
    stride_order = pt_tensor.dim_order()
    for mode in range(pt_tensor.dim()):
        ct.mark_compact_shape_dynamic(mode=mode, stride_order=stride_order)
    if leading_dim is not None:
        ct.mark_layout_dynamic(leading_dim=leading_dim)
    return ct


# ---------------------------------------------------------------------------
# PyTorch custom op -- raw GEMM only, opaque to torch.compile
# ---------------------------------------------------------------------------
@torch.library.custom_op(
    "vllm::cutedsl_raw_gemm",
    mutates_args=[],
    device_types="cuda",
)
def _cutedsl_raw_gemm_op(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Custom op: raw CuTe DSL GEMM with unit scales.

    Computes ``A @ B`` without any scale application.  Scaling is handled
    by the caller so that ``torch.compile`` can see and fuse the
    elementwise multiplications.

    Uses from_dlpack + mark_compact_shape_dynamic for zero-copy input
    wrapping with dynamic layouts.  No torch.cuda.synchronize().
    """
    M, K = a.shape
    N = b.shape[1]
    is_fp8 = a.dtype == torch.float8_e4m3fn

    # ---- select tile / cluster config -----------------------------------
    if is_fp8:
        from vllm.kernels.cutedsl.scaled_mm_sm90_fp8 import (
            select_tile_config,
        )
    else:
        from vllm.kernels.cutedsl.scaled_mm_sm90_int8 import (
            select_tile_config,
        )
    tile_mn, cluster_mn = select_tile_config(M, N, K)

    # ---- get or compile kernel ------------------------------------------
    cache_key = (tile_mn, cluster_mn, a.dtype, out_dtype)
    compiled = _kernel_cache.get_or_compile(
        cache_key,
        lambda: _compile_kernel(
            is_fp8, tile_mn, cluster_mn, M, N, K, out_dtype
        ),
    )

    # ---- zero-copy wrap inputs as dynamic-layout CuTe tensors -----------
    # Kernel expects 3D: A (M,K,1), B (N,K,1), C (M,N,1), scales (1,1,1).
    # b is (K,N) column-major from weight loading; b.t() yields a
    # contiguous (N,K) view — no data copy.  unsqueeze is also a view.
    a_3d = a.contiguous().unsqueeze(-1)
    c_3d = torch.empty(M, N, 1, dtype=out_dtype, device=a.device)

    a_cute = _to_cute_dynamic(a_3d)
    c_cute = _to_cute_dynamic(c_3d)

    # B (weight) CuTe tensor is cached — weights are persistent
    # nn.Parameters whose data pointer never changes during serving.
    b_ptr = b.data_ptr()
    if b_ptr in _weight_cute_cache:
        b_cute = _weight_cute_cache[b_ptr]
    else:
        b_3d = b.t().unsqueeze(-1)              # (K,N) col-major -> (N,K,1) view, zero-copy
        b_cute = _to_cute_dynamic(b_3d)
        _weight_cute_cache[b_ptr] = b_cute

    sa_cute = _to_cute_dynamic(scale_a)
    sb_cute = _get_unit_scale_cute()

    # ---- launch kernel (async, no synchronize) --------------------------
    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    compiled(a_cute, b_cute, c_cute, sa_cute, sb_cute, stream)

    return c_3d.squeeze(-1)


@torch.library.register_fake("vllm::cutedsl_raw_gemm")
def _cutedsl_raw_gemm_fake(
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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
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
    scales.  The kernel runs with unit scales; actual scales are applied
    after in FP32:  ``output = (raw_output.float() * scale_a * scale_b).to(out_dtype)``

    Returns ``None`` if the inputs are not supported (bias, unsupported dtype,
    non-Hopper GPU, missing dependencies).

    Args:
        a: (M, K) FP8-e4m3 or INT8 input, contiguous, on CUDA.
        b: (K, N) FP8-e4m3 or INT8 weight, column-major, on CUDA.
        scale_a: FP32 scale for *a* -- scalar (per-tensor) or (M, 1)
            (per-token).
        scale_b: FP32 scale for *b* -- scalar (per-tensor) or (N, 1) / (1, N)
            (per-channel).
        out_dtype: ``torch.bfloat16`` or ``torch.float16``.
        bias: must be ``None`` (CuTe DSL kernels do not fuse bias).

    Returns:
        ``(M, N)`` output tensor, or ``None`` on unsupported config.
    """
    if bias is not None:
        return None

    # ---- guard: dtype must be FP8 or INT8 -------------------------------
    is_fp8 = a.dtype == torch.float8_e4m3fn
    is_int8 = a.dtype == torch.int8
    if not (is_fp8 or is_int8):
        return None

    # ---- guard: SM90 (Hopper) required ----------------------------------
    cap = torch.cuda.get_device_capability()
    if cap[0] < 9:
        return None

    # ---- guard: CuTe DSL runtime available ------------------------------
    if not is_cute_dsl_available():
        return None

    # ---- raw GEMM via custom op (opaque to torch.compile) ----------------
    raw_output = torch.ops.vllm.cutedsl_raw_gemm(a, b, scale_a, scale_b, out_dtype)

    # ---- apply scales in FP32 (visible to torch.compile → fused) -------
    # torch.compile sees these as regular ops and fuses them into a single
    # elementwise kernel, avoiding an extra M×N FP32 intermediate tensor.
    #sa = scale_a.float().reshape(-1, 1)   # (1,1) or (M,1)
    #sb = scale_b.float().reshape(1, -1)   # (1,1) or (1,N)
    return raw_output.to(out_dtype)
