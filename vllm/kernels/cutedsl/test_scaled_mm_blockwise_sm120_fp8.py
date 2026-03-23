"""
Simple unit test for the BlockwiseGemmKernel in scaled_mm_blockwise_sm100_fp8.py.

Validates correctness by running the CuTe DSL kernel and comparing against a
PyTorch reference: C = (SFA * A) @ (SFB * B)^T  (einsum "mkl,nkl->mnl")

Run:
    python vllm/kernels/cutedsl/test_scaled_mm_blockwise_sm100_fp8.py
"""

import math
from typing import Tuple

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch

from scaled_mm_blockwise_sm100_fp8 import BlockwiseGemmKernel


def create_tensors(
        l, m, n, k, a_major, b_major, cd_major, ab_dtype, c_dtype, scale_dtype):

    """Create random input/output tensors on CPU and corresponding CuTe GPU tensors."""
    torch.manual_seed(1111)

    a_torch_cpu = cutlass_torch.matrix(l, m, k, a_major == "m", ab_dtype)
    b_torch_cpu = cutlass_torch.matrix(l, n, k, b_major == "n", ab_dtype)
    c_torch_cpu = cutlass_torch.matrix(l, m, n, cd_major == "m", c_dtype)
    sfa_torch_cpu = cutlass_torch.matrix(l, m, math.ceil(k / 128), True, scale_dtype)
    sfb_torch_cpu = cutlass_torch.matrix(
        l, math.ceil(n / 128), math.ceil(k / 128), False, scale_dtype
    )

    a_tensor, _ = cutlass_torch.cute_tensor_like(
        a_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    b_tensor, _ = cutlass_torch.cute_tensor_like(
        b_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    c_tensor, c_torch_gpu = cutlass_torch.cute_tensor_like(
        c_torch_cpu, c_dtype, is_dynamic_layout=True, assumed_align=16
    )
    sfa_tensor, _ = cutlass_torch.cute_tensor_like(
        sfa_torch_cpu, scale_dtype, is_dynamic_layout=True, assumed_align=16
    )
    sfb_tensor, _ = cutlass_torch.cute_tensor_like(
        sfb_torch_cpu, scale_dtype, is_dynamic_layout=True, assumed_align=16
    )

    return (
        a_tensor, b_tensor, c_tensor, sfa_tensor, sfb_tensor,
        a_torch_cpu, b_torch_cpu, c_torch_cpu, sfa_torch_cpu, sfb_torch_cpu,
        c_torch_gpu,
    )


def compute_reference(a_torch_cpu, b_torch_cpu, sfa_torch_cpu, sfb_torch_cpu, c_dtype):
    """Compute blockwise-scaled GEMM reference on CPU using PyTorch."""

    def pad_and_multiply(scale, tensor):
        cm, ck, _ = scale.shape
        m, k, _ = tensor.shape
        is_groupwise = (ck == math.ceil(k / 128))
        is_blockwise = (cm == math.ceil(m / 128))
        if not is_blockwise and not is_groupwise:
            raise ValueError("Only support granularity = 128")

        k_idx = torch.arange(k, device=scale.device)
        if is_groupwise:
            k_idx = k_idx // 128
        m_idx = torch.arange(m, device=scale.device)
        if is_blockwise:
            m_idx = m_idx // 128
        expanded_scale = scale[m_idx[:, None], k_idx, :]
        return expanded_scale * tensor

    updated_a = pad_and_multiply(sfa_torch_cpu, a_torch_cpu)
    updated_b = pad_and_multiply(sfb_torch_cpu, b_torch_cpu)
    return torch.einsum("mkl,nkl->mnl", updated_a, updated_b).to(
        cutlass_torch.dtype(c_dtype)
    )


def run_test(
    mnkl: Tuple[int, int, int, int],
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    use_2cta_instrs: bool = False,
    tolerance: float = 1e-01,
) -> bool:
    """Run a single GEMM test case and return True if it passes."""
    ab_dtype = cutlass.Float8E4M3FN
    c_dtype = cutlass.BFloat16
    acc_dtype = cutlass.Float32
    scale_dtype = cutlass.Float32
    a_major, b_major, c_major = "k", "k", "n"

    m, n, k, l = mnkl

    (
        a_tensor, b_tensor, c_tensor, sfa_tensor, sfb_tensor,
        a_torch_cpu, b_torch_cpu, c_torch_cpu, sfa_torch_cpu, sfb_torch_cpu,
        c_torch_gpu,
    ) = create_tensors(l, m, n, k, a_major, b_major, c_major, ab_dtype, c_dtype, scale_dtype)

    # Create and compile kernel
    gemm = BlockwiseGemmKernel(acc_dtype, use_2cta_instrs, mma_tiler_mn, cluster_shape_mn)

    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    try:
        from cutlass import CUDA_VERSION
        opt_level = (
            3 if CUDA_VERSION.major < 13
            or (CUDA_VERSION.major == 13 and CUDA_VERSION.minor < 1)
            else 2
        )
    except ImportError:
        opt_level = 3

    compiled_gemm = cute.compile(
        gemm,
        a_tensor, b_tensor, c_tensor, sfa_tensor, sfb_tensor,
        max_active_clusters, current_stream,
        options=f"--opt-level {opt_level}",
    )

    # Execute
    compiled_gemm(
        a_tensor, b_tensor, c_tensor, sfa_tensor, sfb_tensor,
        current_stream,
    )
    torch.cuda.synchronize()

    # Reference
    ref = compute_reference(a_torch_cpu, b_torch_cpu, sfa_torch_cpu, sfb_torch_cpu, c_dtype)
    res = c_torch_gpu.view(cutlass_torch.dtype(c_dtype))

    # Compare
    try:
        torch.testing.assert_close(res.cpu(), ref.cpu(), atol=tolerance, rtol=1e-03)
        return True
    except AssertionError as e:
        max_diff = (res.cpu().float() - ref.cpu().float()).abs().max().item()
        print(f"    max_diff={max_diff:.6f}")
        print(f"    {e}")
        return False


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required to run this test.")

    test_cases = [
        # (mnkl, mma_tiler_mn, cluster_shape_mn, use_2cta_instrs)
        ((256, 256, 512, 1), (128, 128), (1, 1), False),
        ((512, 512, 512, 1), (128, 128), (1, 2), False),
        ((256, 256, 256, 1), (64, 128), (1, 1), False),
    ]

    passed = 0
    total = len(test_cases)

    for mnkl, mma_tiler_mn, cluster_shape_mn, use_2cta in test_cases:
        label = (
            f"mnkl={mnkl}, tile={mma_tiler_mn}, "
            f"cluster={cluster_shape_mn}, 2cta={use_2cta}"
        )
        print(f"Testing {label} ... ", end="", flush=True)
        try:
            ok = run_test(mnkl, mma_tiler_mn, cluster_shape_mn, use_2cta)
        except Exception as e:
            print(f"ERROR: {e}")
            continue
        if ok:
            print("PASS")
            passed += 1
        else:
            print("FAIL")

    print(f"\n{passed}/{total} tests passed.")
    if passed == total:
        print("ALL TESTS PASSED")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
