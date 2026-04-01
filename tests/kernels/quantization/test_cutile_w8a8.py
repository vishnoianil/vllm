import pytest
import torch
import vllm.kernels.cutile.cutile_w8a8  # noqa: F401
# -------------------
# run this test with  python -m pytest -s tests/kernels/quantization/test_cutile_w8a8.py
# -------------------
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
from tests.kernels.quant_utils import native_w8a8_block_matmul
from vllm.benchmarks.lib.utils import default_vllm_config

@pytest.mark.parametrize("out_dtype", [
    torch.bfloat16,
    torch.float32,
])
@pytest.mark.parametrize("M, N, K", [
      (128, 128, 128),
      (256, 256, 256),
      (512, 512, 512),
      (128, 4096, 4096),
      (256, 4096, 4096),
      (512, 4096, 4096),
      (1024, 4096, 4096),
      (128, 14336, 4096),
      (256, 14336, 4096),
      (128, 4096, 14336),
      (128, 512, 7168),
      (256, 576, 4096),
      (384, 1000, 2048),
])
@pytest.mark.parametrize("use_bias", [False])
def test_cutile_blockwise_fp8_kernel(out_dtype, use_bias, M, N, K, default_vllm_config):
    torch.set_default_device("cuda")

    block_size = [128, 128]
    seed = 0

    torch.manual_seed(seed)
    factor_for_scale = 1e-2
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    A_fp32 = (torch.rand(M, K, dtype=torch.float32) - 0.5) * 100
    A_fp8 = A_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    B_fp32 = (torch.rand(N, K, dtype=torch.float32) - 0.5) * 100
    B_fp8 = B_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    block_n, block_k = block_size[0], block_size[1]
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k

    As = torch.rand(M, k_tiles, dtype=torch.float32) * factor_for_scale
    Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32) * factor_for_scale

    if use_bias:
        bias = torch.rand(N, dtype=torch.float32) * 0.001
    else:
        bias = None

    ref_out = native_w8a8_block_matmul(A_fp8, B_fp8, As, Bs, block_size, out_dtype)
    if use_bias:
        ref_out += bias

    out = torch.empty((M, N), dtype=out_dtype, device='cuda')
    torch.ops.vllm.cutile_scaled_mm(out, A_fp8, B_fp8.t(), As, Bs.t(), bias)

    rel_diff = torch.mean(torch.abs(out.to(out_dtype) - ref_out.to(out_dtype))) / torch.mean(torch.abs(ref_out.to(out_dtype)))
    assert rel_diff < 0.001
    torch.testing.assert_close(out, ref_out, rtol=1e-2, atol=1e-2)