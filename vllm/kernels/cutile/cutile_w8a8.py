import json
import os
from functools import lru_cache

import vllm
import torch
import cuda.tile as ct
from cuda.tile import kernel, ByTarget
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

ConstInt = ct.Constant[int]
DEVICE_NAME = current_platform.get_device_name().lower().replace(" ", "_")


@lru_cache(maxsize=1)
def _load_master_json():
    vllm_root = os.path.dirname(vllm.__file__)
    config_path = os.path.join(
        vllm_root, "model_executor", "layers", "quantization", "utils", "configs", "cutile_w8a8.json"
    )
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


_CONFIG_INDEX = {}

def initialize_registry():
    global _CONFIG_INDEX
    raw_data = _load_master_json() # Your existing JSON loader

    for device, configs in raw_data.items():
        dev_key = device.lower().replace(" ", "_")
        for key, data in configs.items():
            # Index by the exact key: (device, "mperrank_1_n_...")
            _CONFIG_INDEX[(dev_key, key)] = data

            # Index by the shape for fallback: (device, N, K)
            parts = key.split("_")
            if len(parts) >= 6:
                try:
                    n_val, k_val = int(parts[3]), int(parts[5])
                    _CONFIG_INDEX[(dev_key, n_val, k_val)] = data
                except: continue

# Initialize immediately
initialize_registry()

# This function,(adapted from triton/cutile) maps a linear Block ID (bid)
# to a 2D tile coordinate (bid_m, bid_n).
# We group tiles along the M dimension to optimize memory access patterns:
#   - In Matrix A: Each block handles a specific row-tile strip [bid_m, :].
#   - In Matrix B: Multiple blocks in a group share the same column-tile [:, bid_n].
# So instead of loading the whole matrix B to compute a single row of matrix A,
# we process a group of M-rows together to achieve the same N elements
# with better data reuse from the L2 cache.
def map_block_to_tile_grouped(M, N, tm, tn, GROUP_SIZE_M):
    bid = ct.bid(0) # block id
    num_bid_m = ct.cdiv(M, tm)
    num_bid_n = ct.cdiv(N, tn)

    num_tiles_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_tiles_in_group

    first_tile_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_bid_m - first_tile_m, GROUP_SIZE_M)

    bid_m = first_tile_m + ((bid % num_tiles_in_group) % group_size_m)
    bid_n = (bid % num_tiles_in_group) // group_size_m

    return bid_m, bid_n


@ct.kernel(num_ctas=ct.ByTarget(sm_100=8))
def matmul_kernel(A, B, As, Bs, out,
                                 M: ConstInt, N: ConstInt, K: ConstInt,
                                 TILE_M: ConstInt, TILE_N: ConstInt, TILE_K: ConstInt,
                                 GROUP_SIZE_M:ConstInt = 1 ):

    bid_m, bid_n = map_block_to_tile_grouped(M, N, TILE_M, TILE_N, GROUP_SIZE_M)
    num_tiles_k = ct.cdiv(K, TILE_K)

    acc = ct.zeros((TILE_M, TILE_N), dtype=torch.float32)

    for k_idx in range(num_tiles_k):
        a_tile = ct.load(A, index=(bid_m, k_idx), shape=(TILE_M, TILE_K))
        b_tile = ct.load(B, index=(k_idx, bid_n), shape=(TILE_K, TILE_N))

        a_scale = ct.load(As, index=(bid_m, k_idx), shape=(TILE_M, 1))
        b_scale = ct.load(Bs, index=(k_idx, bid_n), shape=(1, 1))

        dot_prod = ct.mma(a_tile, b_tile, ct.zeros((TILE_M, TILE_N), dtype=torch.float32))

        acc += dot_prod * (a_scale * b_scale)

    ct.store(out, index=(bid_m, bid_n), tile=ct.astype(acc, out.dtype))

@ct.kernel(num_ctas=ct.ByTarget(sm_100=8))
def matmul_kernel_use_bias(A, B, As, Bs, out, bias,
                                 M: ConstInt, N: ConstInt, K: ConstInt,
                                 TILE_M: ConstInt, TILE_N: ConstInt, TILE_K: ConstInt,
                                 GROUP_SIZE_M:ConstInt = 1):

    bid_m, bid_n = map_block_to_tile_grouped(M, N, TILE_M, TILE_N, GROUP_SIZE_M)
    num_tiles_k = ct.cdiv(K, TILE_K)

    acc = ct.zeros((TILE_M, TILE_N), dtype=torch.float32)

    for k_idx in range(num_tiles_k):
        a_tile = ct.load(A, index=(bid_m, k_idx), shape=(TILE_M, TILE_K))
        b_tile = ct.load(B, index=(k_idx, bid_n), shape=(TILE_K, TILE_N))

        a_scale = ct.load(As, index=(bid_m, k_idx), shape=(TILE_M, 1))
        b_scale = ct.load(Bs, index=(k_idx, bid_n), shape=(1, 1))

        dot_prod = ct.mma(a_tile, b_tile, ct.zeros((TILE_M, TILE_N), dtype=torch.float32))

        acc += dot_prod * (a_scale * b_scale)

    bias_tile = ct.load(bias, index=(0, bid_n), shape=(1, TILE_N))
    acc += bias_tile
    ct.store(out, index=(bid_m, bid_n), tile=ct.astype(acc, out.dtype))

def _get_sm_count() -> int:
    """Get the number of streaming multiprocessors on the current GPU."""
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return props.multi_processor_count


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

def _select_tile_config_sm100(M: int, N: int, K: int) -> tuple[int, int, int, int]:
    """Select tile config for SM100 (Blackwell).

    Based on scaled_mm_sm100_fp8_dispatch.cuh tile selection.
    CuTile does not support swap_ab, so for small M we use the
    non-swapped configs closest to the CUTLASS choices.

    CUTLASS SM100 per-tensor FP8 configs:
      M in [1, 16]           -> 128x32x128  (swap_ab) -> use 64x64x128
      M in (16, 64], K>=4096 -> 128x64x256  (swap_ab) -> use 64x64x128
      M = 64, K < 4096       -> 64x64x128   (no swap)
      M in (64, 256]         -> 128x128x128  (no swap)
      M in (256, inf)        -> 256x128x128  (no swap)
    """
    if M <= 64:
        # Small M: CUTLASS uses swap_ab here, but CuTile cannot.
        # Use 64x64x128, the non-swapped config for M=64.
        TILE_M, TILE_N, TILE_K = 64, 64, 128
        GROUP_SIZE_M = 8
    elif M <= 256:
        # Medium M
        TILE_M, TILE_N, TILE_K = 128, 128, 128
        GROUP_SIZE_M = 4
    else:
        # Large M
        TILE_M, TILE_N, TILE_K = 256, 128, 128
        GROUP_SIZE_M = 1

    return (TILE_M, TILE_N, TILE_K, GROUP_SIZE_M)


def get_tile_config(M: int, N: int, K: int) -> tuple[int, int, int, int]:
    """Look up tuned tile config, falling back to SM-aware heuristics.

    Returns:
        (TILE_M, TILE_N, TILE_K, GROUP_SIZE_M)
    """
    cap = current_platform.get_device_capability()
    if cap is not None:
        sm_version = cap.major * 10 + cap.minor
        if sm_version >= 100:
            return _select_tile_config_sm100(M, N, K)

    # Default fallback
    return (128, 128, 128, 1)


def cutile_blockwise_mm(out: torch.Tensor, A: torch.Tensor, B: torch.Tensor, As: torch.Tensor, Bs: torch.Tensor, bias: torch.Tensor = None)-> torch.Tensor:
    """
    A: (M, K) in fp8, row-major (stride: (K, 1))
    B: (K, N) in fp8, col-major (stride: (1, K))
    As(A_scale): (M, k_tiles) , col-major (stride: (1, M))
    Bs: (k_tiles, n_tiles), col-major (stride: (1, k_tiles))
    Out: (M, N) in out_dtype, row-major (stride: (N, 1))
    """

    assert As.dtype == torch.float32, "As must be float32"
    assert Bs.dtype == torch.float32, "Bs must be float32"

    # Ensure scale tensors are 2D (can be 1D when n_tiles or k_tiles == 1)
    if Bs.ndim == 1:
        Bs = Bs.unsqueeze(1)
    if As.ndim == 1:
        As = As.unsqueeze(1)

    M, K = A.shape
    K_check, N = B.shape

    assert K == K_check, f"Inner dimension mismatch: A_K={K}, B_K={K_check}"

    TILE_M, TILE_N, TILE_K, GROUP_SIZE_M = get_tile_config(M, N, K)

    grid_m = ct.cdiv(M, TILE_M)
    grid_n = ct.cdiv(N, TILE_N)
    grid_1d = (grid_m * grid_n, 1, 1)

    stream_ptr = torch.cuda.current_stream().cuda_stream
    if bias is not None:
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)  # shape become(1, N)
        assert bias.shape[1] == N, f"Bias shape {bias.shape} doesn't match N={N}"
        assert bias.dtype == out.dtype, f"Bias dtype {bias.dtype} must match out_dtype {out.dtype}"
        ct.launch(stream_ptr, grid_1d, matmul_kernel_use_bias,
              (A, B, As, Bs, out, bias, M, N, K, TILE_M, TILE_N, TILE_K, GROUP_SIZE_M))
    else:
        ct.launch(stream_ptr, grid_1d, matmul_kernel,
              (A, B, As, Bs, out, M, N, K, TILE_M, TILE_N, TILE_K, GROUP_SIZE_M))
    return out


def cutile_blockwise_mm_fake(out: torch.Tensor, A: torch.Tensor, B: torch.Tensor, As: torch.Tensor, Bs: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    return out


direct_register_custom_op(
    op_name="cutile_scaled_mm",
    op_func=cutile_blockwise_mm,
    mutates_args=["out"],
    fake_impl=cutile_blockwise_mm_fake,
)
