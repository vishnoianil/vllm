"""
CuTe DSL implementation of the scaled FP8 GEMM kernel for NVIDIA Hopper (SM90).

Rewrite of the CUTLASS C++ kernel in
csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm90_fp8_dispatch.cuh

Computes: D = scale_a * scale_b * (A @ B^T)
  - A is MxK  (FP8 e4m3, row-major / K-major)
  - B is NxK  (FP8 e4m3, col-major / K-major)
  - D is MxN  (BF16 or FP16, row-major / N-major)
  - scale_a, scale_b are per-tensor FP32 scalars

This kernel targets the NVIDIA Hopper GPU and uses:
  - Tensor Memory Access (TMA) for global<->shared memory transfers
  - WGMMA (Warp Group MMA) for FP8 matrix multiply-accumulate
  - Multi-stage async pipeline for latency hiding
  - TMA multicast with clusters to reduce L2 traffic
  - Per-tensor scaling applied in the epilogue

Run self-test:
    python vllm/kernels/cutedsl/scaled_mm_sm90_fp8.py
"""

import math
from typing import Tuple, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.hopper_helpers as sm90_utils


class ScaledMmSm90Fp8Kernel:
    """Hopper FP8 scaled GEMM kernel: D = scale_a * scale_b * (A @ B^T).

    Mirrors the tile/cluster configurations from the C++ dispatch in
    scaled_mm_sm90_fp8_dispatch.cuh.

    :param acc_dtype: Accumulator data type (must be Float32).
    :param tile_shape_mn: CTA tile shape (M, N).
    :param cluster_shape_mn: Cluster shape (M, N).
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        tile_shape_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
    ):
        self.acc_dtype = acc_dtype
        self.cluster_shape_mn = cluster_shape_mn
        self.tile_shape_mnk = (*tile_shape_mn, 1)
        self.atom_layout_mnk = (
            (2, 1, 1)
            if self.tile_shape_mnk[0] > 64 and self.tile_shape_mnk[1] > 128
            else (1, 1, 1)
        )
        self.num_mcast_ctas_a = None
        self.num_mcast_ctas_b = None
        self.is_a_mcast = False
        self.is_b_mcast = False
        self.tiled_mma = None

        self.occupancy = 1
        self.mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = self.mma_warp_groups * self.num_threads_per_warp_group
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")

        self.ab_stage = None
        self.epi_stage = None
        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None
        self.shared_storage = None
        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        """Configure kernel attributes from input tensor properties."""
        if self.tile_shape_mnk[0] not in [64, 128, 256]:
            raise ValueError("CTA tile shape M must be 64/128/256")
        if self.tile_shape_mnk[1] not in [16, 64, 128, 256]:
            raise ValueError("CTA tile shape N must be 16/64/128/256")

        mma_tiler_m = min(64, self.tile_shape_mnk[0])
        mma_tiler_n = self.tile_shape_mnk[1]

        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(mma_tiler_m, mma_tiler_n),
        )
        mma_inst_shape_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.tile_shape_mnk = (
            self.tile_shape_mnk[0],
            self.tile_shape_mnk[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )

        self.cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))
        self.num_mcast_ctas_a = self.cluster_shape_mn[1]
        self.num_mcast_ctas_b = self.cluster_shape_mn[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        is_cooperative = self.atom_layout_mnk == (2, 1, 1)
        self.epi_tile = sm90_utils.compute_tile_shape_or_override(
            self.tile_shape_mnk, self.c_dtype, is_cooperative=is_cooperative
        )

        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.smem_capacity,
            self.occupancy,
        )

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        scale_a: cute.Tensor,
        scale_b: cute.Tensor,
        stream: cuda.CUstream,
    ):
        """Launch the scaled FP8 GEMM kernel.

        :param a: Input FP8 tensor A (MxKxL).
        :param b: Input FP8 tensor B (NxKxL).
        :param c: Output tensor D (MxNxL).
        :param scale_a: Per-tensor scale for A (scalar FP32).
        :param scale_b: Per-tensor scale for B (scalar FP32).
        :param stream: CUDA stream.
        """
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        self._setup_attributes()

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            self.cluster_shape_mn[1],
        )
        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            self.cluster_shape_mn[0],
        )
        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
            c,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )

        grid = self._compute_grid(c, self.tile_shape_mnk, self.cluster_shape_mn)

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            scale_a,
            scale_b,
            self.tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )

    # -------------------------------------------------------------------------
    #  GPU device kernel
    # -------------------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        gScaleA: cute.Tensor,
        gScaleB: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
    ):
        """Device kernel: TMA load -> WGMMA mainloop -> scaled epilogue."""
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Prefetch TMA descriptors
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

        # Thread / block / cluster indices
        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        cidx, cidy, _ = cute.arch.cluster_idx()
        cdimx, cdimy, _ = cute.arch.cluster_dim()
        cluster_id = cidx + cdimx * cidy

        # CTA swizzle for L2 reuse
        group_size_m = 8
        s_shape = ((group_size_m, cdimx // group_size_m), cdimy)
        s_stride = ((1, cdimy * group_size_m), group_size_m)
        s_layout = cute.make_layout(s_shape, stride=s_stride)
        num_reg_cids = cute.size(s_shape)
        cid_m, cid_n = s_layout.get_flat_coord(cluster_id % num_reg_cids)

        if cluster_id >= num_reg_cids:
            tail_size_m = cdimx % group_size_m
            tail_layout = cute.make_layout(
                (tail_size_m, cdimy), stride=(1, tail_size_m)
            )
            tail_cid = cluster_id - num_reg_cids
            tail_cid_m, tail_cid_n = tail_layout.get_flat_coord(tail_cid)
            cid_m = cute.size(s_shape, mode=[0]) + tail_cid_m
            cid_n = tail_cid_n

        bidx_in_cluster = cute.arch.block_in_cluster_idx()
        pid_m = cid_m * self.cluster_shape_mn[0] + bidx_in_cluster[0]
        pid_n = cid_n * self.cluster_shape_mn[1] + bidx_in_cluster[1]

        tile_coord_mnkl = (pid_m, pid_n, None, bidz)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        # Multicast masks
        a_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=1
        )
        b_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=0
        )
        a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
        b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        # =====================================================================
        #  Shared memory allocation & pipeline init
        # =====================================================================
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        num_warps = self.threads_per_cta // 32
        consumer_arrive_cnt = mcast_size * num_warps
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )

        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mainloop_pipeline_array_ptr,
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # =====================================================================
        #  SMEM tensors
        # =====================================================================
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC_ptr = cute.recast_ptr(
            sA.iterator, epi_smem_layout_staged.inner, dtype=self.c_dtype
        )
        sC = cute.make_tensor(sC_ptr, epi_smem_layout_staged.outer)

        # =====================================================================
        #  Global tile partitioning
        # =====================================================================
        gA_mkl = cute.local_tile(
            mA_mkl, self.tile_shape_mnk, tile_coord_mnkl, proj=(1, None, 1)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, self.tile_shape_mnk, tile_coord_mnkl, proj=(None, 1, 1)
        )
        gC_mnl = cute.local_tile(
            mC_mnl, self.tile_shape_mnk, tile_coord_mnkl, proj=(1, 1, None)
        )

        # =====================================================================
        #  MMA partitioning
        # =====================================================================
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        warp_group_thread_layout = cute.make_layout(
            self.mma_warp_groups, stride=self.num_threads_per_warp_group
        )
        thr_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))
        tCgC = thr_mma.partition_C(gC_mnl)

        # TMA partitions
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        sA_for_tma_partition = cute.group_modes(sA, 0, 2)
        gA_for_tma_partition = cute.group_modes(gA_mkl, 0, 2)
        tAsA, tAgA_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a, a_cta_crd, a_cta_layout,
            sA_for_tma_partition, gA_for_tma_partition,
        )

        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        sB_for_tma_partition = cute.group_modes(sB, 0, 2)
        gB_for_tma_partition = cute.group_modes(gB_nkl, 0, 2)
        tBsB, tBgB_nkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b, b_cta_crd, b_cta_layout,
            sB_for_tma_partition, gB_for_tma_partition,
        )

        # MMA fragments
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        acc_shape = tCgC.shape
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        # =====================================================================
        #  Load per-tensor scales from global memory (scalar values)
        # =====================================================================
        scale_a_val = gScaleA[0]
        scale_b_val = gScaleB[0]
        combined_scale = scale_a_val * scale_b_val

        # =====================================================================
        #  Cluster sync after barrier init
        # =====================================================================
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # =====================================================================
        #  Prefetch TMA loads
        # =====================================================================
        k_tile_cnt = cute.size(gA_mkl, mode=[2])
        prefetch_k_tile_cnt = cutlass.max(cutlass.min(self.ab_stage, k_tile_cnt), 0)

        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        if warp_idx == 0:
            for prefetch_idx in cutlass.range(prefetch_k_tile_cnt, unroll=1):
                mainloop_pipeline.producer_acquire(mainloop_producer_state)
                tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]
                tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

                cute.copy(
                    tma_atom_a, tAgA_k, tAsA_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state),
                    mcast_mask=a_mcast_mask,
                )
                cute.copy(
                    tma_atom_b, tBgB_k, tBsB_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state),
                    mcast_mask=b_mcast_mask,
                )
                mainloop_pipeline.producer_commit(mainloop_producer_state)
                mainloop_producer_state.advance()

        # =====================================================================
        #  Prologue MMA
        # =====================================================================
        k_pipe_mmas = 1

        mainloop_consumer_read_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )
        mainloop_consumer_release_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        peek_ab_full_status = cutlass.Boolean(1)
        if mainloop_consumer_read_state.count < k_tile_cnt:
            peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                mainloop_consumer_read_state
            )

        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
        num_k_blocks = cute.size(tCrA, mode=[2])

        for k_tile in cutlass.range_constexpr(k_pipe_mmas):
            mainloop_pipeline.consumer_wait(
                mainloop_consumer_read_state, peek_ab_full_status
            )
            cute.nvgpu.warpgroup.fence()
            for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                k_block_coord = (
                    None, None, k_block_idx,
                    mainloop_consumer_read_state.index,
                )
                cute.gemm(
                    tiled_mma, accumulators,
                    tCrA[k_block_coord], tCrB[k_block_coord],
                    accumulators,
                )
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

            cute.nvgpu.warpgroup.commit_group()
            mainloop_consumer_read_state.advance()
            peek_ab_full_status = cutlass.Boolean(1)
            if mainloop_consumer_read_state.count < k_tile_cnt:
                peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                    mainloop_consumer_read_state
                )

        # =====================================================================
        #  MAINLOOP
        # =====================================================================
        for k_tile in cutlass.range(k_pipe_mmas, k_tile_cnt, 1, unroll=1):
            mainloop_pipeline.consumer_wait(
                mainloop_consumer_read_state, peek_ab_full_status
            )
            cute.nvgpu.warpgroup.fence()
            for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                k_block_coord = (
                    None, None, k_block_idx,
                    mainloop_consumer_read_state.index,
                )
                cute.gemm(
                    tiled_mma, accumulators,
                    tCrA[k_block_coord], tCrB[k_block_coord],
                    accumulators,
                )

            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)

            mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
            mainloop_consumer_read_state.advance()
            mainloop_consumer_release_state.advance()

            peek_ab_full_status = cutlass.Boolean(1)
            if mainloop_consumer_read_state.count < k_tile_cnt:
                peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                    mainloop_consumer_read_state
                )

            # Overlap TMA load with MMA
            if warp_idx == 0 and mainloop_producer_state.count < k_tile_cnt:
                mainloop_pipeline.producer_acquire(mainloop_producer_state)
                tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]
                tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

                cute.copy(
                    tma_atom_a, tAgA_k, tAsA_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state),
                    mcast_mask=a_mcast_mask,
                )
                cute.copy(
                    tma_atom_b, tBgB_k, tBsB_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state),
                    mcast_mask=b_mcast_mask,
                )
                mainloop_pipeline.producer_commit(mainloop_producer_state)
                mainloop_producer_state.advance()

        # =====================================================================
        #  EPILOGUE: scale accumulators and write back
        # =====================================================================
        cute.nvgpu.warpgroup.wait_group(0)

        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        else:
            cute.arch.sync_threads()

        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            self.c_layout,
            elem_ty_d=self.c_dtype,
            elem_ty_acc=self.acc_dtype,
        )
        copy_atom_C = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(self.c_layout.is_m_major_c(), 4),
            self.c_dtype,
        )
        tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
        tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_Atom)

        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sC)
        tRS_rAcc = tiled_copy_r2s.retile(accumulators)

        rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
        tRS_rD_layout = cute.make_layout(rD_shape[:3])
        tRS_rD = cute.make_rmem_tensor_like(tRS_rD_layout, self.acc_dtype)
        size_tRS_rD = cute.size(tRS_rD)

        sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
        tCgC_for_tma_partition = cute.zipped_divide(gC_mnl, self.epi_tile)

        bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
            tma_atom_c, 0, cute.make_layout(1),
            sepi_for_tma_partition, tCgC_for_tma_partition,
        )

        epi_tile_num = cute.size(tCgC_for_tma_partition, mode=[1])
        epi_tile_shape = tCgC_for_tma_partition.shape[1]
        epi_tile_layout = cute.make_layout(
            epi_tile_shape, stride=(epi_tile_shape[1], 1)
        )

        c_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.threads_per_cta
        )
        c_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stage,
            producer_group=c_producer_group,
        )

        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            # Copy accumulator fragment to D registers
            for epi_v in cutlass.range_constexpr(size_tRS_rD):
                tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]

            # Apply per-tensor scaling: acc *= scale_a * scale_b
            acc_vec = tRS_rD.load()
            scaled_vec = acc_vec * combined_scale

            # Type convert to output dtype and store
            tRS_rD_out = cute.make_rmem_tensor_like(tRS_rD_layout, self.c_dtype)
            tRS_rD_out.store(scaled_vec.to(self.c_dtype))

            epi_buffer = epi_idx % cute.size(tRS_sD, mode=[3])
            cute.copy(
                tiled_copy_r2s, tRS_rD_out, tRS_sD[(None, None, None, epi_buffer)]
            )

            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            pipeline.sync(barrier_id=1)

            gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
            if warp_idx == 0:
                cute.copy(
                    tma_atom_c,
                    bSG_sD[(None, epi_buffer)],
                    bSG_gD[(None, gmem_coord)],
                )
                c_pipeline.producer_commit()
                c_pipeline.producer_acquire()

            pipeline.sync(barrier_id=1)

        if warp_idx == 0:
            c_pipeline.producer_tail()

        return

    # -------------------------------------------------------------------------
    #  Static helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _compute_stages(tile_shape_mnk, a_dtype, b_dtype, smem_capacity, occupancy):
        epi_stage = 4
        epi_bytes = 0
        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
        mbar_helpers_bytes = 1024
        ab_stage = (
            smem_capacity // occupancy - mbar_helpers_bytes - epi_bytes
        ) // ab_bytes_per_stage
        return ab_stage, epi_stage

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk, epi_tile, a_dtype, a_layout, b_dtype, b_layout,
        ab_stage, c_dtype, c_layout, epi_stage,
    ):
        a_smem = sm90_utils.make_smem_layout_a(a_layout, tile_shape_mnk, a_dtype, ab_stage)
        b_smem = sm90_utils.make_smem_layout_b(b_layout, tile_shape_mnk, b_dtype, ab_stage)
        epi_smem = sm90_utils.make_smem_layout_epi(c_dtype, c_layout, epi_tile, epi_stage)
        return a_smem, b_smem, epi_smem

    @staticmethod
    def _compute_grid(c, tile_shape_mnk, cluster_shape_mn):
        c_shape = (tile_shape_mnk[0], tile_shape_mnk[1])
        gc = cute.zipped_divide(c, tiler=c_shape)
        cluster_shape_mnl = (*cluster_shape_mn, 1)
        clusters = cute.ceil_div(cute.get(gc.layout, mode=[1]).shape, cluster_shape_mnl)
        grid = tuple(x * y for x, y in zip(clusters, cluster_shape_mnl))
        return grid

    @staticmethod
    def _make_tma_atoms_and_tensors(tensor, smem_layout_staged, smem_tile, mcast_dim):
        op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op, tensor, smem_layout, smem_tile, num_multicast=mcast_dim,
        )
        return tma_atom, tma_tensor

    @staticmethod
    def _make_tma_store_atoms_and_tensors(tensor_c, epi_smem_layout_staged, epi_tile):
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c, epi_smem_layout, epi_tile,
        )
        return tma_atom_c, tma_tensor_c


# =============================================================================
#  Dispatch helper — mirrors the C++ tile/cluster selection logic
# =============================================================================
def select_tile_config(m: int, n: int, k: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Select (tile_shape_mn, cluster_shape_mn) matching the C++ dispatch.

    Returns the same tile/cluster configurations as
    scaled_mm_sm90_fp8_dispatch.cuh::cutlass_gemm_sm90_fp8_dispatch.
    """
    if m <= 16:
        if n <= 1280:
            return (64, 16), (1, 2)
        return (64, 16), (1, 1)
    elif m <= 64:
        if n <= 1280:
            return (64, 16), (1, 4)
        return (64, 64), (1, 1)
    elif m <= 128:
        return (64, 128), (2, 1)
    elif m >= 8192 and k >= 6144:
        return (256, 128), (2, 1)
    else:
        return (64, 128), (2, 1)


# =============================================================================
#  Correctness test
# =============================================================================
def _create_tensors(m, n, k):
    """Create random FP8 inputs, FP32 scales, and output tensor on GPU."""
    import torch
    import cutlass.torch as cutlass_torch

    torch.manual_seed(42)

    ab_dtype = cutlass.Float8E4M3FN
    c_dtype = cutlass.BFloat16
    scale_dtype = cutlass.Float32
    l = 1  # batch = 1

    # A: MxK, K-major (row-major)
    a_torch_cpu = cutlass_torch.matrix(l, m, k, False, ab_dtype)
    # B: NxK, K-major (col-major)
    b_torch_cpu = cutlass_torch.matrix(l, n, k, False, ab_dtype)
    # C: MxN, N-major (row-major)
    c_torch_cpu = cutlass_torch.matrix(l, m, n, False, c_dtype)

    # Per-tensor scales
    scale_a_cpu = cutlass_torch.matrix(1, 1, 1, False, scale_dtype)
    scale_b_cpu = cutlass_torch.matrix(1, 1, 1, False, scale_dtype)

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
        scale_a_cpu, scale_dtype, is_dynamic_layout=True, assumed_align=16
    )
    sfb_tensor, _ = cutlass_torch.cute_tensor_like(
        scale_b_cpu, scale_dtype, is_dynamic_layout=True, assumed_align=16
    )

    return (
        a_tensor, b_tensor, c_tensor, sfa_tensor, sfb_tensor,
        a_torch_cpu, b_torch_cpu, scale_a_cpu, scale_b_cpu,
        c_torch_gpu,
    )


def _compute_reference(a_cpu, b_cpu, scale_a_cpu, scale_b_cpu, c_dtype):
    """Reference: D = scale_a * scale_b * (A @ B^T), computed in FP32."""
    import torch
    import cutlass.torch as cutlass_torch

    # einsum "mkl,nkl->mnl"
    ref = torch.einsum("mkl,nkl->mnl", a_cpu.float(), b_cpu.float())
    sa = scale_a_cpu.float().flatten()[0]
    sb = scale_b_cpu.float().flatten()[0]
    ref = ref * sa * sb
    return ref.to(cutlass_torch.dtype(c_dtype))


def run_test(
    mnk: Tuple[int, int, int],
    tile_shape_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    tolerance: float = 0.1,
) -> bool:
    """Run a single correctness test case."""
    import torch

    m, n, k = mnk
    (
        a_tensor, b_tensor, c_tensor, sfa_tensor, sfb_tensor,
        a_cpu, b_cpu, sfa_cpu, sfb_cpu, c_gpu,
    ) = _create_tensors(m, n, k)

    gemm = ScaledMmSm90Fp8Kernel(
        acc_dtype=cutlass.Float32,
        tile_shape_mn=tile_shape_mn,
        cluster_shape_mn=cluster_shape_mn,
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

    compiled = cute.compile(
        gemm,
        a_tensor, b_tensor, c_tensor, sfa_tensor, sfb_tensor, current_stream,
        options=f"--opt-level {opt_level}",
    )

    compiled(a_tensor, b_tensor, c_tensor, sfa_tensor, sfb_tensor, current_stream)
    torch.cuda.synchronize()

    import cutlass.torch as cutlass_torch
    ref = _compute_reference(a_cpu, b_cpu, sfa_cpu, sfb_cpu, cutlass.BFloat16)
    res = c_gpu.view(cutlass_torch.dtype(cutlass.BFloat16))

    try:
        print(f"res={res}, ref={ref}")
        torch.testing.assert_close(res.cpu(), ref.cpu(), atol=tolerance, rtol=1e-02)
        return True
    except AssertionError as e:
        max_diff = (res.cpu().float() - ref.cpu().float()).abs().max().item()
        print(f"    max_diff={max_diff:.6f}")
        print(f"    {e}")
        return False


def main():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")

    cap = torch.cuda.get_device_capability()
    if cap[0] < 9:
        raise RuntimeError(f"SM90+ (Hopper) required, got sm_{cap[0]}{cap[1]}")

    test_cases = [
        # (M, N, K), tile_shape_mn, cluster_shape_mn
        ((256, 256, 256), (128, 128), (2, 1)),
        ((512, 512, 512), (128, 128), (2, 1)),
        ((128, 128, 256), (64, 128), (2, 1)),
    ]

    passed = 0
    total = len(test_cases)

    for mnk, tile_mn, cluster_mn in test_cases:
        label = f"mnk={mnk}, tile={tile_mn}, cluster={cluster_mn}"
        print(f"Testing {label} ... ", end="", flush=True)
        try:
            ok = run_test(mnk, tile_mn, cluster_mn)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
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
