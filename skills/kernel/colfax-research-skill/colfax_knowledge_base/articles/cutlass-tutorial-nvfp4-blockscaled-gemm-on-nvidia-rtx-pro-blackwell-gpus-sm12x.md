# NVFP4 Blockscaled GEMM on NVIDIA RTX Pro Blackwell GPUs (SM12x)

**URL:** https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/
**Date:** June 20, 2026
**ISO Date:** 2026-06-20T10:46:55-07:00
**Categories:** 
**Tags:** 
**Content Type:** blog
**PDF URL:** 
**PDF Filename:** 
**Scraped At:** 2026-10-24T14:34:53.547090

---

## Excerpt

In this article, we explore hardware-supported NVFP4 blockscaled GEMM on SM12x GPUs, such as the NVIDIA RTX Pro 6000 Blackwell Server Edition (SM120) or NVIDIA DGX Spark (SM121). We will first discuss features of these GPUs and their kernel programming paradigm, situating them relative to SM10x (e.g. B200 or B300) and SM8x (Ampere/Ada). Then, we will discuss sub-byte blockscaled GEMM on SM12x, covering the necessary PTX background and scale-factor layouts before moving on to implementation detai

---

## Content

In this article, we explore hardware-supported NVFP4 blockscaled GEMM on SM12x GPUs, such as the NVIDIA RTX Pro 6000 Blackwell Server Edition (SM120) or NVIDIA DGX Spark (SM121). We will first discuss features of these GPUs and their kernel programming paradigm, situating them relative to SM10x (e.g. B200 or B300) and SM8x (Ampere/Ada). Then, we will discuss sub-byte blockscaled GEMM on SM12x, covering the necessary PTX background and scale-factor layouts before moving on to implementation details as a CuTe DSL kernel. Finally, we present some GEMM benchmarks, which we will expand on in a subsequent post on optimization.
SM12x architecture
SM12x denotes the feature set available in Blackwell systems such as the RTX Pro 6000 Blackwell Server Edition. However, even though these devices carry the Blackwell family name, their programming model diverges substantially from that for B200/B300 GPUs, instead inheriting much of the design familiar from the older SM8x architectures (Ampere/Ada). In this section, we compare these three architectures in terms of what matters for kernel development.
First, we recall that SM10x systems like the B200 provides the asynchronous
tcgen05.mma
instruction for targeting the tensor cores, which uses Tensor Memory (TMEM) as a dedicated on-chip memory for holding the MMA accumulators. In contrast, SM12x does not use
tcgen05
or TMEM; instead, like with SM8x it uses warp-level
mma.sync
instructions.
This difference has important consequences for kernel structure. On SM10x,
tcgen05.mma
is launched asynchronously from a single thread, sources operands and accumulator from shared memory (SMEM) and TMEM, and is locked to one CTA per SM. As a result, SM10x GEMM kernels follow a warp-specialized paradigm in which all the MMA issue logic is delegated to one warp while other warps are responsible for loading the operands and writing out the result.
On SM12x, one instead uses
mma.sync
, which is a synchronous, warp-collective operation that sources its operands from register memory (RMEM) and also accumulates into RMEM, with fixed partitionings of the tiles over threads. Because it uses register fragments, the instruction tile for
mma.sync
is necessarily much smaller than that for
tcgen05.mma
, and one needs multiple warps handling MMA to achieve good throughput.
We note that
mma.sync
can sometimes be available on SM10x but (a) will never achieve the peak MMA throughput of the device and (b) for certain options like low-precision is also incompatible. So, the takeaways are:
SM10x GEMM kernels are completely incompatible with SM12x and will not run on those devices.
SM12x GEMM kernels may run on SM10x but will perform poorly.
SM8x GEMM kernels will run on SM12x and can often perform reasonably. Indeed, much of the optimization logic around scheduling and pipelining mainloops from SM8x directly applies to SM12x.
Second, we point out that despite the close similarities, SM12x has also advanced in many ways over SM8x:
Sub-byte support and blockscaling:
mma.sync
on SM12x supports lower precision formats than SM8x, as well as hardware-supported blockscaling.
Tensor Memory Accelerator (TMA) is available on SM12x.
Warpgroup register re-allocation is supported only on certain SM12x devices (sm120a/sm120f).
Cluster Launch Control (CLC) – a hardware-supported dynamic work-distribution scheme for persistent scheduling – can be used on SM12x.
Programmatic Dependent Launch (PDL) – kernel parallelism between dependent kernel launches – is supported on SM12x.
In terms of the programming model, the addition of TMA, register reallocation and CLC combine to make persistent scheduling much more favorable. But the SM8x-style static grid scheduling still works fairly well.
Sub-byte datatype and Blockscaling review
We covered both sub-byte datatype and blockscaling in a
previous post
; this section serves as a brief review of these concepts.
Low precision computation: NVFP4
Low-precision data types are useful for reducing model size, memory traffic and computational load. SM12x supports multiple sub-byte data types, which are data types that narrower than 8 bits. We focus on the NVFP4 format.
NVFP4 is an NVIDIA standard data type that bundles a fixed-length vector of low-precision numbers with a scale factor. The operand data type is E2M1, a 4-bit floating-point with 2 exponent bits and 1 mantissa bit. Unlike the IEEE formats, the value with all exponent bits set to 1 is not reserved for NaN or ∞. This modestly extends the range of E2M1. The possible values are:
[0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6]
The scale factor used to dequantize NVFP4 has type UE4M3, a nonnegative 8-bit floating point number with 4 exponent bits and 3 mantissa bits. The vector length, or
micro-block
size, for NVFP4 is 16.
Blockscaling
SM12x has hardware support for blockscaled GEMM. An ordinary GEMM has the form:
D = A @ B + C
In a blockscaled GEMM, each operand is multiplied by a scale factor before the multiply-add:
D = (A * scale_A) @ (B * scale_B) + C
The organization of the scale-factor matrices for A and B depends on the granularity used for scale factors. We refer to these matrices as SFA and SFB.
Figure 2.
Blockscaled GEMM. Adapted from the PTX
documentation
.
Figure 2 depicts a blockscaled GEMM where each row of A and column of B is divided into four chunks and multiplied by four scale factors.
Since NVFP4 has a micro-block size of 16, every 16 elements in the K dimension share a scale factor. In other words, if A has shape MxK, the SFA will have shape Mx(K/16). For B with shape KxN, SFB has shape (K/16)xN.
PTX for SM12x blockscaled MMA
SM12x added blockscaling support to the existing
mma.sync
instruction. There are several different supported datatypes, which can be found in the PTX
programming guide
. In PTX, the NVFP4 datatype is grouped with MXF4 datatype and denominated by the qualifier
.kind::mxf4nvf4
. The full syntax for blockscaled
mma.sync
with this qualifier is:
mma.sync.aligned.m16n8k64.row.col.kind.block_scale.scale_vec_size.f32.e2m1.e2m1.f32.stype
d, a, b, c, scale-a-data, {byte-id-a, thread-id-a}, scale-b-data, {byte-id-b, thread-id-b};

.kind           = {.kind::mxf4nvf4};
.scale_vec_size = {.scale_vec::2X, .scale_vec::4X};
.stype          = {.ue8m0, .ue4m3};
The shape qualifier
.m16n8k64
means that each warp calculates an MMA of shape 16x8x64. This is the only supported shape for NVFP4 blockscaling. The qualifier
.scale_vec_size
determines how many scale factors span the K = 64 dimension of the MMA atom. Since the micro-block size for NVFP4 is 16,
.scale_vec_size
must be set to
.scale_vec::4X
so there are 64/16 = 4 scale factors along each K extent of the MMA shape. The qualifier
.stype
determines the data type of the scale factors and therefore must be set to
.ue4m3
for NVFP4. Thus, the only available PTX instruction for NVFP4 blockscaling is:
mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
The operands are mostly standard for an MMA operation, with the addition of
scale-a-data
and
scale-b-data
. The optional operands
{byte-id-a, thread-id-a}
and
{byte-id-b, thread-id-b}
determine how the scale factors are organized, which we cover in the next section.
All in all, the MMA atom consumes:
A fragment of shape 16×64 with dtype E2M1,
B fragment of shape 64×8 with dtype E2M1,
SFA fragment of shape 16×4 with dtype UE4M3,
SFB fragment of shape 4×8 with dtype UE4M3.
C accumulator fragment of shape 16×8 with dtype F32.
And the instruction computes:
D accumulator fragment of shape 16×8 with dtype F32.
MMA .m16n8k64 fragment layouts for A and B
In this section, we briefly cover how the fragments of A and B that are fed to the NVFP4
mma.sync
instruction are collectively owned by threads in a warp.
The A fragment has shape 16×64, giving a total of 1024 elements. These are divided across the 32 warp threads, so each thread owns 32 elements {a0, a1, …, a31}. Figure 3 shows the (M, K) coordinates owned by each thread. For example, thread 0 owns the elements corresponding to (M, K) coordinates (0, 0:7), (0, 32:39), (8, 0:7), and (8, 32:39). Notice that each row is owned by a single quad (collection of 4 lanes, or threads). This foreshadows the scale factor ownership that will be discussed in the next section.
Figure 3.
MMA
.m16n8k64
fragment layout for matrix A with e2m1 type. Taken from the PTX
documentation
.
The B fragment has shape 64×8, and so each thread owns a length 16 vector {b0, b1, … , b15}. Figure 4 displays which (K, N) coordinates each thread owns for K between 0 and 31. Here, each column is collectively owned by a single quad.
Figure 4.
MMA
.m16n8k64
fragment layout for rows 0–31 of matrix B with e2m1 type. Taken from the PTX
documentation
.
Scale factor layouts on SM12x
As discussed above, the MMA shape for the SM12x NVFP4 atom is 16x8x64. Thus, each MMA operation consumes an A fragment with shape 16×64, a B fragment with shape 64×8, an SFA fragment with shape 16×4, and an SFB fragment with shape 4×8. The way in which SFA and SFB fragments are fed by threads in a warp to the
mma.sync
is based on quads. Figures 3 and 4 illustrated how each row of fragment A and column of fragment B was owned by a single quad. Ultimately, the quad that owns a given row of A or column of B will supply the corresponding scale factors for that row or column. Within a quad, however, only a subset of threads actually feed scale factors to the MMA instruction.
The thread registers used to pass scale factors are determined by the PTX operands
{byte-id-a, thread-id-a}
and
{byte-id-b, thread-id-b}
that we saw earlier. The details of these can be found in the
PTX documentation
. However in the case of CUTLASS, it is hard-coded so that first two threads of the quad (e.g.
thread_id % 4 == 0
and
1
) own SFA, and the first thread (e.g.
thread_id % 4 == 0
) owns SFB. Figure 5 demonstrates this pattern of scale factor passing.
Figure 5.
Thread selector logic for
thread-id-a = thread-id-b = 0
While the PTX documentation describes how to specify which threads in a warp ultimately supply scale factors, it does not specify precisely which scale factors are supplied by which threads.
We can produce such a mapping by considering the thread-value (TV) and SMEM layouts of the scale factors in practice. Let’s begin with SFA. In
mma_traits_sm120.hpp
, CUTLASS encodes the SFA atom TV layout as:
((2, 2, 8), 64) : ((8, 0, 1), 16)
This is the (T, V) -> (M, K) map. For example, evaluating the layout at (1, 0) gives us the coordinates of the 0th value of thread 1, which maps to (M, K) coordinates of (8, 0). Now consider the first quad; the stride of 0 in the middle sub-mode means that threads 0 and 2, as well as 1 and 3, are mapped to the same coordinates. In other words, the data is actually duplicated in the thread-pairs inside the quad. But the
mma.sync
will source the scale factors from threads 0 and 1.
To see the SFA row correspondence to A, let’s take a look at threads 0 and 1. TV-coordinate (0, 0) maps to MK-coordinate of (0, 0), and (1, 0) to (8, 0), we verify that they are in fact contributing the scale factors for the rows of A that they hold. Specifically, thread 0 contributes scale factors to row 0 and thread 1 contributes scale factors to row 8. For the K-mode, the TV layout here goes up to 64, instead of just 4. This contraction by a factor of 16 is done on the tensor layout.
For SFA tensor layout, the shape of a single atom tile is:
(mma_atom_m, (vec_width, num_vec)) : (mma_atom_k/vec_width, (0, 1))
Here, mma_atom_m and mma_atom_k are the size of the m and k-modes of the atom, respectively. So for our m16n8k64 atom we have:
(16, (16, 4)) : (4, (0, 1))
Conceptually, one can think of this as the (M, K) coordinate for A mapped to SFA. Once again, the interesting part is the stride of 0 in the k-mode. We see that coordinates are mapped to one element; this is our 16-width vector of the 4x scale factor mode. To get the SFB layout, simply replace
mma_atom_m
with
mma_atom_n
.
Finally, by taking the composition we get the full SFA TV layout for a single mma tile:
((2, 2, 8),(16, 4)):((32, 0, 4),(0, 1))
We depict this layout in Figure 6.
Figure 6.
Two equivalent views of the SFA TV layout for one m16n8k64 NVFP4 atom. The left view shows the warp-lane/register fragment organization with non-contributing lanes masked out. The right view reindexes the same data by logical (M, K) coordinates.
For SFB, only the first thread holds the scale factor. So the TV layout for the SFB atom is:
((4, 8), 64) : ((0, 1), 8)
Following the exact same derivation as SFA, we get the following SFB TV layout:
((4, 8), (16, 4)):((0, 4), (0, 1))
We will leave it as an exercise for the reader to derive and verify this SFB TV layout. We depict this layout in Figure 7.
Figure 7.
Two equivalent views of the SFB TV layout for one m16n8k64 NVFP4 atom. The left view shows the warp-lane/register fragment organization with non-contributing lanes masked out. The right view reindexes the same data by logical (K, N) coordinates.
Extending a SM12x BF16 GEMM to NVFP4 Blockscaled GEMM
In this section, we describe how to turn a dense SM12x BF16 GEMM into an NVFP4 blockscaled GEMM in CuTe DSL. Although the CUTLASS repository includes
blockscaling GEMM examples
, we will start from the
dense_gemm.py
example.
The dense GEMM tiles the output matrix C across CTAs, loads tiles of A and B from global memory to shared memory using TMA, copies those tiles from shared memory into per-thread register fragments, and then issues warp-level MMA instructions. It also uses a
producer-consumer pipeline
to overlap GMEM -> SMEM loads with Tensor Core computations. The blockscaled kernel follows the same outline, except that the MMA instruction will now consume two additional operands: the scale factor fragments for A and B.
Figure 8.
Producer-consumer pipeline for blockscaled GEMM.
The overall structure of the GEMM kernel will remain the same, with the primary change being the inclusion of the scale factors. We can group the conversion into four main changes:
Create tiled MMA with
MmaMXF4NVF4Op
atom.
Set up TMA with the interleaved GMEM and appropriate SMEM layouts, and add it to the load pipeline.
Copy SFA and SFB from shared memory into per-thread register fragments using SM12x scale factor thread-value layouts.
Pass [A, SFA] and [B, SFB] as operand pairs to
cute.gemm
.
It is worth keeping in mind which scale-factor layouts are forced by the hardware MMA operation and which the programmer is free to choose. The hardware constrains only the per-thread register layout that
mma.sync
reads its scale factors from. How the scale factors are stored in GMEM, how they are staged in SMEM, and whether TMA is used, are all implementation choices.
1. MMA Atom
First, we replace the original MMA operation:
op = cute.nvgpu.warp.MmaF16BF16Op(
    self.a_dtype,
    self.acc_dtype,
    self.mma_inst_mnk,
)
with the blockscaled NVFP4 operation:
op = cute.nvgpu.warp.MmaMXF4NVF4Op(
    self.a_dtype,
    self.acc_dtype,
    self.sf_dtype,
)
MmaMXF4NVF4Op
lowers to the NVFP4 blockscaled
mma.sync
instruction discussed above. Notice that the new MMA operation is not supplied
self.mma_inst_mnk
as an input. Instead, as discussed earlier, the MMA shape for NVFP4 blockscaling on SM12x is fixed as 16x8x64. Thus, the MMA shape is no longer a tunable parameter.
For this example, we consider a CTA tile with (bM, bN, bK) = (128, 128, 128). Note that 128 is the largest bM and bN value currently supported by the CUTLASS helper functions we use for scale factor manipulations. Since the MMA operation computes a 16×8 output tile from a 16×64 fragment of A and a 64×8 fragment of B, a tiled MMA is constructed to cover the larger CTA tile:
permutation_mnk = sm120_utils.get_permutation_mnk(
    self.tile_shape_mnk, self.sf_vec_size, False
)

self.tiled_mma = cute.make_tiled_mma(
    mma_op,
    cute.make_layout((4, 2, 1)),
    permutation_mnk=permutation_mnk,
)
The shape (4, 2, 1) specifies the spatial tiling of the atoms: how many concurrent copies of the 16x8x64 instructions run across the warps of the CTA in each of the M, N, and K directions. This is a tuning parameter that we can choose, so long as the atoms in this layout divide the CTA tile and its total size is the number of warps. In this case, we chose four copies along M, two along N, and one along K, giving 4*2*1 = 8 warps, or 256 threads. This is exactly the eight MMA warps this kernel launches.
The third argument,
permutation_mnk
, defines the M, N, and K extents of the region the TiledMMA covers, and the order in which the atom’s values are arranged inside it. For this configuration,
get_permutation_mnk
returns
perm_m
= 128,
perm_k
= 64, and a strided N layout (8, 2, 2):(1, 16, 8) of size 32. Since this is a blockscaled GEMM, the strided N layout must reorder the values each thread holds so that its fragments of A, B, SFA, and SFB line up in the arrangement the MMA instruction consumes.
2. TMA copy
To set up TMA, we need the GMEM and SMEM tensor layouts for the scale factors. In GMEM, the scale factors use the same interleaved layout as the SM10x NVFP4 blockscaling (see Figure 6 of our
previous post
). We use the CUTLASS provided helper functions to package the logical scale factor tensors in the desired layout:
self.sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a.shape, self.sf_vec_size)
sfa_tensor = cute.make_tensor(sfa.iterator, self.sfa_layout)

self.sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b.shape, self.sf_vec_size)
sfb_tensor = cute.make_tensor(sfb.iterator, self.sfb_layout)
Next, we create TMA atoms for SFA and SFB and add them to the same producer-consumer pipeline as A and B. For the SMEM layouts for SFA and SFB, we use the CUTLASS helper functions:
sfa_smem_layout_staged = blockscaled_utils.sm120_make_smem_layout_sfa(
           tiled_mma, tile_shape_mnk, sf_vec_size, ab_stage,
       )

sfb_smem_layout_staged = blockscaled_utils.sm120_make_smem_layout_sfb(
           tiled_mma, tile_shape_mnk, sf_vec_size, ab_stage,
       )
Dropping the staging mode, the SMEM layouts have the form:
(((32,4), REST_M/N), ((16,4), 1, REST_K)) : (((16, 4), 512 * REST_K), ((0, 1), 4, 512))
Here, the M mode carries a (32, 4) tiling that corresponds to the TMA copy, and the “rest” modes appear because we stage more than a single atom’s worth of data. Underneath, the familiar scale-factor structure is still visible: 32*4=128 in the M mode is the number of rows needed for eight warps (8*16=128). The K mode shows the same (16, 4):(0, 1) layout of the scale factors from before.
The rest of the TMA setup is standard and identical to A and B. We add the transaction count to the main load pipeline so that the mainloop only needs to manage one pipeline in order to get all the data it needs.
3. SMEM -> RMEM copy
Once the scale factors are in shared memory, fragments of SFA and SFB are copied to registers. Unlike the A and B operand copies, which use ldmatrix, the SFA and SFB operand copies use a universal copy atom. The main challenge is constructing the TV-layout that maps the staged SFA/SFB values in SMEM to the per-thread register fragment expected by the MMA instruction. Luckily, CuTe DSL has a helper function
get_layoutSFA_TV
to specify the SM12x scale-factor TV layouts:
atom_copy_SF = cute.make_copy_atom(
    cute.nvgpu.CopyUniversalOp(),
    self.sf_dtype,
)

smem_tiled_copy_SFA = cute.make_tiled_copy(
               atom_copy_SF,
               sm120_utils.get_layoutSFA_TV(tiled_mma),
               (
                   cute.size(tiled_mma.permutation_mnk[0]),
                   cute.size(tiled_mma.permutation_mnk[2]),
               ),
           )

 smem_tiled_copy_SFB = cute.make_tiled_copy(
               atom_copy_SF,
               sm120_utils.get_layoutSFB_TV(tiled_mma),
               (
                   cute.size(tiled_mma.permutation_mnk[1]),
                   cute.size(tiled_mma.permutation_mnk[2]),
               ),
           )
After partitioning the staged SMEM tiles, the thread slice of the tiled copy gives each thread its SMEM source slice. For the thread-level RMEM slice, we use a helper function which is then retiled with the thread copy.
tCrSFA_tile = sm120_utils.partition_fragment_SFA(
    sSFA_tile[None, None, 0], thr_mma, tidx
)

tCrSFA_tile_copy_view = thr_copy_SFA.retile(tCrSFA_tile)

tCrSFB_tile = sm120_utils.partition_fragment_SFB(
    sSFB_tile[None, None, 0], thr_mma, tidx
)

tCrSFB_tile_copy_view = thr_copy_SFB.retile(tCrSFB_tile)
Before moving on, let’s take a close look at what these helper functions create. The layout of tCrSFA_tile and tCrSFB_tile are as follows:
tCrSFA_tile: ((16, 4), 2, 2):((0, 1), 4, 8)
tCrSFB_tile: ((16, 4), (2, 4), 2):((0, 1), (16, 4), 32)
These are the three-dimensional per-thread scale factor layouts across all MMAs in the CTA tile, in the form (SF_VEC, REST_MN, REST_K). The zeroth mode is the familiar broadcast (16, 4):(0, 1), and the other two modes index the repeated MMA blocks in the MN and K dimensions. In our case, we have CTA tile of (bM, bN, bK) = (128, 128, 128), atom_layout of (4, 2, 1) and individual atom shape of 16x8x64. This gives 2 M-blocks and 2 K-blocks for SFA and 8 N-blocks and 2 K-blocks for SFB. Except here the N mode for SFB is factored as (2, 4), inherited from the interleaving in tCrB.
We can also observe the scale-factor replication from these layouts. The physical footprint of
tCrSFA_tile
per lane is 4*2*2 = 16 bytes. This translates to 32*16 bytes = 512 bytes across an entire warp. But each warp is responsible for a (2*16) x (2*64) slice of the A tile, which translates to 256 distinct 1-byte scale factors per warp. Recall, however, that only two lanes per quad contribute scale factor data to the MMA instruction. Thus, there is a 2x replication of data, resulting in the 512 bytes observed above.
Similarly for SFB, there is a 4x replication. The physical footprint of
tCrSFB_tile
is 4*8*2 = 64 bytes per lane, or 32*64 = 2048 bytes per warp. Each warp is responsible for a (8*8) X (2*64) tile of B, or 512 distinct scale factors. This once again is explained by all lanes loading scale factor data, but only one lane per quad contributing the scale factor data to the MMA instruction.
Figure 9.
Layout of the per-thread SFA register fragment
tCrSFA_tile
. The different colors represent the 4 different MMAs.
Figure 10.
Layout of the per-thread SFB register fragment
tCrSFB_tile
, omitting K atom position 1. The different colors represent 8 different MMAs.
Now that we have the tiled copy and the per-thread slices of the source (SMEM) and the destination (RMEM), we call
cute.copy
.
tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_p)
tCrSFA_copy_view_filtered = cute.filter_zeros(tCrSFA_tile_copy_view)

cute.copy(
    smem_tiled_copy_SFA,
    tCsSFA_p_filtered[None, None, k_block_idx],
    tCrSFA_copy_view_filtered[None, None, k_block_idx],
)
The
filter_zeros
calls are necessary to remove the broadcast mode in the scale-factor layout. In particular, the logical layout contains repeated coordinates that refer to the same physical scale-factor entry.
4. MMA
The mainloop structure of the kernel is very similar to the non-blockscaled kernel, as we use the same pipeline for operands and scale factors. Consumer warps wait on the load pipeline, copy A, B, SFA, and SFB from shared memory into registers, and then issue the MMA. The MMA call needs to be altered so that register fragments of A and SFA and B and SFB are passed as pairs:
cute.gemm(
    tiled_mma,
    accumulators,
    [
        tCrA[None, None, k_block_idx],
        tCrSFA_tile[None, None, k_block_idx],
    ],
    [
        tCrB[None, None, k_block_idx],
        tCrSFB_tile[None, None, k_block_idx],
    ],
    accumulators,
)
Benchmarking
We benchmarked the tutorial NVFP4 blockscaled GEMM kernel against PyTorch’s dense FP16 implementation (
torch.mm
), which uses cuBLAS internally, and cuBLAS’s NVFP4 blockscaled GEMM. We used CUDA Toolkit 13.2. All measurements were taken on an
NVIDIA RTX Pro 6000 Blackwell Server Edition GPU
. One thing to note is that quantization time is not included in the benchmark as, depending on the workload, the quantization may be done ahead of time.
Figure 11.
Benchmark results on the RTX Pro 6000. The value above a bar indicates speedup relative to the cuBLAS dense FP16 GEMM.
The RTX Pro 6000 has peak theoretical dense FP16 throughput of 500 TFLOP/s, and peak theoretical dense FP4 throughput of 2000 TFLOP/s. So given the different datatypes, we expect the compute throughput ratio to be 4x. In our benchmark, we observe close to ~3x difference between the tutorial NVFP4 GEMM and the dense FP16 GEMM in performance for most problem shapes tested. The particularly poor performance in 2k problem shape is mainly due to CTA wave quantization effects. In particular, with 128 x 128 CTA tiles, a 2k x 2k problem launches 16 x 16 = 256 CTAs. The RTX 6000 Pro Blackwell GPU has 188 SMs, translating to 256/188 = 1.36 CTA waves. The final partial wave leaves many SMs idle. The tutorial NVFP4 GEMM tops out at around 60% of peak theoretical throughput.
Conclusion
In this blog post, we examined NVFP4 blockscaling on SM12x GPUs. We first compared and contrasted the SM12x warp-level
mma.sync
programming model with the SM10x
tcgen05
model and the older SM8x
mma.sync
model. We then analyzed the PTX instruction for blockscaled
mma.sync
and carefully studied the scale-factor layouts expected by the instruction. We discussed implementation details for converting a SM12x dense BF16 GEMM example into an NVFP4 blockscaled GEMM. Finally, we presented benchmarks of the tutorial NVFP4 GEMM against the FP16
torch.mm
(cuBLAS) GEMM and cuBLAS’s NVFP4 GEMM.
In the next blog post, we will discuss optimization techniques for improving the utilization of the blockscaled GEMM implementation.
Share this:
Share on LinkedIn (Opens in new window)
LinkedIn
Share on X (Opens in new window)
X
Share on Facebook (Opens in new window)
Facebook
Share on Reddit (Opens in new window)
Reddit
Like this:
Like
Loading…
