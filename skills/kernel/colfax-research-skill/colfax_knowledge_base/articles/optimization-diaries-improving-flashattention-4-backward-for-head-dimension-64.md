# Optimization diaries: Improving FlashAttention-4 backward pass kernel design for head dimension 64

**URL:** https://research.colfax-intl.com/optimization-diaries-improving-flashattention-4-backward-for-head-dimension-64/
**Date:** September 6, 2026
**ISO Date:** 2026-09-06T14:45:28-07:00
**Categories:** 
**Tags:** 
**Content Type:** blog
**PDF URL:** 
**PDF Filename:** 
**Scraped At:** 2026-10-24T14:34:48.022500

---

## Excerpt

In this blog post we discuss the backward pass ofFlashAttention-4(FA4) on NVIDIA Blackwell GPUs. For head dimension 128, FA4 backward is highly performant, achieving1237 TFLOP/s, or about55%of peak compute throughput on a B200 GPU. However, at head dimension 64, the same kernel achieves only30–37%of peak compute throughput for long sequence lengths. This suggests that there is plenty of room for improvement in this case. We will describe a optimization that leverages spare tensor memory (TMEM) t

---

## Content

Introduction
In this blog post we discuss the backward pass of
FlashAttention-4
(FA4) on NVIDIA Blackwell GPUs. For head dimension 128, FA4 backward is highly performant, achieving
1237 TFLOP/s
, or about
55%
of peak compute throughput on a B200 GPU. However, at head dimension 64, the same kernel achieves only
30–37%
of peak compute throughput for long sequence lengths. This suggests that there is plenty of room for improvement in this case. We will describe a optimization that leverages spare tensor memory (TMEM) to improve FA4 backward performance for head dimension 64.
The key observation is that for head dimension 64, a quarter of the TMEM now goes unused under the default kernel design. We can leverage this unused TMEM to de-alias certain tensors that reside there: specifically, same iteration
P
P
onto
S
S
and
d
S
dS
onto
d
P
dP
. De-aliasing allows us to remove unnecessary synchronization and to replace 256 thread barrier syncs by plain warp syncs. It also allows us to reorder MMA issuance so that the next iteration’s
Q
K
QK
MMA can overlap the current iteration’s softmax computation. Together, this results in a speed-up of
1.06–1.15×
, up to
903 TFLOP/s (40% of peak)
. The code may be found in
PR #2804
on the FlashAttention repository.
Recap on the FA4 backward pass
The FA4 backward pass computes
S = Q K^T, \: P = exp(S – L), \: dP = dO V^T, \: dS = P \circ (dP – D),
dV = P^T dO, \: dK = dS^T Q, \: dQ = dS K.
The computation is parallelized over batches, attention heads, and
K
V
KV
tiles. Each CTA owns one tile of
K
K
and
V
V
, and loops over
Q
Q
tiles. Each CTA computes the
d
K
dK
and
d
V
dV
tiles corresponding to the
K
K
and
V
V
tiles it owns. At each iteration of the mainloop, each CTA also computes a contribution to the
d
Q
dQ
tile corresponding to the current
Q
Q
tile, which are summed into an fp32 accumulator in global memory.
The FA4 backward kernel is
warp-specialized
and work is divided among the CTA’s sixteen warps as follows.
role
warps
description
load
1
issues the TMA loads
K
K
,
V
V
,
Q
Q
and
d
O
dO
MMA
1
issues every
tcgen05.mma
instruction
compute
8
computes
P
P
from
S
S
, and computes
d
S
dS
from
d
P
dP
reduce
4
reads each partial
d
Q
dQ
from tensor memory and adds it into the global accumulator
relay/empty
2
idle at hdim 64; the relay warp is used only in the hdim 128 case, where 2-CTA MMA instructions are used
We will focus on the MMA warp and the compute warpgroups. For each iteration of the mainloop, the MMA warp issues five GEMMs. Between these, the compute warps do two pointwise computations. Since each CTA owns a
K
V
KV
tile, the kernel computes
S
T
S^T
,
P
T
P^T
,
d
P
T
dP^T
and
d
S
T
dS^T
rather than
S
S
,
P
P
,
d
P
dP
and
d
S
dS
. The computational steps of each mainloop iteration are described in the table below.
step
who
reads
writes
S
T
=
K
Q
T
S^T = K Q^T
MMA warp
K
K
,
Q
Q
(SMEM)
S
T
S^T
(TMEM, fp32)
P
T
=
e
x
p
(
S
T
−
L
)
P^T = exp(S^T − L)
compute warps
S
T
S^T
(TMEM)
P
T
P^T
(TMEM, bf16)
d
S
T
=
P
T
∘
(
d
P
T
−
D
)
dS^T = P^T∘(dP^T − D)
compute warps
P
T
P^T
(RMEM),
d
P
T
dP^T
(TMEM)
d
S
T
dS^T
(TMEM, bf16; and SMEM)
d
V
+
=
P
T
d
O
dV += P^T dO
,
d
K
+
=
d
S
T
Q
dK += dSᵀ Q
,
d
Q
=
d
S
K
dQ = dS \ K
MMA warp
P
T
Pᵀ
,
d
S
T
dSᵀ
(TMEM),
d
S
T
dSᵀ
(SMEM)
d
V
dV
,
d
K
dK
,
d
Q
dQ
(TMEM)
For brevity we will drop the transposes from here on, and simply write
S
S
,
P
P
,
d
P
dP
and
d
S
dS
for the relevant tensors/tiles.
At this point, we can identify a primary cause for the performance disparity in the head dimension 64 versus 128 case: GEMM FLOPs scale with head dimension, so going from hdim 128 to hdim 64
halves
the tensor-core work per mainloop iteration. On the other hand, the FLOPs computed by the compute warps are independent of head dimension. This means that in the hdim 64 case, there is less tensor-core compute to hide the latency of the pointwise compute warp operations.
Pre-optimization: the baseline mainloop
Recall that the accumulators for
tcgen05.mma
GEMMs live in tensor memory (TMEM), a 128-lane × 512-column array of 32-bit cells per SM. Warps read from TMEM and write to TMEM with
tcgen05.ld
and
tcgen05.st
, respectively. The hdim 64 FA4 backward kernel allocates TMEM as depicted below.
Figure 1
. The baseline TMEM allocation scheme at hdim 64. The top row shows the fp32 accumulators written by
tcgen05.mma
; the bottom row shows the bf16 tiles the compute warps write over them:
P
P
over
S
S
, and
d
S
dS
(and the
d
Q
dQ
partial) over
d
P
dP
. Notably, columns [384, 512) are unused.
The four fp32 accumulators occupy columns [0, 384), and the bf16 tiles
P
P
and
d
S
dS
are written over
S
S
and
d
P
dP
, respectively. Because of this, we must ensure that, for example,
S
S
has been read by all of its consumers before
P
P
is written by any producer.
Let’s take a closer look at what could go wrong if these reads/writes were unguarded. An important restriction on TMEM access is that each warp may only access its own 32-lane sector: warps 0, 1, 2, and 3 of a warpgroup own lanes [0, 32), [32, 64), [64, 96), and [96, 128), respectively. In FA4 backward the eight compute warps make up two warpgroups, so warps 4, 5, 6, and 7 share the same sector of TMEM as warps 8, 9, 10, and 11, respectively. Combined with the aliasing, this creates the cross-warp hazard depicted below.
Figure 2
. The hazard created by storing
P
P
over
S
S
. Each compute warp loads its own 32 lanes of
S
S
. In particular, warp 5 could start storing
P
P
over
S
S
before warp 9 has finished reading
S
S
.
We see here that warp 5 stores
P
P
in lanes [32, 64), columns [0, 64), where warp 9 may still be loading
S
S
. We must therefore enforce that every warp has finished loading
S
S
before any warp stores
P
P
. We call this an
alias guard
. There is one for
P
P
over
S
S
, and one for
d
S
dS
over
d
P
dP
. Each is a
compute-wide barrier
: a named barrier that synchronizes all eight compute warps (256 threads). The
P
P
guard is needed once per mainloop iteration, while the
d
S
dS
guard sits inside the two-stage
d
S
dS
loop, hence is needed twice per mainloop iteration.
cute.arch.fence_view_async_tmem_load()
# P overwrites S, and warp w's P columns sit under warp w+4's S lanes:
# every warp must have loaded S before any warp stores P.
self.compute_sync_barrier.arrive_and_wait()
These alias guards are the barriers that the baseline TMEM allocation forces. Each such barrier holds all eight compute warps at the pace of the slowest warp, which leaves performance on the table.
The aliasing also restricts the order in which the MMA warp can issue its GEMMs. In the baseline kernel, the compute warps release
S
S
and commit
P
P
with a fused signal, “
S
S
read and
P
P
written”:
cute.arch.fence_view_async_tmem_store()
cute.arch.fence_view_async_shared()
self.compute_sync_barrier.arrive_and_wait()
with cute.arch.elect_one():
    pipeline_S_P.consumer_release(consumer_state) # "S read and P written"
    pipeline_LSE.consumer_release(consumer_state_LSE)
Because the
S
S
and
P
P
buffers in TMEM are aliased, the MMA warp cannot issue the next tile’s
S
S
MMA until the current tile’s
d
V
dV
MMA executes. This contributes to the
d
V
dV
MMA becoming exposed on its
P
P
dependency, with the tensor cores not doing useful work in between. In fact, with the MMA warp’s default issue order
QK_{t+1}, \: dK_{t}, \: dQ_{t}, \: dP_{t+1}, \: PdO_{t+1}
the IKET trace shows that for hdim 64, the
d
K
dK
,
d
Q
dQ
and
d
P
dP
matmuls aren’t sufficient to hide the softmax.
Figure 3
. One iteration of the baseline mainloop, measured from in-kernel timestamps (one CTA, SM cycles). Top: the GEMMs the MMA warp issues. Below: the eight compute warps. The lanes move in lock-step, because the five compute-wide barriers per tile (red dashed lines, drawn where the last warp arrives) hold every warp to the pace of the slowest.
Q
K
QK
for the next tile is issued only after the softmax has ended and
P
P
is consumed by the
d
V
dV
MMA. During the softmax, the tensor cores compute
d
Q
t
−
1
dQ_{t−1}
and
d
P
t
dP_t
, and sits idle for most of it.
Note:
P
d
O
t
PdO_t
mislabeled as
P
V
t
PV_t
(conceptually it plays the same role in backward as P’s mma consumer).
Note that there are two more compute-wide barriers per tile: one in front of each of the two signals the compute warps send after their stores. Unlike the alias guards, they protect no cross-warp hazard, and we return to them briefly at the end of the next section. Altogether, we have four barrier sites, and five compute-wide barriers executed per mainloop iteration.
Post-optimization: dedicated TMEM slots for P and dS
The key step in our optimization is the de-aliasing of
P
P
and
d
S
dS
:
if self.split_P_dS:
    # P/dS are bf16 packed two per column: a 128-wide tile is tile_m // 2 columns
    self.tmem_P_offset = self.tmem_dK_offset + self.tile_hdim # [384, 448)
    self.tmem_dS_offset = self.tmem_P_offset + self.tile_m // 2 # [448, 512)
Figure 4
. A snippet of the de-aliased TMEM allocation. Note that
P
P
and
d
S
dS
are in half precision, so they require half as many columns as their logical shape would suggest.
With the TMEM allocation changed, we make three modifications to the kernel:
1. Remove the alias guards.
Since
P
P
and
d
S
dS
now have dedicated TMEM buffers, the cross-warp hazard of Figure 2 cannot occur, so we remove both alias guards (of course, the TMEM fence itself stays).
2. Release S earlier.
In the baseline, the shared
S
S
and
P
P
TMEM buffer is controlled by a single pipeline object. The compute warps signal “
S
S
read and
P
P
written” once, after storing
P
P
, and the MMA warp waits on it before both
Q
K
t
+
1
QK_{t+1}
and
P
d
O
t
PdO_t
. With our new TMEM allocation, we split this pipeline in two:
–
pipeline_S_P
now carries only “
S
S
read”. The compute warps signal it as soon as
S
S
is in registers, before the softmax; the MMA warp waits on it before
Q
K
t
+
1
QK_{t+1}
.
– A new one-stage
pipeline_P
carries “
P
P
written” and “
P
P
consumed”. The compute warps signal “
P
P
written” after storing
P
P
, and the MMA warp waits on this signal before issuing
P
d
O
t
PdO_t
. In return, the compute warps wait on “
P
P
consumed” before overwriting
P
P
for the next iteration.
The early release looks like this:
cute.copy(thr_copy_t2r, tStS_t2r, tSrS_t2r) # S -> registers
if const_expr(self.split_P_dS):
	# S is in registers: release it now, before the softmax,
	# so the MMA warp can issue the next QK into the slot.
	cute.arch.fence_view_async_tmem_load()
	cute.arch.sync_warp()
	with cute.arch.elect_one():
		pipeline_S_P.consumer_release(consumer_state_S)
3. Reorder the MMA warp.
With
P
P
in its own buffer,
Q
K
t
+
1
QK_{t+1}
depends only on
S
S
having been consumed by the compute warps. This means the MMA warp can issue
Q
K
t
+
1
QK_{t+1}
as soon as the
S
S
read signal arrives, ahead of
P
d
O
t
PdO_t
. The issue order becomes
QK_{t+1}, \: PdO_t, \: dK_t, \: dQ_t, \: dP_{t+1}.
The tensor-core work available during the softmax of iteration
t
grows to include
Q
K
t
+
1
QK_{t+1}
, which fills most of the idle window seen in Figure 3.
The remaining barriers.
Three compute-wide barriers remain, one in front of each signal the compute warps send to the MMA warp. But since there are no longer any cross-warp hazards, we can replace them with warp sync, leaving the eight warps free to drift apart through the softmax and dS sections. This eliminates all the compute-wide barriers.
With these changes made, the IKET trace shows a shorter makespan:
Figure 5
.
Q
K
QK
for the next tile now issues in the middle of the softmax, and the eight warps drift freely, since nothing in the loop makes them wait for each other. The tile is 19% shorter (median over 56 tiles).
Results
We benchmark on an NVIDIA B200 GPU with bf16 inputs unless noted, 32 query heads, and 64k tokens (batch × sequence length) per call. Each number is the median over back-to-back runs of the kernel. TFLOP/s counts the backward pass as 2.5× the FLOPs of the forward pass, and utilization is relative to the 2250 TFLOP/s dense peak. In deterministic mode, every variant produces gradients bit-identical to the baseline kernel. All benchmarks were produced with PyTorch 2.13.0, nvidia-cutlass-dsl 4.6.2 and driver 595.71.05.
Figure 6
. FA4 backward at hdim 64 before and after, B200, bf16. The right axis is the fraction of the 2250 TFLOP/s dense peak.
Every configuration measured improves by 6–15%, deterministic ones included, and the gain grows with sequence length. Long-sequence dense multi-head attention goes from 731 to 841 TFLOP/s (1.150×); grouped-query attention with 32:8 heads reaches 903 TFLOP/s non-causal and 842 TFLOP/s causal (1.075× and 1.123×, respectively); fp16 behaves like bf16 (1.131×). A wider sweep of 124 configurations — sequence lengths 512 to 32k, MHA/GQA/MQA, dense, causal, local and variable-length, both dtypes, deterministic and not — gives a geometric-mean speed-up of
1.129× non-deterministic
, with a worst non-deterministic cell of 1.023× and no cell in the grid below 0.99×.
shape (64k tokens)
mode
before TFLOPS
after TFLOPS
speed-up
b2 s32k h32:32
dense
731
841
1.150x
b2 s32k h32:32
causal
705
808
1.147x
b4 s16k h32:8
dense
840
903
1.075x
b4 s16k h32:8
causal
750
842
1.123x
b8 s8k h32:32 fp16
dense
692
782
1.131x
b4 s16k h32:32
dense, deterministic
709
774
1.091x
b4 s16k h32:32
causal, deterministic
664
725
1.093x
The ablation in Figure 7 separates the de-aliasing from the per-warp signalling it enables. Dedicated slots alone are worth about 2% on dense and 3–4% on causal. Per-warp signalling without de-aliasing is worth 8–9% on dense and essentially nothing on causal (1.00–1.01×), because the two alias guards inside the per-stage loops have to stay, and those loops are shortest on causal shapes. Together they are worth 13–15% on dense and 11–15% on causal; most of the win is only reachable once the alias guards are gone.
Figure 7
. Speed-up over baseline for each change alone and for both together.
Figure 8
. The same tile with dedicated TMEM slots only, same scale. De-aliasing alone moves
Q
K
QK
under the softmax, and the warps begin to drift within it. But their
S
S
releases (green) line up, their signal
P
P
sections end together, and their
d
S
dS
sections start together. The tile is about 5% shorter, because the warps still wait for the slowest three times per tile.
Deterministic mode.
In deterministic mode the kernel is bottlenecked by the semaphore ordering
d
Q
dQ
accumulation. However, this caps the gain rather than erasing it: the deterministic half of the sweep improves by 1.05× against 1.13× for the rest, and the deterministic rows of the table above still pick up 9%. For further speedup with deterministic mode, one should use 2-CTA MMA instructions like we do for hdim 128, since the
d
S
dS
2-CTA exchange over DSMEM extends the
d
Q
dQ
MMA’s reduction to be over the cluster tile and hence cuts
d
Q
dQ
bulk reduce atomics in half.
Conclusion
In this blog post, we described an optimization for hdim-64 FA4 backward that uses leftover TMEM to assign dedicated slots for
P
P
and
d
S
dS
, so that the compute warps no longer write over the
S
S
and
d
P
dP
buffers from which they read. This removed the hazard between warps that shared TMEM lanes, allowing us to remove the two compute-wide barriers that guarded it. Once those were gone, the three barriers that remained only fenced each warp’s own stores, and a warp-local sync could take their place. The mainloop now runs without any compute-wide barriers and the tensor core starts the next tile’s
Q
K
QK
MMA under the current softmax. Finally, benchmarking show a 6–15% speedup, up to 903 TFLOP/s.
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
