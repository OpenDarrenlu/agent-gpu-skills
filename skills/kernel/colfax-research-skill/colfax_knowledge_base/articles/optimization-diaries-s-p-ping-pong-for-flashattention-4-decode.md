# Optimization diaries: S/P ping-pong for FlashAttention-4 decode

**URL:** https://research.colfax-intl.com/optimization-diaries-s-p-ping-pong-for-flashattention-4-decode/
**Date:** September 7, 2026
**ISO Date:** 2026-09-07T10:59:28-07:00
**Categories:** 
**Tags:** 
**Content Type:** blog
**PDF URL:** 
**PDF Filename:** 
**Scraped At:** 2026-10-24T14:34:45.530811

---

## Excerpt

LLM Inference is divided into a prefill phase and a decode phase. During prefill, the model processes a large number of input tokens and populates a key-value (KV) cache. During decode, it autoregressively generates one or a few new tokens at a time using the cached keys and values.

---

## Content

LLM Inference is divided into a prefill phase and a decode phase. During prefill, the model processes a large number of input tokens and populates a key-value (KV) cache. During decode, it autoregressively generates one or a few new tokens at a time using the cached keys and values.
In this blog post, we discuss an optimization for FlashAttention-4 (FA4) decoding on NVIDIA Blackwell GPUs. Currently, in FA4 decoding, the
Q
K
T
QK^T
matmul for
K
V
KV
block
i
+
1
i+1
is issued only after the softmax for
K
V
KV
block
i
i
has completed, even though there is no mathematical dependency between the two. To instead overlap them, we can leverage spare tensor memory (TMEM) present in the decode path. The spare TMEM is used to ping-pong
S
/
P
S/P
between two slots so that while the softmax warps are writing the output of block
i
i
to one slot, the MMA warp can issue
Q
K
T
QK^T
to the other slot.
This change achieves up to a 16% performance gain on supported single and multi-token decoding configurations for head dimension 64 and 128. The code may be found in
PR #2817
on the FlashAttention repository.
Recap on the FA4 forward pass
Let
Q
Q
,
K
K
, and
V
V
be the query, key, and value matrices. The forward pass of attention calculates the attention output
O
O
as follows:
S
=
1
d
Q
K
T
,
P
=
softmax
⁡
(
S
)
,
O
=
P
V
S = \frac{1}{\sqrt{d}}QK^T, \qquad
P = \operatorname{softmax}(S), \qquad
O = PV
where softmax is applied row-wise. In practice, FA4 does not materialize the full
S
S
or
P
P
matrices and instead processes them tile by tile. As such, softmax is computed “online” with
O
O
rescaled when necessary.
To implement the forward pass, FA4 uses a web of overlapping pipelines across 5 different warp roles: load, MMA, softmax, correction, and epilogue. The load warp copies tiles of
Q
Q
,
K
K
, and
V
V
from global memory (GMEM) to shared memory (SMEM). The MMA warp consumes
Q
Q
and
K
K
from load and issues
S
=
Q
K
T
S=QK^T
for the softmax warps to consume. The softmax warps produce
P
P
and update the online softmax statistics. Based on the softmax statistics, the correction warps rescale
O
O
if necessary. The MMA warp then consumes
P
P
with
V
V
to issue
P
V
PV
.
Moving forward, we suppress the transpose decoration on
K
K
. For prefill, each CTA is assigned two 128-row
Q
Q
tiles, a high
Q
Q
-tile
Q
H
Q^H
and a low
Q
Q
-tile
Q
L
Q^L
. The purpose of this is to overlap softmax(
S
H
S^H
) with
Q
L
K
Q^L K
. In this scheme,
S
H
S^H
is stored in TMEM columns 0-127 and
S
L
S^L
is stored in columns 128-255. These two slots are reused to store
P
H
P^H
and
P
L
P^L
. Figure 1 is taken from the FA4 preprint and depicts this schedule.
Figure 1.
Each CTA is assigned two
Q
Q
-tiles. This figure shows how the two matmuls and softmax for each are scheduled relative to each other during prefill.
For single and multi-token decode, however, there is typically only a single (usually padded) 128-row
Q
Q
tile. The overlap of operations in Figure 1 is not relevant in this case. Even so, TMEM columns 128-255 are still allocated and left unused. Let
Q
K
(
i
)
QK(i)
and
S
(
i
)
S(i)
denote the
Q
K
QK
matmul and corresponding scores for
K
V
KV
block
i
i
. With only a single
Q
Q
-tile,
Q
K
(
i
)
QK(i)
,
softmax
⁡
(
S
(
i
)
)
\operatorname{softmax}(S(i))
, and
Q
K
(
i
+
1
)
QK(i+1)
proceed serially. Thus, nothing hides the latency of softmax, even though
Q
K
(
i
+
1
)
QK(i+1)
does not depend on
softmax
⁡
(
S
(
i
)
)
\operatorname{softmax}(S(i))
. We implement an alternative parallelism strategy to rectify this.
The S/P Ping-Pong
For simplicity, we refer to TMEM columns 0-127 as slot 0 and columns 128-255 as slot 1. Figure 2 shows which operands are resident in each slot during the prologue and the first few iterations of the main loop for the first work for the base path.
Figure 2.
The operands live in slots 0 and 1 for the base path.
S
S
tiles are in blue,
P
P
tiles are in yellow, and grey means empty.
Ping-pong overlaps
Q
K
(
i
+
1
)
with
softmax
⁡
(
S
(
i
)
)
by utilizing the unused TMEM in slot 1. In particular, where the MMA warp writes
S
S
and the softmax warps write
P
P
will ping-pong between the two buffers as shown in Figure 3:
Figure 3.
The operands live in slots 0 and 1 for the S/P ping-pong path.
where the new issue order is as follows:
Figure 4.
MMA/softmax schedule for the S/P ping-pong path.
The cell widths in Figure 2 are not proportional to execution time.
Figure 5 summarizes some of the synchronizations among the MMA, load, softmax, and correction warps with respect to the ping-pong. Dashed arrows indicate a TMEM value being consumed. The
O
O
accumulator uses a single buffer, but is drawn twice for visual clarity. While the correction warps are depicted issuing “rescale” each iteration, whether or not rescaling is actually done ultimately depends on the extent to which the row-max has changed.
Figure 5
. Overview of pipelines and synchronizations for the
S
/
P
S/P
ping-pong. A dotted arrow depicts an operand in TMEM being consumed. The two
O
O
‘s represent the same accumulation buffer.
Implementation
To implement the ping-pong, careful coordination is necessary to ensure warps are waiting on or consuming from the appropriate slot. The original code only needed a single phase bit since, with a single
S
S
slot, there is a single barrier whose phase flips once per block. With two
S
S
slots, each slot’s barrier flips once every other block. So we must keep track of which barrier to use and how many times that barrier has already flipped. Thus, we use a pair of barriers and two bits: bit 0 and bit 1 of the global count of
P
V
PV
matmuls issued (
mma_pv_count
). Bit 0 selects the barrier and bit 1 tracks (mod 2) how many times the given slot’s barrier has flipped.
The prologue proceeds as follows:
Load warp copies
Q
0
Q0
and
K
0
K0
to SMEM.
MMA warp issues
Q
K
(
0
)
QK(0)
.
Load warp copies
K
1
K1
to SMEM.
The important change is the
K
V
KV
load order. Previously, the load warp produced tiles of
K
K
and
V
V
in the following order:
K
0
,
V
0
,
K
1
,
V
1
,
K
2
,
V
2
,
…
,
K0, V0, K1, V1, K2, V2, \dots ,
. The ping-pong path path produces
K
0
,
K
1
,
V
0
,
K
2
,
V
1
,
K
3
,
V
2
,
…
K0, K1, V0, K2, V1, K3, V2, \dots
before appending the final tile of
V
V
at the end. The second
K
K
tile being loaded is to enable the MMA warp to issue
Q
K
(
1
)
QK(1)
immediately upon entering the main loop. Here is the code for the prologue:
if const_expr(self.use_s_ping_pong):
   # Wait for Q
   pipeline_q.consumer_wait_w_index_phase(0, mma_q_consumer_phase)
   # Wait for K(0)
   pipeline_kv.consumer_wait(mma_kv_consumer_state)
   Ki_index, Ki_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
   sK_cur = sK[None, None, None, Ki_index]
   if const_expr(self.uneven_kv_smem):
       sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)
   # Issue QK(0).
   if (mma_pv_count & 1) == 0:
       gemm_Si[0](smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sK_cur.iterator))
       pipeline_s_p_o.producer_commit_w_index(0)
   else:
       gemm_Si[1](smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sK_cur.iterator))
       pipeline_s_p_o.producer_commit_w_index(1)
   mma_q_consumer_phase ^= 1
   # Release K(0)
   pipeline_kv.consumer_release(mma_kv_consumer_state)
   # Advance to K(1)
   mma_kv_consumer_state.advance() 
   O_should_accumulate = False
The variable
mma_pv_count
is a global count of
P
V
PV
matmuls issued. Calling
gemm_Si[0]
instructs
Q
K
(
0
QK(0
) to write to slot 0. The
pipeline_s_p_o.producer_commit_w_index(0)
call signals to the softmax warpgroup to wait for the
Q
K
(
0
)
QK(0)
matmul to finish before giving the go-ahead to consume
S
(
0
)
S(0)
from slot 0. Similarly,
gemm_Si[1]
and
pipeline_s_p_o.producer_commit_w_index(1)
do the same except for slot 1.
The ping-pong main loop proceeds as follows:
for i in cutlass.range(block_iter_count - 1, unroll=1):
   # Wait for K(i+1)
   pipeline_kv.consumer_wait(mma_kv_consumer_state)
   Ki_index, Ki_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
   sK_cur = sK[None, None, None, Ki_index]
   if const_expr(self.uneven_kv_smem):
       sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)
   # Issue QK(i+1). Even global block count writes to TMEM slot 0 (0-127) and odd writes to TMEM slot 1 (128-255)
   if ((mma_pv_count + 1) & 1) == 0:
       gemm_Si[0](smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sK_cur.iterator))
       pipeline_s_p_o.producer_commit_w_index(0)
   else:
       gemm_Si[1](smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sK_cur.iterator))
       pipeline_s_p_o.producer_commit_w_index(1)
   # Release K(i+1)
   pipeline_kv.consumer_release(mma_kv_consumer_state)
   # Advance to V(i)
   mma_kv_consumer_state.advance()
   # Wait for V(i)
   pipeline_kv.consumer_wait(mma_kv_consumer_state)
   Vi_index, Vi_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
   tOrVi = tOrV[None, None, None, Vi_index]
   sV_cur = sV[None, None, None, Vi_index]
   if const_expr(self.uneven_kv_smem):
       sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
   # Phase for this block's slot. Each slot is reused every second block, and its barrier phase flips on each reuse.
   pv_phase = (mma_pv_count >> 1) & 1
   # Issue PV(i)
   if (mma_pv_count & 1) == 0:
       pipeline_s_p_o.producer_acquire_w_index_phase(0, pv_phase)
       gemm_Pi[0](
           tCrB=tOrVi,
           sB=sV_cur,
           zero_init=not O_should_accumulate,
           mbar_ptr=pipeline_p_lastsplit.sync_object_full.get_barrier(0) if self.split_P_arrive > 0 else None,
           mbar_phase=pv_phase,
       )
   else:
       pipeline_s_p_o.producer_acquire_w_index_phase(1, pv_phase)
       gemm_Pi[1](
           tCrB=tOrVi,
           sB=sV_cur,
           zero_init=not O_should_accumulate,
           mbar_ptr=pipeline_p_lastsplit.sync_object_full.get_barrier(1) if self.split_P_arrive > 0 else None,
           mbar_phase=pv_phase,
       )
   pipeline_o_acc.producer_commit_w_index(mma_pv_count & 1)
   mma_pv_count += 1
   # Release V(i)
   pipeline_kv.consumer_release(mma_kv_consumer_state)
   # Advance to K(i+2)
   mma_kv_consumer_state.advance() 
   O_should_accumulate = True
Here are a few notes to supplement the comments in the code block. The slot the
Q
K
QK
matmul writes to depends on the parity of
mma_pv_count + 1
since
Q
K
QK
runs one ahead of
P
V
PV
. The line
pv_phase = (mma_pv_count >> 1) & 1
extracts bit 1 from
mma_pv_count
, which is the parity of the reuse count of slot
mma_pv_count & 1
. Figure 6 summarizes how the slot PV writes to and
pv_phase
change with
mma_pv_count
through the first several iterations.
Figure 6.
PV slot and pv_phase for the given
mma_pv_count
.
The
pipeline_s_p_o.producer_acquire_w_index_phase(0/1, pv_phase)
call serves two purpose:
It waits on the softmax warps to finish producing
P
(
i
)
P(i)
and release the slot.
It waits on the correction warps to finish any required
O
O
rescale and release the slot.
The
pipeline_o_acc.producer_commit_w_index(mma_pv_count & 1)
call signals that the MMA warp is done accumulating this iteration’s
P
V
PV
into the
O
O
accumulation buffer so it is safe for correction warps to consume. Note that this is not required for the original path, since the serialization guaranteed that
P
V
(
i
)
PV(i)
would complete before the potential correction rescale for
S
(
i
+
1
)
S(i+1)
would arrive.
IKET Profiling
NVIDIA’s In-Kernel Event Tracing (
IKET
)  enables us to examine the activity of individual warps over the lifetime of a kernel. While our changes in theory allow the MMA warp to issue
Q
K
(
i
+
1
)
QK(i+1)
while the softmax warps are still processing
S
(
i
)
S(i)
, IKET can verify that this scheduling change actually occurs. To that end, we provide two traces from IKET: base vs ping-pong. Here is the trace from the base path:
Figure 7
. Order of instructions issued by load, MMA, and softmax warps in the FA4 base path during decode.
The relevant sequence is:
Q
K
QK
matmul (
mma_issue_QK
)
softmax(
S
S
) (
sm_compute
)
P
V
PV
matmul (
mma_issue_PV
)
Q
K
QK
matmul (
mma_issue_QK
)
There is little to no overlap between the bars for these operations, so they are effectively serialized. For comparison, here is the trace from the ping-pong path:
Figure 8
. Order of instructions issued by load, MMA, and softmax warps in the FA4 S/P ping-pong path during decode.
Notice how the blue
mma_issue_QK
bars overlap significantly with the purple
sm_compute
bars. This is concrete evidence that the ping-pong path is issuing
Q
K
QK
matmuls and softmax concurrently.
Performance
Decoding is memory-bound so we report achieved memory bandwidth as the performance metric. Figure 9 displays benchmark results measured on an NVIDIA B200 Blackwell GPU for a subset of shapes where the ping-pong path is taken, including grouped-query attention (GQA) with ratios
H
q
:
H
k
v
H_q:H_{kv}
= 16:1 and 16:2. The benefit is small at shorter
K
V
KV
sequence lengths, but grows substantially as the sequences become longer. This is consistent with the change targeting steady-state, where longer sequences expose more iterations in which the
Q
K
QK
and softmax overlap can be benefited from. For head dimension 64, the increase in achieved bandwidth reaches 15.6% for GQA ratio
16
:
2
16:2
and 16.0% for GQA ratio 16:1. For head dimension 128, the increases are up to 9.3% for 16:1 and 14.9% for 16:2. Figure 10 compares both single and multi-token decode at a fixed sequence length of 128k. The improvements are consistent across all query lengths considered.
Figure 9.
Single token FA4 decode benchmark results for base path (in blue) vs ping-pong path (in orange) measured on an NVIDIA B200 Blackwell GPU.
Figure 10.
Single and multi-token FA4 decode benchmark results for base path (in blue) vs ping-pong path (in orange) measured on an NVIDIA B200 Blackwell GPU. Sequence length is fixed to 128k.
Conclusion
In this blog post, we discussed an optimization for FA4 decode that removes an unnecessary serialization between
Q
K
QK
and softmax by ping-ponging
S
/
P
S/P
across two TMEM slots, one of which was previously idle. We illustrated how parallelism strategies for prefill do not always apply to decode, covered implementation details of the ping-pong, and generated IKET traces to confirm the intended overlap. Benchmarks reported gains in performance of up to 16%.
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
