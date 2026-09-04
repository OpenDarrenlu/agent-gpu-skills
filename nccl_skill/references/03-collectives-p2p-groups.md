# 03 - Collectives, Point-to-Point Operations, and Group Semantics

## Primary source files

- `repos/nccl/src/nccl.h.in`: public operation semantics and prototypes.
- `repos/nccl/src/collectives.cc`: API wrappers that create `ncclInfo` and call `ncclEnqueueCheck`.
- `repos/nccl/src/include/info.h`: normalized API-call descriptor.
- `repos/nccl/src/group.cc`: group start/end, async jobs, grouped launch.
- `repos/nccl/src/config/collconfig.cc`: per-collective `ncclCollConfig_t` validation and resolution.
- `repos/nccl/src/ce_coll.cc`: zero-SM Copy-Engine collectives (flat and hierarchical paths).
- `repos/nccl/src/rma/rma_proxy.cc`: RMA CPU proxy backing the hierarchical collectives' inter-node rail.
- `repos/nccl/src/tuning/pat.cc`: PAT algorithm cost model and enablement.
- `repos/nccl/docs/examples/02_point_to_point/*`: P2P ring example.
- `repos/nccl/docs/examples/03_collectives/*`: AllReduce example.

## Enqueue semantics

NCCL collective and P2P API calls generally return after work is **enqueued on a CUDA stream**, not
after communication is complete. Host code must synchronize the relevant CUDA stream or use CUDA events
when it needs the data.

This distinction matters for:

- Correctness checks after collectives.
- Buffer lifetime and reuse.
- Destroy/finalize timing.
- CUDA graph capture.
- Async error handling.

## Collective operation table

| API | Data movement | Reduction op? | Root? | In-place rule |
|---|---|---:|---:|---|
| `ncclReduce` | all ranks reduce into root recv buffer | yes | yes | `sendbuff == recvbuff` |
| `ncclBroadcast` | root data copied to every rank | no | yes | `sendbuff == recvbuff` |
| `ncclBcast` | deprecated in-place broadcast | no | yes | implicitly in-place |
| `ncclAllReduce` | reduce and distribute result to all ranks | yes | no | `sendbuff == recvbuff` |
| `ncclReduceScatter` | reduce then scatter equal blocks | yes | no | `recvbuff == sendbuff + rank * recvcount` |
| `ncclAllGather` | gather equal blocks from all ranks | no | no | `sendbuff == recvbuff + rank * sendcount` |
| `ncclAlltoAll` | each rank sends equal count to every rank | no | no | layout-dependent |
| `ncclGather` | every rank sends equal count to root | no | yes | root: `sendbuff == recvbuff + root * count` |
| `ncclScatter` | root sends equal count to every rank | no | yes | root: `recvbuff == sendbuff + root * count` |

The `root` parameter is always a **rank**, not a CUDA device ID.

## Data types

`ncclDataType_t` values from the public header:

| Type | Aliases/notes |
|---|---|
| `ncclInt8` | `ncclChar` |
| `ncclUint8` | unsigned 8-bit |
| `ncclInt32` | `ncclInt` |
| `ncclUint32` | unsigned 32-bit |
| `ncclInt64` | signed 64-bit |
| `ncclUint64` | unsigned 64-bit |
| `ncclFloat16` | `ncclHalf` |
| `ncclFloat32` | `ncclFloat` |
| `ncclFloat64` | `ncclDouble` |
| `ncclBfloat16` | bfloat16 |
| `ncclFloat8e4m3` | FP8 e4m3 |
| `ncclFloat8e5m2` | FP8 e5m2 |

## Reduction operations

Built-in `ncclRedOp_t` values:

| Op | Meaning |
|---|---|
| `ncclSum` | sum |
| `ncclProd` | product |
| `ncclMax` | maximum |
| `ncclMin` | minimum |
| `ncclAvg` | average |

Dynamic reduction operation support:

```c
ncclRedOpCreatePreMulSum(&op, scalar, datatype, residence, comm);
ncclRedOpDestroy(op, comm);
```

`ncclScalarResidence_t` controls when the scalar is read:

| Residence | Meaning |
|---|---|
| `ncclScalarDevice` | scalar is in device-visible memory and read while the collective runs |
| `ncclScalarHostImmediate` | scalar is read from host-visible memory before create returns |

## Per-collective config variants (`nccl*Config`)

Every collective has a config-taking variant declared in `repos/nccl/src/nccl.h.in` and implemented in
`repos/nccl/src/collectives.cc`:

```c
ncclCollConfig_t cfg = NCCL_COLLCONFIG_INITIALIZER;
cfg.algSelection = "PAT";        // restrict algorithm choice for this call
ncclAllReduceConfig(sendbuff, recvbuff, count, datatype, op, comm, stream, &cfg);
```

The variants are `ncclAllReduceConfig`, `ncclBroadcastConfig`, `ncclReduceConfig`, `ncclAllGatherConfig`,
`ncclReduceScatterConfig`, `ncclAlltoAllConfig`, `ncclGatherConfig`, and `ncclScatterConfig`. Passing
`config == NULL` is equivalent to the plain API. Key `ncclCollConfig_t` fields:

| Field | Meaning |
|---|---|
| `minCTAs` / `maxCTAs` / `nvlsCTAs` | per-call channel/CTA bounds |
| `cgaClusterSize` | thread-block-cluster size (Hopper+); must be consistent within a group |
| `algSelection` | algorithm filter string; NULL/"" means automatic selection |
| `forceAlgSelection` | default 1: an unsatisfiable `algSelection` is a hard error; 0 falls back to automatic |
| `CTAPolicy` | `NCCL_CTA_POLICY_*` bitmask; unset inherits the communicator-level policy |
| `userProfilerTag` | opaque tag delivered verbatim to profiler plugins |

The config must be initialized with `NCCL_COLLCONFIG_INITIALIZER` and set identically on every rank;
NCCL validates it only locally (`repos/nccl/src/config/collconfig.cc`). `ncclSend`/`ncclRecv` have no
config variant.

## Zero-SM (Copy-Engine) collectives

With the CTA policy `NCCL_CTA_POLICY_ZERO` — set via the comm config `CTAPolicy` field, the
`NCCL_CTA_POLICY` environment variable, or per call through `ncclCollConfig_t.CTAPolicy` — the enqueue
path routes eligible collectives to a Copy-Engine (CE) task instead of an SM kernel
(`ceCollTaskAppend` in `repos/nccl/src/enqueue/enqueue.cc`), leaving SMs free for compute/communication
overlap. Eligibility is decided by `ncclCeAvailable` / `ncclHierCeAvailable` in
`repos/nccl/src/ce_coll.cc`:

- CE collectives are implemented for AllGather, AlltoAll, Scatter, and Gather (CUDA driver 12.5+),
  require symmetric memory support, and need registered windows for the communication buffers.
- Single-node: the flat CE path requires all ranks in one NVLink (LSA) team.
- Multi-node: the hierarchical path supports **AllGather and AlltoAll only**. Intra-node movement uses
  CE scatters over LSA pointers; inter-node movement goes through the RMA CPU proxy
  (`ncclRmaWantInternalCtx` in `repos/nccl/src/rma/rma_proxy.cc`), which requires host RMA support.
  Both send and recv buffers must be in registered symmetric windows.

Refinements in the current source:

- Hierarchical collectives distribute inter-node rail puts/waits across multiple internal RMA proxy
  contexts and signals (default 4 contexts, `NCCL_NUM_RMA_INT_CTX`; per-transfer fan-out gated by
  `NCCL_RMA_MULTI_CTX_THRESHOLD`, default 4 MiB; `NCCL_HIER_CE_COLL_NUM_CTX` caps usage). These internal
  contexts sit above the user-addressable range, so `numRmaCtx == 0` does not block the rail.
- CE AllGather can use a single NVLink multicast write covering all LSA peers for smaller per-rank
  messages (`ncclCeAllGatherUseMulticast`); the unicast/multicast choice is cost-model driven, with
  `NCCL_CE_COLL_AG_MULTICAST_THRESHOLD` as a byte-threshold override.

## PAT algorithm

PAT (`NCCL_ALGO_PAT`, SIMPLE protocol only) is registered for AllGather and ReduceScatter
(`repos/nccl/src/config/algorithm_registry.cc`). Its cost model lives in `repos/nccl/src/tuning/pat.cc`:

- Gated by `NCCL_PAT_ENABLE`; requires SM60+ and host net device (no net device offload).
- Multi-node (multi-RPN) PAT is hierarchical: NVLS for the intra-node phase, PAT across nodes; it
  requires NVLS support and is currently opt-in — the cost model prices it out of automatic selection,
  so request it explicitly via `ncclCollConfig_t.algSelection`.

## `ncclInfo`: how public calls enter internals

`collectives.cc` turns each API call into an `ncclInfo` record and calls `ncclEnqueueCheck`.
Examples:

- `ncclAllReduce` sets `func=ncclFuncAllReduce`, operation name `"AllReduce"`, datatype, op, count,
  communicator, stream, and allreduce chunk/slice steps.
- `ncclSend` and `ncclRecv` set `func=ncclFuncSend` / `ncclFuncRecv`, peer in `root`, and `chunkSteps=1`.
- RMA/signal APIs set peer window, signal index, context, flags, or wait descriptors.

This is important for source debugging: public API bugs usually route through:

```text
nccl.h.in declaration
  -> collectives.cc wrapper
  -> ncclEnqueueCheck(info)
  -> taskAppend / scheduling in enqueue.cc
```

## Group semantics

### Why groups exist

The public header explains two major reasons:

1. When one host thread manages multiple GPUs, NCCL calls for different ranks may need inter-CPU
   synchronization. Grouping lets them be submitted as one operation.
2. Grouping fuses multiple operations to improve performance or allow concurrent progress, especially
   for multiple send/recv operations that would otherwise deadlock.

### API

```c
ncclGroupStart();
// NCCL calls only; no dependent CUDA work between start/end.
ncclGroupEnd();
```

`ncclGroupStart` queues NCCL calls until `ncclGroupEnd`. Nothing starts on the CUDA stream until group
end. `ncclGroupEnd` starts the fused operation and returns when operations have been enqueued, not when
they have completed on device.

### Simulate end

```c
ncclSimInfo_t sim = NCCL_SIM_INFO_INITIALIZER;
ncclGroupSimulateEnd(&sim);
printf("estimated time = %f\n", sim.estimatedTime);
```

Use simulation for planning/introspection, not as a replacement for benchmarking.

## Collective call ordering

All ranks in a communicator clique must call matching collective operations in compatible order. Mismatched
operation order, datatype, count, root, or participation usually causes hangs or asynchronous errors.

For example, if rank 0 calls AllReduce then Broadcast while rank 1 calls Broadcast then AllReduce, both
ranks can wait forever because the device kernels and proxy work are not matching the same operation
sequence.

## Single-process multi-GPU collective pattern

```cpp
ncclGroupStart();
for (int r = 0; r < nranks; ++r) {
  cudaSetDevice(devices[r]);
  ncclAllReduce(send[r], recv[r], count, ncclFloat32, ncclSum, comms[r], streams[r]);
}
ncclGroupEnd();

for (int r = 0; r < nranks; ++r) {
  cudaSetDevice(devices[r]);
  cudaStreamSynchronize(streams[r]);
}
```

Use this shape when one host thread loops over devices.

## P2P operations

### `ncclSend`

```c
ncclSend(sendbuff, count, datatype, peer, comm, stream);
```

Sends data from this rank to `peer`. The peer must call `ncclRecv` with the same datatype and count
from this rank.

### `ncclRecv`

```c
ncclRecv(recvbuff, count, datatype, peer, comm, stream);
```

Receives data from `peer`. The peer must call matching `ncclSend`.

### P2P blocking rule

The public header states that P2P operations are blocking for the GPU. If multiple sends and receives
must progress concurrently to complete, they must be fused within `ncclGroupStart` / `ncclGroupEnd`.

### Ring P2P example shape

```cpp
int prev = (rank - 1 + nranks) % nranks;
int next = (rank + 1) % nranks;

ncclGroupStart();
ncclSend(sendbuf, count, ncclFloat32, next, comm, stream);
ncclRecv(recvbuf, count, ncclFloat32, prev, comm, stream);
ncclGroupEnd();
```

In a single process managing all GPUs, the group must include all local ranks' sends and recvs.

## AllReduce example logic

The in-repo collective example uses a simple verification pattern:

1. Create one communicator per GPU with `ncclCommInitAll`.
2. Allocate per-GPU send/recv buffers and streams.
3. Fill send buffer with rank-derived values.
4. Group `ncclAllReduce(..., ncclSum, ...)` across ranks.
5. Synchronize streams.
6. Verify each rank received the expected global sum.
7. Finalize/destroy comms and free buffers.

This is a good template for correctness examples but not for measuring peak performance.

## Rooted collectives: reduce, broadcast, gather, scatter

When using rooted collectives, keep two namespaces separate:

- `root` is a communicator rank.
- CUDA device IDs are chosen by the application.

This is a common bug in MPI applications where `localRank` and global rank differ. Use
`ncclCommUserRank` or application rank mapping when in doubt.

## Count and layout rules

### AllGather

Each rank sends `sendcount` elements. `recvbuff` must contain at least `nranks * sendcount` elements.
Data from rank `i` is at `recvbuff + i * sendcount`.

### ReduceScatter

Each rank receives `recvcount` elements. `sendbuff` must contain at least `nranks * recvcount` elements.
The reduced result block for rank `i` goes to rank `i`.

### AlltoAll

Each rank sends `count` elements to every other rank. Data for destination rank `j` is read from
`sendbuff + j * count`; data received from source rank `i` is written to `recvbuff + i * count`.

### Gather

Each rank sends `count` elements to `root`. On root, data from rank `i` is placed at
`recvbuff + i * count`. Non-root `recvbuff` is unused.

### Scatter

Root sends `count` elements to every rank. Root reads rank `i`'s data at `sendbuff + i * count`.
Non-root `sendbuff` is unused.

## CUDA graph capture considerations

NCCL tracks graph capture in the enqueue planner (`planner->capturingGraph`). Grouped calls, captured
streams, persistent work buffers, and destructors have special handling. When answering graph-capture
questions:

- All streams in a grouped NCCL operation need compatible capture state.
- NCCL may record kernel launches rather than launch immediately during graph capture.
- Persistent plans and registration cleanup differ from normal launches.
- Debug with `NCCL_DEBUG=INFO` and check whether the user is mixing captured and non-captured streams.

## Practical deadlock checklist

For a hang involving collectives/P2P:

1. Does every rank call the same operation sequence?
2. Are `count`, `datatype`, `root`, and `peer` values compatible?
3. Are all ranks in the communicator participating?
4. Are multi-device single-thread calls wrapped in `ncclGroupStart/End`?
5. Are concurrent P2P sends/recvs grouped?
6. Is CUDA stream work before the NCCL operation blocking progress?
7. Did any rank return an immediate error that the application ignored?
8. What is the last `NCCL_DEBUG=INFO` line per rank?
