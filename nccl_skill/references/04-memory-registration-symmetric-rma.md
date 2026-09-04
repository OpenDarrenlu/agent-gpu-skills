# 04 - Memory Registration, Symmetric Windows, Suspend/Resume, and RMA Signals

## Primary source files

- `repos/nccl/src/nccl.h.in`: public memory, registration, window, suspend, RMA/signal APIs.
- `repos/nccl/src/allocator.cc`: `ncclMemAlloc` / `ncclMemFree` implementation.
- `repos/nccl/src/mem_manager.cc`: communicator memory tracking and suspend/resume.
- `repos/nccl/src/include/mem_manager.h`: tracked memory metadata.
- `repos/nccl/src/register/register.cc`, `coll_reg.cc`, `sendrecv_reg.cc`: registration internals.
- `repos/nccl/src/rma/*`: one-sided put/signal/wait implementation.
- `repos/nccl/src/dev_runtime.cc`: window register/deregister, symmetric memory spaces, CFT binding.
- `repos/nccl/src/cft_dev_runtime.cc`: CUDA logical endpoint (CFT) setup and host query APIs.
- `repos/nccl/src/sym_kernels.cc`, `repos/nccl/src/include/sym_kernels.h`: symmetric kernel selection and tuning.
- `repos/nccl/src/plugin/rma.cc`, `repos/nccl/plugins/rma/README.md`: RMA plugin loading and developer guide.
- `repos/nccl/docs/examples/04_user_buffer_registration/*`
- `repos/nccl/docs/examples/05_symmetric_memory/*`

## NCCL allocation helpers

```c
ncclMemAlloc(void** ptr, size_t size);
ncclMemFree(void* ptr);
```

`ncclMemAlloc` allocates memory in a form suitable for NCCL optimizations such as user buffer
registration and symmetric windows. The actual allocated size may be larger than requested due to
alignment/granularity requirements.

Use `ncclMemAlloc` instead of raw `cudaMalloc` when:

- The buffer will be registered with `ncclCommRegister`.
- The buffer will be used as a symmetric memory window.
- You are following Device API examples.
- You want NCCL to choose a compatible allocation strategy for advanced transports.

## User buffer registration

### Public API

```c
ncclCommRegister(comm, buff, size, &handle);
ncclCommDeregister(comm, handle);
```

Registration lets NCCL operate directly on user-allocated buffers, avoiding repeated per-call
registration/staging overhead. The in-repo examples describe it as useful for repeated collectives on
the same buffers and as a prerequisite for advanced features such as symmetric memory or Device API calls.

### Typical shape

```c
void* d_send = NULL;
void* d_recv = NULL;
void* send_handle = NULL;
void* recv_handle = NULL;

ncclMemAlloc(&d_send, size_bytes);
ncclMemAlloc(&d_recv, size_bytes);

ncclCommRegister(comm, d_send, size_bytes, &send_handle);
ncclCommRegister(comm, d_recv, size_bytes, &recv_handle);

ncclAllReduce(d_send, d_recv, count, ncclFloat32, ncclSum, comm, stream);

ncclCommDeregister(comm, send_handle);
ncclCommDeregister(comm, recv_handle);
ncclMemFree(d_send);
ncclMemFree(d_recv);
```

### When registration helps

| Scenario | Why registration helps |
|---|---|
| repeated collectives on same buffers | amortizes registration and transport setup |
| RDMA-capable networking | avoids repeated memory registration with NIC |
| symmetric memory / Device API | required or strongly expected by the feature |
| channel/resource pressure | can reduce internal NCCL buffering/resource usage |

### Pitfalls

- Do not deregister before all CUDA stream work using the buffer has completed.
- Match registration handles with the communicator that created them.
- Use NCCL allocation helpers for examples unless you know the transport's registration requirements.
- Registration may not improve one-off small operations; benchmark representative workloads.

## Symmetric memory windows

### Public API

```c
ncclCommWindowRegister(comm, buff, size, &win, winFlags);
ncclCommWindowDeregister(comm, win);
ncclWinGetUserPtr(comm, win, &ptr);
```

Window flags from `nccl.h.in`:

| Flag | Meaning |
|---|---|
| `NCCL_WIN_DEFAULT` | default behavior |
| `NCCL_WIN_COLL_SYMMETRIC` | register window for symmetric collective optimizations |
| `NCCL_WIN_STRICT_ORDERING` | request stricter ordering behavior |

`NCCL_WIN_REQUIRED_ALIGNMENT` is 4096.

Since NCCL 2.30.7, ranks may register windows with asymmetric sizes. Registration exchanges each
rank's window size with `bootstrapAllGather` and reserves flat VA space using the largest size
across LSA peers (`lsaMinSize` / `lsaMaxSize` in `ncclDevrMemory`, see
`repos/nccl/src/dev_runtime.cc`).

Window registration and deregistration also work during CUDA graph capture: the register,
deregister, and finalize paths wrap their CUDA calls with
`cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureModeRelaxed)` (see
`repos/nccl/src/dev_runtime.cc`, e.g. `ncclDevrWindowRegisterInGroup` and
`ncclCommWindowDeregister`).

Multiple windows can be carved out of one backing cuMem allocation. Each window records its offset
within the backing allocation (`memOffset`), and the RMA/symmetric paths compute addresses relative
to the allocation base via `ncclDevrGetWinOffset` (`repos/nccl/src/dev_runtime.cc`,
`repos/nccl/src/rma/rma_proxy_launch.cc`). Earlier versions could corrupt data in this scenario
(GitHub issue #2198). The window start must stay `NCCL_WIN_REQUIRED_ALIGNMENT`-aligned within the
backing allocation.

### Symmetric collective window shape

```c
void* buffer = NULL;
ncclWindow_t win;

ncclMemAlloc(&buffer, size_bytes);
ncclCommWindowRegister(comm, buffer, size_bytes, &win, NCCL_WIN_COLL_SYMMETRIC);

ncclAllReduce(buffer, buffer, count, ncclFloat32, ncclSum, comm, stream);

ncclCommWindowDeregister(comm, win);
ncclMemFree(buffer);
```

### Why symmetric windows exist

The examples describe symmetric windows as enabling optimized collective protocols when all ranks use
consistent memory layouts. Memory must be allocated through CUDA Virtual Memory Management-compatible
paths and registered with NCCL. Symmetric memory is especially relevant for:

- Very low latency local collectives.
- Device API LSA peer access.
- Multimem-style local collectives where supported.
- Future-proof buffer layouts for NCCL internals.

Symmetric windows back a set of dedicated symmetric kernels, including several ReduceScatter
variants (`ReduceScatter_LL`, `ReduceScatter_TmaLD`, `ReduceScatter_LD`, `ReduceScatter_LDMC`,
`ReduceScatter_RailA2A_LsaLD`, `ReduceScatter_RailA2A_LsaLDMC` in
`repos/nccl/src/include/sym_kernels.h`). Kernel selection and block/buffer sizing are tuning-driven
(see `getRequirements_gin` and the RailA2A tuning helpers in `repos/nccl/src/sym_kernels.cc`).

### CFT logical endpoint host APIs

Since NCCL 2.31, window memory can be mapped to CUDA logical endpoints (LEs). When a communicator
has CFT support, registration binds window memory to the team's LEs with
`cuLogicalEndpointBindAddr` (`symBindTeamLe` in `repos/nccl/src/cft_dev_runtime.cc`). Three host
query APIs (declared in `repos/nccl/src/include/nccl_device/core.h`, implemented in
`repos/nccl/src/cft_dev_runtime.cc`) translate a `(window, offset)` pair into a logical endpoint ID
plus offset:

```c
ncclGetMultimemDeviceLeInfo(window, offset, &leId, &leOffset);
ncclGetCftDeviceLeInfo(window, offset, peerCft, cftTeam, &leId, &leOffset);
ncclGetPeerDeviceLeInfo(window, offset, peerWorld, &leId, &leOffset);
```

These require a CFT-capable communicator/driver (CUDA driver 13.3+; `gpuCftSupport` in
`repos/nccl/src/cft_dev_runtime.cc`). Behavior is controlled by the `hostCftMode` communicator
config field (`ncclHostCftMode_t`: `ncclHostCftEnable` / `ncclHostCftDisable` /
`ncclHostCftFallback` in `repos/nccl/src/nccl.h.in`) and the `NCCL_CFT_ENABLE` environment
variable.

## Communicator memory suspend/resume

### Public API

```c
#define NCCL_SUSPEND_MEM 0x01
ncclCommSuspend(comm, NCCL_SUSPEND_MEM);
ncclCommResume(comm);
```

Suspend releases suspendable dynamic GPU allocations tracked by NCCL. The communicator cannot be used
while suspended. Resume restores previously suspended resources.

### Memory statistics

```c
ncclCommMemStats(comm, ncclStatGpuMemSuspend, &value);
ncclCommMemStats(comm, ncclStatGpuMemSuspended, &value);
ncclCommMemStats(comm, ncclStatGpuMemPersist, &value);
ncclCommMemStats(comm, ncclStatGpuMemTotal, &value);
```

Stats:

| Stat | Meaning |
|---|---|
| `ncclStatGpuMemSuspend` | bytes of GPU memory that can be suspended |
| `ncclStatGpuMemSuspended` | whether suspendable memory is suspended (`0` or `1`) |
| `ncclStatGpuMemPersist` | bytes of GPU memory that cannot be suspended |
| `ncclStatGpuMemTotal` | total NCCL-tracked GPU memory |

Use this when an application needs to temporarily reduce NCCL memory footprint, e.g. between phases or
while another subsystem owns GPU memory.

## One-sided RMA/signal APIs

### `ncclPutSignal`

```c
ncclPutSignal(localbuff, count, datatype,
              peer, peerWin, peerWinOffset,
              sigIdx, ctx, flags, comm, stream);
```

Writes data from a local buffer to a remote peer's registered memory window and associates the operation
with a signal index/context. The target process does not explicitly post a matching receive in the same
way as `ncclRecv`.

Parameters to pay attention to:

| Parameter | Meaning |
|---|---|
| `peer` | target rank |
| `peerWin` | memory window registered by target peer |
| `peerWinOffset` | byte offset inside peer window |
| `sigIdx` | signal index identifier |
| `ctx` | context identifier |
| `flags` | reserved for future use in public header |

### `ncclSignal`

```c
ncclSignal(peer, sigIdx, ctx, flags, comm, stream);
```

Sends a signal to a peer without transferring data.

### `ncclWaitSignal`

```c
typedef struct {
  int opCnt;
  int peer;
  int sigIdx;
  int ctx;
} ncclWaitSignalDesc_t;

ncclWaitSignal(nDesc, signalDescs, comm, stream);
```

Waits for one or more signal descriptors. Each descriptor specifies how many signal operations to wait
for from a given peer/signal/context combination.

### Contexts, signals, and multiple NICs

Since NCCL 2.31, one-sided RMA supports multiple contexts and multiple signal indices per rank:

- The `numRmaCtx` communicator config field (env `NCCL_NUM_RMA_CTX`, see `repos/nccl/src/init.cc`)
  sets the number of RMA contexts; each context has its own task queue
  (`planner.rmaTaskQueues`, `repos/nccl/src/include/comm.h`). Setting `NCCL_NUM_RMA_CTX=0` disables
  host RMA for the communicator.
- The `numRmaSig` config field sets the number of signal indices per context. Each context
  allocates a signal buffer laid out as `numRmaSig * nRanks` 64-bit counters, with the slot for a
  `(sigIdx, rank)` pair at `signalIdx * nRanks + rank` (`ncclRmaSignalSlot` in
  `repos/nccl/src/include/rma/rma.h`, `repos/nccl/src/include/rma/rma_proxy.h`).
- A single rank's one-sided RMA traffic can span multiple NICs: the RMA proxy creates one
  connection (`collComm`) per RMA-capable network device and distributes contexts across them
  round-robin (`collCommIdx = ... % rmaCommCount` in `repos/nccl/src/rma/rma_proxy.cc`).

Ordering guarantees are per-context: completing a signal implies completion of previous puts on the
same context only (see `repos/nccl/plugins/rma/README.md`). Host RMA can also be disabled entirely
with `NCCL_RMA_DISABLE=1` (`repos/nccl/src/rma/rma.cc`).

### RMA plugin architecture

The RMA plugin interface was refactored in NCCL 2.30.7. Loading behavior
(`repos/nccl/src/plugin/rma.cc`, developer guide in `repos/nccl/plugins/rma/README.md`):

- NCCL first tries `libnccl-rma.so`, or the comma-separated plugin list in `NCCL_RMA_PLUGIN`
  (bare name, library file name, or absolute path).
- If no external plugin loads, NCCL looks for RMA symbols in the GIN plugin, then the NET plugin,
  and finally falls back to its internal IB implementation (`ncclRmaIbProxy`).
- Only one implementation is used: the first plugin whose `init` succeeds and reports at least one
  device wins; the remaining external plugins are disabled.
- Plugins export versioned symbols `ncclRmaPlugin_vX`; the in-tree internal implementation supports
  v13-v15 (`getNcclRma_v13/14/15` in `repos/nccl/src/plugin/rma.cc`).
- Plugin memory registration is symmetric (`regMrSym` / `regMrSymDmaBuf`); `mrFlags` such as
  `NCCL_NET_MR_FLAG_FORCE_SO` and `NCCL_NET_MR_FLAG_SIGNAL_NEVER_RESET` mark signal-capable
  regions.
- `plugins/rma/example` contains a minimal compile/load example plugin.

### Copy engine (CE) path

For intra-node (single LSA team) peers, put and wait operations can be executed by CUDA copy
engines instead of the network proxy (`repos/nccl/src/rma/rma_ce.cc`). The CE path batches
operations with `ncclCeBatchOpsParams` / `ncclCeLaunchBatchOps` (`repos/nccl/src/ce_coll.cc`),
which use `cudaMemcpyBatchAsync` on CUDA 12.8+ and fall back to per-op `cudaMemcpyAsync` during
graph capture or on the legacy stream. Data copies and signal writes are submitted as separate
batches because batched copies do not guarantee execution order; the wait path batches
`CUstreamBatchMemOpParams` wait/write ops per `(signalIdx, peer)` slot.

## RMA vs P2P vs collectives

| Need | Prefer |
|---|---|
| standard distributed training gradients | `ncclAllReduce` or reduce-scatter/all-gather decomposition |
| explicit pairwise exchange | `ncclSend` / `ncclRecv` with groups |
| one-sided writes into registered peer windows | `ncclPutSignal` + `ncclWaitSignal` |
| custom in-kernel peer access | Device API with windows/LSA/GIN |
| repeated collectives on fixed buffers | `ncclCommRegister` or symmetric windows where applicable |

## Registration internals map

- `src/register/register.cc`: generic local/user registration logic.
- `src/register/coll_reg.cc`: collective registration paths.
- `src/register/sendrecv_reg.cc`: send/recv registration paths.
- `src/mem_manager.cc`: tracks communicator allocations, peer import/export, suspend/resume, stats.
- `src/transport/net.cc` and `src/transport/net_ib/reg.cc`: network transport registration behavior.
- `src/dev_runtime.cc`: `ncclCommWindowRegister` / `ncclCommWindowDeregister`, `symMemoryObtain`,
  `symWindowCreate`, flat VA space management (`ncclDevrGetWinOffset` maps a window to its offset
  within the backing allocation).
- `src/rma/rma.cc`: RMA task batching across per-context queues.
- `src/rma/rma_proxy*.cc`: network-proxy RMA contexts, signal buffers, descriptor build.
- `src/rma/rma_ce.cc`: copy-engine put/wait path.
- `src/plugin/rma.cc`: RMA plugin discovery/selection (`NCCL_RMA_PLUGIN`).

When modifying registration code, trace both the public handle lifetime and the transport-specific memory
handle lifetime.

## Practical safety checklist

1. Allocate/initialize buffers before registration.
2. Register with every communicator that will use the buffer.
3. Enqueue NCCL work only after registration succeeds.
4. Synchronize streams or otherwise prove work completion before deregistration/free.
5. Keep window offsets aligned where required (`NCCL_WIN_REQUIRED_ALIGNMENT`, 4096, also applies to
   the window start within a larger backing allocation).
6. Do not assume peer window address equality unless using APIs that explicitly expose peer pointers.
7. For hangs, verify all ranks register compatible windows and reach the same signal/wait sequence.
