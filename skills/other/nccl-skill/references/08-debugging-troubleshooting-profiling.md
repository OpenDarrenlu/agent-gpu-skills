# 08 - Debugging, Troubleshooting, Logging, and Profiling

## Primary source files

- `repos/nccl/src/debug.cc`: logging subsystem and debug behavior.
- `repos/nccl/src/collectives.cc`: NVTX payload wrapping of public operations.
- `repos/nccl/src/init.cc`: init-time warnings and setup.
- `repos/nccl/src/enqueue.cc`: launch/capture/scheduler failure points.
- `repos/nccl/src/proxy.cc`: proxy progress engine.
- `repos/nccl/src/ras/*`: fault handling/RAS, including RAS diagnostics (`ras.cc`, `diagnostics_checks.*`).
- `repos/nccl/src/diagnostics.cc` and `repos/nccl/src/include/diagnostics/*`: active (data-path) diagnostics.
- `repos/nccl/src/transport/net_ib/wqe_lat_mon.cc`: per-QP IB WQE post-to-poll latency monitor.
- `repos/nccl/plugins/profiler/README.md`: profiler plugin API.
- `repos/nccl/src/include/plugin/nccl_profiler.h` and `repos/nccl/src/include/plugin/profiler/profiler_v7.h`: profiler interface versions (v7 is the default in 2.31.x).
- `repos/nccl/docs/examples/common/include/nccl_utils.h`: example `NCCLCHECK` macro style.

## First debug command

For most NCCL application issues, start with:

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET,TUNING ./your_app
```

If the hang involves many ranks, capture logs per rank:

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET,TUNING \
  mpirun -np 8 ./your_app 2>&1 | tee nccl.log
```

For production clusters, configure the launcher to write one rank log per file so the last line from
each rank is visible.

## Built-in diagnostics (NCCL 2.31+)

Since 2.31, NCCL ships two categories of built-in diagnostics; on 2.31+ deployments, run these
before manual log archaeology.

### RAS diagnostics (passive)

RAS diagnostics collect and compare GPU/CUDA/NCCL configuration state across ranks without
exercising data paths (`repos/nccl/src/ras/ras.cc`, `repos/nccl/src/ras/diagnostics_checks.cc`).
Reported checks:

- GPU inventory (same GPU count and model across ranks),
- CUDA driver version consistency,
- volatile SRAM/DRAM ECC error counters,
- NVLink count and active state,
- `NCCL_*` environment-variable name/value consistency across ranks.

Trigger at each communicator init:

```bash
NCCL_RUN_RAS_DIAGNOSTICS=1 ./your_app
```

or on demand for all live communicators via the RAS client (`ncclras -D`, or
`echo diagnostics | nc <host> 28028`). Requires the RAS subsystem (enabled by default). The report
is printed on the process hosting rank 0 of the communicator.

### Active diagnostics (P2P connectivity)

Active diagnostics (`repos/nccl/src/diagnostics.cc`) exercise actual communication paths during
communicator initialization. The P2P check transfers and validates data between every GPU pair NCCL
expects to communicate directly (NVLink or PCIe), and on failure identifies the affected GPU pair
and connection path. Enable with:

```bash
NCCL_RUN_DIAGNOSTICS=1 ./your_app
```

Report lines are prefixed with `NCCL DIAG` and tagged `[OK]`/`[INFO]`; findings are informational
and do not abort communicator init. A failed P2P check points at hardware/driver/topology; a clean
pass redirects investigation toward the network or the application.

## Error checking pattern

Educational examples use macros similar to:

```c
#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r != ncclSuccess) {                           \
    fprintf(stderr, "%s:%d NCCL failure %s: %s\n",  \
            __FILE__, __LINE__, #cmd,               \
            ncclGetErrorString(r));                 \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while (0)
```

For libraries, return errors rather than exiting, but keep:

- file/line or callsite context,
- failed operation string,
- `ncclGetErrorString`,
- `ncclGetLastError(comm)` when a communicator exists,
- async error polling when operations may fail later.

## Async errors

NCCL work is asynchronous with CUDA stream execution. A call can enqueue successfully and fail later.
Use:

```c
ncclResult_t asyncErr = ncclSuccess;
ncclCommGetAsyncError(comm, &asyncErr);
if (asyncErr != ncclSuccess) {
  fprintf(stderr, "async NCCL error: %s last=%s\n",
          ncclGetErrorString(asyncErr), ncclGetLastError(comm));
  ncclCommAbort(comm);
}
```

Poll async errors in long-running distributed jobs, especially while waiting for CUDA streams/events or
for remote ranks.

## Hangs: triage checklist

### Communicator init hang

Check:

1. Every rank received the same `ncclUniqueId`.
2. Every rank calls `ncclCommInitRank` with same `nranks` and unique `rank`.
3. Each rank calls `cudaSetDevice` before init.
4. Rank-to-GPU mapping is valid and does not oversubscribe unintentionally.
5. Firewalls/container networking allow bootstrap communication.
6. If MPI is used, `MPI_Bcast` covers exactly the communicator ranks.
7. Logs show all ranks reach the same bootstrap/init phase.

Useful logs:

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,NET
```

### Collective hang

Check:

1. All ranks call the same collective sequence.
2. Counts, datatypes, roots, and communicator objects match.
3. Single-thread multi-device loops are grouped.
4. No rank skipped the collective due to conditional logic.
5. CUDA work before the NCCL call on the same stream is not stuck.
6. Application did not free/deregister buffers too early.

### P2P hang

Check:

1. Every `ncclSend` has a matching `ncclRecv` with same count/datatype and opposite peer.
2. Concurrent send/recv cycles are inside `ncclGroupStart/End`.
3. Peer IDs are ranks, not device IDs.
4. The group includes all P2P operations needed to make progress.

### Shutdown hang

Check:

1. Streams using NCCL work have progressed.
2. Ranks call finalize/destroy in compatible order.
3. No rank aborted while peers wait for normal finalize.
4. Use `ncclCommGetAsyncError` and `ncclCommAbort` on error paths.

## Performance triage order

1. **Establish baseline** with `nccl-tests` on the same nodes/GPU mapping.
2. **Check topology logs** with `NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET,TUNING`.
3. **Check transport**: P2P/SHM/NET/IB/socket/external plugin.
4. **Check algorithm/protocol**: ring/tree/CollNet/NVLS/PAT and LL/LL128/SIMPLE.
5. **Check channel count** and CTA/thread settings.
6. **Check buffer registration** for repeated large operations.
7. **Check CUDA stream synchronization** and whether application serializes communication unintentionally.
8. **Profile** with Nsight Systems/NVTX and NCCL profiler plugin if deeper instrumentation is needed.

## Common symptoms table

| Symptom | Likely causes | Next action |
|---|---|---|
| `ncclCommInitRank` never returns | rank count/ID mismatch, bootstrap networking, wrong launch | print rank/env/device before init; enable `INIT,BOOTSTRAP,NET` logs |
| first collective hangs | mismatched operation order or missing rank | log operation sequence per rank |
| P2P ring hangs | not grouped, wrong peer formula | wrap send/recv in group; verify prev/next ranks |
| slow multi-node bandwidth | socket fallback, wrong NIC, GDR disabled, topology mismatch | inspect `NET` logs; compare `nccl-tests` |
| slow small messages | protocol/algorithm not suitable, launch overhead, no grouping | inspect `TUNING`; test LL/LL128 behavior |
| CUDA graph capture failure | mixed capture streams, unsupported path, registration issue | isolate capture; check `enqueue.cc` graph handling logs |
| errors only after stream sync | async NCCL/CUDA failure | poll `ncclCommGetAsyncError`, print `ncclGetLastError` |
| registered-buffer crash | freed/deregistered before stream completed | synchronize before deregister/free |

## Version-specific known issues (2.31.x)

Known issues called out for the 2.31.x line, with source-verified kill switches:

| Symptom | Workaround | Knob source |
|---|---|---|
| H100: performance regression when the PAT algorithm is selected (multi-RPN PAT uses NVLS for the intra-node phase) | `NCCL_PAT_ENABLE=0` to disable PAT | `repos/nccl/src/tuning/pat.cc` (`NCCL_PARAM(PatEnable, "PAT_ENABLE", 2)`) |
| B40: symmetric TMA kernels can exceed available shared memory | `NCCL_SYM_TMA_ENABLE=0` to fall back to non-TMA symmetric kernels | `repos/nccl/src/sym_kernels.cc` (`NCCL_PARAM(SymTmaEnable, "SYM_TMA_ENABLE", 1)`; TMA path requires `minCompCap >= 100`) |
| B100 (PCIe) under MPS with MLoPart: CUDA error 101 (`cudaErrorInvalidDevice`) during init | `NCCL_CUMEM_ENABLE=0` to fall back to the legacy allocator | `repos/nccl/src/misc/cudawrap.cc`, hint in `repos/nccl/src/include/cudawrap.h` |

These switches are also useful as bisection steps when a regression appears after upgrading to
2.31.x: disable the suspected feature, re-run, and compare.

## NVTX instrumentation

`collectives.cc` wraps public APIs with NVTX payload macros, including operation name, comm hash, count
in bytes, root/peer, and op where relevant. This helps Nsight Systems correlate application calls with
NCCL kernels and proxy/network work.

Disable NVTX if needed:

```bash
export NCCL_NVTX_DISABLE=1
```

## Profiler plugin overview

Profiler plugins were introduced to extract performance data from NCCL and integrate with frameworks.
They load as:

```text
libnccl-profiler.so
libnccl-profiler-${NCCL_PROFILER_PLUGIN}.so
```

or by setting `NCCL_PROFILER_PLUGIN` to a pathname.

The profiler plugin exports versioned symbols such as `ncclProfiler_v5`; 2.31.x loads up to
`ncclProfiler_v7` and uses v7 as the in-tree default (`repos/nccl/src/include/plugin/nccl_profiler.h`),
while still accepting older plugin versions for backward compatibility.

### Main v5 interface

```c
typedef struct {
  const char* name;
  ncclResult_t (*init)(void** context, uint64_t commId, int* eActivationMask,
                       const char* commName, int nNodes, int nranks, int rank,
                       ncclDebugLogger_t logfn);
  ncclResult_t (*startEvent)(void* context, void** eHandle,
                             ncclProfilerEventDescr_v5_t* eDescr);
  ncclResult_t (*stopEvent)(void* eHandle);
  ncclResult_t (*recordEventState)(void* eHandle,
                                   ncclProfilerEventState_v5_t eState,
                                   ncclProfilerEventStateArgs_v5_t* eStateArgs);
  ncclResult_t (*finalize)(void* context);
} ncclProfiler_v5_t;
```

Profiler generated errors generally should not alter normal NCCL behavior. The docs advise returning
`ncclSuccess` except `init`, where failure can disable the plugin.

### Profiler v7 additions

Interface v7 (`repos/nccl/src/include/plugin/profiler/profiler_v7.h`) keeps the v5 callback shape and
extends event descriptors:

- `ncclProfileKernelPhase` (`1 << 15`) kernel-barrier phase sub-events: each per-channel kernel event
  is split into `initial_sync` / `compute` / `final_sync` phases (`phaseId` 0/1/2, GPU globaltimer
  timestamps), exposing barrier/sync overhead separately from compute. Enabling KernelPhase implicitly
  enables `ncclProfileKernelCh` (`repos/nccl/src/plugin/profiler.cc`).
- Symmetric-kernel metadata on collective events: `kernelVariant` and `isSymColl` in the `coll`
  descriptor identify which symmetric kernel variant ran.
- `userTag`: per-call profiler annotation carried on API and task descriptors (0 = untagged).

### Kernel-channel events use a dedicated per-communicator thread

KernelCh timestamps were historically polled by the proxy progress thread, which both delayed kernel
profiling under network load and left proxy-less paths (intra-node NVLink/SHM, symmetric kernels)
uncovered. Current source instead runs a dedicated profiler thread per communicator
(`ncclProfilerThreadCreate` / `ncclProfilerThreadDestroy` in `repos/nccl/src/include/profiler.h`,
implementation in `repos/nccl/src/plugin/profiler.cc`); communicators split with shared resources
reuse the parent's thread. Device kernels write GPU globaltimer counters into host-pinned arrays
(e.g., `repos/nccl/src/device/symmetric/kernel.cuh` for symmetric kernels), and the thread fires
KernelCh events from them. KernelCh work is posted from a captured host callback
(`ncclProfilerPostPlanWork`, gated by `hasProfilerOps` in `repos/nccl/src/enqueue/enqueue.cc`), so
per-channel events also work for graph-captured collectives on every replay. Note that the plugin
`README.md` "Known Limitations" section still describes the older proxy-thread design.

### Event types

The profiler docs list event categories including:

- `ncclProfileGroupApi`
- `ncclProfileCollApi`
- `ncclProfileP2pApi`
- `ncclProfileKernelLaunch`
- `ncclProfileGroup`
- `ncclProfileColl`
- `ncclProfileP2p`
- `ncclProfileProxyOp`
- `ncclProfileProxyStep`
- `ncclProfileProxyCtrl`
- `ncclProfileKernelCh`
- `ncclProfileKernelPhase` (v7)
- `ncclProfileNetPlugin`

This is the right extension surface when a framework wants structured NCCL timing without parsing logs.

## Proxy progress debugging

Network and some transport paths rely on CPU proxy progress. Relevant source:

- `repos/nccl/src/proxy.cc`
- `repos/nccl/src/include/proxy.h`
- transport implementations that enqueue `ncclProxyOp`

Proxy-related profiler events distinguish:

- proxy op state,
- individual proxy steps,
- proxy control idle/active/sleep/wakeup,
- append of new network work.

If GPU kernels are launched but network communication does not progress, inspect proxy thread state,
network plugin completions, and whether helper threads are running.

### IB WQE post-to-poll latency monitoring

The IB transport can time CPU WQE post-to-poll latency per queue pair
(`repos/nccl/src/transport/net_ib/wqe_lat_mon.cc`). Set a threshold in nanoseconds to enable it:

```bash
NCCL_IB_WQE_LATENCY_THRESHOLD_NS=100000  # 100 us; default 0 disables monitoring
```

When the post-to-poll delta of a tracked WQE exceeds the threshold, NCCL logs a
`NET/IB: WQE slow` line (qpn, HCA, LID/GID of both ends, mean/stddev/p50/p90/p99/p99.9/max); a
posted WQE with no completion within the threshold logs `NET/IB: WQE stall (no CQE)`. With
`NCCL_IB_WQE_LATENCY_REPORT=1` (default), a per-QP latency summary is also printed at connection
teardown. Use this to separate host-side posting/completion delays from fabric issues when
multi-node bandwidth is low.

## RAS and fault handling

RAS source files (`repos/nccl/src/ras/*`) implement a background reliability/availability/serviceability subsystem
with peer tracking, keepalive/retry behavior, local notifications, and a diagnostic client. For user-facing
failure handling, combine:

- async error polling,
- `ncclCommRevoke`,
- `ncclCommShrink(... NCCL_SHRINK_ABORT ...)`,
- `ncclCommAbort` on unrecoverable paths.

See `16-ras-fault-handling.md` for subsystem details.

## Minimal data to request from a user reporting NCCL issues

Ask for:

1. NCCL version and how it was installed/built.
2. CUDA driver/runtime version and GPU model.
3. Number of nodes, GPUs per node, process count, launcher command.
4. Rank-to-GPU mapping logic.
5. Relevant `NCCL_*` variables.
6. `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET,TUNING` logs from all ranks.
7. Whether `nccl-tests` passes on the same allocation.
8. Exact operation where hang/error occurs.

This usually separates application ordering bugs from NCCL topology/network problems.
