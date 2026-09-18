# 16 - RAS, Fault Handling, Revoke, Shrink, and Resiliency

## Primary source files

- `repos/nccl/src/include/ras.h`: RAS integration declarations.
- `repos/nccl/src/ras/ras.cc`: RAS thread, local notifications, message framing, polling loop.
- `repos/nccl/src/ras/rasnet.cc`: RAS socket lifecycle, keepalive, retries, fallback links, timeouts.
- `repos/nccl/src/ras/peers.cc`: peer discovery/update propagation, dead-peer tracking, link reinit.
- `repos/nccl/src/ras/collectives.cc`: RAS-level distributed collectives.
- `repos/nccl/src/ras/client.cc`: `ncclras` diagnostic client.
- `repos/nccl/src/ras/diagnostics.cc`, `diagnostics.h`: RAS diagnostics orchestration, check dispatch, report formatting.
- `repos/nccl/src/ras/diagnostics_checks_common.cc`, `diagnostics_checks.h`: shared check helpers and report tags.
- `repos/nccl/src/ras/diagnostics_gpu.cc`: GPU inventory, CUDA driver version, ECC, and NVLink checks.
- `repos/nccl/src/ras/diagnostics_env.cc`: `NCCL_*` environment-variable consistency check.
- `repos/nccl/src/init.cc`: communicator revoke/shrink/grow/finalize/abort APIs.
- `repos/nccl/src/transport/net_ib/p2p_resiliency*.cc`: IB resiliency/failover/recovery.

## RAS purpose

RAS stands for reliability, availability, and serviceability. NCCL's RAS subsystem provides background
monitoring and peer-state propagation so failures and diagnostic state can be detected and surfaced more
systematically than a raw transport hang.

RAS is not a replacement for application-level failure handling. Applications still need to poll async
errors, abort/revoke failed communicators, and decide whether to shrink/restart.

## RAS integration API

`include/ras.h` exposes a small internal integration surface:

```c
ncclRasCommInit(...);
ncclRasCommFini(...);
ncclRasAddRanks(...);
```

These are called as communicators are initialized/finalized or ranks are added.

## RAS source components

| File | Role |
|---|---|
| `ras.cc` | central RAS thread, local notifications, message handling, polling, termination |
| `rasnet.cc` | socket connections, keepalive, retry/fallback, timeouts |
| `peers.cc` | peer database, peer updates, dead-peer tracking, link reinitialization |
| `collectives.cc` | RAS-level collectives to aggregate/distribute RAS state |
| `client.cc` | diagnostic client (`ncclras`) |
| `diagnostics.cc` | diagnostics request orchestration, check dispatch table, report formatting |
| `diagnostics_gpu.cc` | GPU inventory, CUDA driver version, ECC, NVLink checks (via NVML) |
| `diagnostics_env.cc` | `NCCL_*` environment-variable consistency check |

## RAS-related parameters

Source-derived variables:

| Variable | Default | Meaning |
|---|---:|---|
| `NCCL_RAS_ENABLE` | `1` | enable RAS subsystem |
| `NCCL_RAS_TIMEOUT_FACTOR` | `1` | scale RAS timeouts |
| `NCCL_RUN_RAS_DIAGNOSTICS` | `0` | run RAS diagnostics at each communicator initialization |
| `NCCL_DIAGNOSTICS_ECC_THRESHOLD` | `0` | corrected ECC error count above which the ECC check is flagged (`0` disables corrected reporting) |

IB resiliency variables:

| Variable | Default | Meaning |
|---|---:|---|
| `NCCL_IB_RESILIENCY_PORT_FAILOVER` | `0` | enable port failover |
| `NCCL_IB_RESILIENCY_PORT_FAILOVER_MAX_ATTEMPTS` | `1` | max failover attempts |
| `NCCL_IB_RESILIENCY_PORT_FAILOVER_PROBE_DELAY` | `10 ms` | probe delay |
| `NCCL_IB_RESILIENCY_PORT_RECOVERY` | `0` | enable port recovery |
| `NCCL_IB_RESILIENCY_PORT_RECOVERY_START_DELAY` | `200 ms` | recovery start delay |
| `NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_INTERVAL` | `500 ms` | alive batch interval |
| `NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE` | `5` | alive batch size |
| `NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_SEQUENCE_SIZE` | `5` | alive sequence size |
| `NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_TIMEOUT` | `4000 ms` | alive timeout |
| `NCCL_IB_RESILIENCY_PORT_RECOVERY_ACK_TIMEOUT` | `5000 ms` | ack timeout |
| `NCCL_IB_RESILIENCY_PORT_RECOVERY_ATTEMPTS_MAX` | `5` | max recovery attempts |

Treat resiliency variables as advanced; validate against current NCCL docs/source and cluster vendor
guidance before recommending deployment changes.

## RAS diagnostics

RAS diagnostics are a readiness probe that gathers per-rank state through the RAS peer network and reports
consistency issues across ranks. They do not exercise NCCL data paths. Triggering modes:

- Set `NCCL_RUN_RAS_DIAGNOSTICS=1` before starting the application: diagnostics run at each communicator
  initialization, scoped to the new communicator, and the report is printed on the process hosting rank 0.
  The init path goes through `ncclRunDiagnosticsPassive` (`ras.cc`), which forwards the request to the RAS
  thread via `rasLocalHandleRunDiag`.
- On demand while the application runs: `ncclras -D` (`client.cc`, sends `DIAGNOSTICS` over the RAS text
  protocol) or `echo diagnostics | nc <host> 28028`. An on-demand request covers all communicators known to
  the responding RAS peers. If another request is already in progress, RAS answers
  `BUSY: diagnostics already in progress` (`client_support.cc`).

Checks (dispatch table `rasDiagnosticsChecks` in `diagnostics.cc`, IDs in `diagnostics.h`):

| Check | ID | What it verifies |
|---|---|---|
| GPU inventory | `RAS_DIAG_CHECK_GPU_MODEL` | same GPU count per node and same GPU model across ranks (NVML) |
| CUDA driver version | `RAS_DIAG_CHECK_CUDA_DRIVER_VERSION` | same driver-supported CUDA version across ranks |
| ECC | `RAS_DIAG_CHECK_ECC` | volatile SRAM/DRAM ECC counters; uncorrected errors always flagged, corrected errors flagged above `NCCL_DIAGNOSTICS_ECC_THRESHOLD` |
| NVLink | `RAS_DIAG_CHECK_NVLINK` | same NVLink count per rank and all links active (omitted on PCIe-only systems) |
| NCCL environment | `RAS_DIAG_CHECK_NCCL_ENV` | `NCCL_*` environment-variable names and values consistent across ranks |

Each report line is prefixed `<host>:<pid> NCCL DIAG`. Result lines carry `[OK]` (no issue) or `[INFO]`
(condition to review: mismatch, unavailable or partial data); tags are defined in
`diagnostics_checks_common.h` as `RAS_DIAG_TAG_OK`/`RAS_DIAG_TAG_INFO`. Mismatches group ranks by value,
e.g. `NCCL environment: NCCL_DEBUG=INFO on rank(s) {0,1,2,3}`.

## Public failure-handling APIs

### Async error polling

```c
ncclResult_t asyncErr;
ncclCommGetAsyncError(comm, &asyncErr);
```

Poll during long waits. If `asyncErr != ncclSuccess`, use `ncclGetLastError(comm)` for details and decide
whether to abort/revoke/shrink.

### Abort

```c
ncclCommAbort(comm);
```

Use when a communicator cannot complete outstanding operations normally. Abort frees resources and stops
operations that may still run on device.

### Revoke

```c
ncclCommRevoke(comm, NCCL_REVOKE_DEFAULT);
```

Stops in-flight operations and waits for quiescence. After revoke, destroy/split/shrink can proceed.
Calling `ncclCommFinalize` after revoke is invalid.

### Shrink

```c
ncclCommShrink(comm, excludeRanksList, excludeRanksCount, &newcomm,
               config, NCCL_SHRINK_ABORT);
```

Use to remove failed ranks and continue with a smaller communicator. `NCCL_SHRINK_ABORT` first terminates
ongoing parent operations, then shrinks.

## Failure-handling flow patterns

### Normal shutdown

```text
all ranks finish enqueued work
  -> ncclCommFinalize
  -> wait/poll until quiescent if needed
  -> ncclCommDestroy
```

### Fatal error shutdown

```text
rank detects immediate or async NCCL error
  -> notify application control plane if any
  -> ncclCommAbort on affected comms
  -> clean up CUDA/application resources
  -> restart or fail job
```

### Recover by shrink

```text
detect failed/excluded ranks
  -> revoke or shrink with NCCL_SHRINK_ABORT
  -> new communicator returned for surviving ranks
  -> rebuild application parallel groups/state
  -> resume with smaller world if algorithm supports it
```

This requires application-level support. NCCL can produce the smaller communicator, but it cannot fix model
parallel layouts, optimizer sharding, checkpoint consistency, or data-loader state by itself.

## Diagnosing suspected rank failure

Ask for:

1. Which rank first observed error/hang.
2. Last log line for every rank.
3. Whether one process died or was OOM-killed.
4. Network error counters/logs if multi-node.
5. `ncclCommGetAsyncError` values on surviving ranks.
6. RAS logs if enabled, or an on-demand RAS diagnostics report (`ncclras -D`) if the job is still running.
7. Whether the application used abort/revoke/shrink or waited indefinitely.

## Interaction with proxy and transports

Network failures often surface through proxy progress and transport request completion. A GPU kernel may
be waiting on connector state while proxy/network cannot make progress. Profiler plugin events can expose
proxy op/step states, and RAS can expose peer-level state.

For IB-specific failures, inspect both NCCL logs and verbs/network-driver logs. NCCL variables can adjust
retry/failover behavior, but physical/network misconfiguration must be fixed outside NCCL.

## Source modification cautions

- Failure paths must be async-safe with respect to in-flight kernels and proxy ops.
- Do not free connector/proxy memory while device work can still reference it.
- Revoke/shrink/grow interact with resource sharing; test with split/shrink shared resources enabled and disabled.
- RAS message handling must avoid blocking the progress path.
- Keep diagnostic client compatibility in mind if changing message formats.
