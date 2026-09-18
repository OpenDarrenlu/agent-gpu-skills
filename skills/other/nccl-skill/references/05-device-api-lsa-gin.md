# 05 - NCCL Device API, LSA, GIN, Teams, and Device-Side Communication

## Primary source files

- `repos/nccl/src/include/nccl_device.h`: umbrella header for Device API.
- `repos/nccl/src/include/nccl_device/core.h`: host/device core types and declarations.
- `repos/nccl/src/include/nccl_device/coop.h`: cooperative launch/device coordination helpers.
- `repos/nccl/src/include/nccl_device/barrier.h`, `lsa_barrier`, `gin_barrier`: barrier surfaces.
- `repos/nccl/src/include/nccl_device/ll_a2a.h`: low-latency all-to-all helpers.
- `repos/nccl/src/include/nccl_device/ptr.h`: pointer/window helpers.
- `repos/nccl/src/include/nccl_device/reduce_copy.h`: reduce/copy primitives for device code.
- `repos/nccl/src/include/nccl_device/cft.h`: Compute Fabric Transport (CFT) device API.
- `repos/nccl/src/include/nccl_device/gin/`: per-backend GIN device headers (`proxy`, `gdaki`, `gpi`, `efa_gda`).
- `repos/nccl/src/dev_runtime.cc`: host runtime implementation.
- `repos/nccl/src/cft_dev_runtime.cc`: CFT host runtime (logical endpoint support, `NCCL_CFT_ENABLE`).
- `repos/nccl/src/devcomm/*`: versioned device-communicator compatibility shims (`devcomm_v22902` ... `devcomm_v23100`).
- `repos/nccl/src/gin/*`: GIN host/proxy support.
- `repos/nccl/src/plugin/gin/`: GIN plugin loading (`gin_v13.cc`, `gin_v14.cc`).
- `repos/nccl/plugins/gin/`: GIN plugin developer guide plus `example/` plugin.
- `repos/nccl/bindings/nccl4py/nccl/core/device/cute/`: CuTeDSL Device API bindings (LSA, GIN, barriers).
- `repos/nccl/docs/examples/06_device_api/*`: LSA, GIN, CFT barrier, and hybrid examples.

## What the Device API enables

NCCL's host collectives enqueue NCCL-owned kernels. The Device API enables application CUDA kernels to
perform communication directly. This lets applications:

- Fuse communication with custom compute.
- Implement custom collective algorithms in CUDA kernels.
- Use load/store-accessible peer memory for local ranks.
- Use GPU-Initiated Networking (GIN) for remote peers where supported.
- Build MoE/expert-parallel primitives such as NCCL EP dispatch/combine.

The examples frame the Device API as a way to schedule communication from inside CUDA kernels.

## Required include and build context

Host and device code include:

```cpp
#include <nccl.h>
#include <nccl_device.h>
```

Device API examples are CUDA `.cu` programs and require a NCCL build with Device API support. For
multi-node GIN paths, the hardware/network stack must support the chosen GIN backend.

## Capability discovery

Before using Device API features, query communicator properties:

```cpp
ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
NCCLCHECK(ncclCommQueryProperties(comm, &props));

if (!props.deviceApiSupport) {
  // Device API not available for this communicator/platform.
}
if (props.ginType == NCCL_GIN_TYPE_NONE) {
  // GIN not available.
}
```

Important fields from `ncclCommProperties_t`:

| Field | Meaning |
|---|---|
| `rank` | this communicator rank |
| `nRanks` | communicator size |
| `cudaDev` | associated CUDA device |
| `nvmlDev` | associated NVML device |
| `deviceApiSupport` | whether `ncclDevCommCreate` can be used |
| `multimemSupport` | whether multimem pointers/handles are supported |
| `ginType` | GIN backend of the communicator; see the `ncclGinType_t` table below |
| `nLsaTeams` | number of load-store-accessible teams |
| `hostRmaSupport` | host RMA availability |
| `railedGinType` | GIN type for railed GIN |
| `commHash` | communicator hash identifier shared across all ranks (NCCL 2.31+) |
| `ginMinStride` | minimum legal `ginCustomStride` for custom-stride GIN connections (NCCL 2.31+) |
| `ginConnectionType` | communicator-wide GIN connection type (NCCL 2.31+) |
| `ginSupport[64]` | per-type availability bitmap: `ginSupport[i]` true if GIN type `i` is supported (NCCL 2.31+) |
| `devCommRuntimeVersionSize` | size of `ncclDevComm_t` for runtime-versioned (JIT) device code (NCCL 2.31+) |

GIN type values (`repos/nccl/src/include/nccl_device/core.h`):

| Value | Meaning |
|---|---|
| `NCCL_GIN_TYPE_NONE` (=0) | no GIN support; sentinel meaning "accept any available backend" in `ncclDevCommRequirements` |
| `NCCL_GIN_TYPE_PROXY` (=2) | host proxy-backed GIN path |
| `NCCL_GIN_TYPE_GDAKI` (=3) | GPU Direct Async Kernel-Initiated path (mlx5 IB) |
| `NCCL_GIN_TYPE_GPI` (=4) | GPU-Push Interface; experimental, requires SpectrumX |
| `NCCL_GIN_TYPE_EFA_GDA` (=5) | AWS EFA GPUDirect Async; requires the AWS OFI NCCL plugin (NCCL 2.31+) |

## Device communicator creation

### Requirements structure

```cpp
ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
```

Important fields:

| Field | Meaning |
|---|---|
| `resourceRequirementsList` | linked list of device resource buffer requirements |
| `teamRequirementsList` | linked list of team/multimem requirements |
| `lsaMultimem` | enable multimem on LSA team |
| `barrierCount` | generic barrier count |
| `lsaBarrierCount` | LSA barrier count |
| `railGinBarrierCount` | rail GIN barrier count |
| `lsaLLA2ABlockCount`, `lsaLLA2ASlotCount` | LSA low-latency all-to-all resources |
| `ginForceEnable` | force GIN enablement request |
| `ginContextCount` | GIN context hint |
| `ginSignalCount`, `ginCounterCount` | allocated signal/counter ranges |
| `ginConnectionType` | none/full/rail GIN connection type |
| `ginExclusiveContexts` | exclusive GIN context request |
| `ginQueueDepth` | queue depth hint |
| `ginTrafficClass` | traffic class/QoS |
| `worldGinBarrierCount` | world-level GIN barrier count |
| `ginStrongSignalsRequired` | set false if kernels will not use GIN strong signals (default true) |
| `ginVaSignalsRequired` | set false if kernels will not use GIN VA signals (default true) |
| `ginCustomStride` | rank stride when `ginConnectionType` is `NCCL_GIN_CONNECTION_CUSTOM_STRIDE`; must be >= `props.ginMinStride` |
| `ginType` | requested GIN backend for this DevComm (`NCCL_GIN_TYPE_NONE` accepts any) |
| `useRuntimeVersion` | device code is JIT-compiled against the runtime NCCL version |
| `cftCaps`, `cftBarrierCount` | CFT capability bitmask (`ncclCftCap_t`) and CFT barrier slots |

### Host creation/destruction

```cpp
ncclDevComm_t devComm;
ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
reqs.lsaBarrierCount = NCCL_DEVICE_CTA_COUNT;

NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));
// launch kernels using devComm
NCCLCHECK(ncclDevCommDestroy(comm, &devComm));
```

The actual example code registers windows and creates the device communicator before launching custom
kernels.

## Teams

`ncclTeam_t` describes a rank subset with fields:

```c
struct ncclTeam {
  int nRanks;
  int rank;
  int stride;
};
```

Team helpers include:

| Helper | Meaning |
|---|---|
| `ncclTeamWorld` | all ranks in communicator |
| `ncclTeamLsa` | load-store-accessible local team |
| `ncclTeamRail` | rail team, equivalent to outer factor of LSA team |
| `ncclTeamRankIsMember` | membership test |
| `ncclTeamRankToTeam` | translate rank from one team to another |
| `ncclTeamRankToWorld` | translate team rank to world rank |
| `ncclTeamRankToLsa` | translate team rank to LSA rank |
| `ncclTeamInnerFactor`, `ncclTeamOuterFactor` | derive subteams from layout factors |
| `ncclTeamRankInDifference` | rank in set difference of parent and subset |

Use teams when writing kernels that choose local LSA behavior for some peers and GIN/network behavior
for others.

## Windows and device pointers

Host-side pointer helpers:

```c
ncclGetLsaMultimemDevicePointer(window, offset, &ptr);
ncclGetMultimemDevicePointer(window, offset, multimemHandle, &ptr);
ncclGetLsaDevicePointer(window, offset, lsaRank, &ptr);
ncclGetPeerDevicePointer(window, offset, peer, &ptr);
```

Device-side pointer helpers include:

```cpp
ncclGetLocalPointer(window, offset);
ncclGetLsaPointer(window, offset, peer);
ncclGetPeerPointer(window, offset, peer);
ncclGetPeerPointer(window, offset, team, peer);
ncclGetMultimemPointer(window, offset, multimemHandle);
ncclGetLsaMultimemPointer(window, offset, devComm);
```

Resource-buffer helper variants map `ncclDevResourceHandle` to local, LSA, peer, and multimem pointers.

## LSA: Load Store Access

LSA is the local peer-memory path exposed through windows and teams. The LSA allreduce example uses:

1. Device communicator with LSA barrier support.
2. Symmetric memory windows for send/recv buffers.
3. `ncclGetLsaPointer` or related pointer helpers to access peer memory.
4. Device-side barriers for correctness.
5. Manual reduction inside a CUDA kernel.

Typical use case: local GPUs in the same LSA team where peer memory is load/store accessible.

## GIN: GPU-Initiated Networking

GIN lets GPU kernels initiate network operations for remote peers. The examples cover:

- Pure GIN AlltoAll: use GIN for all peers.
- Hybrid AlltoAll: use LSA for local peers and GIN for remote peers.
- GIN barriers/signals to order puts and detect completion.

For multi-node RDMA GIN in NCCL EP docs, a recommended environment example is:

```bash
export NCCL_GIN_TYPE=3  # GDAKI
```

GIN availability depends on hardware, CUDA/NCCL build, net device support, and plugin/backend support.
Always query `ncclCommProperties_t` before assuming support.

### GIN backends and per-DevComm backend selection

`ncclGinType_t` in `repos/nccl/src/include/nccl_device/core.h` enumerates four backends: `NCCL_GIN_TYPE_PROXY`,
`NCCL_GIN_TYPE_GDAKI`, `NCCL_GIN_TYPE_GPI` (experimental; requires SpectrumX), and `NCCL_GIN_TYPE_EFA_GDA`
(AWS EFA GPUDirect Async, via the AWS OFI NCCL plugin). Communication between different GIN types is not
supported on a single communicator.

A communicator can have multiple active GIN backends (up to `NCCL_GIN_MAX_ACTIVE_BACKENDS`); query
`props.ginSupport[i]` to see which types are available. Each device communicator selects its own backend via
`reqs.ginType` in `ncclDevCommRequirements` (`repos/nccl/src/dev_runtime.cc`), so different DevComms on the
same communicator may use different backends. On the device side, `ncclGin` is an alias for
`ncclGin_BackendMask<NCCL_GIN_BACKEND_MASK_ALL>` (`repos/nccl/src/include/nccl_device/gin.h`); use
`ncclGin_BackendOne<NCCL_NET_DEVICE_GIN_GDAKI>` (or another `NCCL_NET_DEVICE_GIN_*` value) to compile a kernel
against a single backend.

The EFA GDA device implementation (`repos/nccl/src/include/nccl_device/gin/efa_gda/gin_efa_gda.h`, backed by
`src/transport/net_efa_gda/` and efa-dp-direct) currently implements Put (data + signal/counter endpoints and
signal-only), PutValue, Flush, signal/counter pointer access, and ResetSignal/ResetCounter; Get, FlushAsync,
and Wait are stubs.

### Custom-stride connections

Beyond `NCCL_GIN_CONNECTION_NONE`/`FULL`/`RAIL`, `ncclGinConnectionType_t` has
`NCCL_GIN_CONNECTION_CUSTOM_STRIDE`: connect only to ranks at multiples of `reqs.ginCustomStride`. The minimum
legal stride is reported as `props.ginMinStride` (`INT_MAX` when connectivity is not uniform, in which case
custom stride cannot be used); enforced in `repos/nccl/src/dev_runtime.cc`.

### Strong vs. weak signal semantics

`repos/nccl/src/include/nccl_device/gin.h` splits put/signal completion actions into explicit strength classes:

- **Strong** (`ncclGin_StrongSignalInc/Add`, `ncclGin_StrongVASignalInc/Add`): signal visibility implies this
  put *and all preceding puts* on the same context to that peer are settled.
- **Weak** (`ncclGin_WeakSignalInc/Add`, `ncclGin_WeakVASignalInc/Add`): only the bundled put is guaranteed
  settled. Counter increments (`ncclGin_WeakCounterInc`) are weak by definition.
- The unqualified names (`ncclGin_SignalInc`, `ncclGin_VASignalAdd`, ...) are deprecated aliases; use the
  explicit Strong/Weak forms.

Not every backend supports strong signals; `ncclGin::_supportsStrongSignal()` reports it at runtime, and
`reqs.ginStrongSignalsRequired` / `reqs.ginVaSignalsRequired` let the host declare which signal classes the
kernels need.

### GIN fence levels and barriers

`ncclGinFenceLevel` (`repos/nccl/src/include/nccl_device/gin_barrier.h`) is a composable bit-flag enum:

- `ncclGinFenceLevel::None`: pure synchronization, no drain.
- `ncclGinFenceLevel::Put`: after the barrier, puts issued by other team members toward the calling rank are
  visible in the calling rank's memory.
- `ncclGinFenceLevel::Get`: after the barrier, gets issued by the calling rank have landed locally.

`ncclBarrierSession::sync` and `ncclGinBarrierSession::sync` default to `Put | Get`. Passing
`ncclGinAllContexts(comm)` instead of a single `ncclGin` context expands the fence across every GIN context on
the communicator.

### Device-side timeouts on blocking GIN calls

Blocking GIN device APIs have timeout overloads that take a `uint64_t timeoutCycles` argument and return
`ncclResult_t` instead of spinning forever: `flush`, `wait` (on a request from `flushAsync`), `waitSignal`,
and `waitCounter` in `repos/nccl/src/include/nccl_device/gin.h`, plus
`ncclGinBarrierSession::sync(..., timeoutCycles)` in `repos/nccl/src/include/nccl_device/gin_barrier.h`.
A non-`ncclSuccess` return means the timeout expired.

### GDAKI backend notes

Host side: `repos/nccl/src/transport/net_ib/gdaki/gin_host_gdaki.cc`.

- Out-of-order (DDP) delivery: `NCCL_GIN_IB_OOO_OPT=1` sets the QP ordering semantic
  (`DOCA_VERBS_QP_ORDERING_SEMANTIC_OOO_ALL`) for out-of-order data placement.
- GRH/global routing: GIDs are exchanged between peers and a DOCA verbs address handle
  (`gdakiCreateVerbsAh`) is created when the peer is on a different subnet or `IBV_QPF_GRH_REQUIRED` is set.
- Path MTU discovery: `gdakiGetPathMtu` derives the connection path MTU from the minimum of the local and
  remote port `active_mtu`.
- `NCCL_GIN_IB_TC` overrides the GIN IB traffic class when `reqs.ginTrafficClass` is not set
  (`repos/nccl/src/transport/net_ib/gin.cc`).

### GIN plugins

GIN plugins are shared libraries discovered via `NCCL_GIN_PLUGIN` (bare name, library name, absolute path, or
comma-separated list; see `repos/nccl/plugins/gin/README.md`). Only the v13/v14 plugin API symbols are loaded
(`ncclGinPlugin_v13`/`ncclGinPlugin_v14` in `repos/nccl/src/plugin/gin/gin_v13.cc` and `gin_v14.cc`); older
v11/v12 GIN plugin APIs are no longer supported. Every GIN implementation that initializes stays available,
and the backend to use is picked per device communicator. `repos/nccl/plugins/gin/example/` contains a minimal
example plugin.

## CFT: Compute Fabric Transport device API

Starting with NCCL 2.31, the Device API includes Compute Fabric Transport (CFT) helpers for kernels that
communicate through CUDA fabric logical endpoints (`repos/nccl/src/include/nccl_device/cft.h`,
`repos/nccl/src/cft_dev_runtime.cc`, user guide `repos/nccl/docs/userguide/source/usage/cft.rst`).

Requirements:

- SM_100 (Blackwell) or newer and CUDA Toolkit 13.3+: the device helpers are compiled only when
  `__CUDA_ARCH__ >= 1000 && CUDART_VERSION >= 13030` (`NCCL_CFT_ENABLE` in
  `repos/nccl/src/include/nccl_device/impl/cft__funcs.h`), and the host runtime requires CUDA/driver >= 13.3
  with logical-endpoint unicast and multicast support (`ncclGpuCftSupport` in
  `repos/nccl/src/cft_dev_runtime.cc`).
- Host setup: request `reqs.cftCaps = NCCL_CFT` (plus `NCCL_CFT_MULTIMEM` for multicast/multimem operations)
  and `reqs.cftBarrierCount`; `ncclDevCommCreate` fails if any rank cannot provide the requested capability.

Device-side `ncclCft<Coop>` provides `put`/`get`/`red`/`pullRed` (plus `putMultimem`/`redMultimem` and
`*CpMask` variants) against `(ncclCftLeId, leOffset)` logical-endpoint addresses obtained host-side via
`ncclGetPeerDeviceLeInfo`, `ncclGetCftDeviceLeInfo`, or `ncclGetMultimemDeviceLeInfo`. Peers are addressed
through CFT teams from `ncclTeamCft`/`ncclTeamCftMultimem`. A runnable CFT barrier example lives in
`repos/nccl/docs/examples/06_device_api/04_cft_barrier/`.

## CuTeDSL (nccl4py) Device API bindings

`repos/nccl/bindings/nccl4py/nccl/core/device/cute/` provides CuTeDSL bindings over the Device API. The `Gin`
class (`gin.py`) covers `put`, `put_value`, `get`, `signal`, `read_signal`, `wait_signal`, `read_counter`,
`wait_counter`, `flush`, and signal-shadow access; `types.py` mirrors `ncclGinFenceLevel`
(`GinFenceLevel.PUT | GET`) and the backend mask (`GinBackendMask.PROXY | GDAKI | GPI | EFA_GDA`). Runnable
examples are in `repos/nccl/bindings/nccl4py/examples/cute/` (see `04_gin_ops.py`).

## Example 1: LSA AllReduce structure

Host:

```cpp
ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
ncclCommQueryProperties(comm, &props);
if (!props.deviceApiSupport || props.nLsaTeams != 1) { /* fallback or exit */ }

ncclWindow_t sendWin, recvWin;
ncclCommWindowRegister(comm, d_send, bytes, &sendWin, NCCL_WIN_COLL_SYMMETRIC);
ncclCommWindowRegister(comm, d_recv, bytes, &recvWin, NCCL_WIN_COLL_SYMMETRIC);

ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
reqs.lsaBarrierCount = NCCL_DEVICE_CTA_COUNT;
ncclDevComm_t devComm;
ncclDevCommCreate(comm, &reqs, &devComm);

simpleAllReduceKernel<<<NCCL_DEVICE_CTA_COUNT, NCCL_DEVICE_THREADS_PER_CTA, 0, stream>>>(
    sendWin, 0, recvWin, 0, count, devComm);
```

Device:

- Use LSA barriers for cross-GPU synchronization.
- Load peer values through LSA pointers.
- Reduce in-kernel.
- Store result to local output.

## Example 2: Pure GIN AlltoAll

The pure GIN example creates a device communicator with GIN support and uses network barriers/signals.
Its communication is network-only, so it is useful as a baseline for multi-node all-to-all behavior.

Checklist:

1. Query `props.ginType != NCCL_GIN_TYPE_NONE`.
2. Configure `reqs` for GIN contexts/signals/barriers.
3. Create `devComm`.
4. Launch kernel that uses GIN `put`, barriers, and completion signaling.
5. Destroy resources and windows after stream completion.

## Example 3: Hybrid LSA + GIN AlltoAll

Hybrid kernels classify peers:

- **Local**: ranks in the LSA team (`ncclTeamLsa`), typically same node or same NVLink domain.
- **Remote**: world ranks outside the local LSA team; use GIN.

This is the production-shaped pattern for multi-node custom kernels: choose the lowest-overhead path per
peer instead of forcing all communication through one mechanism.

## Device API troubleshooting

1. Query properties and print `deviceApiSupport`, `ginType`, `nLsaTeams`, `multimemSupport`.
2. Confirm buffers are allocated/registered as windows with compatible flags.
3. Confirm all ranks create compatible device communicator requirements.
4. Confirm kernel launch dimensions match the barriers/resources requested.
5. For GIN, confirm `NCCL_GIN_TYPE`, net plugin/device support, and multi-node network setup.
6. Use `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH` before assuming a kernel bug.
7. If a custom kernel hangs, distinguish barrier mismatch from network put/signal mismatch.

## Source modification map

| Task | Start files |
|---|---|
| add/query a property | `src/include/nccl_device/core.h`, `src/dev_runtime.cc` |
| change device communicator creation | `src/dev_runtime.cc`, `src/include/dev_runtime.h`, version shims in `src/devcomm/` |
| add LSA pointer/team helper | `src/include/nccl_device/core.h`, implementation headers under `nccl_device/impl` |
| change GIN host behavior | `src/gin/gin_host.cc`, `src/gin/gin_host_proxy.cc` |
| change net-backed GIN (GDAKI) | `src/transport/net_ib/gdaki/*`, device side in `src/include/nccl_device/gin/gdaki/` |
| change EFA GDA GIN | `src/transport/net_efa_gda/*`, device side in `src/include/nccl_device/gin/efa_gda/` |
| add/change a GIN backend | `src/include/nccl_device/gin/gin_device_common.h` (backend mask/dispatch), `src/gin/gin_host.cc` (version compat arrays) |
| write a GIN plugin | `repos/nccl/plugins/gin/README.md`, `repos/nccl/plugins/gin/example/`, ABI headers in `src/include/plugin/gin/` |
| change CFT host/device behavior | `src/cft_dev_runtime.cc`, `src/include/nccl_device/cft.h`, `src/include/nccl_device/impl/cft__funcs.h` |
| CuTeDSL bindings | `repos/nccl/bindings/nccl4py/nccl/core/device/cute/` |
| update examples | `docs/examples/06_device_api/*` |
