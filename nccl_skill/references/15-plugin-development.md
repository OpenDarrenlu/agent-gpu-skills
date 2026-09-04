# 15 - NCCL Plugin Development: Net, Tuner, Profiler, Env, GIN, RMA, and Mixed Plugins

## Primary source files

- `repos/nccl/plugins/net/README.md`
- `repos/nccl/plugins/tuner/README.md`
- `repos/nccl/plugins/profiler/README.md`
- `repos/nccl/plugins/env/README.md`
- `repos/nccl/plugins/gin/README.md`
- `repos/nccl/plugins/rma/README.md`
- `repos/nccl/plugins/mixed/README.md`
- `repos/nccl/src/include/plugin/nccl_net.h`
- `repos/nccl/src/include/plugin/nccl_tuner.h`
- `repos/nccl/src/include/plugin/nccl_profiler.h`
- `repos/nccl/src/include/plugin/nccl_env.h`
- `repos/nccl/src/include/plugin/nccl_gin.h`
- `repos/nccl/src/include/plugin/nccl_rma.h`
- `repos/nccl/src/plugin/plugin_open.cc`
- `repos/nccl/src/plugin/net.cc`
- `repos/nccl/src/plugin/tuner.cc`
- `repos/nccl/src/plugin/profiler.cc`
- `repos/nccl/src/plugin/env.cc`
- `repos/nccl/src/plugin/gin.cc`
- `repos/nccl/src/plugin/rma.cc`

## Plugin architecture overview

NCCL plugins are shared libraries loaded dynamically at runtime. They expose versioned structs filled with
function pointers. NCCL probes symbols by version so one plugin can support multiple NCCL versions and
NCCL can fall back to older plugin ABIs when needed.

Common plugin properties:

- shared library naming convention,
- environment variable for selection/path,
- versioned exported symbol,
- `name` field used in logs,
- NCCL error-code return values,
- NCCL-provided logging callback for consistent logs.

## Plugin library names and variables

| Plugin type | Default library | Selection variable | Versioned symbol pattern |
|---|---|---|---|
| Net | `libnccl-net.so` | `NCCL_NET_PLUGIN` | `ncclNetPlugin_vX` (current: v12), optional `ncclCollNetPlugin_vX` structs |
| Tuner | `libnccl-tuner.so` or named examples | `NCCL_TUNER_PLUGIN` | `ncclTunerPlugin_vX` (current: v6) |
| Profiler | `libnccl-profiler.so` | `NCCL_PROFILER_PLUGIN` | `ncclProfiler_vX` (current: v7) |
| Env | `libnccl-env.so` | `NCCL_ENV_PLUGIN` | `ncclEnvPlugin_vX` (current: v2) |
| GIN | `libnccl-gin.so` | `NCCL_GIN_PLUGIN` | `ncclGinPlugin_vX` (current: v14) |
| RMA | `libnccl-rma.so` | `NCCL_RMA_PLUGIN` | `ncclRmaPlugin_vX` (current: v15) |

For suffix-based loading, NCCL looks for names like:

```text
libnccl-net-${NCCL_NET_PLUGIN}.so
libnccl-profiler-${NCCL_PROFILER_PLUGIN}.so
libnccl-env-${NCCL_ENV_PLUGIN}.so
libnccl-gin-${NCCL_GIN_PLUGIN}.so
libnccl-rma-${NCCL_RMA_PLUGIN}.so
```

Many plugin variables also accept an absolute pathname. `NCCL_GIN_PLUGIN` and `NCCL_RMA_PLUGIN` accept
a comma-separated list of plugin names, up to `NCCL_GIN_MAX_PLUGINS`/`NCCL_RMA_MAX_PLUGINS` (16).

Current ABI support in this source tree (see `repos/nccl/src/plugin/*/` for the per-version wrappers):

| Plugin type | Versions NCCL probes | Notes |
|---|---|---|
| Net | v6-v12 | `ncclNet_t` is `ncclNet_v12_t` |
| Tuner | v2-v6 | `NCCL_TUNER_PLUGIN_SYMBOL` is `ncclTunerPlugin_v6` |
| Profiler | v1-v7 | `ncclProfiler_t` is `ncclProfiler_v7_t` |
| Env | v1-v2 | `NCCL_ENV_PLUGIN_SYMBOL` is `ncclEnvPlugin_v2` |
| GIN | v13-v14 | v11/v12 support has been removed from NCCL core; only `gin_v13.cc`/`gin_v14.cc` remain under `repos/nccl/src/plugin/gin/` |
| RMA | v13-v15 | `ncclRma_t` is `ncclRma_v15_t`; RMA plugins can also be probed through the `ncclGinPlugin_v13` symbol for GIN-based implementations |

## Net plugin

### Purpose

Net plugins decouple NCCL core builds from network stack builds. They allow NCCL to work on external or
vendor-specific networks without recompiling NCCL.

### Load/selection

```bash
export LD_LIBRARY_PATH=/path/to/plugin:$LD_LIBRARY_PATH
export NCCL_NET_PLUGIN=myplugin
export NCCL_NET=PluginReportedName
```

`NCCL_NET_PLUGIN` controls which library is loaded. `NCCL_NET` controls which implementation name is used
after the plugin reports one.

### v11/v12 interface shape

The current net ABI is v12 (`ncclNetPlugin_v12`; `ncclNet_t` is `ncclNet_v12_t` in
`repos/nccl/src/include/plugin/nccl_net.h`). v12 keeps the same function table as v11; it adds
`railId`/`planeId` to `ncclNetDeviceProps` and raises `NCCL_NET_MAX_DEVS_PER_NIC` from 4 to 8 (see
`repos/nccl/src/include/plugin/net/net_v12.h`). From the plugin docs, `ncclNet_v11` includes:

```c
typedef struct {
  const char* name;
  ncclResult_t (*init)(void** ctx, uint64_t commId, ncclNetCommConfig_v11_t* config,
                       ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction);
  ncclResult_t (*devices)(int* ndev);
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v11_t* props);
  ncclResult_t (*listen)(void* ctx, int dev, void* handle, void** listenComm);
  ncclResult_t (*connect)(void* ctx, int dev, void* handle, void** sendComm,
                          ncclNetDeviceHandle_v11_t** sendDevComm);
  ncclResult_t (*accept)(void* listenComm, void** recvComm,
                         ncclNetDeviceHandle_v11_t** recvDevComm);
  ncclResult_t (*regMr)(void* comm, void* data, size_t size, int type, void** mhandle);
  ncclResult_t (*regMrDmaBuf)(void* comm, void* data, size_t size, int type,
                              uint64_t offset, int fd, void** mhandle);
  ncclResult_t (*deregMr)(void* comm, void* mhandle);
  ncclResult_t (*isend)(void* sendComm, void* data, size_t size, int tag,
                        void* mhandle, void* pHandle, void** request);
  ncclResult_t (*irecv)(void* recvComm, int n, void** data, size_t* sizes, int* tags,
                        void** mhandles, void** pHandles, void** request);
  ncclResult_t (*iflush)(void* recvComm, int n, void** data, int* sizes,
                         void** mhandles, void** request);
  ncclResult_t (*test)(void* request, int* done, int* sizes);
  ncclResult_t (*closeSend)(void* sendComm);
  ncclResult_t (*closeRecv)(void* recvComm);
  ncclResult_t (*closeListen)(void* listenComm);
  ncclResult_t (*getDeviceMr)(void* comm, void* mhandle, void** dptr_mhandle);
  ncclResult_t (*irecvConsumed)(void* recvComm, int n, void* request);
  ncclResult_t (*makeVDevice)(int* d, ncclNetVDeviceProps_t* props);
} ncclNet_t;
```

### Operation flow

```text
init
  -> devices
  -> getProperties for each device
  -> listen on receiver
  -> exchange handle through NCCL bootstrap
  -> connect on sender until sendComm != NULL
  -> accept on receiver until recvComm != NULL
  -> regMr/regMrDmaBuf
  -> isend/irecv/iflush/test
  -> closeSend/closeRecv/closeListen
  -> deregMr
```

### Nonblocking requirements

The net docs require several calls not to block:

- `connect`: may return success with `sendComm == NULL`, NCCL retries.
- `accept`: may return success with `recvComm == NULL`, NCCL retries.
- `isend`: may return success with `request == NULL`, NCCL retries.
- `irecv`: may return success with `request == NULL`, NCCL retries.

Blocking in these functions can hang NCCL progress.

### Device properties

Net plugin `getProperties` fields influence topology and scheduling:

| Field | Importance |
|---|---|
| `name` | logs and `NCCL_NET` selection |
| `pciPath` | topology/NIC locality; `NULL` for virtual devices |
| `guid` | detects shared physical ports/endpoints |
| `ptrSupport` | host/CUDA/DMABUF pointer support |
| `regIsGlobal` | registration cache/global registration behavior |
| `forceFlush` | asks NCCL to flush all transfers |
| `speed` | port speed in Mbps |
| `port` | physical port number |
| `latency` | network latency in microseconds |
| `maxComms` | max connections |
| `maxRecvs` | grouped receive capability |
| `netDeviceType`, `netDeviceVersion` | device networking support |
| `maxP2pBytes`, `maxCollBytes` | chunking limits |
| `vProps` | virtual NIC child devices |

### Net plugin error-code guidance

Common plugin return codes:

- `ncclSuccess`: success.
- `ncclSystemError`: kernel/system/network/hardware/library failure.
- `ncclInternalError`: NCCL core used plugin incorrectly or plugin invariant failed.
- `ncclInvalidUsage`: likely user misconfiguration or size mismatch.
- `ncclInvalidArgument`: rarely needed; NCCL core usually checks arguments.
- `ncclUnhandledCudaError`: CUDA error, uncommon for net plugins.

## CollNet plugin support

Network plugins can expose an optional CollNet structure for in-network collective operations. CollNet is
tied to net plugin versioning and shares many functions. It can accelerate inter-node reductions in
AllReduce when network hardware supports it.

## Tuner plugin

### Purpose

Tuner plugins customize NCCL's algorithm/protocol/channel selection without recompiling NCCL.

### Interface

The current tuner ABI is v6 (`NCCL_TUNER_PLUGIN_SYMBOL` is `ncclTunerPlugin_v6`; NCCL probes v2-v6).
Relative to older versions, v6 `init` takes the communicator ID and an `ncclNvlDomainInfo_v6_t*`
NVL-domain info struct, and v6 adds an optional `getChunkSize` callback that lets the plugin override
NCCL's computed chunk size (see `repos/nccl/src/include/plugin/tuner/tuner_v6.h`).

```c
ncclResult_t (*init)(void** ctx, uint64_t commId, size_t nRanks, size_t nNodes,
                     ncclDebugLogger_t logFunction,
                     ncclNvlDomainInfo_v6_t* nvlDomainInfo, ncclTunerConstants_v6_t* constants);

ncclResult_t (*getCollInfo)(void* context, ncclFunc_t collType, size_t nBytes,
                            int numPipeOps, float** collCostTable,
                            int numAlgo, int numProto,
                            int regBuff, int* nChannels);

ncclResult_t (*finalize)(void* context);

ncclResult_t (*getChunkSize)(void* context, ncclFunc_t collType, size_t nBytes,
                             int algo, int proto, int nChannels, size_t* chunkSize);
```

### Cost table behavior

- Lower costs are preferred.
- `0.0` strongly prefers a combination.
- `NCCL_ALGO_PROTO_IGNORE` disables a combination.
- `nChannels` can be changed or left as default.

### Loading

```bash
export LD_LIBRARY_PATH=/path/to/plugin:$LD_LIBRARY_PATH
export NCCL_TUNER_PLUGIN=example
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=TUNING
```

## Profiler plugin

### Purpose

Profiler plugins provide structured NCCL performance events to frameworks and analysis tools.

### Load naming

```bash
export NCCL_PROFILER_PLUGIN=myprofiler
# loads libnccl-profiler-myprofiler.so
```

or set an absolute path.

### v7 interface

The current profiler ABI is v7 (`ncclProfiler_t` is `ncclProfiler_v7_t`; NCCL probes v1-v7).
See `repos/nccl/src/include/plugin/profiler/profiler_v7.h`:

```c
typedef struct {
  const char* name;
  ncclResult_t (*init)(void** context, uint64_t commId, int* eActivationMask,
                       const char* commName, int nNodes, int nranks, int rank,
                       ncclDebugLogger_t logfn);
  ncclResult_t (*startEvent)(void* context, void** eHandle,
                             ncclProfilerEventDescr_v7_t* eDescr);
  ncclResult_t (*stopEvent)(void* eHandle);
  ncclResult_t (*recordEventState)(void* eHandle,
                                   ncclProfilerEventState_v7_t eState,
                                   ncclProfilerEventStateArgs_v7_t* eStateArgs);
  ncclResult_t (*finalize)(void* context);
} ncclProfiler_v7_t;
```

v7 additions over v6 (all in `ncclProfilerEventDescr_v7_t` and `repos/nccl/src/include/plugin/nccl_profiler.h`):

- `userTag` on the `collApi`, `p2pApi`, `coll`, `p2p`, and `ceColl` descriptors: a per-call user
  profiler tag (`0` == untagged). Applications set it through the `userProfilerTag` field of
  `ncclCollConfig_t` on the `nccl*Config()` collective entry points
  (`repos/nccl/src/nccl.h.in`); NCCL copies it into each profiler event of that call. Values with
  the MSB set are reserved by NCCL.
- `kernelVariant` and `isSymColl` on the `coll` descriptor: kernel variant metadata, including
  whether the launched kernel is a symmetric-memory collective variant.
- `ncclProfileKernelPhase` event type (`1 << 15`) with the `kernelPhase` descriptor: per-kernel
  barrier phase sub-events. `phaseId` is `0=initial_sync, 1=compute, 2=final_sync`, with
  `phaseName` and GPU-globaltimer start/stop timestamps (`pTimer`; stop reuses the `kernelCh`
  state-args shape via `ncclProfilerKernelPhaseStop`). KernelPhase events are gated on
  `ncclProfileKernelCh`: NCCL enables KernelCh implicitly when a plugin requests KernelPhase
  (`repos/nccl/src/plugin/profiler.cc`).

Profiler errors generally should not alter NCCL behavior; return `ncclSuccess` except `init` may fail to
disable plugin.

## Env plugin

### Purpose

Env plugins customize environment variable resolution, validation, transformation, or integration with
configuration management systems.

### v2 interface

The current env ABI is v2 (`NCCL_ENV_PLUGIN_SYMBOL` is `ncclEnvPlugin_v2`; NCCL still probes v1).
v2 `init` adds an NCCL debug logging callback compared to v1
(`repos/nccl/src/include/plugin/env/env_v2.h`):

```c
typedef struct {
  const char* name;
  ncclResult_t (*init)(uint8_t ncclMajor, uint8_t ncclMinor, uint8_t ncclPatch,
                       const char* suffix, ncclDebugLogger_t logFunction);
  ncclResult_t (*finalize)(void);
  const char* (*getEnv)(const char* name);
} ncclEnv_v2_t;
```

`getEnv` returns a pointer that must remain valid until plugin finalize or another `getEnv` call for the
same variable. Avoid blocking in `getEnv` because NCCL calls it synchronously.

### Loading

```bash
export LD_LIBRARY_PATH=/path/to/plugin:$LD_LIBRARY_PATH
export NCCL_ENV_PLUGIN=myenv
```

## GIN plugin

GIN (GPU-Initiated Networking) plugins provide device-side networking for symmetric-memory kernels.
Selection follows the same pattern as other plugins: `NCCL_GIN_PLUGIN=mygin` loads
`libnccl-gin-mygin.so`, or an absolute path (`repos/nccl/plugins/gin/README.md`).

The current GIN ABI is v14 (`NCCL_GIN_PLUGIN_SYMBOL` is `ncclGinPlugin_v14`). NCCL core only probes
v13 and v14 (`repos/nccl/src/plugin/gin/gin_v13.cc`, `gin_v14.cc`); the older v11/v12 GIN plugin APIs
are no longer supported by NCCL core, even though the example plugin still exports those symbols for
older NCCL releases. `repos/nccl/plugins/gin/example/plugin.c` is the reference GIN plugin and shows
the multi-version export pattern (`ncclGinPlugin_v11` through `ncclGinPlugin_v14` in one library).

## RMA plugin

RMA plugins back NCCL's symmetric-memory RMA path (put/get/signal) used by CE collectives. Selection
uses `NCCL_RMA_PLUGIN=myrma` (loads `libnccl-rma-myrma.so`, a comma-separated list, or an absolute
path; see `repos/nccl/plugins/rma/README.md` and `repos/nccl/src/plugin/rma.cc`).

The current RMA ABI is v15 (`ncclRma_t` is `ncclRma_v15_t`; NCCL probes v13-v15). v15 adds the
`optFlags` argument carrying `ncclRmaOptFlags` to the `iput`/`iputSignal`/`iget` op-table entry
points; the first defined flag is `ncclRmaOptFlagsAggregateRequests` (`1 << 0`), an aggregation hint
telling the plugin that NCCL may batch/aggregate the requests
(`repos/nccl/src/include/plugin/rma/rma_v15.h`). `ncclRmaConfig_v15_t` is unchanged from v14.

## Mixed plugin

The mixed plugin example demonstrates combining multiple plugin APIs in one library, such as Net and
Tuner. This is useful for vendor packages that ship network support and topology-specific tuning together.

When building mixed plugins, keep symbol/version exports clear and test each interface independently.

## Plugin development checklist

1. Copy/fork the relevant NCCL plugin header versions into the plugin source tree.
2. Export all versioned symbols needed by target NCCL versions.
3. Implement `name` consistently with expected `NCCL_NET` or logs.
4. Use NCCL logging callback, not ad hoc stdout spam.
5. Keep retry/nonblocking functions nonblocking.
6. Treat memory ownership and returned pointer lifetime as ABI contracts.
7. Support graceful fallback: failed init should let NCCL choose built-ins where appropriate.
8. Test with `NCCL_DEBUG=INFO` and subsystem-specific logs.
9. Benchmark against built-in plugins and no tuner plugin.
10. Validate under multi-rank, multi-node, CUDA graph, and buffer-registration workloads if supported.

## Source modification map

| Task | Files |
|---|---|
| plugin loading behavior | `src/plugin/plugin_open.cc`, type-specific `src/plugin/*.cc` |
| net ABI update | `src/include/plugin/nccl_net.h`, net wrappers, plugin examples |
| tuner ABI update | `src/include/plugin/nccl_tuner.h`, `src/plugin/tuner.cc`, tuning integration |
| profiler event update | `src/include/plugin/nccl_profiler.h`, instrumentation callsites |
| env behavior | `src/include/plugin/nccl_env.h`, `src/plugin/env.cc`, `param` system |
| GIN behavior | `src/include/plugin/nccl_gin.h`, `src/plugin/gin.cc`, `src/plugin/gin/gin_v*.cc` |
| RMA behavior | `src/include/plugin/nccl_rma.h`, `src/plugin/rma.cc`, `src/plugin/rma/rma_v*.cc` |
| per-call profiler tag | `ncclCollConfig_t.userProfilerTag` in `src/nccl.h.in`, propagation in `src/enqueue/enqueue.cc`, delivery in `src/plugin/profiler.cc` |
| example plugin docs | `plugins/<type>/README.md`, examples under `plugins/<type>/` |
