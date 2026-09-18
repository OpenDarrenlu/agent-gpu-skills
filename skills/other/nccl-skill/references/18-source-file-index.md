# 18 - NCCL Source File Index

Use this index to jump to the right source area before making claims or edits.

## Public API and examples

| Path | Purpose |
|---|---|
| `repos/nccl/src/nccl.h.in` | public C API template: types, errors, communicator APIs, collectives, memory/window/RMA/parameter APIs |
| `repos/nccl/README.md` | top-level overview, build/install/test commands |
| `repos/nccl/docs/examples/README.md` | examples overview and build/run variables |
| `repos/nccl/docs/examples/01_communicators` | communicator init patterns |
| `repos/nccl/docs/examples/02_point_to_point` | P2P ring pattern |
| `repos/nccl/docs/examples/03_collectives` | basic collective examples |
| `repos/nccl/docs/examples/04_user_buffer_registration` | buffer registration examples |
| `repos/nccl/docs/examples/05_symmetric_memory` | symmetric window examples |
| `repos/nccl/docs/examples/06_device_api` | Device API LSA/GIN/hybrid examples |
| `repos/nccl/docs/examples/07_kernel_fusion` | fused RMSNorm + communication examples (LSA/multimem/GIN/hybrid) |
| `repos/nccl/docs/examples/08_ras` | RAS fault-detection example |
| `repos/nccl/docs/examples/common` | shared example utilities |
| `repos/nccl/docs/dev_guide/nccl_coding_style.md` | NCCL C/C++ coding style guide |

## Build and packaging

| Path | Purpose |
|---|---|
| `repos/nccl/Makefile` | top-level Make targets |
| `repos/nccl/CMakeLists.txt` | CMake build entry |
| `repos/nccl/makefiles/common.mk` | common Make settings, CUDA arch defaults |
| `repos/nccl/makefiles/version.mk` | NCCL version (`2.31.2` in this checkout) |
| `repos/nccl/pkg/debian` | Debian package files |
| `repos/nccl/pkg/redhat` | RPM package files |
| `repos/nccl/pkg/txz`, `pkg/srctxz` | tarball packaging |

## Core host runtime

| Path | Purpose |
|---|---|
| `repos/nccl/src/init.cc` | communicator initialization/lifecycle/split/shrink/grow/revoke |
| `repos/nccl/src/collectives.cc` | public collective/P2P/RMA API wrappers and string conversions |
| `repos/nccl/src/group.cc` | group semantics, async job handling, grouped launch |
| `repos/nccl/src/enqueue/*` | task append, prepare, schedule, upload, launch; mgmt-task enqueue |
| `repos/nccl/src/channel.cc` | channel helpers |
| `repos/nccl/src/bootstrap.cc` | bootstrap communication before transports are ready |
| `repos/nccl/src/debug.cc` | logging/debug subsystem |
| `repos/nccl/src/allocator.cc` | `ncclMemAlloc` / `ncclMemFree` |
| `repos/nccl/src/mem_manager.cc` | tracked communicator memory, suspend/resume/stats |
| `repos/nccl/src/dev_runtime.cc` | Device API host runtime |
| `repos/nccl/src/dev_runtime_segments.cc` | Device API memory segment validation/layout (Elastic GIN) |
| `repos/nccl/src/cft_dev_runtime.cc` | CFT (Compute Fabric Transport) host runtime, `NCCL_CFT_ENABLE` |
| `repos/nccl/src/enhcompat.cc` | enhanced compatibility behavior |

## Core internal headers

| Path | Purpose |
|---|---|
| `repos/nccl/src/include/comm.h` | central structs: communicator, channel, tasks, planner, plans |
| `repos/nccl/src/include/info.h` | `ncclInfo` API-call descriptor |
| `repos/nccl/src/include/group.h` | group helpers |
| `repos/nccl/src/include/enqueue.h` | enqueue/launch declarations |
| `repos/nccl/src/include/collectives.h` | collective enums/constants |
| `repos/nccl/src/include/device.h` | device work structures/constants |
| `repos/nccl/src/include/transport.h` | transport interfaces |
| `repos/nccl/src/include/proxy.h` | proxy structures/messages |
| `repos/nccl/src/include/graph.h` | topology graph declarations |
| `repos/nccl/src/include/param.h` | legacy/internal `NCCL_PARAM` macro |
| `repos/nccl/src/include/param/param.h` | typed parameter registry macros |
| `repos/nccl/src/include/checks.h` | error-checking helpers |
| `repos/nccl/src/include/argcheck.h` | argument checks |

## Device kernels

| Path | Purpose |
|---|---|
| `repos/nccl/src/device/common.cu` | generic device kernel entry |
| `repos/nccl/src/device/common_kernel.h` | work-batch execution and dispatch framework |
| `repos/nccl/src/device/primitives.h` | protocol primitive abstraction |
| `repos/nccl/src/device/prims_simple.h` | SIMPLE protocol |
| `repos/nccl/src/device/prims_ll.h` | LL protocol |
| `repos/nccl/src/device/prims_ll128.h` | LL128 protocol |
| `repos/nccl/src/device/all_reduce.h` | AllReduce device implementation |
| `repos/nccl/src/device/broadcast.h` | Broadcast device implementation |
| `repos/nccl/src/device/reduce.h` | Reduce device implementation |
| `repos/nccl/src/device/reduce_scatter.h` | ReduceScatter device implementation |
| `repos/nccl/src/device/all_gather.h` | AllGather device implementation |
| `repos/nccl/src/device/all_gather_v.h` | AllGatherV device implementation |
| `repos/nccl/src/device/sendrecv.h` | P2P send/recv device implementation |
| `repos/nccl/src/device/reduce_kernel.h` | reduce kernel helpers |
| `repos/nccl/src/device/generate.py` | device code generation support |
| `repos/nccl/src/device/onerank.cu` | single-rank degenerate path |

## Device API

| Path | Purpose |
|---|---|
| `repos/nccl/src/include/nccl_device.h` | Device API umbrella header |
| `repos/nccl/src/include/nccl_device/core.h` | core Device API types and host/device helpers |
| `repos/nccl/src/include/nccl_device/coop.h` | cooperative helpers |
| `repos/nccl/src/include/nccl_device/barrier.h` | barrier API |
| `repos/nccl/src/include/nccl_device/cft.h` | CFT device API (fabric endpoint teams, memory fences) |
| `repos/nccl/src/include/nccl_device/ptr.h` | pointer/window helpers |
| `repos/nccl/src/include/nccl_device/reduce_copy.h` | reduce/copy helpers |
| `repos/nccl/src/include/nccl_device/ll_a2a.h` | low-latency all-to-all helpers |
| `repos/nccl/src/include/nccl_device/impl/*` | inline implementation/types |
| `repos/nccl/src/devcomm/*` | Device API compatibility versions |

## Topology and graph

| Path | Purpose |
|---|---|
| `repos/nccl/src/graph/topo.cc` | topology discovery/build |
| `repos/nccl/src/graph/paths.cc` | path quality/bandwidth computation |
| `repos/nccl/src/graph/search.cc` | topology graph search |
| `repos/nccl/src/graph/connect.cc` | channel connectivity from selected graphs |
| `repos/nccl/src/graph/rings.cc` | ring helpers |
| `repos/nccl/src/graph/trees.cc` | tree helpers |
| `repos/nccl/src/tuning/*` | algorithm/protocol cost model and per-algorithm tuning |
| `repos/nccl/src/graph/xml.cc` | topology XML |

## Transports and proxy

| Path | Purpose |
|---|---|
| `repos/nccl/src/transport.cc` | transport multiplexer and connection setup |
| `repos/nccl/src/transport/p2p.cc` | CUDA P2P transport |
| `repos/nccl/src/transport/shm.cc` | shared-memory transport |
| `repos/nccl/src/transport/net.cc` | network transport layer |
| `repos/nccl/src/transport/net_socket.cc` | socket net implementation |
| `repos/nccl/src/transport/net_ib/init.cc` | IB init/device discovery |
| `repos/nccl/src/transport/net_ib/connect.cc` | IB connection/QP setup |
| `repos/nccl/src/transport/net_ib/p2p.cc` | IB P2P send/recv behavior |
| `repos/nccl/src/transport/net_ib/reg.cc` | IB memory registration |
| `repos/nccl/src/transport/net_ib/common.cc` | IB common helpers |
| `repos/nccl/src/transport/net_ib/p2p_resiliency*.cc` | IB resiliency/failover/recovery |
| `repos/nccl/src/transport/net_ib/gdaki/*` | GDAKI/GIN IB support |
| `repos/nccl/src/transport/coll_net.cc` | CollNet transport |
| `repos/nccl/src/transport/nvls.cc` | NVLS transport |
| `repos/nccl/src/proxy.cc` | proxy progress engine |

## Plugins

| Path | Purpose |
|---|---|
| `repos/nccl/src/plugin/plugin_open.cc` | dynamic library open/probe helpers |
| `repos/nccl/src/plugin/net.cc` | net plugin integration |
| `repos/nccl/src/plugin/tuner.cc` | tuner plugin integration |
| `repos/nccl/src/plugin/profiler.cc` | profiler plugin integration |
| `repos/nccl/src/plugin/env.cc` | env plugin integration |
| `repos/nccl/src/plugin/gin.cc` | GIN plugin integration |
| `repos/nccl/src/plugin/rma.cc` | RMA plugin integration |
| `repos/nccl/src/include/plugin/nccl_net.h` | net plugin ABI |
| `repos/nccl/src/include/plugin/nccl_tuner.h` | tuner plugin ABI |
| `repos/nccl/src/include/plugin/nccl_profiler.h` | profiler plugin ABI |
| `repos/nccl/src/include/plugin/nccl_env.h` | env plugin ABI |
| `repos/nccl/src/include/plugin/nccl_gin.h` | GIN plugin ABI (`ncclGin_v14_t`) |
| `repos/nccl/src/include/plugin/nccl_rma.h` | RMA plugin ABI (`ncclRma_v15_t`) |
| `repos/nccl/plugins/net` | net plugin docs/examples |
| `repos/nccl/plugins/tuner` | tuner plugin docs/examples |
| `repos/nccl/plugins/profiler` | profiler plugin docs/examples |
| `repos/nccl/plugins/env` | env plugin docs/examples |
| `repos/nccl/plugins/mixed` | combined plugin example |
| `repos/nccl/plugins/gin` | GIN plugin docs and example |
| `repos/nccl/plugins/rma` | RMA plugin docs and example |

## Registration, memory, RMA, RAS

| Path | Purpose |
|---|---|
| `repos/nccl/src/register/register.cc` | generic registration |
| `repos/nccl/src/register/coll_reg.cc` | collective registration |
| `repos/nccl/src/register/sendrecv_reg.cc` | P2P registration |
| `repos/nccl/src/rma/*` | one-sided RMA/signal implementation |
| `repos/nccl/src/ras/ras.cc` | RAS main thread/message handling |
| `repos/nccl/src/ras/rasnet.cc` | RAS networking/keepalive/retry |
| `repos/nccl/src/ras/peers.cc` | peer state tracking |
| `repos/nccl/src/ras/collectives.cc` | RAS collectives |
| `repos/nccl/src/ras/client.cc` | RAS diagnostic client |

## Scheduler extensions and special subsystems

| Path | Purpose |
|---|---|
| `repos/nccl/src/scheduler/symmetric_sched.cc` | symmetric collective scheduling |
| `repos/nccl/src/scheduler/allgatherv_sched.cc` | AllGatherV scheduling |
| `repos/nccl/src/ce_coll.cc` | CE/special collective support |
| `repos/nccl/src/gin/gin_host.cc` | GIN host support |
| `repos/nccl/src/gin/gin_host_proxy.cc` | GIN proxy support |
| `repos/nccl/src/gin/proxy_gpucontext/*` | GIN proxy GPU context (versioned) |
| `repos/nccl/src/config/*` | collective config, algorithm selection parsing/registry |
| `repos/nccl/src/init_nvtx.cc` | NVTX init/disable behavior |
| `repos/nccl/src/misc/*` | sockets, CUDA wrappers, NVML/IB wrappers, utilities |
| `repos/nccl/src/os/*` | OS abstractions |

## Bindings and contrib

| Path | Purpose |
|---|---|
| `repos/nccl/bindings/nccl4py` | Python/Cython bindings |
| `repos/nccl/bindings/ir` | Device API/IR binding wrapper |
| `repos/nccl/contrib/nccl_ep` | Expert Parallelism dispatch/combine extension |
| `repos/nccl/contrib/pace` | PACE parallelism-aware collective engine on the GIN Device API |
| `repos/nccl/contrib/niin` | experimental NVSHMEM API implemented on NCCL |
| `repos/nccl/contrib/nccl4rust` | experimental Rust interface to NCCL host/device APIs |
| `repos/nccl/contrib/nccl_checkpoint` | collective checkpoint/restore shim (`LD_PRELOAD`) |
| `repos/nccl/contrib/nccl_m2n` | cross-group reshard data movement (moved to NVIDIA/nccl-extensions) |
| `repos/nccl/contrib/nccl_ubx` | UB-X experimental fused-collective library (symmetric allocator) |
| `repos/nccl/contrib/custom_algos` | reference custom collective kernels on the Device API |

## Fast grep recipes

```bash
# Find public API declarations
grep -n "ncclResult_t .*nccl" repos/nccl/src/nccl.h.in

# Find environment parameters
grep -R "NCCL_PARAM(" -n repos/nccl/src

# Trace a public operation
grep -R "ncclAllReduce\|ncclFuncAllReduce" -n repos/nccl/src

# Trace a plugin ABI version
grep -R "ncclNet_v\|ncclProfiler_v\|ncclEnvPlugin_v\|ncclTuner\|ncclGin_v\|ncclRma_v" -n repos/nccl

# Find Device API symbols
grep -R "ncclDevCommCreate\|ncclCommQueryProperties" -n repos/nccl/src repos/nccl/docs/examples
```
