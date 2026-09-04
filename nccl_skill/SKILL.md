---
name: nccl-skill
description: >
  Comprehensive reference documentation and skill for NVIDIA NCCL (Collective Communications Library),
  the GPU communication library for multi-GPU and multi-node collectives. Use this skill whenever the
  user mentions NCCL, all-reduce, all-gather, reduce-scatter, broadcast, gather/scatter, all-to-all,
  ncclSend/ncclRecv, communicator initialization, CUDA stream group semantics, distributed training
  communication, NVLink/NVSwitch/InfiniBand/TCP transport behavior, NCCL environment variables,
  debugging NCCL hangs or performance, NCCL plugins (net/tuner/profiler/env), Device API, GIN, LSA,
  symmetric memory, user buffer registration, RAS, or NCCL source-code internals. Common queries:
  NCCL allreduce, NCCL hang, NCCL_DEBUG, NCCL_ALGO, NCCL_PROTO, ncclCommInitRank, ncclGroupStart,
  NCCL net plugin, tuner plugin, GIN, LSA, symmetric memory, ncclMemAlloc, ncclCommSplit,
  per-collective config, ncclCollConfig, zero-SM collectives, CFT, PAT.
  中文触发词：NCCL、集合通信、多卡通信、allreduce、alltoall、通信 hang 住、NCCL 环境变量、NCCL 调优、NCCL 源码、Device API、GIN、对称内存。
version: 2.31.2
triggers:
  - "NCCL"
  - "allreduce"
  - "all-gather"
  - "reduce-scatter"
  - "alltoall"
  - "ncclSend"
  - "ncclRecv"
  - "NCCL_DEBUG"
  - "NCCL_ALGO"
  - "NCCL_PROTO"
  - "GIN"
  - "LSA"
  - "symmetric memory"
  - "ncclCommInitRank"
  - "NCCL 调优"
  - "集合通信"
  - "通信 hang"
---

# NCCL - NVIDIA Collective Communications Library

## Source Code Locations

NCCL 源码位于此 skill 安装目录下的 `repos/nccl/`（NVIDIA/nccl，sparse checkout）。
实际路径取决于所用工具:
- Cursor: `~/.cursor/skills/nccl-skill/repos/nccl/`
- Claude Code: `~/.claude/skills/nccl-skill/repos/nccl/`
- Codex: `~/.codex/skills/nccl-skill/repos/nccl/`
- Kimi Code: `~/.agents/skills/nccl-skill/repos/nccl/`

**NCCL_REPO**: 下文示例用 `~/.cursor/skills/nccl-skill/repos/nccl/` 作占位符，**替换为实际路径**。

如果该路径不存在，在项目目录下运行 `bash update-repos.sh nccl`。

### NCCL 源码结构

```
NCCL_REPO/
├── src/
│   ├── nccl.h.in          # 公共 API 头文件模板（生成 nccl.h，含 ncclCollConfig_t）
│   ├── collectives.cc     # 集合通信入口（ncclAllReduce 及 nccl*Config 变体）
│   ├── group.cc           # ncclGroupStart/End 语义
│   ├── enqueue/           # 任务入队、kernel plan、launch（task_prep/、task_sched/）
│   ├── init.cc            # communicator 初始化
│   ├── device/            # 设备端 kernel 与协议原语
│   ├── devcomm/           # Device API communicator 版本 shim
│   ├── gin/               # GIN host/proxy/device 后端（gdaki、efa_gda、gpi、proxy）
│   ├── cft_dev_runtime.cc # CFT（Compute Fabric Transport）runtime
│   ├── ce_coll.cc         # Zero-SM / Copy Engine collectives
│   ├── sym_kernels.cc     # 对称内存 kernel（含 Blackwell TMA kernel）
│   ├── config/            # collConfig 解析、算法注册表
│   ├── tuning/            # 统一 cost model（ring/tree/nvls/collnet/pat/ce/sym）
│   ├── transport/         # P2P / SHM / NET / CollNet / NVLS 传输
│   ├── graph/             # 拓扑发现与 ring/tree/NVLS 图搜索
│   ├── include/           # 内部头文件（comm.h、device.h、info.h、nccl_device/ 等）
│   └── ...
├── plugins/               # 插件头文件与示例：net / tuner / profiler / env / gin / rma / mixed
├── contrib/               # nccl_ep、pace、niin、nccl4rust、custom_algos 等
├── bindings/              # nccl4py（Python 绑定，含 CuTe DSL device API）
├── docs/                  # userguide / dev_guide / examples / perf
└── makefiles/             # 构建系统
```

## How to use this skill

Use this skill as a code-aware NCCL reference manual. When a user asks about NCCL usage, debugging,
performance, plugin development, or internals, first identify which layer they are working at:

1. **Application API usage**: communicator lifecycle, collectives, P2P, group semantics, errors.
2. **Advanced user features**: buffer registration, symmetric memory windows, one-sided RMA, Device API.
3. **Runtime tuning/debugging**: environment variables, topology, algorithms/protocols, transport choice.
4. **Plugin development**: net, tuner, profiler, env, GIN/plugin loading.
5. **Source-code internals**: init, group/enqueue/scheduler, device kernels, topology graph, proxy/transport.
6. **Bindings/contrib**: nccl4py and NCCL EP for MoE expert parallel communication.

When the answer needs precise behavior, cite the source file path from `repos/nccl` and, if you read
current files, cite `path:line`. If a reference here names a symbol and the user is about to modify code,
verify the current source first because NCCL internals change quickly.

## NCCL in one page

NCCL is a standalone GPU communication library implementing collective and point-to-point primitives
optimized for PCIe, NVLink, NVSwitch, InfiniBand Verbs, and TCP/IP sockets. It supports arbitrary GPU
counts in a single node or across multiple nodes, and can be used from single-process, multi-threaded,
or multi-process/MPI applications.

**Core public API families:**
- Communicator lifecycle: `ncclGetUniqueId`, `ncclCommInitRank`, `ncclCommInitAll`,
  `ncclCommInitRankConfig`, `ncclCommInitRankScalable`, `ncclCommFinalize`, `ncclCommDestroy`,
  `ncclCommAbort`, `ncclCommSplit`, `ncclCommShrink`, `ncclCommGrow`, `ncclCommRevoke`.
- Collectives: `ncclReduce`, `ncclBroadcast`, `ncclAllReduce`, `ncclReduceScatter`,
  `ncclAllGather`, `ncclAlltoAll`, `ncclGather`, `ncclScatter`.
- Per-collective configuration (2.31+): `ncclCollConfig_t` + `nccl*Config` variants of every
  collective (e.g. `ncclAllReduceConfig`) for per-call algorithm selection, CTA/CGA size and
  CTA policy overrides, and profiler `userProfilerTag`.
- P2P and one-sided: `ncclSend`, `ncclRecv`, `ncclPutSignal`, `ncclSignal`, `ncclWaitSignal`.
- Group semantics: `ncclGroupStart`, `ncclGroupEnd`, `ncclGroupSimulateEnd`.
- Memory/registration: `ncclMemAlloc`, `ncclMemFree`, `ncclCommRegister`, `ncclCommDeregister`,
  `ncclCommWindowRegister`, `ncclCommWindowDeregister`, `ncclWinGetUserPtr`.
- Device API: `ncclCommQueryProperties`, `ncclDevCommCreate`, `ncclDevCommDestroy`, LSA/GIN pointer,
  team, barrier, and device communication helpers from `nccl_device` headers; CFT (Compute Fabric
  Transport) device-side Put/Get/Red/NVLS on Blackwell with CUDA 13.3+.
- Parameters: `ncclParamBind`, typed `ncclParamGet*`, `ncclParamGetParameter`,
  `ncclParamGetAllParameterKeys`, `ncclParamDumpAll`.

**Notable 2.30.7–2.31.2 features covered in the references:**
- Zero-SM collectives: hierarchical AllGather/AllToAll via RMA CPU proxy (inter-node) + Copy
  Engine (intra-node), enabled with `NCCL_CTA_POLICY_ZERO` / `NCCL_CTA_POLICY=ZERO`.
- PAT (Parallel Aggregated Tree) hierarchical kernels for ReduceScatter/AllGather
  (NVLS intra-node + PAT inter-node), opt-in via `NCCL_ALGO=PAT` or `collConfig`.
- GIN: EFA GDA backend, per-DevComm backend selection, custom strides, device-side timeouts,
  Strong/Weak signal semantics, GDAKI DDP/GRH/path-MTU; GIN plugin ABI is v13/v14 only.
- Symmetric memory: asymmetric window sizes, window registration during CUDA graph capture,
  one-sided RMA multi-context/multi-signal across multiple NICs.
- Diagnostics: `NCCL_RUN_DIAGNOSTICS=1`, `NCCL_RUN_RAS_DIAGNOSTICS=1`; Profiler V7 per-kernel
  phase events; RMA plugin ABI v15; net plugin v12; tuner v6; env v2.
- Contrib additions: PACE, NIIN, nccl4rust, nccl_checkpoint, nccl_m2n, nccl_ubx, custom_algos.

## Quick usage patterns

### Single-process multi-GPU allreduce
```cpp
int ndev = 0;
cudaGetDeviceCount(&ndev);
std::vector<ncclComm_t> comms(ndev);
ncclCommInitAll(comms.data(), ndev, nullptr);

ncclGroupStart();
for (int r = 0; r < ndev; ++r) {
  cudaSetDevice(r);
  ncclAllReduce(send[r], recv[r], count, ncclFloat32, ncclSum, comms[r], streams[r]);
}
ncclGroupEnd();

for (int r = 0; r < ndev; ++r) cudaStreamSynchronize(streams[r]);
for (int r = 0; r < ndev; ++r) {
  ncclCommFinalize(comms[r]);
  ncclCommDestroy(comms[r]);
}
```

### Multi-process initialization shape
```cpp
ncclUniqueId id;
if (rank == 0) ncclGetUniqueId(&id);
// Broadcast id with MPI, sockets, shared memory, or another launcher mechanism.
cudaSetDevice(local_device);
ncclComm_t comm;
ncclCommInitRank(&comm, world_size, id, rank);
```

### P2P ring pattern must use grouping
```cpp
ncclGroupStart();
ncclSend(sendbuf, count, ncclFloat32, next_rank, comm, stream);
ncclRecv(recvbuf, count, ncclFloat32, prev_rank, comm, stream);
ncclGroupEnd();
```

### Debugging first line
```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET,TUNING ./your_app
```

## Code Examples

Official NCCL performance tests from [NVIDIA/nccl-tests](https://github.com/NVIDIA/nccl-tests), covering all collective operations:

| Example | Collective Operation |
|---------|---------------------|
| `examples/src/all_reduce.cu` | AllReduce — sum/avg/min/max across all ranks |
| `examples/src/all_gather.cu` | AllGather — gather data from all ranks |
| `examples/src/alltoall.cu` | All-to-All — personalized communication |
| `examples/src/broadcast.cu` | Broadcast — one-to-all distribution |
| `examples/src/reduce.cu` | Reduce — reduce to single rank |
| `examples/src/reduce_scatter.cu` | ReduceScatter — scatter reduced data |
| `examples/src/gather.cu` | Gather — collect to single rank |
| `examples/src/scatter.cu` | Scatter — distribute from single rank |
| `examples/src/sendrecv.cu` | Send/Recv — point-to-point communication |
| `examples/src/hypercube.cu` | Hypercube collective algorithm |
| `examples/src/common.cu/h` | Common infrastructure: timing, data validation |
| `examples/verifiable/` | Verifiable NCCL operations with correctness checks |

Each test supports configurable message sizes, data types, and iteration counts for benchmarking NCCL performance.

## Search Strategy

**用 Grep/rg 工具搜索**，不要整文件加载。

```bash
NCCL_REPO="$HOME/.cursor/skills/nccl-skill/repos/nccl"

# 查找公共 API 定义
rg "ncclAllReduce|ncclCommInitRank" $NCCL_REPO/src/nccl.h.in

# 查找环境变量定义
rg "NCCL_PARAM" $NCCL_REPO/src/ -l

# 查找算法/协议选择
rg "NCCL_ALGO|NCCL_PROTO" $NCCL_REPO/src/

# 查找设备端 kernel 与协议原语
rg "RunWork|LL128|primitives" $NCCL_REPO/src/device/

# 查找插件接口头文件
ls $NCCL_REPO/plugins/net/ $NCCL_REPO/plugins/tuner/ $NCCL_REPO/plugins/profiler/

# 查找 NCCL EP（MoE expert parallel）
ls $NCCL_REPO/contrib/nccl_ep/
```

## Documentation map

### User-facing API and usage
- [01 - Overview, Build, Install](references/01-overview-build-install.md): what NCCL is, build/package/test flow, examples layout.
- [02 - Public API and Communicators](references/02-public-api-communicators.md): public types, errors, config, initialization, lifecycle, split/shrink/grow.
- [03 - Collectives, P2P, and Groups](references/03-collectives-p2p-groups.md): operation semantics, in-place rules, count/layout rules, P2P deadlocks, grouping.
- [04 - Memory Registration, Symmetric Windows, RMA](references/04-memory-registration-symmetric-rma.md): `ncclMemAlloc`, buffer registration, symmetric windows, suspend/resume, signals.
- [05 - Device API, LSA, GIN](references/05-device-api-lsa-gin.md): device communicators, teams, windows, LSA/GPU-Initiated Networking, examples.
- [06 - Example Patterns](references/06-example-patterns.md): single process, pthread, MPI, AllReduce, ring P2P, advanced examples.

### Runtime configuration and performance
- [07 - Environment Variables and Parameters](references/07-environment-parameters.md): source-derived parameter map and public parameter API.
- [08 - Debugging, Troubleshooting, Profiling](references/08-debugging-troubleshooting-profiling.md): hangs, async errors, transport visibility, profiler plugins, logging.
- [09 - Algorithms, Protocols, Tuning](references/09-algorithms-protocols-tuning.md): Ring/Tree/CollNet/NVLS/PAT, LL/LL128/SIMPLE, cost model and tuner plugin hooks.

### Source internals
- [10 - Source Architecture and Communicator Lifecycle](references/10-source-architecture-communicator-lifecycle.md): key source files, `ncclComm`, init pipeline.
- [11 - Group, Enqueue, Scheduler, Launch Pipeline](references/11-group-enqueue-scheduler-launch.md): `ncclInfo` → tasks → kernel plans → CUDA launch/proxy ops.
- [12 - Device Kernels and Protocol Primitives](references/12-device-kernels-protocols.md): generated kernels, `RunWork*`, SIMPLE/LL/LL128 primitives.
- [13 - Topology, Graph Search, Channels](references/13-topology-graph-channels.md): topology discovery, ring/tree/NVLS/CollNet graph search, channel connection.
- [14 - Transports, Proxy, Networking](references/14-transports-proxy-networking.md): P2P/SHM/NET/CollNet/NVLS transports, proxy progress, sockets/IB.

### Extension surfaces
- [15 - Plugin Development](references/15-plugin-development.md): net, tuner, profiler, env, mixed plugins, ABI/versioning/load names.
- [16 - RAS and Fault Handling](references/16-ras-fault-handling.md): RAS thread, peers, keepalive, communicator revoke/shrink patterns.
- [17 - Python Bindings and NCCL EP](references/17-python-bindings-nccl-ep.md): nccl4py, NCCL EP dispatch/combine for MoE.
- [18 - Source File Index](references/18-source-file-index.md): source-code map by module for fast navigation.

## Answering rules for NCCL tasks

- For API questions, explain both **host-side enqueue semantics** and **CUDA stream completion semantics**. NCCL calls usually return after enqueueing work, not after communication finishes.
- For P2P questions, check whether multiple sends/recvs must progress concurrently. If yes, recommend `ncclGroupStart/End`.
- For multi-GPU single-thread code, default to grouping per-device NCCL calls.
- For multi-process code, make `cudaSetDevice(localRank)` before communicator init explicit.
- For hangs, ask for `NCCL_DEBUG=INFO`, rank count, launcher, device mapping, network interface/NIC, and the last log lines per rank.
- For performance, distinguish topology/algorithm/protocol/channel issues from application stream synchronization and buffer registration issues.
- For source changes, start from `repos/nccl/src/nccl.h.in`, `collectives.cc`, `group.cc`, `enqueue/`, `init.cc`, and `src/include/comm.h` depending on the layer.

## 更新 NCCL 源码

```bash
# 在 agent-gpu-skills 项目目录下
bash update-repos.sh nccl
```

## Additional References

- NCCL 官方仓库: https://github.com/NVIDIA/nccl
- NCCL 官方文档: https://docs.nvidia.com/deeplearning/nccl/
- NCCL Tests: https://github.com/NVIDIA/nccl-tests
- 本 skill 内容改编自: https://github.com/jstzwj/ai-infra-plugins (plugins/nccl/skills/nccl)
