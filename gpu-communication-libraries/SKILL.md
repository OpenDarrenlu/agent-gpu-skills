---
name: gpu-communication-libraries
description: |
  GPU communication libraries and distributed GPU communication skill. Use for NCCL,
  NVSHMEM, CUDA P2P/IPC, GPUDirect, RDMA, UCX/UCC, Gloo, RCCL, MSCCL, DeepEP,
  GIN, collectives, allreduce, allgather, reduce-scatter, alltoall, multi-GPU,
  multi-node training/inference, tensor/expert/data parallel communication, and
  communication-kernel fusion. 中文触发词：GPU 通信库、多卡通信、多节点通信、
  NCCL、NVSHMEM、RDMA、GPUDirect、P2P、CUDA IPC、allreduce、alltoall、通信优化。

triggers:
  - "GPU 通信库"
  - "NCCL"
  - "NVSHMEM"
  - "collective communication"
  - "P2P"
  - "GPUDirect"
  - "RDMA"
  - "allreduce"
  - "alltoall"
  - "多卡通信"
  - "分布式通信"
  - "通信优化"
  - "Gloo"
  - "UCX"
  - "DeepEP"
  - "通信 kernel"
  - "GPU P2P"
  - "CUDA IPC"
---

# GPU 通信代码库全景指南

## 1. 核心 NVIDIA 官方库

### 1.1 NCCL (NVIDIA Collective Communications Library)
**GitHub**: https://github.com/NVIDIA/nccl
**Docs**: https://docs.nvidia.com/deeplearning/nccl/

**核心特性**:
- 业界最广泛使用的 GPU 集体通信库
- 支持 Ring / Tree / NVLS / CollNet 四种算法
- 支持 Simple / LL / LL128 三种协议（带宽-延迟权衡）
- 自动拓扑感知（NVLink、PCIe、InfiniBand、Ethernet）
- 支持多维度并行：Data Parallel / Tensor Parallel / Pipeline Parallel / Expert Parallel
- **NCCL 2.28+ Device API**: 支持 GPU 发起通信（GIN、LSA、Multimem）

**支持的操作**:
```
AllReduce, AllGather, ReduceScatter, Broadcast, Reduce, SendRecv, AllToAll, AllToAllv, Gather, Scatter
```

**关键环境变量**:
```bash
NCCL_DEBUG=INFO              # 查看拓扑检测和算法选择
NCCL_ALGO=RING|TREE|NVLS    # 强制算法
NCCL_PROTO=LL|LL128|SIMPLE  # 强制协议
NCCL_IB_HCA=mlx5_0          # 指定 IB 网卡
NCCL_P2P_DISABLE=0            # 启用 P2P
NCCL_NET_GDR_LEVEL=5        # GPUDirect RDMA 阈值
```

**适用场景**: 分布式训练（PyTorch DDP/FSDP、DeepSpeed、Megatron-LM）、分布式推理

---

### 1.2 NVSHMEM (NVIDIA OpenSHMEM for GPU)
**GitHub**: https://github.com/NVIDIA/nvshmem
**Docs**: https://docs.nvidia.com/nvshmem/

**核心特性**:
- **唯一支持 device-initiated 通信的模型**（GPU 内核中直接发起通信）
- Partitioned Global Address Space (PGAS) 编程模型
- 支持 GPU 端直接调用 put/get/atomic 操作
- 支持 device-initiated collectives（需 cooperative group launch）
- 无需返回 host 即可完成全部通信

**关键概念**:
```c
// 在 GPU kernel 中直接调用
__global__ void kernel() {
    nvshmem_put(target, source, nelems, target_pe);  // 设备端发起 put
    nvshmem_get(target, source, nelems, target_pe);  // 设备端发起 get
    nvshmem_barrier_all();                           // 设备端 barrier
}
```

**实现原理**:
- GPU 线程通过 pinned host memory 与 host progress 线程通信
- Host 线程调用 Verbs/UCX 发起实际 RDMA 操作
- 较新版本的 NVSHMEM 也在探索 direct GPU-to-NIC 路径

**适用场景**:
- 自定义通信 kernel（fused compute-comm kernel）
- 零 SM 占用、低延迟 collectives
- 需要 kernel 内精细控制通信的 HPC/AI 场景

---

### 1.3 CUDA P2P / IPC (Peer-to-Peer & Inter-Process Communication)
**Docs**: CUDA Runtime API 文档

**P2P Direct（单进程多 GPU）**:
```c
// 检查并启用 peer access
int canAccess;
cudaDeviceCanAccessPeer(&canAccess, gpu0, gpu1);
if (canAccess) {
    cudaSetDevice(gpu0);
    cudaDeviceEnablePeerAccess(gpu1, 0);  // 允许 GPU0 直接访问 GPU1 内存
}
// 之后 GPU0 的 kernel 可以直接 dereference GPU1 的指针
```

**IPC（多进程）**:
```c
// 进程 A: 导出内存句柄
cudaIpcMemHandle_t handle;
cudaIpcGetMemHandle(&handle, d_buf);
// 通过 MPI/socket 发送 handle

// 进程 B: 导入并访问
void* peer_buf;
cudaIpcOpenMemHandle(&peer_buf, handle, cudaIpcMemLazyEnablePeerAccess);
// 使用 peer_buf 如同本地指针
```

**硬件路径**: NVLink（高带宽）或 PCIe（通过 PCIe switch/root complex）

**适用场景**: 单进程多 GPU、MPI 多进程多 GPU、NCCL/NVSHMEM 内部实现基础

---

### 1.4 GPUDirect 技术族

| 技术 | 全称 | 功能 | 场景 |
|------|------|------|------|
| **GPUDirect P2P** | Peer-to-Peer | GPU 间直接内存访问 | 单机多卡 |
| **GPUDirect RDMA** | Remote DMA | GPU 内存直接通过网络 RDMA | 跨节点通信 |
| **GPUDirect Storage** | - | GPU 直接访问存储 | 数据加载 |
| **GPUDirect Async** | - | GPU 直接发起网络操作 | GIN / DOCA GPUNetIO |

**关键**: GPUDirect RDMA 绕过 CPU 内存，实现 GPU memory ↔ NIC 的直接数据传输，是 NCCL 跨节点高性能的基础。

---

## 2. 其他厂商/开源集体通信库

### 2.1 RCCL (ROCm Collective Communications Library)
**GitHub**: https://github.com/ROCm/rccl
**描述**: AMD GPU 版的 NCCL，兼容 NCCL API，用于 MI100/MI200/MI300 系列 GPU

### 2.2 MSCCL (Microsoft Collective Communication Library)
**GitHub**: https://github.com/microsoft/msccl
**描述**: 
- 微软开发的自定义 collective 算法编译器
- 支持编写自定义通信算法（如自定义 ring、tree）
- 通过 XML/DSL 描述算法，编译为 CUDA kernel
- 与 NCCL 集成，可作为 NCCL 的后端算法

### 2.3 Gloo
**GitHub**: https://github.com/facebookincubator/gloo
**描述**:
- Facebook 开发的集体通信库
- 支持 CPU 和 GPU（CUDA-aware）
- 主要用于 CPU 通信，GPU 支持较弱
- PyTorch 早期 DDP 的默认后端
- 支持 AllReduce、Broadcast、Barrier 等

### 2.4 oneCCL (Intel oneAPI Collective Communications Library)
**GitHub**: https://github.com/oneapi-src/oneCCL
**描述**: Intel 的集体通信库，支持 CPU 和 GPU（Intel GPU），兼容 MPI 风格 API

### 2.5 ACCL (Alibaba Collective Communication Library)
**描述**: 阿里内部开发的集体通信库，针对大规模集群优化

---

## 3. 通信中间层 / 框架

### 3.1 UCX (Unified Communication X)
**GitHub**: https://github.com/openucx/ucx
**Docs**: https://openucx.readthedocs.io/

**核心特性**:
- 统一的通信框架，抽象底层传输（IB Verbs、RoCE、TCP、Shared Memory、CUDA IPC）
- 自动选择最优传输路径
- 支持 GPU 内存传输（cuda_copy、cuda_ipc、gdr_copy）
- 被 OpenMPI、MPICH、UCC 等广泛使用

**关键环境变量**:
```bash
UCX_TLS=rc,sm,cuda_copy,cuda_ipc,gdr_copy  # 启用传输层
UCX_NET_DEVICES=mlx5_0                     # 指定网卡
UCX_IB_GPU_DIRECT_RDMA=yes                 # 启用 GDR
```

---

### 3.2 UCC (Unified Collective Communications)
**GitHub**: https://github.com/openucx/ucc
**描述**:
- 基于 UCX 的集体通信库
- 支持 CUDA-aware collectives
- 与 MPI 集成，可作为 MPI 的集体通信后端

---

### 3.3 MPI (Message Passing Interface) - CUDA-aware
**实现**: OpenMPI、MPICH、MVAPICH、Intel MPI

**CUDA-aware MPI 特性**:
```c
// 直接传递 GPU 指针给 MPI
MPI_Send(d_buf, size, MPI_BYTE, dest, tag, MPI_COMM_WORLD);
MPI_Recv(d_buf, size, MPI_BYTE, src, tag, MPI_COMM_WORLD, &status);
```
- 内部自动使用 CUDA IPC / P2P / GPUDirect RDMA
- 简化编程，无需手动管理 host staging buffer
- 但性能通常不如 NCCL（缺乏拓扑优化和自定义算法）

---

## 4. 新兴 / 专用通信库

### 4.1 DeepEP (DeepSeek Expert Parallelism)
**GitHub**: https://github.com/deepseek-ai/DeepEP
**描述**:
- DeepSeek 开源的 MoE 训练/推理通信库
- 支持 low-latency 和 high-throughput 两种内核模式
- 针对 all-to-all（expert dispatch）场景极致优化
- 支持 GPU-initiated RDMA（通过 GIN/NCCL Device API）
- 与 NCCL 2.28+ GIN 集成

**核心特性**:
- 针对 MoE 的 irregular all-to-all 优化
- 支持 FP8 通信
- 支持 dual-stream 重叠（compute + comm）

---

### 4.2 GIN (GPU-Initiated Networking) - NCCL 2.28+
**来源**: NCCL 2.28 Device API
**论文**: https://arxiv.org/abs/2511.15076

**核心特性**:
- GPU 线程直接从 CUDA kernel 发起 one-sided RDMA 操作
- 双后端架构：
  - **GDAKI**: 通过 DOCA GPUNetIO 直接 GPU-to-NIC 通信（16.7μs RTT）
  - **Proxy**: CPU 辅助操作，兼容标准 RDMA 硬件
- 与 NCCL 现有基础设施集成（hierarchical communicator、fault tolerance、topology-aware）
- 统一三种原语：
  - **LSA**: NVLink/PCIe 的 Load/Store Accessible
  - **Multimem**: NVLink SHARP 网络内计算
  - **GIN**: 网络 RDMA

**适用场景**: MoE 推理、编译器生成的融合 kernel、需要 kernel 内直接控制网络的 workload

---

### 4.3 pplx-kernels
**来源**: Perplexity AI 开源
**描述**: 提供 low-latency GPU-initiated 通信原语，独立于 collective 通信框架

---

### 4.4 FasterMoE / Tutel
- **FasterMoE**: 针对 MoE 的专家调度优化
- **Tutel**: 微软的 MoE 优化库，包含通信优化

---

## 5. 快速选型指南

### 5.1 按场景选择

| 场景 | 推荐库 | 理由 |
|------|--------|------|
| 通用分布式训练（PyTorch/TensorFlow） | **NCCL** | 生态最成熟、拓扑感知最优、与框架深度集成 |
| 需要 kernel 内融合通信 | **NVSHMEM** | 唯一支持 device-initiated，PGAS 模型 |
| 需要 GPU 直接控制网络（MoE 推理） | **GIN / NCCL Device API** | GPU 直接发起 RDMA，延迟最低 |
| MoE all-to-all 极致优化 | **DeepEP** | 针对 MoE 场景专门优化 |
| AMD GPU 训练 | **RCCL** | NCCL API 兼容 |
| 自定义 collective 算法 | **MSCCL** | 可编写自定义算法并编译为 kernel |
| CPU 为主 + GPU 辅助 | **Gloo** | 对 CPU 友好，PyTorch 早期后端 |
| 与 MPI 生态集成 | **CUDA-aware MPI + UCX/UCC** | 标准 MPI 编程模型 |
| 自定义 HPC 通信 kernel | **CUDA P2P/IPC + NVSHMEM** | 最底层控制 |

### 5.2 按延迟/带宽需求选择

| 需求 | 选择 |
|------|------|
| 最低延迟（< 20μs RTT） | GIN (GDAKI) / DeepEP |
| 高带宽大消息 | NCCL (Ring + Simple) |
| 小消息低延迟 | NCCL (Tree + LL) / NVSHMEM |
| 内核内零开销 | NVSHMEM device-initiated |
| 跨节点 GPUDirect RDMA | NCCL + GPUDirect RDMA |
| 单节点 NVLink | NCCL / CUDA P2P |

---

## 6. 关键 API 速查

### NCCL 核心 API
```c
ncclCommInitRankConfig(&comm, nranks, ncclId, rank, &config);
ncclAllReduce(sendbuf, recvtbuf, count, ncclFloat32, ncclSum, comm, stream);
ncclAllToAll(sendbuf, recvbuf, count, ncclFloat32, comm, stream);
ncclGroupStart(); ncclSend(...); ncclRecv(...); ncclGroupEnd();
ncclCommDestroy(comm);
```

### NVSHMEM 核心 API
```c
// Host API
nvshmem_init();
void *ptr = nvshmem_malloc(size);

// Device API (在 kernel 中调用)
__device__ void nvshmem_put(void *dest, const void *source, size_t nelems, int pe);
__device__ void nvshmem_get(void *dest, const void *source, size_t nelems, int pe);
__device__ void nvshmem_barrier_all();
__device__ void nvshmemx_putmem_block(void *dest, const void *source, size_t bytes, int pe);
```

### CUDA P2P/IPC API
```c
cudaDeviceCanAccessPeer(&accessible, dev0, dev1);
cudaDeviceEnablePeerAccess(peerDev, 0);
cudaMemcpyPeerAsync(dst, dstDev, src, srcDev, size, stream);

cudaIpcGetMemHandle(&handle, d_ptr);
cudaIpcOpenMemHandle(&d_peer, handle, cudaIpcMemLazyEnablePeerAccess);
cudaIpcCloseMemHandle(d_peer);
```

---

## 7. 性能调试工具

| 工具 | 用途 |
|------|------|
| **Nsight Systems** | 时间线分析，查看通信与计算重叠 |
| **Nsight Compute** | Kernel 级性能分析 |
| **NCCL_DEBUG=INFO** | 查看 NCCL 拓扑检测、算法选择、通道配置 |
| **UCX_LOG_LEVEL=debug** | UCX 传输层调试 |
| **nvidia-smi topo -m** | 查看 GPU 拓扑（NVLink/PCIe 连接） |
| **ib_write_bw / ib_read_bw** | 测试 RDMA 带宽 |
| **nccl-tests** | NCCL 性能基准测试 |

---

## 8. 相关资源

- **NCCL GitHub**: https://github.com/NVIDIA/nccl
- **NVSHMEM GitHub**: https://github.com/NVIDIA/nvshmem
- **UCX GitHub**: https://github.com/openucx/ucx
- **UCC GitHub**: https://github.com/openucx/ucc
- **RCCL GitHub**: https://github.com/ROCm/rccl
- **MSCCL GitHub**: https://github.com/microsoft/msccl
- **Gloo GitHub**: https://github.com/facebookincubator/gloo
- **DeepEP GitHub**: https://github.com/deepseek-ai/DeepEP
- **NCCL Tests**: https://github.com/nvidia/nccl-tests
- **CUDA P2P/IPC 示例**: https://github.com/NVIDIA/cuda-samples
- **GIN 论文**: https://arxiv.org/abs/2511.15076
- **xCCL 综述论文**: https://jcst.ict.ac.cn/cn/article/pdf/preview/10.1007/s11390-023-2894-6.pdf
