---
name: persistent-kernel-scheduling
description: |
  CUDA Persistent Kernel 调度策略完整指南：静态调度 (Static Persistent)、动态调度
  (Dynamic Persistent / CLC)、单 Tile 调度 (Non-persistent) 的对比分析。
  覆盖 Hopper (SM90) 与 Blackwell (SM100) 架构，包含 Cluster Launch Control (CLC)
  硬件特性详解、Stream-K 负载均衡、尾效量化、决策树。
  适用场景：选择 persistent kernel 调度策略、理解 CLC 工作原理、诊断负载不均问题、
  优化 grouped GEMM / MoE / 变长 Attention 等负载不均场景。
triggers:
  - "persistent kernel"
  - "static persistent"
  - "dynamic persistent"
  - "CLC"
  - "cluster launch control"
  - "tile scheduler"
  - "Stream-K"
  - "负载不均"
  - "尾效"
  - "wave quantization"
  - "grouped GEMM 调度"
  - "MoE 调度"
  - "CTA 调度"
---

# CUDA Persistent Kernel：静态调度 vs 动态调度 资料汇总

> 整理时间：2025年6月  
> 覆盖架构：Hopper (SM90) 与 Blackwell (SM100)  
> 核心主题：Persistent Kernel 的 Tile Scheduling 策略演进

---

## 目录

1. [Persistent Kernel 基础概念](#1-persistent-kernel-基础概念)
2. [三种调度策略概览](#2-三种调度策略概览)
3. [静态 Persistent 调度 (Static Persistent)](#3-静态-persistent-调度-static-persistent)
4. [动态 Persistent 调度 (Dynamic Persistent / CLC)](#4-动态-persistent-调度-dynamic-persistent--clc)
5. [单 Tile 调度 (Single Tile / Non-persistent)](#5-单-tile-调度-single-tile--non-persistent)
6. [性能对比与实验数据](#6-性能对比与实验数据)
7. [代码实现参考](#7-代码实现参考)
8. [关键参考资料](#8-关键参考资料)

---

## 1. Persistent Kernel 基础概念

### 1.1 什么是 Persistent Kernel？

Persistent Kernel 是一种 GPU kernel 设计模式，其核心思想是：**让 CTA (Thread Block) 在 SM 上长期驻留，处理多个工作单元（tiles），而不是每个 tile 都重新启动一次 kernel**。

在传统的 "单 tile 调度" 中：
- 每个 CTA 只处理一个输出 tile
- 处理完就退出
- 硬件需要不断 launch 新的 CTA

在 Persistent Kernel 中：
- 启动的 CTA 数量 ≈ 可用 SM 数量（或 cluster 数量）
- 每个 CTA 处理完一个 tile 后，通过 tile scheduler 获取下一个 tile 的坐标
- 循环往复直到所有 tile 处理完毕

### 1.2 为什么需要 Persistent Kernel？

**主要收益：**

| 收益 | 说明 |
|------|------|
| **消除 CTA launch 开销** | 避免每波（wave）重新初始化 pipeline、barrier、shared memory |
| **Epilogue-Mainloop 重叠** | 当前 tile 的 epilogue（写回 C）可以与下一个 tile 的 mainloop（加载 A/B）重叠 |
| **减少 pipeline 重启** | TMA descriptor、barrier、shared memory 只需初始化一次 |
| **提高 SM 利用率** | 减少 wave 之间的空闲间隙 |

**Modular Blog 的总结**（Matrix Multiplication on Blackwell Part 4）：

> "By keeping the CTA resident on the SM, we can eliminate the overhead time taken to launch the next block."

### 1.3 Wave 概念

在 GPU 中，**wave** 指的是一批可以同时被调度到所有可用 SM 上的 thread blocks。Kernel 执行可以看作是一系列 wave 依次处理，直到所有工作完成。

对于 Persistent Kernel，kernel 作者（而非硬件）控制这些 block tile 坐标的调度。

---

## 2. 三种调度策略概览

| 调度策略 | 架构支持 | Grid 大小 | 工作分配方式 | 负载均衡 |
|---------|---------|----------|------------|---------|
| **Single Tile (Non-persistent)** | 所有架构 | = 总 tile 数 | 硬件调度，每个 CTA 一个 tile | ✅ 好 |
| **Static Persistent** | Hopper+ | = SM/Cluster 数 | 静态算术分配：tile_id += gridDim | ❌ 差（负载不均时） |
| **Dynamic Persistent (CLC)** | Blackwell+ | = 总 tile 数 | 运行时动态偷取：try_cancel | ✅ 好 |

### 2.1 编程模型对比（CUTLASS 官方文档）

```cuda
// ========== Non-persistent kernel ==========
__device__ non_persistent_kernel(...) {
  setup_common_data_structures();
  dim3 workCoordinates = blockIdx;
  coordinate_specific_compute(workCoordinates);
}

// ========== Static Persistent Kernel ==========
__device__ static_persistent_kernel(...) {
  setup_common_data_structures(...);
  dim3 workCoordinates = blockIdx;
  do {
    coordinate_specific_compute(workCoordinates);
    isValidId, workCoordinates = staticTileScheduler.fetch_next_work();
  } while (isValidId);
}

// ========== Blackwell Dynamic Persistent Kernel (CLC) ==========
__device__ clc_dynamic_persistent_kernel(...) {
  setup_common_data_structures(...);
  dim3 workCoordinates = blockIdx;  // 预加载第一个 ClcID
  do {
    coordinate_specific_compute(workCoordinates);
    isValidId, newClcID = clcTileScheduler.fetch_next_work();
    workCoordinates = newClcID;
  } while (isValidId);
}
```

---

## 3. 静态 Persistent 调度 (Static Persistent)

### 3.1 工作原理

静态 Persistent 调度在 **Hopper (SM90)** 上引入，核心思想：

1. **Launch 固定数量的 CTA/Cluster**：通常等于 GPU 上可同时驻留的 SM 或 cluster 数量
2. **静态算术分配 tile**：每个 CTA 通过简单的算术计算下一个 tile
   - 初始 tile = `blockIdx.x`
   - 下一个 tile = `current + gridDim.x`（即 stride = 总 CTA 数）
3. **循环直到无 tile 可处理**

### 3.2 代码示例（Modular Blog）

```mojo
fn fetch_next_work(mut self) -> WorkInfo:
  self.idx += num_idle_ctas
  return self.idx
```

```mojo
# every CTA in cluster participates in loads
if WarpRole.is_producer():
  while work_info.is_valid():
    # load ...
    work_info = scheduler.fetch_next_work()

# only leader CTAs in cluster participate in MMA
if WarpRole.is_consumer():
  for work_info.is_valid():
    # MMA ...
    store_C()
    work_info = scheduler.fetch_next_work()
```

### 3.3 静态调度的缺陷

**核心问题：负载不均 (Load Imbalance)**

来自 Colfax Research 的经典例子（Grouped GEMM）：

| Problem | Shape | K 维度 |
|---------|-------|--------|
| 0 | (256, 256, 128) | 小 |
| 1 | (256, 256, 2048) | 大 |
| 2 | (256, 256, 128) | 小 |
| 3 | (256, 256, 2048) | 大 |

使用 tile shape (128, 128, 128)，每个 problem 产生 4 个 work tiles：
- Problems 0 和 2 的 tile：2^22 FLOPs
- Problems 1 和 3 的 tile：2^26 FLOPs（16 倍！）

**静态分配结果**（8 个 cluster，每个处理每隔 8 个 tile）：
- Cluster 0: tiles 0, 8 → 都来自 problem 0/2（小 K）
- Cluster 1: tiles 1, 9 → 都来自 problem 1/3（大 K）

→ **Cluster 1 的计算量是 Cluster 0 的 16 倍！**

当其他 cluster 已完成工作进入空闲时，处理大 K tile 的 cluster 还在"补作业"，整个 Grid 的执行时间被这根"长尾"严重拖慢。

### 3.4 静态调度的其他问题

1. **无法感知实时 SM 可用性**：如果某些 SM 被其他 kernel 占用，静态调度不知道，仍然按固定数量 launch
2. **无法处理抢占 (pre-emption)**：被抢占的 SM 上的工作无法迁移到其他空闲 SM
3. **长尾效应 (Tail Effect)**：最后一波 wave 往往只有部分 SM 在工作

### 3.5 适用场景

- 负载均衡的标准 GEMM（M/N/K 均匀）
- 问题规模足够大，wave 数量多，尾效影响小
- Blackwell 上作为 CLC 的 fallback/对比基准

---

## 4. 动态 Persistent 调度 (Dynamic Persistent / CLC)

### 4.1 Cluster Launch Control (CLC) 概述

**CLC 是 Blackwell (SM100) 引入的硬件特性**，是硬件支持的动态 persistent tile 调度。

**核心思想**：
- 像单 tile 调度一样 launch 完整的逻辑 grid（所有 work tiles 都有对应的 ClcID）
- 但第一波 active cluster 会循环尝试"偷取"（cancel）尚未 launch 的 cluster 的工作
- 取消成功后，该 cluster 获得被取消 cluster 的 tile 坐标，自己完成这项工作

### 4.2 CLC 的六大规则（NVIDIA 官方文档）

1. 当资源可用时，ClcID 会被 launch 为 Worker
2. 已存在的 Worker 可通过 `clusterlaunchcontrol.try_cancel` 查询并取消未 launch 的 ClcID
3. 每个 ClcID 保证被 (1) 或 (2) 处理
4. 每个 Worker 预加载一个 ClcID（即 `{blockIdx.x, blockIdx.y, blockIdx.z}`）
5. `try_cancel` 返回成功信号（含 ClcID）或拒绝信号（所有 ClcID 已处理完）
6. CLC 工作在 cluster 粒度。例如 2×2 persistent worker cluster 的查询一次消耗 2×2 个 ClcID

### 4.3 CLC 与软件原子计数器的对比

**软件方案（Hopper 可用）**：
```cuda
// 全局原子计数器
__device__ int next_tile;
// 每个 cluster 完成后原子 fetch-and-inc
int my_tile = atomicAdd(&next_tile, 1);
```

问题：
- 所有 cluster 反复对同一全局计数器做原子操作
- 引入序列化
- 需要反复访问 global memory
- 每次 kernel launch 前需清零计数器

**CLC 硬件方案（Blackwell）**：
- 硬件管理 ClcID 的分配和取消
- 无需全局原子操作
- 支持抢占和并发 kernel 调度
- 自动处理资源回收

### 4.4 CLC Pipeline 设计

CLC 查询可以被 pipeline 化以隐藏延迟。CUTLASS C++ kernel 使用 depth=3 的 pipeline：

```python
# CuTeDSL 示例 (dense_gemm_persistent_dynamic.py)
clc_pipeline = pipeline.PipelineClcFetchAsync.create(
    barrier_storage=storage.clc_mbar_ptr.data_ptr(),
    num_stages=self.num_clc_stage,  # 1 in example, 3 in CUTLASS C++
    producer_group=clc_pipeline_producer_group,
    consumer_group=clc_pipeline_consumer_group,
    tx_count=self.num_clc_response_bytes,  # 16 bytes (ClcID response)
    cta_layout_vmnk=cluster_layout_vmnk,
    defer_sync=True,
)
```

**Warp 分工**（Blackwell warp-specialized kernel）：

| Warp 角色 | Warp ID | 职责 |
|-----------|---------|------|
| MMA | 0 | 执行 tcgen05.mma |
| Scheduler | 1 | **CLC producer**：发起 try_cancel 查询 |
| Mainloop Load | 2 | TMA load A/B |
| Epilogue Load | 3 | TMA load C (for epilogue) |
| Epilogue | 4-7 | 写回结果 |

### 4.5 CLC 与并发 Kernel / 抢占

CLC 的一个重要优势是支持**动态资源释放**：

> "Another reason that a `try_cancel` can fail... a second, higher-priority kernel was launched... After observing the failure, CTAs of the first kernel will exit, yielding GPU resources... Then, after the higher-priority kernel finishes... new clusters will be launched to finish off the rest of the grid."

这是静态 persistent 调度无法做到的——静态调度一旦 launch，资源分配就固定了。

### 4.6 Multi-stage CLC Pipeline 的权衡

虽然可以增加 CLC pipeline stage 数来隐藏调度延迟，但：

> "The larger the number of stages, the more CLC will resemble static persistent scheduling."

原因：
- 深 pipeline 会提前为不同 SM 分配不等量的工作
- 失去了动态负载均衡的优势
- 对于负载严重不均的 grouped GEMM，甚至应该**阻塞 scheduler warp**，等 MMA mainloop 完成后再发起 try_cancel

### 4.7 适用场景

- **Grouped GEMM / MoE**：不同 problem 的 tile 工作量差异大
- **变长 Attention**：不同 sequence length 的 tile 工作量差异大
- **多 kernel 并发**：需要动态资源调整
- **Blackwell 上的默认选择**：CUTLASS 将 CLC 作为 Blackwell 的默认调度器

---

## 5. 单 Tile 调度 (Single Tile / Non-persistent)

### 5.1 工作原理

最简单的调度方式：
- Grid 大小 = 总 work tile 数
- 每个 CTA/Cluster 处理一个 tile 后退出
- 硬件负责调度新的 CTA 到空闲 SM

### 5.2 优缺点

| 优点 | 缺点 |
|------|------|
| 负载均衡天然好 | 每波都要重新初始化 pipeline |
| 实现简单 | 无法重叠 epilogue 和下一个 tile 的 mainloop |
| 无调度开销 | CTA launch 开销累积 |
| | 尾效问题（最后一波 SM 利用率低） |

### 5.3 适用场景

- 小规模问题（tile 数 < SM 数）
- 对 launch 开销不敏感的场景
- 作为 baseline 对比

---

## 6. 性能对比与实验数据

### 6.1 负载不均场景（Colfax Research，B200）

**测试配置**：
- GPU: B200 (148 SMs, 74 clusters of shape 2×1)
- Data type: mxfp4
- MMA tile: 256×128, 2CTA MMA
- Grouped GEMM: 4 problems, K 从 1024 到 8192 变化

**结果**：

> "When work tiles become highly load-imbalanced, the dynamic scheduler significantly outperforms the static scheduler."

### 6.2 负载均衡场景（Colfax Research，B200）

**测试配置**：
- Standard dense GEMM
- M=N from 1024 to 32768
- K in [2048, 8192]
- Float8E4M3FN, MMA tile 256×256, cluster 2×1

**发现**：

| 观察 | 说明 |
|------|------|
| Persistent > Single Tile | Persistent 调度器能重叠 epilogue 和 mainloop |
| CLC vs Static 差异复杂 | CLC 在大 workload 上有时反而略差 |
| L2 Hit Rate 差异 | (32768, 32768, 2048) 时，CLC L2 hit 35% vs Static 52% |
| Tensor Pipe 利用率 | CLC 在末尾阶段能更好地利用所有 SM |

**结论**：
> "Even for balanced workloads one should keep both static scheduling and CLC for tuning purposes."

### 6.3 三种调度器的理论对比

| 维度 | Single Tile | Static Persistent | Dynamic Persistent (CLC) |
|------|------------|-----------------|--------------------------|
| **Launch 开销** | 高（每 tile 都 launch） | 低（一次 launch） | 中（完整 grid launch，但可能不都执行） |
| **Epilogue 重叠** | ❌ 无 | ✅ 有 | ✅ 有 |
| **负载均衡** | ✅ 好 | ❌ 差 | ✅ 好 |
| **尾效** | 有 | 有 | 较小 |
| **抢占支持** | ✅ 天然支持 | ❌ 不支持 | ✅ 硬件支持 |
| **调度开销** | 无 | 无（算术计算） | 低（硬件 CLC） |
| **L2 局部性** | 中 | 好（固定模式） | 中（动态可能打乱） |
| **实现复杂度** | 低 | 中 | 高 |

---

## 7. 代码实现参考

### 7.1 CUTLASS Tile Scheduler 抽象

```cpp
// CUTLASS 的 Tile Scheduler 通用接口
for (auto worktile = scheduler.get_initial_tile();
     scheduler.is_valid(worktile);
     worktile = scheduler.get_next_tile(worktile)) {
    
    auto [m_block, n_block, k_block_start, k_block_stop] = worktile.get_block_coord();
    
    for (k_block = k_block_start; k_block < k_block_stop; ++k_block) {
        // mainloop: load A/B via TMA, compute MMA
    }
    // epilogue: write back C
}
```

### 7.2 Static Persistent 的 Next Tile 计算

```cpp
// 简单线性调度
class StaticPersistentTileScheduler {
  int idx;
  
  WorkInfo get_next_work() {
    idx += gridDim.x;  // stride = 总 CTA 数
    return idx < total_tiles ? WorkInfo(idx) : InvalidWork();
  }
};
```

### 7.3 CLC 的 Mojo 实现（Modular Blog）

```mojo
struct TileScheduler[num_pipeline_stages: Int]:
  
  fn fetch_next_work(self, ...) -> WorkInfo:
    # Wait for the 16 bytes arrival (ClcID response)
    index, phase = consumer_state.index(), consumer_state.phase()
    self.full_mbar[index].wait(phase)
    
    # Read work coordinate from shared memory
    var work_tile = work_info_from_clc_response(self.clc_response[index])
    
    # Signal scheduler CTA that work coordinate is fetched
    self.empty_mbar[index].arrive_cluster(0)
    
    return work_tile
  
  fn advance_to_next_work(self, ...) -> PipelineState:
    index, phase = producer_state.index(), producer_state.phase()
    
    # Wait for work coordinate to be fetched
    self.empty_mbar[index].wait(phase())
    
    # Set arrival signal of 16 bytes
    self.full_mbar[index].arrive_and_expect_bytes()
    
    if elect_one_sync():
      # Try to cancel thread blocks and write coordinate to clc_response
      clusterlaunchcontrol_try_cancel[multicast=True](
        self.clc_response + index, 
        self.full_mbar + index,
      )
    
    return producer_state.next()
```

### 7.4 CLC Pipeline 参数设置

```python
# CLC pipeline 的关键参数
# transaction_bytes = 16 (CLC 返回 16B 的 ClcID 响应)
# consumer_arv_count = 所有 consumer warps 的线程数
# producer_arv_count = 1 (只有 scheduler warp 的一个线程发起 try_cancel)
# producer_blockid = 0 (cluster 中第一个 CTA 作为 producer)
```

### 7.5 PTX 指令

```asm
// CLC 核心 PTX 指令 (Blackwell)
clusterlaunchcontrol.try_cancel  // 尝试取消未 launch 的 cluster
clusterlaunchcontrol.query       // 查询 CLC 状态
```

---

## 8. 关键参考资料

### 8.1 核心文章（按重要性排序）

| # | 标题 | 作者 | 链接 | 内容 |
|---|------|------|------|------|
| 1 | **Dynamic persistent tile scheduling with CLC on Blackwell** | Colfax Research | [链接](https://research.colfax-intl.com/dynamic-persistent-tile-scheduling-with-cluster-launch-control-clc-on-nvidia-blackwell-gpus/) | 最全面的 CLC 教程，含实验数据 |
| 2 | **CUTLASS Tutorial: Persistent Kernels and Stream-K** | Colfax Research | [链接](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/) | Persistent kernel 基础概念 |
| 3 | **Matrix Multiplication on Blackwell: Part 4** | Modular | [链接](https://www.modular.com/blog/matrix-multiplication-on-blackwell-part-4---breaking-sota) | CLC 实现细节，Mojo 代码 |
| 4 | **Dissecting Nvidia Blackwell** | SemiAnalysis | [链接](https://newsletter.semianalysis.com/p/dissecting-nvidia-blackwell-tensor) | 架构层面介绍 CLC |
| 5 | **基于 CuTe 和 CUTLASS 的 Blackwell Tensor Core 编程** | 知乎 | [链接](https://zhuanlan.zhihu.com/p/2008547341972574634) | 中文，静态 vs 动态调度图解 |
| 6 | **Blackwell Cluster Launch Control 文档** | NVIDIA CUTLASS | [链接](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_cluster_launch_control.md) | 官方伪代码和规则 |
| 7 | **Tile Scheduling Strategies** | MIT KernelWiki | [链接](https://github.com/mit-han-lab/KernelWiki/blob/master/wiki/techniques/tile-scheduling.md) | 调度策略速查 |
| 8 | **GEMM with Thread Block Clusters on Blackwell** | Colfax Research | [链接](https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/) | Thread Block Cluster 基础 |

### 8.2 官方代码仓库

| 仓库 | 路径 | 说明 |
|------|------|------|
| CUTLASS | `include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp` | Blackwell CLC tile scheduler |
| CUTLASS | `include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp` | Hopper static tile scheduler |
| CUTLASS | `include/cutlass/pipeline/sm100_pipeline.hpp` | CLC pipeline 实现 |
| CUTLASS | `examples/python/CuTeDSL/blackwell/dense_gemm_persistent_dynamic.py` | CuTeDSL CLC 示例 |
| CUTLASS | `examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py` | CuTeDSL static persistent 示例 |
| CUTLASS | `media/docs/cpp/blackwell_cluster_launch_control.md` | 官方 CLC 文档 |

### 8.3 相关概念

- **Thread Block Clusters (CGA)**：Hopper 引入，将多个 CTA 编组到同一 GPC
- **TMA Multicast**：同一 cluster 内多个 CTA 共享数据加载
- **Warp Specialization**：不同 warp 承担不同角色（producer/consumer/scheduler）
- **Stream-K**：另一种负载均衡策略，在 K 维度拆分 tile
- **Programmatic Dependent Launch (PDL)**：Hopper 引入，隐藏 kernel launch 延迟

---

## 9. 总结

### 9.1 决策树

```
问题规模小（tile 数 < SM 数）?
  → Single Tile 调度

Blackwell (SM100) 可用?
  → 负载不均? 
    → 是: Dynamic Persistent (CLC)
    → 否: 两者都试，调优选择

Hopper (SM90) 或仅静态调度可用?
  → 负载均衡的标准 GEMM?
    → 是: Static Persistent
    → 否: 考虑软件原子计数器方案，或接受负载不均
```

### 9.2 关键要点

1. **Persistent Kernel 的核心价值**：消除 CTA launch 开销 + 重叠 epilogue/mainloop
2. **静态调度的致命弱点**：负载不均时产生严重长尾效应
3. **CLC 的革命性**：硬件支持的动态工作偷取，无需全局原子操作
4. **CLC 不是银弹**：
   - 深 pipeline stage 会削弱动态均衡效果
   - 某些均衡 workload 上 L2 hit rate 可能不如静态调度
   - 需要与 static persistent 一起作为调优选项
5. **未来方向**：CLC 与 Stream-K、PDL 等特性结合，进一步优化 GPU 利用率

---

*本文档基于 NVIDIA 官方文档、CUTLASS 源码、Colfax Research 技术博客、Modular 博客、SemiAnalysis 架构分析以及 MIT KernelWiki 整理而成。*
