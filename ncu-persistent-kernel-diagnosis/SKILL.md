---
name: ncu-persistent-kernel-diagnosis
description: |
  NCU/Nsight Compute persistent kernel diagnosis skill. Use for persistent kernel
  performance diagnosis, SM idle bubbles, Tensor Core idle bubbles, pipeline stall,
  long scoreboard, barrier stall, warp issue stalled, PmSampling, load imbalance,
  tail effect, Hopper sm_90, Blackwell sm_100, B200, TC utilization, SM utilization,
  and evidence-based optimization from NCU metrics. 中文触发词：NCU 分析 persistent kernel、
  SM 空泡、TC 空泡、流水线空泡、尾效、负载不均、为什么 persistent kernel 慢。
triggers:
  - "ncu persistent"
  - "SM 空泡"
  - "TC 空泡"
  - "pipeline stall"
  - "long scoreboard"
  - "barrier stall"
  - "warp issue stalled"
  - "PmSampling"
  - "负载不均"
  - "尾效"
  - "persistent kernel 诊断"
---

# 通过 NCU 分析 Persistent Kernel 的 SM 空泡与流水线掩盖

> 针对 Hopper (SM90) / Blackwell (SM100) 上 Persistent Kernel 的性能诊断指南

---

## 一、核心问题定义

Persistent Kernel 的典型性能陷阱：

| 问题 | 表现 | 根因 |
|------|------|------|
| **SM 空泡 (Idle SM)** | 部分 SM 早结束，其他 SM 还在跑 | 静态调度负载不均 / 尾效 |
| **Tensor Core 空泡** | TC pipe 利用率低，大量 cycle 无指令发射 | 数据未就绪（TMA/WGMMA 等待） |
| **流水线未掩盖** | LSU stall 暴露到 TC，或 TC stall 暴露到 LSU | Producer-Consumer 同步间隙 |

---

## 二、NCU 采集命令

### 2.1 基础采集（必做）

```bash
# 全量指标 + PM Sampling（看时间线）
ncu --set full \
    --section PmSampling \
    -o report_full \
    ./your_persistent_kernel

# 如果只需要特定指标（更快）
ncu --metrics \
    sm__cycles_active.avg,sm__cycles_active.min,sm__cycles_active.max,\
    sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active,\
    sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active,\
    sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active,\
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    sm__inst_issued.avg.per_cycle_active,\
    sm__cycles_elapsed.avg,\
    smsp__warp_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_active,\
    smsp__warp_issue_stalled_math_pipe_throttle.avg.pct_of_peak_sustained_active,\
    smsp__warp_issue_stalled_barrier.avg.pct_of_peak_sustained_active,\
    smsp__warp_issue_stalled_mio_throttle.avg.pct_of_peak_sustained_active,\
    smsp__warp_issue_stalled_short_scoreboard.avg.pct_of_peak_sustained_active,\
    smsp__warp_issue_stalled_not_selected.avg.pct_of_peak_sustained_active,\
    smsp__warp_issue_stalled_selected.avg.pct_of_peak_sustained_active,\
    sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
    sm__pipe_lsu_cycles_active.avg.pct_of_peak_sustained_active \
    -o report_metrics \
    ./your_persistent_kernel
```

### 2.2 Source-Level 采集（定位代码热点）

```bash
ncu --set source --section SourceCounters \
    -o report_source \
    ./your_persistent_kernel
```

> **必须加 `-lineinfo` 编译**，否则 source view 为空。

---

## 三、SM 空泡诊断（负载不均 / 尾效）

### 3.1 关键指标

| 指标 | NCU 名称 | 诊断意义 |
|------|---------|---------|
| **SM Active Cycles (avg/min/max)** | `sm__cycles_active.[avg/min/max]` | **max >> min** = 严重负载不均 |
| **SM Elapsed Cycles** | `sm__cycles_elapsed.avg` | 总执行时间 |
| **Achieved Occupancy** | `sm__warps_active.avg.pct_of_peak_sustained_active` | 低 = warp 不足或调度问题 |

### 3.2 判断标准

```
# 计算 SM 负载不均衡度
imbalance_ratio = sm__cycles_active.max / sm__cycles_active.min

if imbalance_ratio > 1.5:
    → 严重负载不均（静态 persistent 调度的典型症状）
    → 考虑换 Dynamic Persistent (CLC) 或 Stream-K
    
if imbalance_ratio > 1.1:
    → 轻度不均，可能来自尾效或输入 shape 差异
    → 检查 grid size 是否为 SM 数的整数倍
```

### 3.3 Persistent Kernel 特有的尾效检查

**Tail Effect**：最后一波 wave 只有部分 SM 有工作。

```
# 理论尾效损失
total_tiles = (M/bM) * (N/bN)
num_sms = 148  # B200
clusters_per_sm = 1  # 假设 cluster size = 1
wave_size = num_sms * clusters_per_sm
tail_tiles = total_tiles % wave_size
tail_waste = tail_tiles / wave_size * 100%

if tail_waste > 20%:
    → 考虑调整 tile size 或 grid size 减少尾效
```

**NCU 中看尾效**：
- `PmSampling` 时间线：末尾阶段 SM 利用率骤降
- `sm__cycles_active.min` 远小于 `max`：部分 SM 提前空闲

---

## 四、Tensor Core 空泡诊断

### 4.1 核心指标

| 指标 | NCU 名称 | 含义 |
|------|---------|------|
| **Tensor Pipe Utilization** | `sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active` | TC 指令发射占峰值比例 |
| **Tensor Pipe Active** | `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | TC pipe 有指令的 cycle 占比 |
| **TC Op Count** | `sm__inst_executed_pipe_tensor_op_hmma.sum` (Hopper) / `sm__inst_executed_pipe_tensor_op_dmma.sum` | 实际执行的 TC 指令数 |

> **注意**：Blackwell (SM100) 的 metric 名称可能与 Hopper 不同，建议先用 `ncu --query-metrics` 枚举。

### 4.2 判断标准

```
tensor_util = sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active

if tensor_util < 30%:
    → 严重 TC 空泡
    → 检查数据依赖：TMA/WGMMA 等待
    
if 30% <= tensor_util < 60%:
    → 中度空泡
    → 可能来自 epilogue-mainloop 同步间隙
    
if tensor_util >= 80%:
    → TC 利用率良好
    → 瓶颈可能在别处（内存带宽、launch 开销）
```

### 4.3 TC 空泡的根因分析

TC 空泡通常来自**数据未就绪**，需要结合 stall 指标：

| Stall 原因 | NCU 指标 | 含义 |
|-----------|---------|------|
| **Long Scoreboard** | `smsp__warp_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_active` | 等待 global/L2 数据（TMA load 未完成） |
| **Short Scoreboard** | `smsp__warp_issue_stalled_short_scoreboard.avg.pct_of_peak_sustained_active` | 等待 shared memory / barrier |
| **Math Pipe Throttle** | `smsp__warp_issue_stalled_math_pipe_throttle.avg.pct_of_peak_sustained_active` | TC 指令背压（前一条 TC 未完成） |
| **Barrier** | `smsp__warp_issue_stalled_barrier.avg.pct_of_peak_sustained_active` | `cluster.sync()` / `mbarrier.wait()` 等待 |
| **MIO Throttle** | `smsp__warp_issue_stalled_mio_throttle.avg.pct_of_peak_sustained_active` | 发射队列满（指令密度过高） |

**诊断流程**：

```
TC 利用率低?
  → Long Scoreboard 高?
    → TMA/WGMMA 数据未就绪，检查 memory pipeline
    → 可能 TMA transaction_bytes 设置不当
    → 可能 LSU 与 TC 未充分重叠
  
  → Short Scoreboard 高?
    → 等待 shared memory 数据（producer-consumer 同步）
    → 检查 mbarrier 同步点是否过早/过晚
    → 检查 cluster.sync() 是否过度同步
  
  → Math Pipe Throttle 高?
    → TC 指令发射太密集，前一条未完成
    → 正常情况，但如果伴随 TC 利用率低，说明 warp 数不足
  
  → Barrier 高?
    → 同步开销过大
    → 检查是否可以用 targeted barrier 替代 cluster.sync()
    → 检查 mbarrier 的 arrival count 设置
```

---

## 五、流水线掩盖诊断（Producer-Consumer Overlap）

### 5.1 Persistent Kernel 的理想流水线

```
Time →
Tile N:   [TMA Load A/B] ---- [WGMMA] ---- [TMA Store C]
Tile N+1:              [TMA Load A/B] ---- [WGMMA] ---- [TMA Store C]
Tile N+2:                           [TMA Load A/B] ---- [WGMMA] ----

理想情况：TMA Load (Tile N+1) 与 WGMMA (Tile N) 完全重叠
```

### 5.2 关键指标组合

| 指标组合 | 诊断 |
|---------|------|
| `pipe_lsu` 高 + `pipe_tensor` 高 + 两者交替 | ✅ 良好掩盖 |
| `pipe_lsu` 高 + `pipe_tensor` 低 | ❌ TC 等待数据，LSU 未充分提前启动 |
| `pipe_lsu` 低 + `pipe_tensor` 高 | ⚠️ 可能数据已缓存，或 LSU 被 throttle |
| `pipe_lsu` 和 `pipe_tensor` 同时高 | ⚠️ 可能 stage 数不足，pipeline 未填满 |

### 5.3 用 PM Sampling 看时间线

```bash
ncu --section PmSampling -o report_pm ./kernel
```

分析 `report_pm.ncu-rep` 中的时间序列数据：

```python
# 伪代码：提取 PM Sampling 数据
import ncu_report
report = ncu_report.load("report_pm.ncu-rep")
action = report["your_kernel"]

# 获取时间序列指标
for sample in action.pmsampling_samples():
    time = sample.time
    sm_active = sample.metric("sm__cycles_active.avg")
    tensor_active = sample.metric("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active")
    lsu_active = sample.metric("sm__pipe_lsu_cycles_active.avg.pct_of_peak_sustained_active")
    # 绘制时间线，观察是否有周期性低谷
```

**典型模式**：

| 时间线模式 | 含义 |
|-----------|------|
| 周期性尖峰-低谷 | Pipeline stage 未填满，或 barrier 同步间隙 |
| 末尾骤降 | 尾效：最后几波 SM 利用率下降 |
| 持续低利用率 | 全局瓶颈（内存带宽、 occupancy 不足） |
| LSU 和 TC 交替高 | 良好掩盖，但可能有优化空间（增加 stage） |

### 5.4 Warp Specialization 的掩盖检查

对于 warp-specialized persistent kernel（Hopper/Blackwell 典型设计）：

| Warp 角色 | 期望行为 | 异常信号 |
|-----------|---------|---------|
| **Producer (TMA Load)** | 持续发射 TMA 指令 | `pipe_lsu` 低 = 等待 mbarrier 或 descriptor |
| **Consumer (WGMMA)** | 持续发射 TC 指令 | `pipe_tensor` 低 = 等待数据或 barrier |
| **Scheduler (CLC)** | 周期性 CLC 查询 | `barrier` stall 高 = CLC 响应延迟 |

**检查 Producer-Consumer 同步**：

```
# 理想：mbarrier 等待时间 ≈ 0（数据已就绪）
# 异常：mbarrier 等待时间 >> 0（consumer 等 producer）

smsp__warp_issue_stalled_barrier.avg.pct_of_peak_sustained_active
→ 如果 > 10%，检查：
  1. producer 是否提前足够多启动？
  2. stage 数是否足够？（通常 3-5 stage）
  3. TMA transaction_bytes 是否匹配实际数据量？
```

---

## 六、Persistent Kernel 专项检查清单

### 6.1 静态 Persistent (Hopper)

```
□ sm__cycles_active.max / min > 1.2?
  → 负载不均，考虑：
    - 输入 shape 是否均匀？
    - tile scheduler 是否线性分配？
    - 是否可用 CTA swizzling 改善 L2 locality？

□ 尾效检查：
  total_clusters = gridDim.x * gridDim.y * gridDim.z
  max_concurrent = num_sms / cluster_size
  tail = total_clusters % max_concurrent
  if tail > 0: 考虑调整 grid 或 tile size

□ TC 利用率 < 60%?
  → 检查 epilogue-mainloop 重叠：
    - 当前 tile 的 epilogue 是否与下一个 tile 的 mainloop 重叠？
    - 静态调度中，epilogue 和 mainloop 在同一 CTA 内串行？
```

### 6.2 Dynamic Persistent / CLC (Blackwell)

```
□ CLC try_cancel 成功率？
  → 需要 PM Sampling 或 custom logging（ncu 不直接暴露）
  → 替代：检查是否有异常多的 "early exit" 模式

□ CLC Pipeline depth 是否合适？
  - depth=1：CLC 查询延迟可能暴露
  - depth=3（CUTLASS 默认）：通常最优
  - depth>3：趋近静态调度，失去动态均衡优势

□ 抢占场景？
  → CLC 天然支持，但需验证 try_cancel 失败后 CTA 是否正确退出
```

---

## 七、实战案例：诊断流程

### 场景：Persistent GEMM 性能不达预期

**Step 1：SM 负载不均检查**
```bash
ncu --metrics sm__cycles_active.avg,sm__cycles_active.min,sm__cycles_active.max ./gemm
```
- max/min = 1.8 → **严重不均** → 怀疑静态调度 + 输入 shape 差异

**Step 2：TC 利用率检查**
```bash
ncu --metrics sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active ./gemm
```
- 45% → **中度空泡**

**Step 3：Stall 分解**
```bash
ncu --metrics \
  smsp__warp_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_active,\
  smsp__warp_issue_stalled_short_scoreboard.avg.pct_of_peak_sustained_active,\
  smsp__warp_issue_stalled_barrier.avg.pct_of_peak_sustained_active,\
  smsp__warp_issue_stalled_math_pipe_throttle.avg.pct_of_peak_sustained_active \
  ./gemm
```
- Long Scoreboard: 35% → **TMA 数据等待是主因**
- Barrier: 15% → **同步也有影响**

**Step 4：PM Sampling 时间线**
```bash
ncu --section PmSampling -o report_pm ./gemm
```
- 观察到周期性 LSU 尖峰后 TC 尖峰，中间有 ~10% gap
- 末尾 20% 时间 SM 利用率骤降 → **尾效**

**诊断结论**：
1. 静态调度导致负载不均（max/min=1.8）
2. TMA 与 WGMMA 重叠不足（Long Scoreboard 35%）
3. 尾效损失约 20%

**优化方向**：
1. 换 Dynamic Persistent (CLC) 或增加 CTA swizzling
2. 增加 pipeline stage 数，或调整 TMA 提前启动时机
3. 调整 tile size 使 grid 为 SM 数的整数倍

---

## 八、参考指标速查表

### 8.1 通用指标（Hopper/Blackwell）

| 指标 | 健康范围 | 异常信号 |
|------|---------|---------|
| `sm__warps_active.avg.pct_of_peak_sustained_active` | > 50% | < 30% = occupancy 问题 |
| `sm__inst_issued.avg.per_cycle_active` | 2-4 | < 1 = 严重 stall |
| `sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active` | > 60% | < 30% = TC 空泡 |
| `sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active` | > 40% | < 20% = LSU 未充分利用 |
| `smsp__warp_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_active` | < 20% | > 40% = 内存延迟瓶颈 |
| `smsp__warp_issue_stalled_short_scoreboard.avg.pct_of_peak_sustained_active` | < 10% | > 20% = SMEM 同步问题 |
| `smsp__warp_issue_stalled_barrier.avg.pct_of_peak_sustained_active` | < 5% | > 15% = 过度同步 |
| `smsp__warp_issue_stalled_math_pipe_throttle.avg.pct_of_peak_sustained_active` | < 10% | > 30% = TC 指令背压 |

### 8.2 Blackwell (SM100) 特殊注意

- Metric 名称可能与 Hopper 不同，建议先用 `ncu --query-metrics | grep tensor` 确认
- `sm__pipe_tensor_cycles_active` 可能比 `sm__inst_executed_pipe_tensor` 更能反映实际 TC 活跃周期
- CLC 相关指标目前 ncu 不直接暴露，需通过 PM Sampling 间接推断

---

## 九、工具链推荐

| 工具 | 用途 |
|------|------|
| `ncu --set full` | 全面指标采集 |
| `ncu --section PmSampling` | 时间线分析（尾效、周期性模式） |
| `ncu --set source` | 代码级 stall 热点定位 |
| `ncu_report` Python API | 程序化解析报告 |
| `helpers/plot_timeline.py` (ncu-report-skill) | ASCII 时间线可视化 |
| `ncu-cli` (第三方) | 自动化诊断 |

---

*本文档基于 NCU 官方文档、NVIDIA CUTLASS 实践、以及 Colfax Research / Modular Blog 的 Blackwell 分析经验整理。*
