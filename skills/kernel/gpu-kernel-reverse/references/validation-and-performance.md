# 验证、重放与性能验收

## 让原始 kernel 成为真实对照

优先复用已知正确的 caller。独立 harness 可用 Driver API：初始化 context，`cuModuleLoad`/`cuModuleLoadData`，`cuModuleGetFunction`，分配/初始化缓冲区、global/constant/workspace，设置函数属性，再按已恢复的 launch 配置调用。

`cuLaunchKernel` 的 `kernelParams` 是 **host 端各参数值的地址数组**，不是 device pointer 数组。例如指针参数应提供 `&device_ptr_value`。打包 `extra` buffer 时按真实 offset/size/alignment 填入，并使用该 API 要求的终止项；不要同时提供两种参数路径。

读取真实头文件/Driver API 文档确认 `cuFuncGetParamInfo`、扩展 launch/cluster API 与当前 driver 版本。cluster、cooperative launch、动态 smem opt-in、texture/tensor map、device link 依赖不能被普通 `cuLaunchKernel` 猜测替代。裸 cubin 不保证信息足够建立通用 launcher。

原始与候选各用 adapter，输入来自同一逻辑数据。保留 alias 拓扑和 in-place 语义；需相同地址条件时可顺序复用 allocation，但每次必须恢复初始状态。检查每次 API 返回值和同步后的异步错误。

## 正确性矩阵

先冻结数值契约，再优化：

- integer/bitwise 任务按位验证；浮点采用用户或原 workload 的 atol/rtol/ULP，另查 NaN/Inf/±0。不要为通过候选而放宽阈值。
- 原 kernel vs 候选之外，尽量加入独立 CPU/高精度数学 oracle，防止两边共享错误。oracle 的舍入/归约顺序不同需解释，不把 f64 数学结果机械当硬件 bitwise 标准。
- 有效域内的最小/边界/常见/大 shape、tile 边界前后、尾块、stride/转置/合法非连续布局、alignment、alias、in-place、mask、空输入（若支持）。
- zeros/ones/ramp/one-hot、正负/大动态范围、多个随机种子、特殊浮点、极端合法 scale/sparse metadata。
- 多次 launch、不同合法 grid、workspace/RNG/counter reset、并发或 cluster 场景（若契约包含）。stochastic 输出按声明的统计契约验证，不能因有随机性而忽略偏差。

在适用的 GPU/工具版本上使用 Compute Sanitizer 的 memcheck、racecheck、synccheck、initcheck，覆盖代表性边界路径。它们分别有覆盖限制，racecheck 主要针对 shared hazards，并非所有 global races/弱内存协议的证明。sanitize 构建和运行不用于性能计时。

原始不支持的输入不进入等价比较；如果实现扩展输入域，扩展部分另用 oracle 验证。任何越界、同步错误或错误输出都阻止性能验收。

## 可比的计时

冻结设备 UUID/model/SM/MIG、driver/toolkit、功耗/时钟/温度状态、输入 shape/stride/dtype、launch/cluster、dynamic smem、stream、缓存策略、graph/replay 模式、workspace/packing 范围和候选 build hash。主动更改系统时钟/功耗不是本流程的默认动作。

1. 在兼容的同一 GPU 上，初始化/分配/JIT/编译与测量分离，先预热；保持相同状态恢复语义。
2. 用同一 stream 的 CUDA events 或已验证 harness。短 kernel 按多次 launch 的 batch 计时再除次数，使窗口足够长；确保批次重复运行不会改变算法状态。CUDA Graph 对双方一致，并单独说明普通 launch 与 graph 指标。
3. 以相邻 A/B 或 B/A 为一对，随机/交替顺序，收集多个独立 batch pair；默认至少 20 对。相同一轮内的重复 launch 不是 20 个独立样本。漂移/自相关明显时增加独立批次或用适合的 block bootstrap，不盲信独立采样 CI。
4. 原始和候选不同时占用 GPU；避免其他负载污染。报告 median、分位数、每对数据、时序与置信区间，保留原始数据而不是只保留最佳一次。
5. 按真实应用分别定义 warm/cold cache；固定重复同一小输入可能只测到 L2。需要时轮换大于 cache 的工作集，并对双方同样执行。
6. 单 kernel latency 与包括 packing/setup/多 kernel 的端到端 latency 分开。不能删去候选额外工作来获得表面加速；如果基准是 kernel-only，同样解释额外成本。

## Nsight Compute 用于解释差距

先从本地 `ncu --list-sections`/`--query-metrics` 选择可用的小集合，收集 launch、occupancy、memory、scheduler/warp state、source 证据；不要硬编码跨架构通用的 metric 名。从实际热点 PC/依赖解释 stall，单一高百分比不一定是根因。

NCU 可能序列化、replay、保存/恢复内存、清 cache 和控制时钟，所以 profile 中 duration 不等于原始应用 latency。应用/范围 replay 与 kernel replay 对交互式/persistent/concurrent kernel 的兼容性不同；核对工具文档并保持两侧设置一致。插桩结果也不能作为无插桩性能。

## 性能 gate

默认工程阈值 `margin=0.05`，用户要求优先。对**每个约定 case**计算 `median(candidate_us) / median(reference_us)`，保持 batch pairing 做 bootstrap，报告 95% 区间：

- 上界 ≤ `1 + margin`：该 case 的计时 gate 通过；更快也通过。
- 下界 > `1 + margin`：存在实测回退。
- 区间跨阈值：证据不确定，先改善测量或增加独立样本。
- 样本不足、case 缺失、NaN/Inf/非正时间：不能通过。

`scripts/compare_timings.py` 提供这个辅助计算。输入 JSON 结构如下（`...` 表示省略的真实测量，不能直接执行）：

```json
{
  "expected_cases": ["shape-A", "shape-B"],
  "cases": [
    {"name": "shape-A", "reference_us": [12.1, 12.0, "..."],
     "candidate_us": [12.0, 12.1, "..."]}
  ]
}
```

实际输入必须包含全部 case、每对双方等长且有限正数的数组。case 名应唯一标识 shape/dtype/layout/launch/测量模式。此脚本不启动 GPU、不验证样本独立性、不证明正确性，也不核对上面要求的环境；最终 `performance-matched` 还需要正确性记录和可比环境证据。期望 case 列表必须在测量前冻结，不能为通过而删减。

```bash
python3 /path/to/gpu-kernel-reverse/scripts/compare_timings.py timings.json --margin 0.05
```

默认 10,000 次 bootstrap，`--bootstrap` 至少 1,000；默认至少 20 对 batch。返回码 `0/1/2/3/4/5` 分别为全部 case 计时通过/数据无效/CLI 用法错误/回退/不确定/样本不足，具体情况见输出 JSON。调整统计参数应基于测量设计，不应反复选择随机种子直到通过。

多 case 的区间是逐项区间，不能把它们宣称为整个 suite 的同时 95% 保证；严谨的全套统计结论需多重比较校正。性能范围也仅限已覆盖设备与输入域。

## 按差距选择实验

| 证据 | 优先实验 |
|---|---|
| transaction/实际搬运字节更多 | 地址、vector width、mask、layout、额外临时缓冲 |
| spills/local traffic/占用率变化 | live range、unroll、tile、launch bounds；不要机械限制寄存器 |
| barrier/wait 很多，TC 有空隙 | stage 数、producer/consumer 分工、wait 位置和循环尾部 |
| 原始与候选数学相同但 MMA 路径不同 | dtype/accumulator、tile/atom、target、编译器、库 schedule |
| 访存相似但长依赖链 | ILP、循环展开、accumulator 依赖、地址提前计算 |
| kernel 快但端到端慢 | packing、workspace、额外 launch、host/graph 开销 |
| SASS 很相似但 latency 不稳定 | 时钟、缓存、地址对齐、温度、launch 模式、外部负载 |

一次改一个主要因素，正确性先过再计时；保留最优已验证候选。编译器不保证暴露所有调度控制，无法对齐时交付性能差距和证据，不隐瞒未达标 case。
