# Kernel 重建报告

状态：选择 static-only / candidate-unverified / correctness-verified / performance-matched / performance-gap。

## 结论与覆盖

说明恢复了什么、使用什么 CUDA/CUTLASS 实现、在哪些真实 GPU/输入域验证、尚缺什么。性能结论引用逐 case 的数据与阈值；未测项写“未测”。

## 基准和重现

记录原始/candidate SHA256、container image、mangled entry、target SM、compiler/driver/library commit、完整 build/launch/test/benchmark 命令及依赖。链接 launch-contract 与原始 evidence manifest。

## 语义与并行结构

| 结论 | 原始 PC/基本块/元数据/实验 | 证据等级 | 对应源码 |
|---|---|---|---|

补充参数、地址方程、lane/tile、同步/异步依赖图、数值策略。把未解 opaque 和假设列出，不用 heuristic 标签替代证明。

## 正确性

| case / 有效域 | seed / 输入分布 | 比较契约 | 误差/失败定位 | 原始 vs 候选 | 独立 oracle / sanitizer |
|---|---|---|---|---|---|

## 性能

注明设备/时钟/缓存/graph/计时范围、预热、样本单位、paired 顺序、样本数、CI 方法与阈值。

| case | 原始 median | 候选 median | 候选/原始 | CI | 结论 |
|---|---|---|---|---|---|

链接原始时间数组和 profiler 报告，说明 kernel-only 与端到端差异。资源/指令特征仅作为诊断证据。

## 迭代与未解问题

| 假设/主要改动 | 正确性 | 性能 | 决策/证据 |
|---|---|---|---|

每个未解项附最小可执行补证动作；区分缺硬件/缺 ABI/缺语义与已经测到的性能差距。
