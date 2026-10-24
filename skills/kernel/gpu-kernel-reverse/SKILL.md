---
name: gpu-kernel-reverse
description: >-
  Reverse engineer NVIDIA cubin, fatbin, embedded CUDA device code, or SASS into
  explainable CUDA C++ or CUTLASS/CuTe implementations, then validate semantics
  and benchmark performance against the original kernel. Use for GPU kernel
  reverse engineering, cubin-to-CUDA reconstruction, recovering launch ABI and
  thread/data layouts, or reproducing a binary kernel's performance. 中文：GPU
  kernel 逆向、cubin 反编译、SASS 还原 CUDA/CUTLASS、复现二进制算子及性能。
  Ordinary source-only optimization does not need this workflow.
---

# GPU Kernel Reverse Engineering

从给定 NVIDIA GPU 二进制开始调查，产出可解释、可编译、经过差分验证的 CUDA C++ 或 CUTLASS/CuTe 实现，在相同 GPU 和工作负载上追近或超过原 kernel 性能。优先完成用户指定的 kernel 和输入域；未指定时先枚举入口，从实际调用或可独立分析的入口开始。

编译是多对一映射，变量名、模板源码、原始控制结构通常不可唯一恢复；裸 cubin 还可能缺少输入域、launch 配置和运行时描述符。不要把“可以开展逆向”说成“所有 cubin 都能自动生成等性能源码”。条件缺失时继续静态分析，给出最小补证动作。

## 工作契约

- 分开记录语义恢复、输入覆盖、源码生成和实测性能。静态相似、CPU 仿真通过、GPU 正确、GPU 等性能分别报告。
- 原文件及 hash 是基准；提取、实验、候选放进独立目录，不改写原始 cubin。
- 结论带文件/hash、入口、PC/基本块、工具版本和目标 SM；区分 **观察、推断、待证**。heuristic 标签不是语义证明。
- 没有原始 GPU 执行结果，不声称 GPU 等价；没有同条件原始计时，不声称性能达标。候选不能偷偷加载原 cubin 或用测试输入预填输出。
- 默认交付 CUDA C++；tile/矩阵结构明确时使用 CUTLASS/CuTe；必要时使用局部 inline PTX。SASS patch 或原 cubin 包装器不算源码重建。

## 输入分流

先从文件、caller、测试和 profile 获取信息；缺失且影响下一步时才询问。

| 输入/状态 | 立即进行 | 执行阶段需要 |
|---|---|---|
| 单独 `.cubin` | 枚举函数、架构、资源、常量、参数元数据、SASS/CFG | 参数语义、有效输入、launch、兼容 GPU |
| `.fatbin`、`.so`、可执行文件、框架缓存 | 提取设备 image，关联 host launch 与实际选择 | image/入口/调用实例，不能默认第一个 |
| 只有 SASS 文本 | 按入口/PC 分析控制流和数据流 | 原编码、元数据或 caller；缺失部分保持未知 |
| 保留 PTX/源码/lineinfo | 缩小搜索，再对照实际 SASS | 同一 image/变体的证据；JIT 可能改变代码 |
| 无 GPU / 无兼容解码器 | 静态恢复、受控编译、适用的 CPU 模型 | 目标环境执行；新 SM 不能强行当旧 SM 解码 |

复制 [launch-contract.json](assets/launch-contract.json) 到工作目录，填入真实值，未知项保留 `null`。这是调查记录，不是通用 launcher 输入。至少记录参数 offset/size/type、缓冲区 shape/stride/alias、grid/block、动态 shared memory、cluster/合作 launch、全局状态与有效域。

## 推进流程

### 1. 定位与冻结基准

读 [工具与提取](references/tools-and-extraction.md)，用 `scripts/collect_evidence.py --help` 确认调用参数，保存原始输出、工具版本、逐命令状态。容器提取后每个目标 cubin 单独收集，避免混合多 image 的资源和指令。

建立 mangled entry 与 caller 的映射，检查寄存器、shared memory、spill/stack、常量、重定位和依赖函数。优先恢复原始 kernel 的正确 harness；未知 ABI 时先做静态工作。

### 2. 恢复语义和并行布局

读 [语义重建](references/reconstruction.md)。从输出 store/atomic 反向切片，恢复参数依赖、地址公式、谓词与循环；先写有位宽/舍入语义的模型，再提升成线程、warp、CTA 和 tile 操作。

产出 `semantic-map.md`：参数表、PC→操作映射、CFG、地址方程、跨线程通信、数值行为、假设及能区分假设的实验。不能凭 EX2 宣布 softmax，或只凭 HMMA 数量决定 tile。

涉及 tensor core、TMA、warp specialization 时读 [矩阵与异步流水线](references/tensor-and-pipelines.md)，保持 warp/warpgroup/CTA/cluster 的参与范围和同步协议。

### 3. 生成并验证源码

保存清楚的语义基线，再生成结构接近原 kernel 的 CUDA/CUTLASS 候选。明确 compiler、目标 SM（含特性后缀）、库 commit、flags；不要擅自打开 fast math、更换 dtype 或放宽误差。

交付真实计算、build 命令和 host 参数适配器。CUTLASS `Arguments`/`Params` 不能当作原始 ABI；不同 ABI 可经显式适配保持调用契约，用户要求二进制 drop-in 时另行满足。

按 [验证与性能](references/validation-and-performance.md) 对照原 kernel 和独立参考，覆盖有效域边界、尾块、别名、特殊数值和同步。先定位错误到地址/值/同步的首次分歧，再优化。

### 4. 编译与测量闭环

循环：**候选源码 → 相同配置编译 → 反汇编 → 核对结构 → GPU 差分 → 成对计时 → 按瓶颈改动**。

记录假设、主要变量、正确性、寄存器/shared/spill、每步主循环指令、原始计时和决策。优先对齐访存、tile/分工、异步 stage、归约/epilogue、占用率约束，再处理调度和编译器差异。opcode histogram 相同不等于性能相同。

默认将“同等性能”操作化为：约定的每个 case 正确性通过，候选相对原始时间的置信区间上界不超过 `1.05`。这是可修改的工程阈值，用户阈值优先；更快也达标。逐 case 报告，不能用平均提升掩盖回退。`scripts/compare_timings.py` 只评估给定计时，不证明数据来源与正确性。

### 5. 交付

用 [报告模板](assets/report-template.md) 保存源码/build、launch 适配器、正确性 harness、原始/候选 evidence、测量数据、依赖版本和未解问题。

结论选择：`static-only`、`candidate-unverified`、`correctness-verified`、`performance-matched`、`performance-gap`。条件缺失时明确已完成内容和下一条补证动作；正常迭代仍有可检验假设时继续推进，不用无限随机调参替代缺失证据。

## 按需资料

- [工具与提取](references/tools-and-extraction.md)：命令、容器、JSON、反编译与跟踪。
- [语义重建](references/reconstruction.md)：ABI、切片、地址、数值、受控实验。
- [矩阵与异步流水线](references/tensor-and-pipelines.md)：现代 SM、tile、pipeline 与 CUTLASS。
- [验证与性能](references/validation-and-performance.md)：replay、差分、计时、gate。
- [来源与取舍](references/sources.md)：联网核对的官方文档和工具及其限制。

本 skill 不依赖特定 agent 的委托功能。当前用户授权独立实现器时，可委托有明确输入/输出的候选或 harness，由主 agent 审查 diff、重跑 gates；不要自动启动未授权的付费外部 session。
