# LMSYS 博客写作模式

需要参考优秀公开文章来组织技术原理叙事或实验叙事时，读取本文件。本文只给结构上的示例，规则仍以 [报告结构模板](gpu-report-structures.md) 和 [实验报告结构](experiment-report-structures.md) 为准。

## 范围与方法

- 2026-11-01 获取了 [LMSYS 博客](https://www.lmsys.org/blog/) 目录和 sitemap 公开列出的全部 136 篇文章正文，两份列表一致。
- 对 136 篇做了解析器结构统计：79 篇含 HTML 表格，88 篇含代码块，128 篇含图。图片没有 OCR，图中的数据和条件没有纳入分析。
- 按提纲和选定段落深入分析了 15 篇，代表段落已对照原文核对。没有逐篇逐行通读，没有验证作者的实验，也没有跟进外部链接。
- 本文是对精选文章中可复用方法的原创中文总结，不代表所有 LMSYS 文章都这样写，也不评价文章质量。本文只总结结构，不复用其中的性能数字和实验阈值。

## 技术原理叙事

精选文章的机制部分大多沿着「约束 → 思路 → 机制 → 验证与代价 → 边界」展开，与 [报告结构模板](gpu-report-structures.md#主线) 的主线一致。可以借鉴的具体做法：

- 从约束推出方案。先写硬件或系统的限制和服务目标，再推出部署拓扑或执行路径，读者能看出每个选择对应哪条约束。
- 每个机制成对写出问题和设计。先说明生产负载为什么不同于孤立的 benchmark，再写设计；精度与速度并列给出，不只报速度。
- 被否方案单独成节。用「为什么不直接用 X」一两段写清既有方案为何不适用，比在文末罗列备选方案更容易读。
- 按职责拆分机制。例如把数据搬运路径与调度控制路径分开写，每部分有自己的数据流和接口。
- 代价与来源一起写。报告性能开销时，同时说明开销主要来自哪一部分，以及哪些部分尚未优化。

短篇机制说明从 [通用技术原理模板](../assets/technical-principle.md) 起步。

## 实验与测量叙事

- 先校准测量再给数字。在下结论之前，先说明 profiler 对 host 侧事件的影响、热缓存微基准与真实执行的差别、单窗口峰值的波动，再据此选定 A/B 用的指标。
- 正确性门槛先于性能。每项改动先通过约定的输出一致性或任务指标检查，再进入性能比较；会合法改变舍入的改动单独说明。
- 比较问题分开写。同并发比较回答「同样负载下谁更快」，最大并发比较回答「容量变大后能多跑多少」，两者分节给出。
- 越界条件和混杂因素显式披露。超出模型原生能力的测试点写明只作性能实验；两臂在某一档的配置不得不不同时，写在条件段里。
- 单项 A/B 与累计阶梯分开。逐项开关先单独比较，再组合；累计增益图只说明叠加后的效果。
- 组合收益显式归因。一个结果同时来自扩容、组件替换和限制解除时，在表后直接写出这几项共同构成收益。
- 结果带样本范围。评测分数写明子集大小和每题尝试次数；吞吐写明节点规模和固定不变的部分。
- 跨来源参考值单独标注。引用其他平台或其他团队的数字时写明是来源报告值，并指向本文自己的测量方法。
- 数值问题按诊断线写：异常现象 → 候选误差来源 → 逐一隔离 → 假设 → 验证。实验分组先说明隔离的是哪个变量，指标先定义含义和方向。
- 诊断指标写参照对象。mismatch 或 KL 相对哪个参考计算、包含哪些成分写清楚，并与 reward 等目标指标分开读。
- 分层验证，负结果进摘要。数值正确、持续学习、局部加速和集成代价分层报告；在线收益被同步或转换成本抵消这类结果，直接写进摘要。
- 推断和适用范围标出来。推测的原因标为推测；自建评测写明它不是唯一指标。

这些做法对应的规则见 [实验报告结构](experiment-report-structures.md)；描述性测量从 [测量记录模板](../assets/measurement-note.md) 起步。

## 摘要、正文与附录

- 摘要只写一次：主结果、关键条件和影响结论的限制。TL;DR、Abstract、Introduction 与结论节重复同一组数字时，读者无法判断哪一处是权威版本。
- 正文保留影响解读的条件，例如模型、硬件、workload、并发、计时口径和 cache 设置；完整数据表、启动参数和复现命令放附录，并给出 commit。
- 发布后发现描述有误时，用带日期的更正段落说明改了什么，不悄悄替换。

## 按需选图表

- 跨档位、跨条件的数字用表；趋势、Pareto 前沿和时间线用图；启动命令和配置用代码块，放在复现部分或附录。
- 一张表能说清时不必画图，也不必为每节配图。图的选择和图注规则见 [图表、公式与代码](figures-formulas-code.md)。

## 不照搬的写法

- 宣传式开头、以用户证言开篇、号召读者「立即体验收益」。
- 多层摘要重复同一组数字。
- 只写 `up to` 或最佳点而不给范围。这类表述可以用，但要同时给出测量范围和最佳点的条件。
- 平均倍数不写平均方式（算术平均、几何平均或加权）以及参与平均的用例。
- 用「目测没有明显相关」「在合理范围内」代替统计量或判定依据。没有预设阈值时，写观测值本身。
- 一侧用默认参数、另一侧调过参数，却不说明比较类型。
- 关键条件或数字只出现在图里。正文或表中也要给出，便于检索、引用和核对。
- 把博客中的数字当成本环境的实测结果，或当成匹配的基线。

## 精选来源

以下章节标题为原文标题；「可借鉴」一列是本文的归纳，不是原文摘录。

| 文章 | 章节 | 可借鉴 |
| --- | --- | --- |
| [Ling-3.0-flash 投机解码（2026-08-21）](https://www.lmsys.org/blog/2026-08-21-ling3-flash-spec-decode-blackwell) | 「Fixing the ruler before fixing the machine」；「DSpark: high-quality block drafting」下的「Results」 | 先校准测量工具；每项改动的正确性门槛；开关先单独 A/B 再组合；粗略一致性检查写明局限 |
| [NVFP4 KV cache（2026-09-16）](https://www.lmsys.org/blog/2026-09-16-nvfp4-kv-cache) | 「Performance」；「Limitations and Ongoing Items」 | 同并发与最大并发分开；越界测试点和不同配置显式披露；限制单独成节 |
| [DeepSeek-V4-Pro on H20（2026-08-19）](https://www.lmsys.org/blog/2026-08-19-deepseek-v4-pro-engine-optimization-h20) | 「1. Introduction」；「2. From Hardware Constraints to Serving Profiles」；「B.3 Benchmark Settings」 | 从硬件约束推出 serving profile；跨平台数字标为来源报告值并指向方法；正文写汇总，全表与启动参数放附录 |
| [HPC-Ops × SGLang（2026-08-07）](https://www.lmsys.org/blog/2026-08-07-hpc-ops-sglang) | 「Attention, routing, and experts: three hot paths in MoE model serving」；「Performance evaluation」 | 每个热点按问题 → 设计 → 精度与速度展开；评估先声明硬件范围，区分本地测量与上游 PR 中的结果 |
| [MXFP8 与 NVFP4 RL（2026-07-29）](https://www.lmsys.org/blog/2026-07-29-mxfp8-nvfp4-rl) | 「Why Blackwell-Native Recipes?」；「Why not directly use the NVFP4 pretraining recipe?」；「Train-inference mismatch」 | 被否方案单独成节；结果前声明实验只是 recipe 消融；诊断指标写明参照对象，与 reward 分开读 |
| [Unified FP8 RL（2025-11-25）](https://www.lmsys.org/blog/2025-11-25-fp8-rl) | 「Challenges of FP8 Training」；「Locating the Source of Error」；「MoE Experiment Design」 | 理论收益与实际收益的差距；从异常现象出发逐一隔离误差来源；实验分组和指标含义先定义 |
| [miniTE 技术报告（2026-10-25）](https://www.lmsys.org/blog/2026-10-25-minite) | 「TL;DR」；「6.1 Setup and Measurement Scope」；「6.6 What the Experiments Establish」 | 负结果和未决问题进摘要；算法与工程两条线分开评估；按层级说明实验能证明什么 |
| [MI355X agentic RL（2026-10-29）](https://www.lmsys.org/blog/2026-10-22-miles-mi355x-agentic-rl) | 「TL;DR」；「Results」下的「Throughput」 | 结果带样本范围和固定项；组合收益在表后直接归因 |
| [SGLang Llama3 serving（2024-07-25）](https://www.lmsys.org/blog/2024-07-25-sglang-llama3) | 「Benchmark Setup」；「Appendix A: Detailed Benchmark Setups」 | 先定义 offline 与 online 口径和指标分母；复现脚本放附录；带日期的更正段落 |
| [SGLang 确定性推理（2025-09-22）](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic) | 「Verifying Deterministic Behavior」；「Measuring Offline Inference Performance」 | 验证用例由简到难；开销与主要来源一起写 |
| [SGLang HiCache（2025-09-10）](https://www.lmsys.org/blog/2025-09-10-sglang-hicache) | 「Design of SGLang HiCache:」下的「Optimized data plane」「Versatile control plane」 | 按数据路径与控制路径拆分机制 |
| [LongChat 与 LongEval（2023-06-29）](https://www.lmsys.org/blog/2023-06-29-longchat) | 「Evaluation toolkits: LongEval」；「Discussion」；「Disclaimer」 | 评测方法从「怎样验证宣称」出发；推测原因标为推测；写明评测的适用范围 |
