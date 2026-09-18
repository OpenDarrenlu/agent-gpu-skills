# AKO：Agentic Kernel Optimization，使用 Claude Code 在几小时内优化出超越 FlashInfer 专家实现的 Kernel

**作者**: sha7dow宅😐

**原文链接**: https://zhuanlan.zhihu.com/p/2044554843948308001

---

​
目录
收起
优化结果
简单介绍
一些思考
相对于 FlashInfer 专家实现在 5 个 MLSys-2026 竞赛内核（NVIDIA B200）上的几何平均加速比 —— AKO4X 在 DSA 稀疏注意力上最高超越专家 30.71×，并在全部五项上都领先于 MIT 的 KDA。

我们开源了完整的 AKO: Agentic Kernel Optimization，包括：
项目主页：https://tongminglaic.github.io/AKO/
AKO4ALL：https://github.com/TongmingLAIC/AKO4ALL
AKO4X：https://github.com/TongmingLAIC/AKO4X

优化结果

在 MLSys 2026 Nvidia Track（FlashInfer AI Kernel Generation Contest）里，MIT 的方案 Kernel Design Agents（KDA）在 Full Agent 赛道的最终结果是：

Track	Result
MoE Track	1st place
DSA Track	2nd place
GDN Track	3rd place

但实际我们 AKO4X 优化的 5 个 kernel（3 个 Track 一共 5 个 kernel），在我们公开的同一套 B200 评测口径下（官方环境），性能全部显著超过了 KDA 提交的 kernel：

Kernel	AKO4X vs FlashInfer expert	KDA vs FlashInfer expert	Workloads
DSA sparse attention	30.71×	3.78×	23/23
GDN prefill	2.30×	1.53×	100/100
MoE fp8 block-scale	1.19×	0.59×	19/19
GDN decode	1.17×	0.81×	54/54
DSA top-k indexer †	1.27× faster than KDA	— (ref)	128/128

注意这里的 baseline 不是 naive PyTorch baseline，而是 FlashInfer 高性能算子库里的专家实现。

所以从技术结果看，我们本来应该排在 KDA 前面。但比较遗憾的是，由于 “we also removed (after careful review and discussion) teams with members based in restricted regions from the winner candidates.” 的资格限制（比赛奖品的显卡有出口管控），我们作为中国团队最终没有进入官方 winner candidates。我们尊重比赛规则和合规要求，但也希望把技术结果、代码、数据和完整优化轨迹都公开出来，一方面想向大家展示我们这几个月的工作和成果，另一方面也是希望能加速这个方向的发展。

除了比赛的这 5 个 kernel，我们还在 B200 上测试了 FlashInfer-Bench 中的其他几个 kernel（数量不多，因为 B200 太贵了😢），除了 GEMM 外都在几个小时甚至一个小时内超越了 FlashInfer 中的专家实现，具体结果可以见上面的项目主页。

优化轨迹和结果

简单介绍

和两个多月前我们在 为什么 AI Infra 里还在大量招聘 GPU Kernel Engineer？的回答中说的一样，AKO 不是一个新模型，也不是一个复杂的搜索算法。它就是一个任务特定的 harness：一个给 coding agent 使用的 kernel 优化环境。

它做的事情很简单：给 coding agent 提供 benchmark、profiler、迭代记录、候选 kernel 管理、反作弊检查等工具，然后让 coding agent 自己读代码、改 kernel、跑 benchmark、看 profiling、换 DSL、继续迭代，每一步做什么都由它自己决定。

我们的设计理念是：尽可能少地限制 agent 本身的能力 （对比 fixed pipeline 方法）；提供工具、信息，使 agent 能够在 kernel 优化任务下更好地发挥能力

我们开源了两个工具：

AKO4ALL：这个实际上在两个月前的回答中已经开源了，这次我们只是对它进行了一次重构。AKO4ALL 现在是一个非常轻量的 SKILL。安装后把 coding agent 指到你的 kernel（+ benchmark / reference）上，就可以开始迭代优化。它更适合快速优化一个 kernel 或者你已经有自己 benchmark / 快速适配新硬件等场景。

AKO4X：更完整的 campaign-grade 优化框架。它会为每轮优化创建隔离环境，维护每个 operator 的 archive，把历史 variant、lessons、dead-ends 记录下来，让后续轮次基于前面的经验继续优化。它更适合固定 operator 上的多轮、可复现、可审计优化，比如这次 FlashInfer-Bench / MLSys contest 这种场景。

比较有意思的是，AKO4ALL 虽然比 AKO4X 轻很多，但效果仍然非常好。我们在 FlashInfer-Bench 上用 AKO4ALL 测了 4 个 inference operators：GQA decode、MLA decode、MLA prefill、RMSNorm。它在不到一小时、单次 prompt 的设置下，四个都超过了 FlashInfer expert kernel，其中一个的结果还超越了 AKO4X 花费更多时间的优化结果。不过，这里也可能部分受益于 AKO4ALL 我们测试时使用了更新的 Claude Opus 4.8（AKO4X 测试用的是 Opus 4.7）。

详细内容同样见我们的项目主页和代码仓库。

一些思考

今年 3 月 24 号，在 为什么 AI Infra 里还在大量招聘 GPU Kernel Engineer？的回答中，我们开源了初版的 AKO4ALL，并且展示了不错的效果。第二天 NVIDIA 发的 AVO: Agentic Variation Operators for Autonomous Evolutionary Search 也做了和我们几乎一样的事情（使用 Agentic 的方式去做 kernel 优化，而不是之前很多固定 pipeline 的方法），并且做得更加系统，实验也更加完善，不过并没有开源。

实际上从那时开始，大家逐渐发现 Agentic 的方式做 kernel 优化的效果非常好。我们当时也正在参加 MLSys 2026 的比赛，所以虽然当时我们已经有了 AKO4X 的初版，但是并没有开源。

为什么 coding agent 做 kernel 优化的效果这么好呢？我们目前的思考是：

第一，最重要的当然还是模型本身。比如 Opus 在读 profiling、理解 kernel bottleneck、改 Triton/CUDA/TileLang 这些方面已经很强。

第二，kernel 优化这个任务本身非常适合 coding agent 做闭环迭代：反馈快、指标明确、改动范围相对小、边界也相对清楚。我们的 harness 不需要发明很多复杂机制，只要把 benchmark、profiler、日志、候选管理这些东西放到 agent 手边，再增加一些辅助机制比如允许它用更快的 subset workload 先拿到大致信号，就能让它比较快速稳定地逐步优化。同时 kernel 优化本身也需要更丰富的上下文、动态地决定下一步做什么，这也是 coding agent 的优势。

第三，这次比赛的 workload 集合是固定的，而且模型可以直接访问到。虽然 FlashInfer-Bench 里的 workloads 来自真实工业推理场景，不是随便挑几个 shape；但从优化角度看，这仍然给了 agent 做 workload-specific optimization 的空间。而 agent 本来就很擅长根据具体输入分布写特化代码。比如我们 DSA Sparse Attention 的最终提交里，就混合了 Triton 和 TileLang，并且针对 T = 1、T = 2、T > 2 分别写了不同 kernel。当然能写大量的 workload-specific 代码本身也是 LLM 相比于专家手写最大的优势之一吧，因为 coding agent 是不知疲倦的（

那是不是 kernel 优化已经被 agent 解决了？那肯定远远没有。

现在大部分 LLM-driven kernel optimization 仍然是在一个或少数几个 kernel 上单独做优化和评测，而且通常满足上面这些条件：目标明确、benchmark 固定、反馈快、改动边界小。但真实工业场景要复杂得多。

即使是在 FlashInfer-Bench 这种已经很规范的 benchmark 里，reward hacking 也会非常严重。我们在比赛过程中就遇到过至少几类 agent 利用 benchmark 漏洞、CUDA graph 行为或计时/缓存机制得到“假加速”的情况。它们看起来通过了 correctness，也能让 measured latency 变好，但并不是一个真实、可移植、可部署的 kernel 优化。实际上我们也并不能保证我们展示的最终优化结果一定有实际意义，因为总得来说我们只是在 FlashInfer-Bench 官方的规则下去优化（

真实场景下显然要麻烦得多：很多时候即使是非常简单的端到端测试也很难给出干净、低噪声、可归因的信号；性能问题也不一定只在一个 kernel 文件里，可能牵涉更多更复杂的系统。

所以我们并不觉得 “agent 已经解决 kernel 优化” 了。更准确的说法是：在一个边界清楚、反馈明确、workload 可访问的环境里，强 coding agent 已经可以非常有效地做 kernel 优化，甚至在一些真实高性能库的 expert kernel 上继续推进性能。（当然满足了这些条件，不只是 kernel，也许在其他领域也都可以利用 agent 进行类似的优化了）

也许随着模型能力继续增强，未来真的会出现这样的流程：新硬件出来后，agent 能自动读 profiling、理解架构、在真实工业场景下生成并验证一批高质量 kernels，很快把常见 workloads 调到接近专家水平，甚至超过专家。

希望大家以后都不用痛苦地手写 kernel 了。

我们也希望 AKO 能往这个方向推进一点点。

欢迎大家使用、Star ✨、提 Issue，也欢迎一起讨论！
