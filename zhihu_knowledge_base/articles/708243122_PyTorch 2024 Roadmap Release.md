# PyTorch 2024 Roadmap Release

**作者**: maja性别男，爱好女

**原文链接**: https://zhuanlan.zhihu.com/p/708243122

---




Pytorch 2024 Roadmap 正式发布，我们可以看到最关键的两个词：编译器，分布式。其中PT2-D，接过 TPU team，Modoluar team，以及Dataflow 硬件厂商的旗帜 开始发力：

> 编译器 和 分布式的结合。

而这块算法上, 已然没有什么太大瓶颈，现在用NV 手动版 Megatron，都需要自己去改 PP 避免<头重脚轻>，跑一次训练几十个配置（虽然大部分已经固定），且版本间会时常不一致：因此采用DTensor和编译器技术，有望让新团队快速绕过逻辑分支，直达问题核心：

> 指定/自动 目标代码到特定设备，无需大规模调整单机形状

关键合作伙伴回应




Roadmap

https://dev-discuss.pytorch.org/t/meta-pytorch-team-2024-h2-roadmaps/2226

本次发布中，最重要的 Compiler feature 会正式确定（是否移除nvfusion, 增加NCC backend支持；如何完善 DTensor, 在 5D parallel的机制，以及支持 IO/Compute Overlap （只需要一个pass即可，无需手动写各种专属的overlap代码处理分支））。

core compiler Vision-OKR：

构建统一PT2优化栈，合并训练，推理的优化路径，并提供了6个可量化目标支撑.

在目标6中Meta意识到 Peak Memory Optimization (也就是我们编译优化常说的activation liveness) 是LLM优先项（priority）.




目前在Megatron- LM 中经过dynamo优化的模型要比不优化的快20%左右，是一个性能大杀器。




Pytorch- Distributed Vision-OKR：

从早期的 DDP开始 ，在Google提出 Device Mesh概念，以及NV提出的<手动>优化架构Megatron后，一直处于技术跟随状态：比如模仿借鉴SBP技术方案的Dtensor。




从sharded-dataparallel (MS zero-1 算法）, PP,TP 到全面支持的5D parallel （async 非独立纬度SP + 独立维度 CP, 不含 MoE parallel）并“尝试”在E2E训练中解决EP问题，作为底座全面支持针对大模型的解决方案torchtitan.




本项目共提出 2 个可量化的 OKR ，平均每个O 包含 6个KR. 在13个关键可量化结果中，llama内存优化；支持titan; add zero bubble scheduler （ by sea lab, 论文还在预审阶段就被广泛复现，应用起来了；其本质就是 将 activation 和 weights 更新分离，使得NCCL 和 计算互相掩盖，对应到编译器技术就是拓扑排序问题，一个pass即可解决) 被高亮。




产业应用：

2024年3月字节跳动发布12228张卡训练Megascale 我就开始跟踪，当时提出想和字节的研发team沟通下（我提了好多问题，还欠 @姜宁 老师 一篇文稿，一直没写完）, 从4月份提交的代码上看，字节在尝试使用DTensor技术，估计会继续很近。

总结：

本次发布会着眼于 compiler 和 分布式 达成Native Support LLM training 和 inference. 这将对未来技术格局产生不可忽视的影响。
