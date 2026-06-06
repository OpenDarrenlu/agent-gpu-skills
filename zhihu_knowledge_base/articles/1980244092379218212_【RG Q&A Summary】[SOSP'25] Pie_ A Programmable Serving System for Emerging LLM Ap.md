# 【RG Q&A Summary】[SOSP'25] Pie: A Programmable Serving System for Emerging LLM Applications

**作者**: USTC-NHPCC中国科学技术大学-国家高性能计算中心-先进数据系统实验室

**原文链接**: https://zhuanlan.zhihu.com/p/1980244092379218212

---

这篇文章来自中国科学技术大学 ADSL 实验室的系统论文阅读小组，我们每学期举办关于系统领域最新论文的阅读分享。本篇文章主要是对讨论过程中问答环节的总结。
Reading Group 的主页地址：ADSL Reading Group
bilibili 链接：USTC-NHPCC的个人空间
Pie：面相新兴大语言模型应用的可编程服务系统

作者：In Gim, Yale University; Zhiyao Ma, Yale University; Seung-seob Lee, Yale University; Lin Zhong, Yale University




内容总结：

现有的 LLM 服务系统（如 vLLM）通常采用固定且整体式的“Prefill-Decode”循环，将每个用户请求都视作单个输入 prompt 进行处理。在实现高吞吐文本生成的同时，这种设计也导致了隐式的 KV Cache 管理、僵化的解码过程以及低效的外部 I/O 集成。难以满足新兴的各类解码策略（如投机推理）、生成策略（如 Graph/Recursion-of-Thought）和 LLM 应用（如 Agent）等对细粒度控制、定制化生成逻辑以及与外部 I/O 无缝集成的需求。

为了突破这些限制，Pie 提出了将传统整体式生成循环分解为细粒度的服务处理程序（handlers），并将生成过程的控制权移交给用户提供的程序（称为 inferlet)。这些 inferlet 运行在轻量级的 WebAssembly 沙箱环境中，允许开发者通过 API 显式地管理 KV 缓存的分配与复用、定义定制化的解码流程，并在生成过程中直接集成外部计算或 API 调用，从而使得应用逻辑能够完全在服务端闭环运行，而无需修改底层服务系统。

在系统实现上，Pie 采用了包含应用层、控制层和推理层的分层架构，以高效地服务多个并发的 inferlet。控制层负责处理资源虚拟化和自适应批处理调度，通过水平和垂直批处理技术将来自不同 inferlet 的细粒度 API 调用合并，从而在保证编程灵活性的同时最大化 GPU 的利用率。

文章中的实验评估显示，在标准文本生成任务上能匹配现有最先进系统的性能，仅带来极小的延迟开销（3-12%）。同时，得益于允许制定应用特定优化的能力，Pie 在 Agentic Workflow 中实现了显著的延迟降低和吞吐量提升（1.3×-3.4×）。

Q&A

Q：Pie 的 Batching 策略能否横跨多个 inferlet？

A：Pie 的 Batching 策略可以横跨多个 inferlet。在应用层，每个 inferlet 通过一个或多个 command queue 定义生成逻辑；在控制层，Batch Scheduler 会尝试横跨多个 command queue 打包同一类 API 调用（Horizontal Batching），并且在每个 command queue 内部，无数据依赖的同一类 API 调用也会被打包在一起（Vertical Batching）。
