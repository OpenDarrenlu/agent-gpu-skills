# 【RG Q&A Summary】[SOSP'24] Ozz & [arXiv] AsyncFS

**作者**: USTC-NHPCC中国科学技术大学-国家高性能计算中心-先进数据系统实验室

**原文链接**: https://zhuanlan.zhihu.com/p/1915442048447808124

---

这篇文章来自中国科学技术大学 ADSL 实验室的系统论文阅读小组，我们每学期举办关于系统领域最新论文的阅读分享。本篇文章主要是对讨论过程中问答环节的总结。
Reading Group 的主页地址：ADSL Reading Group
bilibili 链接：USTC-NHPCC的个人空间

OZZ: Identifying Kernel Out-of-Order Concurrency Bugs with In-Vivo Memory Access Reordering

作者：Dae R. Jeong, Yewon Choi, Byoungyoung Lee, Insik Shin, Youngjin Kwon







内核并发性bug因其极难识别而广泛威胁系统的可靠性和安全性。在内核中，开发者不仅要考虑锁，还需使用内存屏障以防止指令的乱序执行，避免产生难以预料的并发错误，如乱序执行的bug。这种bug是由线程调度和乱序执行的非确定特性共同导致，极难检测。

这篇文章提出了一种名为Ozz的基于Fuzz的内核测试工具，用于检测由于处理器乱序执行引发的bug。作者首先设计了OEMU来通过控制内存访问顺序而非指令的执行顺序以实现动态模拟处理器乱序执行。接着作者提出了Ozz系统，该系统由三部分构成：分析评估内存访问、启发式的计算假设性内存屏障位置、运行与验证。

实验证明，Ozz不仅能重现大部分已知的内核乱序执行bug，还在Linux内核中发现了11个新的真实缺陷，已由开发者确认和修复。这为内核并发错误检测提供了一种高效、系统的解决方案。

Q&A

Q1: load-store的乱序情况为什么没有考虑？

A1: 文章中说，这种乱序虽然理论上确实会发生，但乱序发生的目的是优化性能，但这种乱序对硬件的提升微乎其微，因此实际上很难发生，因此作者把这种乱序排除在外。作者也说会在后续工作考虑这种乱序问题。

Q2: 乱序执行的bug是如何确定的？

A2: 文章是运用了前人的bug探测工作，如KASAN来发现bug的，作者没有自己实现bug探测工具。然后通过宕机或是错误的结果值，来得知乱序执行的发生。

AsyncFS: Metadata Updates Made Asynchronous for Distributed Filesystems with In-Network Coordination

作者： Jingwei Xu, Mingkai Dong, Qiulin Tian, Ziyi Tian, Tong Xin, Haibo Chen







论文提出了AsyncFS，一种支持异步元数据更新的分布式元数据服务。该方法在实现负载均衡的同时有效降低了元数据更新的开销，并提出目录属性的更新可以延迟处理这一关键见解。然而，该方案也带来了一定的代价，例如 statdir 操作的开销增加，以及依赖中心化服务器来跟踪目录的脏/干净状态，可能存在性能瓶颈或单点故障的风险。

Q&A

Q1: WAL 和 Change-Log 的作用分别是什么？两者的作用用什么区别？

A1: WAL 会来存放所有到达该服务器的更新，以此来保证 Crash 情况下数据不会丢失。Change-Log 是用来暂存目录属性的更新，用于 aggregate 来进行最终值的计算。

Q2: 他们也是使用的分布式数据库来进行元数据的存储吗？

A2: 他们论文中只提到了 RocksDB，并且没有提到分布式事务相关的内容。

Q3: 对于写后读的场景，AsyncFS 表现会是什么样的？

A3: 相关的实验中，他们测试 burst write 下的表现，其延迟会随着 bust write 大小的增加逐步平稳（主要是由于 change-log 满了之后就会进行 aggregate）。
