# 【RG Q&A Summary】[OSDI'24] Motor: Enabling Multi-Versioning for Distributed Transactions on Disagg...

**作者**: USTC-NHPCC中国科学技术大学-国家高性能计算中心-先进数据系统实验室

**原文链接**: https://zhuanlan.zhihu.com/p/15998395738

---

这里是中国科学技术大学 ADSL 实验室的系统论文阅读小组，我们每学期举办关于系统领域最新论文的阅读分享。本篇文章主要是对讨论过程中问答环节的总结。
Reading Group 的主页地址：ADSL Reading Group
bilibili 链接：USTC-NHPCC的个人空间
Motor(为分离式内存上的分布式事务启用多版本控制)

作者：Ming Zhang, Yu Hua, and Zhijun Yang, Wuhan National Laboratory for Optoelectronics, School of Computer, Huazhong University of Science and Technology

内存分离架构是在现代数据中心中备受关注的一种部署方式，他将计算资源和内存资源池化以提升硬件资源的利用率和资源扩展的弹性。在这种架构下，计算节点可以通过 RDMA 或者 CXL 来访问其存储在内存节点上的数据。 为计算节点提供分布式事务的抽象来可以保证不同计算节点访问远端数据的强一致性，比如FORD（FAST 22）。然而 FORD 只为每份数据（对象）维护一个版本，然而这种单版本的并发控制导致（1）写者会阻塞读者；（2）需要 undo log 来保证事务处理的原子性，这消耗了网络带宽，降低了吞吐。

本文提出使用多版本并发控制（MVCC）来提高事务执行的并发度，同时也避免了记录 undo log 带来的开销，然而如何在内存分离架构下高效地实现 MVCC 面临以下两点挑战：

复杂的事务协议和受限的内存节点算力之间的不匹配：内存分离架构下，内存节点没有算力或者算力非常有限（比如仅有几个 ARM 核心），无法处理大量的事务协议处理。

实现高效的多版本数据结构：通常 MVCC 的实现使用链表将同一份数据的不同版本连接起来，通过遍历链表的方式来读取某个特定版本的数据，然而在内存分离架构下遍历链表意味着需要多个网络往返（round trip）来实现，不够高效。

Motor 主要由两部分构成：首先是内存节点上的 Memory Store，包括维护每份数据不同版本元数据的 Consecutive Version Table（CVT）、存储数据内容的 Value Region、和维护数据不同版本之间内容变更的 Delta Area；其次是位于计算节点上的 Transaction Protocol，它负责利用 Motor 提供的 MVCC 支持，使用单边 RDMA 访问内存节点上的 Memory Store 来进行事务处理。

作者使用 TATP 、TPCC 和 SmallBank 三种工作负载测试 Motor 的性能表现。相比与 FORD，Motor 在三个工作负载下分别可以提升 14.4%、98.1% 和 65.4% 的事务吞吐，原因是 FORD 的单版本并发控制限制了事务处理的并发度，以及记录 undo log 消耗了额外的网络带宽。相比于 FaRMv2-DM，Motor 在三个工作负载下分别可以提升 18.9%、44.3% 和 29.5% 的事务吞吐，原因是 FaRMv2-DM 使用链表来存储不同版本的数据，带来了额外的网络往返。

Q&A

Q：这个存储当中不涉及到ssd吗？
A：这篇文章主要说的是in-memory的database，不涉及持久化的部分

Q：memory store里的key的8B是什么意思？如果说的是最大主键长度不符合实际的生产情况
A：这个地方应该是一种模型的简化，后面的实验使用的也是8B的key长度

Q：memory store里如果是存储最新的版本和修改过的属性，这里会不会有double write的问题
A：这里在写入的时候是会有batch writes合成一个batch写入，确实是要同时修改value最新值和修改的属性

Q：memory store里如果是存储最新的版本和修改过的属性，这里会不会有double write的问题
A：这里在写入的时候是会有batch writes合成一个batch写入，确实是要同时修改value最新值和修改的属性

Q：为什么TATP在P99的延迟结果中，FORD会和Motor持平
A：P99这里应该是在high contention的情况下，Motor相较FORD有高并发的特点，FORD在流程上少一个RTT，两者的优势互相抵消了
