# 优化NCCL运行时的启动kernel的部分没有意义

**作者**: yylloc以太网超节点

**原文链接**: https://zhuanlan.zhihu.com/p/711160136

---

GTC&#39;21 S31880

阅读源码不难发现，前3步在ncclCommInitRank就做完了，第4步启动kernel的函数调度图如下…

https://github.com/YconquestY/nccl，基于v2.22.3-1

这仅仅是ncclSend/ncclRecv，会不会很复杂？感不感觉看到了屎山？想不想优化它？觉不觉得简化后能大幅缩短通信时间？

没有意义。跑个模型做做性能分析吧，这么复杂的CPU侧调用，时延会被异步执行完全隐藏，性能瓶颈依旧在GPU侧…

3周白干。
