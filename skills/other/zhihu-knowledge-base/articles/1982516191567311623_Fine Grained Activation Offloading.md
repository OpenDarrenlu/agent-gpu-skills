# Fine Grained Activation Offloading

**作者**: 朱然拉航母的小朱

**原文链接**: https://zhuanlan.zhihu.com/p/1982516191567311623

---

转载自我们的公众号 大模型训练的高效内存解决方案：流水线感知的细粒度激活卸载，实现显存开销与吞吐性能的联合最优
欢迎加入 小红书大模型AI Infra团队介绍&招聘信息

虽迟但到，CPU offloading不算是什么创新的技术，想法源于1年前再次启动做MoE准备训练dots1 llm的时候，显存成为了制约训练系统效率的最大因素，那时候我们还没有DeepEp、AlltoAll hiding这些技术手段，expert parallel的分布式扩展性并不好，为了能在更小model parallel下跑起来，我们不得不选择了full recompute。full recompute 叠加无法被overlap的forward dispatch/combine 通信，就更加拉垮了。

如何能节省activation显存从而不那么依赖recompute呢，自然想到了“以存换算”，把前向计算的activation offload到CPU memory上，反向计算需要时再load回显存。

仅仅synchronous offload/load 会有比较大的PCIe传输开销，由于activation 都是按照layer去计算和使用的，所以我们可以利用相邻layer的计算，来跟activation的load/offload做重叠。

所有activation 都做offload/load，叠加上述overlap的效果可能依然是不够理想的，因为PCIe的带宽有限，如果load/offload的整体开销大于计算开销，依然会出现GPU bubble。因此我们可以进一步结合不同算子的 计算/activation显存比，来混合使用offloading和recompute来达到最优。

理论是比较简单的，之前行业里也有不同的tech report中声称在预训练中使用了相关技术，但是我们实际实现的过程中还是遇到了不少细节问题，要达到“完美overlap”并不容易。最大的挑战是 implicit synchronization 。在训练代码的某些操作会引发device level的sync，从而破坏load/compute/offload的流水，导致性能收益消失，需要精细排查和review每个无法被overlap的H2D/D2H 拷贝，才能拿到最优的效果。
