# [ISCA'18] RANA: Towards Efficient Neural Acceleration with Refresh-Optimized Embedded DRAM

**作者**: 涂锋斌​清华大学 微电子与纳电子学系博士

**原文链接**: https://zhuanlan.zhihu.com/p/36733625

---

转眼距离上一次写专栏已经过去一年了。这期间，一直处于很忙碌的状态。现在终于有新成果可以和大家分享了。

这一年我一直在从事第二代DNA（Deep Neural Architecture）架构的研究。如果说我们的第一代DNA架构（Thinker芯片原型架构，参见我的第一篇专栏文章）着力于计算数据流的优化，那么在第二代架构研究中我们更加注重片上存储的改进。我们使用eDRAM（Embedded DRAM）替代SRAM作为片上存储以获得更大的存储容量，减少片外访存，并提出了RANA（Retention-Aware Neural Acceleration）框架来降低eDRAM刷新所带来的额外能耗，最终大大提升了整个系统的能量效率。这份工作在今年3月被计算机体系结构领域的顶级会议ISCA'18接收。会议日期是6月2日至6日，地点是那个我梦想中的城市-洛杉矶。

今年的ISCA设置了一个online lightning talk环节。大家在开会前一个月上传一个2分钟左右的视频到YouTube，简要介绍自己的工作。这样方便大家提前了解今年的文章内容，在会上可以充分交流。不方便上YouTube的同学，可以直接观看下面视频，欢迎一起讨论。

01:54

这份工作的题目是“RANA: Towards Efficient Neural Acceleration with Refresh-Optimized Embedded DRAM”，主要有三个层次的主要贡献：

Training Level: A retention-aware training method is proposed to improve eDRAM's tolerable retention time with no accuracy loss. Bit-level retention errors are injected during training, so the network' s tolerance to retention failures is improved. A higher tolerable failure rate leads to longer tolerable retention time, so more refresh can be removed.
Scheduling Level: A system energy consumption model is built in consideration of computing energy, on-chip buffer access energy, refresh energy and off-chip memory access energy. RANA schedules networks in a hybrid computation pattern based on this model. Each layer is assigned with the computation pattern that costs the lowest energy.
Architecture Level: RANA independently disables refresh to eDRAM banks based on their storing data's lifetime, saving more refresh energy. A programmable eDRAM controller is proposed to enable the above fine-grained refresh controls.

暂时先透露这么多啦，更多的信息会在开会之后公布。如果有时间的话，我希望自己能写一篇ISCA的开会见闻，或者总结总结这爆炸的一年中学术圈发生了些什么。

最后还是那句话，希望下次可以把更好的成果分享给大家。

2018.06.13更新：开会的这些天，结识了不少小伙伴，讨论了很多对于AI芯片现状和未来发展的看法。这种感觉，让我更加憧憬美国的学术氛围。我对自己的这份工作做了一些整理，在奕欣沐沐们的帮助下，刊登在了雷锋网AI科技评论上：

欢迎大家对我的工作提出宝贵意见。
