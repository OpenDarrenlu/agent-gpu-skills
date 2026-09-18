# Neural Networks on Silicon

**作者**: 涂锋斌​清华大学 微电子与纳电子学系博士

**原文链接**: https://zhuanlan.zhihu.com/p/25890632

---

我的名字叫涂锋斌，是清华大学微纳电子系魏少军老师和尹首一老师的博士生。我的博士课题是高能效神经网络加速芯片设计。这个领域现在非常火热，为了紧跟科技的前沿，我在GitHub上做了一个reading list，里面囊括了自2014年以来神经网络硬件架构这个领域在顶会上发表的文章，以及一些我自己很感兴趣的研究热点。这个project的名字叫做Neural Networks on Silicon ，很酷吧？ 希望借此平台可以和国内外同行们一起交流最新的研究成果。

我们组在去年设计了一款可重构多模态神经网络芯片Thinker，已经成功流片（封面图）。我有幸代表全组成员在今年的ISSCC上做了展示。我自己的一部分工作也刚刚被TVLSI接收，题目是Deep Convolutional Neural Network Architecture with Reconfigurable Computation Patterns。这份工作有三个主要贡献：

This is the first work to assign Input/Output/Weight Reuse to different layers of a CNN, which optimizes system-level energy consumption based on different CONV parameters.
A 4-level CONV engine is designed to to support different tiling parameters for higher resource utilization and performance.
A layer-based scheduling framework is proposed to optimize both system-level energy efficiency and performance.

我现在正在进行第二代架构的开发，希望能尽快把最新的研究成果分享给大家。

感谢师弟师妹和同学们的分享，让更多的人能读到我的第一篇专栏文章。我最开始写这篇文章的目的是想贯彻自己的一个科研理念：分享。我觉得科研的一个理想状态应该是，“我有一个了不起的想法，好想好想分享给你。”

我从2016年1月开始维护那个GitHub项目，感觉还是蛮有收获的，现在自己基本上把握了这个领域的发展脉络。不过现在真的太忙了，很多文章来不及仔细阅读，希望有更多的人加入进来。


我觉得自己还没有达到自己所理想的高度，所以还得努力，希望有一天可以把更好的成果分享给大家。
