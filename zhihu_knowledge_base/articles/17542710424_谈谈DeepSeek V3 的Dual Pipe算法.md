# 谈谈DeepSeek V3 的Dual Pipe算法

**作者**: Fazzie白天摸鱼，摸黑干活

**原文链接**: https://zhuanlan.zhihu.com/p/17542710424

---

昨天的回答收到很多反馈

DeepSeek V3推理时的Dual Pipe算法到底是怎么做掩盖的？
129 赞同 · 9 评论 回答

训练的双流PP最早可以追溯到 21年SC奇美拉 Chimera: Efficiently Training Large-Scale Neural Networks with Bidirectional Pipelines 这篇文章，由ETHz的Torsten Hoefler和 当时应该还在ETHz做博后的 Shigang Li老师做的。

双流水线进行交叉排布，可以减小的bubble rate，但是增加了一倍weight的显存占用。


添加图片注释，不超过 140 字（可选）


添加图片注释，不超过 140 字（可选）

但是在过去几年为什么基本没人用呢？

个人觉得这么几点

实现复杂，在Megatron，Deepspeed，Colossal AI几个常见并行库中都没有实现，基本还是以最简单的1F1B为主，没有好的开源实现
看文章吐需要在超多节点才能拿到超过PipeDream的收益，而且后期被其他改进的PP超越，2122年基本没有几家真的训这么大的模型
模型的显存要double，彼时attention训练的瓶颈还不在序列长度，不像现在基本pretrain 8k起步，激活显存占比很小，模型显存double在大家没这个多卡的情况下基本不可接受，这么玩提升MFU还不如直接拉大batchsize提throughoutput

为什么现在又有人愿意去尝试了？

我觉得有这么几点

正如Deepseek文章中提到的 （Although DualPipe requires keeping two copies of the model parameters, this does not significantly increase the memory consumption since we use a large EP size during training.）因为大稀疏的MOE配合大EP，模型参数显存double不会显著增加内存消耗。
DeepSeek的Dual Pipe本质上是 Zero Bubble Pipeline Parallelism 和 Chimera 两旁paper的合成，有了Zero Bubble的 BW分离backward加上双流PP，在增加一点显存的小代价下，进一步减少Bubble提升MFU。
这点是我觉得最牛的，有MOE引入了all to all 通信，有双流正好可以在bwd的同时做另一个流fwd的all to all，理论上完美的把all to all overlap了，单流的话就无法实现了。




Dual Pipe未来会成为主流吗？

个人觉得不会，场景太有限的，如果不是大MOE这样的大EP和大集群，拿到的收益可能不大，但工程复杂度会增加很多，根据奥卡姆剃刀原则，没有大的必要，完全可以从其他更简单的地方拿收益

这个算法只会和模型和集群size强绑定，在有大MOE的情况下，MOE这里完全可以拿更多收益。这就要求Team工程团队和算法Team强绑定合作，基本只会集中在几个继续做基座的group且人才密度足够，否则工程随算法维护和迭代更不上不如选择更加简单正交的优化




PP的发展历史可以看看我去年写的总结

流水线并行论文总结 - Fazzie的文章 - 知乎

Fazzie：流水线并行论文总结
