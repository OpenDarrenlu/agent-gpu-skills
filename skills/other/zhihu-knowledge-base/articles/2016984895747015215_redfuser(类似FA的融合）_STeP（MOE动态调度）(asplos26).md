# redfuser(类似FA的融合）/STeP（MOE动态调度）(asplos26)

**作者**: Arsmart​上海交通大学 计算机科学技术博士在读

**原文链接**: https://zhuanlan.zhihu.com/p/2016984895747015215

---

这是两篇论文，不相关，只是我恰好一起看完了。

redfuser介绍了类似FA的计算，都可以按FA的方式来融合。哪些是这一类呢？比如quant+gemm；MOE routing等。这里还提出可以在thread，warp，CTA，GPU级别来融合。挺有意思的。

优点：揭示了这一类计算都可以进行融合，新奇的角度！

缺点：着眼点稍微有点小啊。如果我们看mirage，trinity这种编译器工作，理论上是可以搜到本文的方案的，视野更广阔（虽然这些工具很难用就是了…）

有点好奇这种真的会被落地到工业界吗 发现的这些融合算子

step介绍了MOE可以如何利用动态性：

其中第二点是结合ASIC的特点，其他两点不强绑定ASIC。

优点：三种动态性挺有启发性的

缺点：还搞了一套很复杂很不自然的原语来套壳子（很多都是自然语言表述，不是连续变量。我很反对，在搜索空间表达上应该是完备的，用自然语言约束应该是在执行搜索的时候用heuristic来压缩），虽然我觉得完全没必要，真不如好好分析这三个case。第一点的怎么做动态block_m没讲（选多大的尺寸，不同尺寸间的trade-off）。第二点仅针对ASIC，在GPU上是不会这样的，是计算与内存都在一个SM内。第三点就是任务的全局抢占嘛，在FA3里已经做过了。（笑，这么看每个创新点都很有问题吗）

未来的方向：试试看GPU上做？




RedFuser: An Automatic Operator Fusion Framework for Cascaded Reductions on AI Accelerators

Streaming Tensor Programs: A Streaming Abstraction for Dynamic Parallelism
