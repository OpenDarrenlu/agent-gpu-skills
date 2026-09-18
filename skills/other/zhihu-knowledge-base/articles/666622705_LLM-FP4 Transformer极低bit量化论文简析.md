# LLM-FP4 Transformer极低bit量化论文简析

**作者**: NULL让智能更廉价: AI芯片软件栈与异构计算系统

**原文链接**: https://zhuanlan.zhihu.com/p/666622705

---

这几个月没有research大模型量化，直觉告诉我应该会有不错的论文出现，果不其然这篇LLM-FP4值得推荐；

众数周知activation量化收益很高，这篇论文与RPTQ一样都想通过碰瓷perchannel activation来提高activation量化精度，但RPTQ"畏难而退"选择group channel浅尝代替perchannel，而这篇LLM-FP4直接"捅穿"了perchannel activation；

再加上采用了FP4这种比INT4更有"表达力"的格式，收获大模型 4bit后量化 SOTA意料之中，FP4顺带也避免了4bit中的非对称量化中零点补偿问题；

既然是简析，大致过程就略掉了，主要分析正面硬刚perchannel activation后如何解决计算爆炸问题；这篇也属于一"挖坑"制作，相信后序会有更好优化方法迭代；

perchannel activation * perchannel weight 矩阵乘，这是一个"量化参数-全连接"的计算量，既然量化是为了加速计算，而引入这种"量化参数-全连接"计算量得不偿失，所以通常采用的是pertensor * perchannel 这种的量化方式；

这篇论文通过一个"预移位指数偏差"，来提前补偿到weight的perchannel(类似soothquant,不过soothquant是基于INT8)。

为了处理高通道间差异性，引入预移位指数偏差，将每个通道的实值缩放因子b从通道的最大值计算而来，然后将 b分解为张量级的实值缩放因子 ρ 和通道级的整数缩放因子​​bori；




weight量化参数经过bori重新参数变成：

而activation依旧用pertensor来计算；

RPTQ是借用SoothQaunt思路，借助重排聚类之后组队，使用group activation量化；

LLM-FP4借用SoothQaunt思路在FP4数量类型上，实现perchannel activation量化；
