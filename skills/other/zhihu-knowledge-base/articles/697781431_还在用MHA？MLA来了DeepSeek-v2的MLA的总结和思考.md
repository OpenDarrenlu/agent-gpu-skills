# 还在用MHA？MLA来了DeepSeek-v2的MLA的总结和思考

**作者**: HongzhuzhuOptimistic Pessimist

**原文链接**: https://zhuanlan.zhihu.com/p/697781431

---

​
目录
收起
Deepseek-v2结构
MLA解决了什么问题
MLA的核心：权重矩阵合并
位置编码的解耦
MLA的计算流程
MLA训练阶段
Deepseek-v2结构

Deepseek的创新主要体现在两个方面，分别对应attention部分的优化Multi-head Latent Attention (MLA) 和 FFN部分的优化DeepSeekMoE。这篇文章先讲MLA，下一篇再讲DeepSeekMoE。

MLA解决了什么问题

MLA的提出主要就是减少推理过程的KV Cache，从而实现在更少的设备上推理更长的Context，或者在相同的Context长度下增大batch size，实现更快的推理速度或者更大的吞吐总量，最终降低推理成本。

首先与经典的MHA和GQA，MQA比较，我们可以看出，MLA在优化kv cache和保证模型效果上有很强的优越性。如下表所示，与GQA相比，MLA相当于GQA中的group数量 
𝑛
𝑔
 =2.25，小于大多数Model里的 group数量，例如Llama2-70b和Llama3 
𝑛
𝑔
是8， 由此可见，其kv cache的size大大减小。与MQA相比，MLA的性能和效果显著优于MQA，甚至强于MHA和GQA，真正实现了即降低推理成本，又保证了模型性能。

MLA的核心：权重矩阵合并

首先复习一下MHA：

假设输入序列为 
ℎ
1
,
ℎ
2
,
.
.
.
,
ℎ
𝑙
 ，其中 
ℎ
𝑡
∈
𝑅
𝑑

𝑞
𝑡
=
𝑊
𝑄
ℎ
𝑡
,
𝑊
𝑄
∈
𝑅
𝑑
ℎ
𝑛
ℎ
×
𝑑

𝑘
𝑡
=
𝑊
𝐾
ℎ
𝑡
,
𝑊
𝐾
∈
𝑅
𝑑
ℎ
𝑛
ℎ
×
𝑑

𝑣
𝑡
=
𝑊
𝑉
ℎ
𝑡
,
𝑊
𝑉
∈
𝑅
𝑑
ℎ
𝑛
ℎ
×
𝑑

𝑜
𝑡
,
𝑖
=
𝐴
𝑡
𝑡
𝑒
𝑛
𝑡
𝑖
𝑜
𝑛
(
𝑞
𝑡
,
𝑘
𝑡
,
𝑣
𝑡
)
=
∑
𝑗
=
1
𝑡
𝑆
𝑜
𝑓
𝑡
𝑚
𝑎
𝑥
𝑗
(
𝑞
𝑡
,
𝑖
𝑇
𝑘
𝑗
,
𝑖
𝑑
ℎ
)
𝑣
𝑗
,
𝑖

其中，
𝑑
ℎ
𝑛
ℎ
=
𝑑
 ，
𝑑
 表示hidden dim， 
𝑛
ℎ
 表示head数量， 
𝑑
ℎ
 表示一个head 的hidden dim。

例如，Llama2-7b，
𝑑
=
4096
，
𝑛
ℎ
=
32
，
𝑑
ℎ
=
128

可以发现，推理阶段，attention中的计算：

(q_{t})^{T}k_{i}=(W^{Q}h_{t})^{T}(W^{K}h_{i})=h_{t}^T(W^{Q})^{T}W^{K}h_{i}

(W^{Q})^{T}W^{K} 可以进行合并作为Q的投影矩阵，同理，v_{t}=W^{V}h_{t}中 W^{V} ，可以与 o_{t} 后面的投影计算， 也就是u_{t}=W^Oo_{t} 中的权重矩阵 W^O 也可以进行合并。

既然MHA也可以进行合并，为什么没有这样做呢？而MLA可以呢？

我们首先主要探讨 (q_{t})^{T}k_{i}=(W^{Q}h_{t})^{T}(W^{K}h_{i})=h_{t}^T(W^{Q})^{T}W^{K}h_{i}，对一个head（ n_{h}=1 )而言，根据每个矩阵分量的shape，对应的矩阵乘就是 [1, d]\times[d,d_{h}]\times[d_{h}, d]\times[d, 1]，我们可以将其拆分为一部分计算一部分存储，那么有以下几种可能：

标准的kv cache，存储角度，需要存 W^{K}h_{i}（这里只写了k，v的大小一致），则kv cache的大小是 2d_{h}*n_{h}*layer=2dl （考虑所有 n_{h} 个head）；计算角度，每个head实例化的参数就是 W^{Q} 和 W^{K}W^{V} 和 W^{O}，大小为 4dd_{h} 。
(W^{Q})^{T}W^{K} 结合到一起，并把结合后的权重apply到x上，存储角度，存储 (W^{Q})^{T}W^{K}h_{i} 作为新的cache，cache大小为 2d*n_{h}*layer=2n_{h}dl ，这样cache扩大了head number 倍；计算角度，每个head示例化的参数量为 (W^{Q})^{T}W^{K} 和 W^{V}W^{O} ，大小为 2d^{2}
(W^{Q})^{T}W^{K} 结合到一起，但是只cache x，不cache k和v的权重，存储角度，需要存储的cache大小是 dl ,相比标准kv cache减少了一半；计算角度，每个头实例化的参数为(W^{Q})^{T}W^{K} 和 W^{V}W^{O} ，大小为 2d^{2}

结合上面的分析，标准的kv cache已经相对而言在空间开销上和计算上是最优的了，尽管我们可以通过只 cache x减少一半的kv cache，但是结合后的矩阵放到运行时计算也增加了计算量，权衡下并不是好的方案。

而 W^{K} 做了低秩变换后， 从[d_{h}, d] 变成 [d_{h}, r]\times[r, d]， h_{t}^T(W^{Q})^{T}W^{K}h_{i} 变成 h_{t}^T(W^{Q})^{T}W^{UK}W^{DKV}h_{i} ：

[1, d]\times[d,d_{h}]\times[d_{h}, d_{c}]\times[d_{c}, d]\times[d, 1]

从存储的角度：此时存储的kv cache就是 W^{DKV}h_{i}, cache大小是 d_{c}l ,加上旋转位置编码的部分，总的kv cache是 (d_{c}+d_{h}^{R})l ,对应是MHA的 (d_{c}+d_{h}^{R})/2d=(512+64)/(2*5120) =5.58%
从计算的角度： W^{UK} 可以merge到 W^{Q} 中，类似地，W^{UV} 可以merge到 W^{O}中。这样实例化的权重就变成了原来的 d/r 分之一
无论是存储还是计算的角度，MLA的拆分方法都优于MHA。所以到这里我们就明白了，MLA的好处来源于两个方面，一个是kv cache的显著降低，另一个是权重的合并和吸收。
位置编码的解耦

接下来解决下一个问题：为什么要做位置编码的解耦呢？

这部分是一开始我最感到疑惑的，原文里是这样解释的：

直到看了苏神的分析缓存与效果的极限拉扯：从MHA、MQA、GQA到MLA，我突然恍然大悟，其实本质上就是权重的吸收合并问题。有了前面的分析，我们就知道了，MLA的核心是需要实现权重的合并吸收的，也就是W^{UK} 可以merge到 W^{Q} 中，W^{UV} 可以merge到 W^{O}中。那么如果我们直接对 k_{i}^{c} 进行位置编码的话，计算 (q_{t})^{T}k_{i}^{c} 可得：

(q_{t})^{T}k_{i}^{c}=(W^{Q}h_{t})^{T}(W^{UK}c_{i}^{KV})=h_{t}^T(W^{Q})^{T}W^{KV}c_{i}^{KV}

复习一下RoPE的操作，

对于query向量 q_{m} 和key向量 k_{n} 之间的内积操作可以被一个函数 g 表示，该函数 g 的输入是词嵌入向量x_{m}

和 x_{n} 之间的相对位置 m-n ,

<f_{q}(x_{m},m),f_{k}(x_{n},n)>=g(x_{m},x_{n},m-n)

对于2维词嵌入向量d=2, 利用2维平面向量的几何性质（此处省略推到过程）， g(x_{m},x_{n},m-n) 可以表示如下：

2维扩展到任意维度，表示为：

为了简化，我们直接用 R_{i} 指代对于位置i的query向量q_{i}加上的位置编码矩阵，所以有了下面的变换：

q_{t}=h_{t}W^{Q}R_{t}

k_{i}=c_{i}W^{KV}R_{i}

q_{t}^{(s)}k_{i}^{(s)T}=q_{t}k_{i}^{c}=(h_{t}W^{Q}R_{t})(c_{i}^{KV}W^{UK}R_{i})^{T}=h_{t}W^{Q}R_{t-i}(W^{KV})^{T}(c_{i}^{KV})^{T}

由于矩阵乘法没有交换率，W_{q}^{(s)}R_{t-i}W_{k}^{(s)T}就无法合并为一个固定的投影矩阵了（与位置差t-i相关），所以需要进行拆分。

其实，MHA如果进行矩阵的合并吸收，也会存在这样的问题，但是我们上面分析过，MHA合并不会带来什么好处，所以合并这个设计就不存在了，自然不会暴露出这个问题。而MLA的优化刚好暴露出这个问题，所以只好进行信息存储矩阵和旋转位置编码矩阵的解耦和拆分。

MLA的计算流程

这部分借鉴了如何看待 DeepSeek 发布的 MoE 大模型 DeepSeek-V2？其中的解读，我重新梳理了一遍，方便理解。

首先对照原文中的公式：

MLA流程图

说明：

虚线框表示权重矩阵，实线框表示输入或者activation
矩阵大小并不是完全按照比例缩放，但是整体上呈现了 d_{c} 和 d_{c}^{'} 之间的比例关系，也就是说， W^{DQ} 和W^{DKV}大小并不相等，这和我们通常在MHA中用到的不一样。
假设head number n_{h} =2

主要流程：

从上到下分为Q、K、V三条path，其中Q和K里面又都细分为latent / low-rank部分和decoupled RoPE path
W^{DKV}会分化成 W^{UK} 和W^{UV}
推理阶段进行cache的数据：用紫色框线表示，包括两部分，也就是压缩后的kv 矩阵和位置编码后的矩阵
会被合并吸收的数据： W^{UK} 吸收进 W^{UQ}，W^{UV} 吸收进 W^{O}
注意：这里K的path位置编码的部分，接受的输入还是原始的h_{t}而不是压缩后的c_{t}

合并吸收后的情况：

推理阶段要cache的东西不变，位置编码计算的逻辑也没有变
W^{UK} 吸收进 W^{UQ} ：Q的计算逻辑没有变，但是权重和激活值的shape都有相应的调整，在图中用红色标识， d_{h} 变成了d_{c}
W^{UV} 吸收进 W^{O}：K少线性映射的计算逻辑，变成了重复拷贝 n_h 份，K也是这样；同样，最后输出权重和激活值的shape有相应的调整，在图中用绿色标识，d_{h} 变成了d_{c}
MLA训练阶段

需要注意的是，为了减少训练期间的激活值，Q的输入也改为了低秩投影的形式。这里不是很理解激活值，按照苏神的说法，是训练期间的参数量和相应的梯度？

后续仔细分析一下再更新。。




References:

deepseek-ai/DeepSeek-V2 (github.com)

Deepseek-V2技术报告解读！全网最细！ (qq.com)

缓存与效果的极限拉扯：从MHA、MQA、GQA到MLA - 科学空间|Scientific Spaces (kexue.fm)

(99+ 封私信 / 81 条消息) 如何看待 DeepSeek 发布的 MoE 大模型 DeepSeek-V2？ - 知乎 (zhihu.com)

DeepSeek-V2：一款强大、经济高效的专家混合语言模型 (qq.com)
