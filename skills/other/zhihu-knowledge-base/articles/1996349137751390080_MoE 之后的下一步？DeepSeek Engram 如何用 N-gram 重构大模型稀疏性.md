# MoE 之后的下一步？DeepSeek Engram 如何用 N-gram 重构大模型稀疏性

**作者**: 李安渝infra小菜鸡

**原文链接**: https://zhuanlan.zhihu.com/p/1996349137751390080

---

​
目录
收起
痛点与直觉
方案：Engram = N-gram 记忆表 + 上下文门控融合
Tokenizer Compression（分词器压缩）
Multi-Head Hashing（多头哈希）
Context-Aware Gating（上下文感知门控）
实验意义
痛点与直觉

论文把语言建模拆成两类“性质完全不同”的活：组合式推理（compositional reasoning）和知识检索（knowledge retrieval）。后者里大量内容是局部、固定、高频、模板化的——比如实体名、多 token 专有名词、公式化表达等。经典 N-gram 模型之所以在捕捉局部依赖上很强，是因为它本质上就是O(1) 查表。问题在于：标准 Transformer（包括 MoE）没有“原生的知识 lookup primitive（查表原语）”。于是今天的 LLM 往往被迫用多层 attention + FFN 去“模拟检索”：为了拼出一个常见实体，早期层得反复做注意力聚合和前馈变换，这近似于“运行时重建一张静态查表”，把宝贵的顺序深度浪费在本该很便宜的操作上。而 MoE（以 DeepSeek-V3 这类模型为代表）解决的是另一件事：conditional computation（条件计算）——模型容量变大，但每 token 激活的 FLOPs 不按比例增长。它仍然是在“算”。

所以论文提出一个互补的新稀疏轴：conditional memory（条件记忆）——把“静态模式/知识”的那部分，改成稀疏检索拿回来，而不是继续用深层计算去“演”。

方案：Engram = N-gram 记忆表 + 上下文门控融合

Engram 的直观定位：给模型装一套“可扩展的静态记忆查表模块”，让它更像是在分工：

MoE：负责动态、上下文依赖的推理计算（conditional computation）
Engram：负责静态、高频、局部模式的检索回忆（conditional memory）

Engram 的关键是：每个 token 只检索常数个槽位，因此扩表会增加总参数，但不会增加每 token FLOPs。这给了模型一个“存算分离”的扩容旋钮。

Engram架构图
Tokenizer Compression（分词器压缩）

这个组件是为了让查表过程更加精确而设计，具体问题是在 BPE 词表中，"Apple"（句首大写）、" apple"（句中小写带前导空格）、"APPLE"（全大写）是完全不同的三个 Token ID，如果我们直接统计 N-gram，那么 ["eat", " apple"] 和 ["eat", " Apple"] 会被当作两个完全不相关的模式。这导致统计数据极其稀疏（Sparsity），同一个语义被迫占用了多个存储槽位，浪费了宝贵的参数空间。 所以DeepSeek-Engram在这一步进行建立映射层，在查表之前引入一个映射函数，来将所有语义上等价的 Token ID（例如忽略大小写、忽略前导空格、Unicode NFKC 标准化），全部映射到一个规范化 ID 。

Multi-Head Hashing（多头哈希）

在N-gram中，记忆表格的数据量是呈现指数级增长 。对于 128k 词表，3-gram 的理论组合数是 128,000^3，这根本存不下。为了存下这些数据，必须用哈希表映射到有限的内存里。如果用单个哈希函数，"New York" 和 "Good Morning" 可能会不幸撞车（Hash Collision），映射到同一个地址。如果只存一个向量，模型就会学傻了：既要像 New York 又要像 Good Morning，最后啥也不是。所以DeepSeek-Engram在这一步借鉴Bloom Filter（布隆过滤器）和Product Quantization（乘积量化）提出了多头哈希。

使用 K 个互不相同的哈希函数。
对于同一个 N-gram（如 “New York”），分别计算 K 个索引，去表中取出 K 个不同的向量。
拼接（Concatenation）：e_t = [e_{t,1} || e_{t,2} || ... || e_{t,K}]。 不存 Key，只存 Value：为了极致压缩，Engram 表里不存对应的文本（Key），只存 Embedding（Value）。这意味着哈希冲突是不可避免的。但是只要 K 个头里有一个是对的，模型就能通过训练学会依赖那个对的头，忽略错的头。这使得 Engram 可以用有限的参数（如 20亿）去隐式地存储数万亿级别的 N-gram 组合。
Context-Aware Gating（上下文感知门控）

这一步是为了让查表的动作更加准确。原来的N-gram 查出来的向量是静态数据，没有办法处理一次多义，比如"Bank" 在 "River bank"（河岸）和 "Bank account"（银行）中完全不同。但 N-gram 查出来的向量可能是一样的。而且由于多头注意力机制的“不存 Key”策略，查出来的 K 个头里，可能混入了几个完全无关的哈希冲突项，如果简单粗暴地把 e_t 加到 Transformer 的残差流里，这些错误的、静态的信息会干扰模型的推理。所以DeepSeek-Engram 引入一个这上下文感知门控来进行融合控制：

动态裁判机制：
Query（裁判）：Transformer 当前层的隐状态 h_t。这是模型经过深思熟虑后的“当前理解”，它包含了全局上下文（Global Context）。
Key/Value（嫌疑人）：检索回来的 N-gram 向量 e_t。
计算门控值：\alpha_t = \sigma\left(\frac{\text{RMSNorm}(h_t)^T \cdot \text{RMSNorm}(W_K e_t)}{\sqrt{d}}\right),本质上是计算 h_t 和 e_t 的语义相似度。
如果哈希冲突导致查出来的是个无关词，它和 h_t 的相似度极低，\alpha_t \approx 0，噪声直接被过滤。
如果是一词多义，模型会根据 h_t（比如当前在谈论金融），给代表“银行”维度的特征更高的权重,使得静态的死记忆，在动态的上下文中“活”了过来。
实验意义
Engram U型实验图

论文发现了一个 U 型曲线：把所有的参数都给 MoE 不好，都给记忆也不好。黄金比例是大约 20%-25% 的稀疏参数分给 Engram（做记忆），剩下的给 MoE（做计算），效果最好。 其他的实验也表示，Engram-27B 在知识类任务（MMLU）、代码和数学上都超过了同等规模的 MoE 模型。长文本检索任务（NIAH）从 84.2 分暴涨到 97.0 分。 表明，Engram架构让浅层网络就能获得深层网络的知识储备，变相增加了模型的“有效深度”，实现了有效的存算分离。
