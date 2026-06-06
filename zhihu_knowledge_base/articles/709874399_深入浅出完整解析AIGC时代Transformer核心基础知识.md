# 深入浅出完整解析AIGC时代Transformer核心基础知识

**作者**: Rocky Ding​北京科技大学 工学硕士

**原文链接**: https://zhuanlan.zhihu.com/p/709874399

---

​
目录
收起
1. Transformer系列资源
2. Transformer整体架构初识
2.1 原生Transformer工作流程初识
2.2 Transformer在AIGC时代的各领域应用初识
2.3 Transformer在AIGC时代的AI绘画领域应用
2.4 Transformer在AIGC时代的AI视频领域应用
2.5 Transformer在AIGC时代的大模型领域应用
2.6 Transformer在AIGC时代的AI多模态领域应用
2.7 Transformer在传统深度学习领域应用
3. Transformer架构的输入端思想详解
3.1 原生Transformer输入端讲解
3.2 在AIGC时代，Transformer的多模态大一统能力
4. 零基础理解Transformer的Self-Attention（自注意力）机制
4.1 零基础深入浅出理解Self-Attention机制
4.2 零基础深入浅出理解Self-Attention机制Q, K, V的计算过程
4.3 零基础深入浅出理解Self-Attention的输出过程
4.4 零基础深入浅出理解Multi-Head Attention机制
5. 零基础理解Transformer的Encoder结构
5.1 零基础理解Add & Norm层的核心原理
5.2 零基础理解Feed Forward层的核心原理
5.3 Tranformer Encoder架构的工作流程
6. 零基础理解Transformer的Decoder结构
6.1 零基础理解Masked Multi-Head Attention核心基础知识
6.2 零基础理解Decoder中Encoder-Decoder Attention核心基础知识
6.3 零基础理解Softmax预测输出单词完整过程
6.4 Transformer的原理总结
7. Transformer系列模型的性能优化
8. 推荐阅读
8.1 深入浅出完整解析Stable Diffusion 3（SD 3）和FLUX.1系列核心基础知识
8.2 深入浅出完整解析Stable Diffusion XL（SDXL）核心基础知识
8.3 深入浅出完整解析Stable Diffusion（SD）核心基础知识
8.4 深入浅出完整解析Stable Diffusion中U-Net的前世今生与核心知识
8.5 深入浅出完整解析LoRA（Low-Rank Adaptation）模型核心基础知识
8.6 深入浅出完整解析ControlNet核心基础知识
8.7 深入浅出完整解析主流AI绘画框架核心基础知识
8.8 手把手教你成为AIGC算法工程师，斩获AIGC算法offer！
8.9 AIGC产业的深度思考与分析
8.10 算法工程师的独孤九剑秘籍
8.11 深入浅出完整解析AIGC时代中GAN系列模型的前世今生与核心知识
本文的专栏：Rocky Ding的AI算法兵器谱
我的公众号：WeThinkIn
更多AI行业干货内容欢迎关注我的知乎，公众号，专栏～

码字不易，希望大家能多多点赞，给我更多坚持写下去的动力，谢谢大家！

大家好，我是Rocky。

Transformer是由Google在2017年发布的论文《Attention is All You Need》中提出，全文加上Reference一共短短的15页内容，却为AI行业带来了深刻的变革。不管是在传统深度学习时代（CV、NLP等），还是在当前的AIGC时代（AI绘画、AI视频、大模型、AI多模态等），都有Transformer架构的身影。Transformer架构正在一步一步重构所有的AI技术方向，成为AI技术架构与多模态整合的关键核心，大有一统“AI江湖”之势。

在进入AIGC时代后，Transformer架构早已不是2017年的基础模样，有了更多的内涵与扩展。

因此，在本文中Rocky深入浅出讲解Transformer的核心基础知识，包括Transofrmer在传统深度学习时代中的核心价值与在AICG时代中的核心价值，让我们来看看Transformer是如何在两个时代中同时从容，并大放异彩的。

同时，Rocky也希望我们能借助本文更好的入门AIGC时代。

话不多说，在Rocky毫无保留的分享下，让我们开始学习吧！

1. Transformer系列资源
Transformer核心论文：Attention Is All You Need
Transformer官方项目库：https://github.com/huggingface/transformers
Transformer视频讲解：李宏毅老师讲解Transformer
The Illustrated Transformer讲解：The Illustrated Transformer
2. Transformer整体架构初识

原生Transformer模型整体上是一个End-to-End的架构，主要由Encoder和Decode两个大的核心模块构成，其中Encoder和Decode中都包含了6个Block结构，具体结构图如下所示：


原生Transformer的整体结构，左图Encoder部分和右图Decoder部分
2.1 原生Transformer工作流程初识

接下来，大家跟着Rocky的脚步，一起来感受一下原生Transformer的整体工作流程：

首先，第一步：我们需要先将输入文本用Tokenizer（分词器）转换成一系列的Tokens，接着再提取这些Tokens的Text Embeddings特征向量，并与输入文本Tokens对应的位置Embeddings相加（add），并将所有的Text Embeddings特征向量进行拼接（concat），最终获得了输入文本的完整特征矩阵 X ，作为Transformer的输入。

接着，第二步：我们将上面得到的输入文本的完整特征矩阵 X (每一行是一个单词的Text Embeddings x ) 传入 Transformer的Encoder中，经过6个Encoder block后得到了输入文本的高危编码矩阵 C ，如下图所示。

输入文本特征矩阵我们可以用 X_{n\times d} 来表示，其中 n 是输入文本中的单词个数， d 是表示矩阵的维度 （Transformer的论文中设置d=512）。同时每一个Encoder block输出矩阵的维度与输入矩阵的维度完全相同。

最后，第三步：我们将Encoder输出的高维编码矩阵 C 传入Decoder中，Decoder会根据当前已经翻译过的1-i 个单词去接着翻译下一个（ i+1 ）单词，如下图所示。在Decoder的推理过程中，翻译到第 i+1 单词的时候需要通过 Mask (掩膜) 操作遮盖住 i+1之后的单词。

上图中Transformer的Decoder接收了Encoder输出的高维编码矩阵 C ，然后首先输入一个翻译开始符 "<Begin>"，预测第一个单词 "I"；接着输入翻译开始符 "<Begin>" 和单词 "I"，预测单词 "have"，以此类推，直至翻译完所有的文本。

上面就是原生Transformer在最初的应用领域——文本翻译上的主要流程。

接下来，Rocky再向大家介绍一下Transformer是如何在AI行业的各个领域都同时繁荣的。

2.2 Transformer在AIGC时代的各领域应用初识

AI行业在2022年进入AIGC时代后，以Stable Diffusion、Sora、GPT等大模型为核心的AIGC技术浪潮爆发。

无一例外的是，不管是AI绘画领域的核心大模型Stable Diffusion、AI视频领域的开创性大模型Sora还是文本对话领域的现象级大模型GPT-4，其底座都是以Transformer架构为基础进行构建的。

在传统深度学习时代，在图像分类领域出现了ViT，在图像分割领域出现了SAM、在目标检测领域出现了GroundingDINO，在自动驾驶领域出现了BEVFormer，无不是以Transformer为核心思想进行设计优化的领域新核心模型。

到目前为止，Transformer已经渗透到AI领域的各个方向中，逐步对各个AI方向进行模型架构的“大一统”，这也是最让我们兴奋的关键一点，在AIGC时代之后，统一的模型架构+海量的数据+坚实的算力，很可能迎来元宇宙初级阶段与AGI初级阶段这一新的红利期。

所以，接下来，Rocky将深入浅出完整分析Transormer在AIGC时代所有的可能性！

2.3 Transformer在AIGC时代的AI绘画领域应用

Transformer架构具有良好的通用性，在AI绘画领域基于Transformer出现了ViT（Vision Transformer）、DiT（Diffusion Transformer）等模型。

2.4 Transformer在AIGC时代的AI视频领域应用




2.5 Transformer在AIGC时代的大模型领域应用

AIGC时代的大模型领域可以说是传统深度学习时代的NLP领域的继承与发展，当仁不让的全面使用Transformer架构，比如DeepSeek、GPT-4o、Llama等大模型。

2.6 Transformer在AIGC时代的AI多模态领域应用

Transformer具备的跨模态对齐能力已经打通视觉、语言、语音等信息壁垒，因此在AI多模态领域，已经成为AIGC模型的基石架构。

2.7 Transformer在传统深度学习领域应用

码字不易，希望大家能多多点赞，支持Rocky的创作呀！！！

3. Transformer架构的输入端思想详解

Rocky认为Transformer架构的输入端思想对于整个AI行业来说都是非常有价值的，在传统深度学习时代体现的并不明显，但是在进入AIGC时代后，才真正意义上的大放光彩和让Transformer拥有了跨越周期的力量。

Transformer的输入端思想——将所有不同模态的数据Token化，进行特征对齐，从而为AI模型的性能爆发打下坚实的基础。

在传统深度学习时代，Transfomer主要是对文本数据进行Token化，到了AIGC时代后，图像、视频、文本、音频、3D等多种模态的数据都可以进行Token化，从而形成AIGC多模态模型。

3.1 原生Transformer输入端讲解

整体上看，Transformer中的输入特征表示x由单词Embedding（Token Embedding）和位置Embedding（Positional Encoding）相加（add操作）得到。Token Embedding提供了语义信息，Positional Encoding提供了位置信息，两者的结合使Transformer能够理解输入的内容和顺序：

Transformer输入端

接下来Rocky和大家一起详细讲解一下单词Embedding和位置Embedding（Positional Encoding）的基础知识。

【单词Embedding详解】

那么什么是单词Embedding（Token Embedding）呢？

我们知道文本数据通常是由离散的单词或子词（如词片段）组成。为了让AI模型能够处理这些离散的数据，首先我们需要将它们转换为连续的数值表示，也就是单词Embedding。

获取单词Embedding方式有很多种，每个输入词（或子词）通过一个嵌入层（采用 Word2Vec、Glove 等算法预训练得到，也可以在 Transformer 中训练得到）映射到一个固定维度的向量空间。最终我们将输入的句子转换成Embedding矩阵，其每一行对应一个词或子词，并且该行中的数值表示这个词的特征。

如果我们使用维度为d_{model}的嵌入向量，并且输入序列长度为 n，那么嵌入后的结果是一个大小为n \times d_{model} 的矩阵。下面我们举一个详细的了例子方便大家理解：

假设我们有一个句子 "We love WeThinkIn"，并且我们使用的词汇表包含这些词。那么这时Transformer如何处理这个输入呢？

假设我们定义词嵌入维度为 d_{model} = 512，那么每个单词都会被映射为一个512维的向量：

`We` -> 一个512维向量 (例如 `[0.1, 0.3, ..., -0.2]`)
`love` -> 一个512维向量 (例如 `[0.2, -0.1, ..., 0.4]`)
`WeThinkIn` -> 一个512维向量 (例如 `[0.7, 0.5, ..., -0.3]`)

接着这些向量被组合成一个矩阵 \text{Embedding}，其形状为 3 \times 512，这就是Transformer的完整的词Embeddings输入。

【位置Embedding详解】

在Transformer中除了单词Embedding，我们还需要使用位置Embedding来表示单词在句子中的对应位置。因为 Transformer 不采用 RNN 的结构，而是使用全局信息，因此不能利用单词的顺序信息，需要显式地为每个输入位置添加位置信息，这样模型才能识别输入的顺序。所以位置Embedding作为额外信息保存了单词在整个句子中的相对或绝对位置注入到Transformer中。

我们可以将位置Embedding（Positional Encoding）用 PE 表示，PE 的特征维度与单词Embedding是一样的。PE 可以通过训练得到，也可以通过预先定义的规则和公式计算得到。在Transformer中采用了正弦和余弦函数来计算位置Embedding，具体的公式如下：

PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right) \\PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right) \\其中，pos 表示单词在整句话中的位置，i 表示维度索引， d 表示PE的维度 (与词Embedding一样)，2i表示偶数的维度，2i+1 表示奇数维度（ 2i≤d, 2i+1≤d ）。那么大家可能会问，为什么使用上述的公式计算 PE 呢？

Rocky帮大家总结了一下，这样计算一共有以下的优势：

1.增强Transformer的扩展性与泛化性：Transformer的PE能够更好的兼容在训练集里未见过的句子，尤其是长序列句子。假设训练集里面最长的句子是有 20 个单词，在推理时输入了一个长度为 21 的句子，则可以使用上述公式计算出第 21 位的Positional Embedding。

2.可以让Transformer容易地计算出输入内容的相对位置：正弦和余弦函数具有周期性，能够捕捉序列中的相对位置信息。这种周期性能够帮助模型理解序列中的重复模式或周期性结构。对于固定长度的间距 k ，PE(pos+k) 可以用PE(pos) 计算得到，因为正弦函数和余弦函数有以下的计算公式： Sin(A+B) = Sin(A)Cos(B) + Cos(A)Sin(B) \\Cos(A+B) = Cos(A)Cos(B) - Sin(A)Sin(B) \\

我们再对上面的“We love WeThinkIn”例子生成位置编码，位置Embedding的维度同样为 d_{model} = 512，每个单词都会被映射为一个512维的向量：

位置0 (`We`) -> 一个512维位置编码向量
位置1 (`love`) -> 一个512维位置编码向量
位置2 (`WeThinkIn`) -> 一个512维位置编码向量

这些位置编码向量也被组合成一个矩阵 \text{Positional Encoding} ，其形状为 3 \times 512。

接下来，Rocky将通过一个具体的例子详细讲解Transformer进行位置编码的全过程，让大家能够更加深入浅出的理解其中的内涵。

还是拿之前的“We love WeThinkIn”为例，其位置为 pos = 0, 1, 2 ，并且假设词嵌入的维度 d_{model} = 8，接下来我们计算每个位置的固定位置编码。

对于位置 pos = 0 ，计算第一个维度的正弦和余弦值：

PE_{(0, 0)} = \sin\left(\frac{0}{10000^{\frac{0}{8}}}\right) = \sin(0) = 0 \\

PE_{(0, 1)} = \cos\left(\frac{0}{10000^{\frac{0}{8}}}\right) = \cos(0) = 1 \\ 我们再计算第二个维度的值：

PE_{(0, 2)} = \sin\left(\frac{0}{10000^{\frac{2}{8}}}\right) = \sin(0) = 0 \\PE_{(0, 3)} = \cos\left(\frac{0}{10000^{\frac{2}{8}}}\right) = \cos(0) = 1 \\ 依次类推，我们最终可以得到： PE_{(0, :)} = [0, 1, 0, 1, 0, 1, 0, 1] \\

我们再来计算位置 pos = 1，按照上面的计算流程，我们可以得到： PE_{(1, :)} = [0.8415, 0.5403, 0.01, 0.99995, 0.001, 0.9999995, 0.0001, 0.999999995] \\对于位置 pos = 2，我们同样可以计算得到： PE_{(2, :)} = [0.9093, -0.4161, 0.02, 0.9998, 0.002, 0.999998, 0.0002, 0.99999998] \\

总的来说，Rocky认为固定位置编码在Transformer模型中发挥着关键作用，它通过使用正弦和余弦函数为每个位置编码提供了一个不依赖于训练的、周期性的表示。这种方法不仅简单有效，而且具有很强的泛化能力，能够捕捉到序列中多尺度的位置信息。这使得Transformer能够在处理长序列时保持稳定的性能，并且不会像RNN那样受到序列长度的限制。通过固定位置编码，Transformer能够理解和处理序列数据中的顺序关系，这也是其在AI的各个领域取得巨大成功的重要因素之一。

有了词Embedding和位置Embedding，我们将两者进行相加（add）操作，就可以得到了最终的Transformer输入表示向量 x： \text{Input Representation} = \text{Embedding} + \text{Positional Encoding} \\

这个结果仍然是一个大小为3 \times 512 的特征矩阵。

总的来说，我们理解Transformer的输入端是理解整个模型工作原理的基础。输入嵌入和位置编码的正确设计对于Transformer能否有效学习和处理序列数据至关重要。

3.2 在AIGC时代，Transformer的多模态大一统能力

在AIGC时代到来后，各个AI技术方向都可以被Transformers大一统，也就是输入的Tokens对齐。

数据（广义的所有模态数据）的Tokens化

将各种模态的输入与输出整合到一起，是AI行业未来发展的必然方向。

4. 零基础理解Transformer的Self-Attention（自注意力）机制

可以说，Transformer的核心跨周期价值主要体现在长距离依赖关系处理和并行计算两个维度，而这两点都离不开自注意力机制。

首先，Transformer引入的自注意力机制能够有效捕捉序列信息中长距离依赖关系，相比于RNN架构，它在处理长序列时的表现更好。

同时，自注意力机制的另一个特点是允许模型进行并行计算，而无需像RNN一样t步骤的计算必须依赖t-1步骤的结果，因此Transformer结构让AI模型的计算效率更高，加速训练和推理速度。

Transformer Encoder 和 Decoder

上图是初始论文中Transformer的结构示意图，左侧是Encoder Blocks结构，右侧是Decoder Blocks结构。红色框部分是Multi-Head Attention机制，它是由多个Self-Attention机制组成的。我们可以看到Encoder Blocks中包含了一个Multi-Head Attention机制，同时Decoder Blocks中包含了两个Multi-Head Attention (其中一个包含了Mask)机制。

Multi-Head Attention机制后面一般接一个Add & Norm层，其中Add表示使用残差连接 (Residual Connection)进行加和来防止网络的退化，Norm则表示Layer Normalization层，用于对每一层的网络激活值进行归一化。

Self-Attention机制和Multi-Head Attention机制可以说是Transformer的核心关键，让Transformer能够跨越周期从传统深度学习时代一直繁荣到AIGC时代，并且势头不减，大有对不同细分领域模型架构进行大一统之势。

接下来，Rocky将带着大家深入浅出全面解析Self-Attention机制 - >Multi-Head Attention机制。

4.1 零基础深入浅出理解Self-Attention机制

我们人类在感知信息时（比如看一张图像、一个视频、一个句子或者一个音频），大脑能够让我们分清哪部分是重要的，哪部分是次要的，从而让我们聚焦在更重要的内容中获得对应的信息。

注意力机制就是希望AI模型也能具备这样的拟人能力。例如AI模型在对图像进行分类、生成视频的下一帧内容、预测句子中的下一个单词以及预测音频的下一句话时，使用一个注意力向量来估计这个Token与其他Token的相关性。

总的来说，注意力机制的核心思想是动态分配权重，根据输入的不同部分对当前任务的重要程度来进行加权处理。通过这种方式，AI模型能够聚焦于关键信息，提升处理效率和准确性。

上图是Self-Attention机制的完整流程图，也是AIGC时代与传统深度学习时代AI模型的核心关键结构之一。在计算的时候需要用到矩阵Q（查询）、K（键值）以及V（值）。在实际应用中，Self-Attention接收的是输入(单词的表示向量x组成的矩阵X) 或者上一个Encoder block的输出。而Q、K、V正是通过Self-Attention的输入进行线性变换得到的。

4.2 零基础深入浅出理解Self-Attention机制Q, K, V的计算过程

Self-Attention的输入用矩阵X进行表示，我们可以使用线性变换矩阵WQ，WK，WV计算得到矩阵Q，K，V。计算过程如下图所示，其中X，Q，K，V的每一行都表示一个单词。

对于输入序列 X \in \mathbb{R}^{n \times d_{\text{model}}} ，通过线性变换得到：

Q = XW^Q \\ K = XW^K \\ V = XW^V \\

其中：W^Q, W^K, W^V \in \mathbb{R}^{d_{\text{model}} \times d_k} 为可学习的权重矩阵。n 为序列长度，d_{\text{model}} 为模型的隐藏层维度， d_k 为键和查询的维度。

4.3 零基础深入浅出理解Self-Attention的输出过程

得到矩阵Q（Query），K（Key），V（Value）之后，我们就可以计算出Self-Attention的输出注意力得分（Scaled Dot-Product Attention）了，计算的公式如下： \text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^\top}{\sqrt{d_k}} \right)V \\其中QK^\top计算Q（查询）与K（键）转置的点积（内积）相似度，得到一个n \times n的矩阵，n表示句子的单词数，表示序列中每个位置之间的相关性（Attention强度）。 \frac{1}{\sqrt{d_k}} 是缩放因子，防止点积值过大导致梯度消失\text{softmax}函数将得分归一化为概率分布。

到这里，大家光看自注意力机制的公式可能还是会不太理解其本质的原理。接下来，Rocky就带着大家深入浅出详细讲解一下自注意力机制的本质意义。

首先我们来理解一下QK^\top的含义代表什么呢？

我们已经知道Q（Query）和K（Key）的根源都来自于输入特征X，其本质都是X的线性变换。所以我们可以将 QK^\top 转化成 W^QW^K(XX^\top) ，这样一来，我们暂时先不看通过训练学习到的权重矩阵W，只看其中的 XX^\top ，我们可以知道，这本质上是X矩阵乘以它自己的转置，也就是X矩阵中每行向量与其他向量计算内积。

那么，向量内积的意义什么呢？主要是用来表示两个向量的夹角，或者说一个向量在另一个向量上的投影。而且投影的值大，说明两个向量相关度高。如果两个向量夹角是九十度，那么这两个向量线性无关，代表完全没有相关性！这样一来，我们就可以用特征的相关性来代表注意力值了！

更进一步的，矩阵 XX^T 是一个方阵，里面已经保存了每个向量与自己以及其他向量进行内积运算的结果，也就是对自己以及其他向量的注意力值，如下图所示：

有了上述的本质理解，下面Rocky带着大家对这个计算过程进行详细图示拆解。首先下图为Q乘以 K^{T} 的详细图解，其中1234表示的是句子中的单词：

Q乘以K的转置的计算

在得到QK^{T} 后，我们需要除以\sqrt{d_k} 进行缩放，这代表什么含义呢？

我们可以假设 Q,K矩阵的均值为0，方差为1，那么A=QK^T矩阵的均值为0，方差为d。当d变得很大时，A 中元素的方差也会变得很大，如果A中的元素方差很大，那么后续进行Softmax(A) 计算时的分布会趋于陡峭(分布的方差大，分布集中在绝对值大的区域)。因此A中每一个元素除以\sqrt{d_k} 后，方差就又变成1。这使得计算 Softmax(A) 后的特征分布“陡峭”程度与d解耦了，从而使得大模型训练过程中的梯度值保持稳定。

接着我们再通过Softmax操作将将 A=QK^T 矩阵归一化，获得每一个单词的加权求和的权重（即Attention系数）。当我们关注"你"这个字的时候，我们应当分配0.3的注意力给它本身，剩下0.1关注"好"，0.2关注"AIGC"，0.2关注“时”，0.2关注“代”。公式中的Softmax是对矩阵的每一行进行归一化操作，即每一行的权重和都变为1：


对矩阵的每一行进行 Softmax

在通过Softmax获得加权权重矩阵之后，再与矩阵 V相乘得到矩阵Z，作为Self-Attention的最终输出：


Self-Attention 输出

上图中加权权重矩阵的第 1 行表示单词 1 与其他所有单词的Attention 系数，最终单词 1 的输出 Z_{1} 等于所有单词 i 的值 V_{i} 根据 attention 系数的比例加在一起得到，如下图所示：


Zi 的计算方法

观察上图，我们可以发现行向量与 V 相乘，得到了一个新的行向量，且这个行向量与 X 中对应的行向量维度相同。

在新的向量中，每一个维度的数值都是由所有词向量在这一维度的数值加权求和得来的，这个新的行向量就是"你"字词向量经过注意力机制加权求和之后的表示。

4.4 零基础深入浅出理解Multi-Head Attention机制

在上一小节中，Rocky已经详细讲解了Self-Attention机制的完整计算过程，在本小节中，Rocky再向大家介绍一下Multi-Head Attention机制（多头注意力机制）。Multi-Head Attention机制是由多个Self-Attention组合构成的，可以说是基于Self-Attention的一个上层建筑，它通过并行地计算多个注意力（Attention）头，使模型能够从不同的表示子空间中捕获信息，从而提升模型的表现力和泛化能力。下面是Multi-Head Attention的结构示意图：

Multi-Head Attention机制示意图

从上图可以看出，多头注意力机制的主要流程包括以下步骤：

线性映射：将输入序列通过线性变换，得到查询（Query）、键（Key）和值（Value）矩阵。
划分头：将查询、键和值的线性变换结果划分为多个头。
并行计算注意力：对每个头，独立地计算注意力输出。
拼接结果：将所有头的注意力输出拼接在一起。
最终线性映射：将拼接结果通过线性变换，得到多头注意力的输出。

下面Rocky带着大家详细解析一下多头注意力机制的完整处理过程。我们设定以下参数：

输入序列长度为 n，输入向量维度为 d_{\text{model}}。
头的数量为 h，每个头的维度为 d_k = d_v = d_{\text{model}} / h 。

从上图可以看到 Multi-Head Attention 包含多个 Self-Attention 层，首先将输入X分别传递到 h 个不同的 Self-Attention 中，计算得到 h 个输出矩阵Z。下图是 h=8 时候的情况，此时会得到 8 个输出矩阵Z。

多个 Self-Attention

我们首先对输入X \in \mathbb{R}^{n \times d_{\text{model}}}进行线性变换，得到查询、键和值矩阵：

\begin{aligned} Q &= XW^Q, \quad W^Q \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}} \\ K &= XW^K, \quad W^K \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}} \\ V &= XW^V, \quad W^V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}} \end{aligned} \\接着我们再将Q, K, V划分为h个头，每个头的维度为 d_k：

\begin{aligned} Q &= [Q_1; Q_2; \dots; Q_h], \quad Q_i \in \mathbb{R}^{n \times d_k} \\ K &= [K_1; K_2; \dots; K_h], \quad K_i \in \mathbb{R}^{n \times d_k} \\ V &= [V_1; V_2; \dots; V_h], \quad V_i \in \mathbb{R}^{n \times d_v} \end{aligned} \\

得到上面的并行注意力头后，我们最后对于每个头 i \in \{1, 2, ..., h\}，计算注意力输出：
\text{head}_i = \text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left( \frac{Q_i K_i^\top}{\sqrt{d_k}} \right)V_i \\

得到 8 个输出矩阵 Z_{1} 到 Z_{8} 之后，Multi-Head Attention 将它们拼接在一起 (Concat)，然后传入一个Linear层，得到 Multi-Head Attention 最终的输出Z：


Multi-Head Attention 的输出

将所有头的输出拼接：\text{Concat} = [\text{head}_1; \text{head}_2; \dots; \text{head}_h], \quad \text{Concat} \in \mathbb{R}^{n \times d_{\text{model}}} \\

将拼接结果通过线性变换，得到多头注意力的输出：\text{Output} = \text{Concat} \cdot W^O, \quad W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}} \\可以看到 Multi-Head Attention 输出的矩阵Z与其输入的矩阵X的维度是一样的。

最后，我们再对 Multi-Head Attention 完整流程进行汇总，完整计算公式为：

\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O \\ 其中： \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) ，W_i^Q, W_i^K, W_i^V 是第i 个头的参数矩阵。

Rocky也总结了多头注意力机制的优势：

捕获多样性特征：多个头允许模型从不同的表示子空间中学习，从而捕获输入序列中丰富的特征。
提升模型性能：在实践中，多头注意力机制证明能够提升模型在各类任务中的性能。
稳定训练过程：通过分散注意力机制，减轻了单个注意力头可能出现的过拟合问题。

多头注意力机制通过并行地关注输入的不同子空间，大大增强了模型的表达能力。在AIGC时代的类Transformer模型中，它已成为不可或缺的核心组件，对推动AIGC领域的发展起到了重要作用。

最后，我们回顾一下Self-Attention的完整流程，可以发现它会和每一个Input Vector都进行Attention，也就没有考虑到Input Sequence的顺序了。这样的话，得到的Attention就丢失了原来输入特征的顺序信息。对比来说，LSTM是通过输出词向量的先后顺序来判定文本顺序信息，因此为了确定位置信息，Transformer中设计了位置编码的概念，大家可以在本文3.1章节中回顾学习。

5. 零基础理解Transformer的Encoder结构

在本章节中，Rocky将详细讲解Transformer中Encoder结构的核心基础知识。

Transformer Encoder block

我们可以看到，上图中红色部分是Transformer的 Encoder block 结构，主要是由Multi-Head Attention机制、Add & Norm层、Feed Forward（前馈神经网络，Feed-Forward Neural Network）层组成的。同时每个子层中还包含残差连接（Residual Connection）和层归一化（Layer Normalization）结构。在之前的章节中，我们已经了解Multi-Head Attention机制的核心原理，接下来我们再了解一下Add & Norm层和Feed Forward曾的核心原理。

编码器的结构流程图：

输入 --> [多头自注意力] --> [残差连接和层归一化] --> [前馈神经网络] --> [残差连接和层归一化] --> 输出
5.1 零基础理解Add & Norm层的核心原理

Add & Norm 层由 Add 和 Norm 两部分组成，其计算公式如下：

Add & Norm公式

其中X表示 Multi-Head Attention 或者 Feed Forward 的输入，MultiHeadAttention(X) 和 FeedForward(X) 表示输出 (输出与输入 X 维度是一样的，所以可以相加)。

Add指 X+MultiHeadAttention(X)，是一种残差连接，通常用于解决多层网络训练的问题，可以让网络只关注当前差异的部分，在传统深度学习的“基座”模型ResNet中经常用到：


残差连接架构

在Transformer Encoder的每个子层中，都会添加输入与子层输出的残差连接（Residual Connection）： \text{Output} = \text{LayerNorm}(X + \text{SubLayer}(X)) \\

其作用作用主要有：（1）缓解深层网络中的梯度消失问题。（2）促进信息的直接传播，提升模型的训练效果。

Norm指Layer Normalization，通常用于 RNN 结构，在完成残差连接操作后，使用Layer Normalization会将每一层神经元的输入都转成均值方差都一样的，这样可以加快训练过程的收敛，同时稳定训练。具体公式如下所示： \text{LayerNorm}(X) = \frac{X - \mu}{\sigma} \cdot \gamma + \beta \\ 其中\mu 和\sigma分别是输入的均值和标准差；\gamma 和\beta是可学习的参数。

5.2 零基础理解Feed Forward层的核心原理

前馈神经网络（Feed-Forward Neural Network）层比较简单，是一个两层的全连接层。第一层的激活函数为 ReLU激活函数，引入非线性，增强模型的表达能力。第二层不使用激活函数，对每个位置的向量独立地进行非线性变换。对应的公式如下所示\text{FFN}(X) = \text{max}(0, XW_1 + b_1)W_2 + b_2 \\其中 X 代表输入；W_1, W_2代表权重矩阵； b_1, b_2代表偏置，Feed Forward 最终得到的输出矩阵的维度与 X 一致。

5.3 Tranformer Encoder架构的工作流程

通过上面讲解的Multi-Head Attention、Feed Forward、Add & Norm模块，我们就可以构造出一个Transformer的Encoder block，Encoder block接收输入矩阵X_{(n\times d)} ，并输出一个矩阵O_{(n\times d)} 。通过多个Encoder block叠加就可以组成完整的Encoder架构。

第一个Encoder block的输入为句子单词的表示向量矩阵，后续Encoder block的输入是前一个Encoder block的输出，最后一个Encoder block输出的矩阵就是编码信息矩阵 C，这个特征矩阵后续会输入到Transformer的Decoder架构中。


Encoder 编码句子信息

总的来说，Transformer的编码器的工作流程包括：

输入嵌入（Input Embedding）：将输入的词序列转换为词向量表示。
位置编码（Positional Encoding）：将位置编码添加到输入嵌入中，形成带有位置信息的输入。
编码器层堆叠：将上述输入依次通过N 个编码器层，每个层都包含多头自注意力和前馈神经网络。
输出生成：编码器的最终输出是一个包含输入序列每个位置的表示向量的序列，供后续的解码器或其他任务使用。

从AIGC视角看，我们可以将Transformer的编码器想象成一个信息处理的流水线：

多头自注意力：像是一群专家团队，每个专家关注输入信息的不同方面，彼此交流协作。
前馈神经网络：对经过交流的信息进行深入处理和特征提取。
残差连接和层归一化：确保信息在处理过程中不失真，保持数据的稳定性。

Transformer的编码器同时具备以下特点：

并行计算：由于不依赖序列的时间步，Transformer 可以对整个序列进行并行计算，大大提高了训练效率。
长距离依赖：自注意力机制使得模型能够直接捕获序列中任意两个位置之间的依赖关系，无论距离多远。
模型可解释性：注意力权重提供了模型在决策过程中关注哪些部分的线索，提升了模型的可解释性。

总的来说，Transformer 的编码器结构通过巧妙地设计，使得模型能够高效地学习序列数据的全局特征。其核心在于多头自注意力机制和前馈神经网络的结合，再辅以残差连接和层归一化，构建了一个强大而高效的编码器模块。

6. 零基础理解Transformer的Decoder结构

接下来，Rocky再详细拆解一下Transformer的Decoder结构。

Transformer Decoder Block架构

可以看到，Transformer的解码器由多个相同的层（Layer）堆叠而成，每一层包含三个主要的子层：

掩码多头自注意力机制（Masked Multi-Head Self-Attention）
编码器-解码器注意力机制（Encoder-Decoder Attention）
前馈神经网络（Feed-Forward Neural Network）

与此同时，每个子层后同样也添加了残差连接（Residual Connection）和层归一化（Layer Normalization）。

下面是解码器的完整结构流程图：

输入 --> [掩码多头自注意力] --> [残差连接和层归一化] -->
       [编码器-解码器注意力] --> [残差连接和层归一化] -->
       [前馈神经网络] --> [残差连接和层归一化] --> 输出

上图红色部分为Transformer的Decoder block结构，与Encoder block相似，但是存在一些区别：

一共包含了两个Multi-Head Attention层。
第一个Multi-Head Attention层采用了Masked操作。
第二个Multi-Head Attention层的K, V矩阵使用Encoder的编码信息矩阵C进行特征计算，而Q则使用上一个 Decoder block的输出进行特征计算。
最后包含一个Softmax层计算下一个翻译单词的概率。
6.1 零基础理解Masked Multi-Head Attention核心基础知识

在Transformer中Decoder Block的第一个Multi-Head Attention采用了Masked操作，即Masked Multi-Head Attention（掩码多头自注意力机制）。因为在翻译的过程中是顺序翻译的，模型在生成第t个词时，只能依赖已生成的词（即第1 到第t-1个词）。通过 Masked 操作可以防止第 t 个单词知道 t+1 个单词之后的信息，从而防止模型“偷看”未来的信息。下面以 "我们喜欢WeThinkIn" 翻译成 "We love WeThinkIn" 为例，了解一下 Masked 操作。

Transformer在进行Decoder的时候，需要根据之前的翻译结果，求解当前最有可能的翻译，如下图所示。首先根据输入 "<Begin>" 预测出第一个单词为 "I"，然后根据输入 "<Begin> I" 预测下一个单词 "have"。


Decoder的预测流程

Decoder 可以在训练的过程中使用 Teacher Forcing 并且并行化训练，即将正确的单词序列 (<Begin> I have a cat) 和对应输出 (I have a cat <end>) 传递到 Decoder。那么在预测第 t 个输出时，就要将第 t+1 之后的单词掩盖住，注意 Mask 操作是在 Self-Attention 的 Softmax 之前使用的，下面用 0 1 2 3 4 5 分别表示 "<Begin> I have a cat <end>"。

Transformer中掩码（Mask）主要通过上三角掩码矩阵实现：使用一个上三角矩阵作为掩码，对角线以下的元素为 0，对角线以上的元素为负无穷大。同时在在计算注意力得分时，将掩码矩阵添加到QK^\top 之前，确保未来位置的注意力得分为负无穷大，经过 softmax 后为 0。

第一步：将Decoder 的输入矩阵和 Mask 矩阵，输入矩阵包含 "<Begin> I have a cat" (0, 1, 2, 3, 4) 五个单词的表示向量，Mask 是一个 5×5 的矩阵。在 Mask 可以发现单词 0 只能使用单词 0 的信息，而单词 1 可以使用单词 0, 1 的信息，即只能使用之前的信息。

输入矩阵与 Mask 矩阵

第二步：接下来的操作和之前的 Self-Attention 一样，通过输入矩阵X计算得到Q,K,V矩阵。然后计算Q和 K^{T} 的乘积 QK^{T} 。

Q乘以K的转置

第三步：在得到 QK^{T} 之后需要进行 Softmax，计算 Attention Score，我们在 Softmax 之前需要使用Mask矩阵遮挡住每一个单词之后的信息，遮挡操作如下：

Softmax 之前 Mask

得到 Mask QK^{T} 之后在 Mask QK^{T}上进行 Softmax，每一行的和都为 1。但是单词 0 在单词 1, 2, 3, 4 上的 attention score 都为 0。

第四步：使用 Mask QK^{T}与矩阵 V相乘，得到输出 Z，则单词 1 的输出向量 Z_{1} 是只包含单词 1 信息的。


Mask 之后的输出

第五步：通过上述步骤就可以得到一个 Mask Self-Attention 的输出矩阵 Z_{i} ，然后和 Encoder 类似，通过 Multi-Head Attention 拼接多个输出Z_{i} 然后计算得到第一个 Multi-Head Attention 的输出Z，Z与输入X维度一样。

总的计算过程包含了计算查询（Q）、键（K）和值（V）： Q = XW^Q, \quad K = XW^K, \quad V = XW^V \\

和计算注意力得分并应用掩码： \text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^\top}{\sqrt{d_k}} + \text{Mask} \right)V \\

其中Mask是上三角掩码矩阵。

6.2 零基础理解Decoder中Encoder-Decoder Attention核心基础知识

Transformer中Decoder Block的第二个Multi-Head Attention和常规Multi-Head Attention架构一致，主要的区别在于其中Self-Attention的K, V矩阵不是使用 上一个 Dcoder block 的输出计算的，而是使用 Encoder 的编码信息矩阵 C 计算的。所以也可以称为编码器-解码器注意力机制（Encoder-Decoder Attention）。

我们根据Encoder的输出 C计算得到 K, V，根据上一个Decoder block 的输出 Z 计算 Q (如果是第一个 Decoder block 则使用输入矩阵 X 进行计算)，后续的计算方法与之前讲解的一致。

这样做的好处是在 Decoder 的时候，每一位单词都可以利用到 Encoder 所有单词的信息 (这些信息无需 Mask)。

连接编码器和解码器：使解码器能够关注编码器输出的相关信息，实现信息交互。

实现翻译或序列生成：在机器翻译中，解码器需要根据源语言（编码器输出）生成目标语言。

完整计算步骤包括接收编码器的输出作为键（K）和值（V），解码器自身的输出作为查询（Q）：

Q_{\text{dec}} = X_{\text{dec}}W^Q, \quad K_{\text{enc}} = X_{\text{enc}}W^K, \quad V_{\text{enc}} = X_{\text{enc}}W^V \\

再计算注意力得分：

\text{Attention}(Q_{\text{dec}}, K_{\text{enc}}, V_{\text{enc}}) = \text{softmax}\left( \frac{Q_{\text{dec}} K_{\text{enc}}^\top}{\sqrt{d_k}} \right)V_{\text{enc}} \\

6.3 零基础理解Softmax预测输出单词完整过程

Decoder block 最后的部分是利用 Softmax 预测下一个单词，在之前的网络层我们可以得到一个最终的输出 Z，因为 Mask 的存在，使得单词 0 的输出 Z0 只包含单词 0 的信息，如下：


Decoder Softmax 之前的 Z

Softmax 根据输出矩阵的每一行预测下一个单词：


Decoder Softmax 预测

这就是 Decoder block 的定义，与 Encoder 一样，Decoder 是由多个 Decoder block 组合而成

到此为止，Rocky再详细总结一下解码器的工作流程：

输入嵌入（Input Embedding）：将目标序列（可能是已生成的词或特殊的起始符）转换为词向量表示。
位置编码（Positional Encoding）：将位置编码添加到输入嵌入中，形成带有位置信息的输入。
掩码多头自注意力：对解码器输入进行自注意力计算，使用掩码防止关注未来的信息。
编码器-解码器注意力：解码器的输出作为查询，编码器的输出作为键和值，计算跨注意力，获取编码器的信息。
前馈神经网络：对每个位置的向量进行非线性变换，增强模型的表达能力。
层堆叠：上述过程在多个解码器层中重复，通常与编码器的层数相同。
输出生成：解码器的最终输出通过线性层和 softmax，生成下一个词的概率分布。

Transformer解码器的主要特点如下所示：

自回归生成：通过掩码多头自注意力机制，解码器能够在生成序列时，只依赖于已生成的词，实现自回归生成
全局信息交互：编码器-解码器注意力机制使解码器能够关注编码器输出的任意位置，捕获全局信息，提高生成质量。
并行化计算：虽然解码器在训练时可以并行计算，但在推理（生成）时，由于自回归的性质，仍需要逐步生成。

Transformer 的解码器结构通过巧妙地设计，使模型能够高效地生成序列。其核心在于掩码多头自注意力机制和编码器-解码器注意力机制的结合，再辅以前馈神经网络、残差连接和层归一化，构建了一个强大而高效的解码器模块。

6.4 Transformer的原理总结
Transformer和RNN不同，能够比较好地并行训练，训练效率较高。
Transformer本身是不能利用单词的顺序信息的，因此需要在输入中添加位置Embedding，否则Transformer就会退化成一个词袋模型。
Transformer的核心思想是提出了Self-Attention结构，其中用到了Q, K, V特征矩阵。
Transformer 中 Multi-Head Attention 中有多个 Self-Attention，可以捕获单词之间多种维度上的相关系数 attention score。
7. Transformer系列模型的性能优化
8. 推荐阅读

Rocky会持续分享AIGC的干货文章、实用教程、商业应用/变现案例以及对AIGC行业的深度思考与分析，欢迎大家多多点赞、喜欢、收藏和转发，给Rocky的义务劳动多一些动力吧，谢谢各位！

8.1 深入浅出完整解析Stable Diffusion 3（SD 3）和FLUX.1系列核心基础知识

Rocky也对Stable Diffusion 3和FLUX.1的核心基础知识作了全面系统的梳理与解析：

8.2 深入浅出完整解析Stable Diffusion XL（SDXL）核心基础知识

Rocky也对Stable Diffusion XL的核心基础知识作了全面系统的梳理与解析：

8.3 深入浅出完整解析Stable Diffusion（SD）核心基础知识

Rocky也对Stable Diffusion 1.x-2.x系列模型的核心基础知识做了全面系统的梳理与解析：

8.4 深入浅出完整解析Stable Diffusion中U-Net的前世今生与核心知识

Rocky对Stable Diffusion中最为关键的U-Net结构进行了深入浅出的全面解析，包括其在传统深度学习中的价值和在AIGC中的价值：

8.5 深入浅出完整解析LoRA（Low-Rank Adaptation）模型核心基础知识

对于AIGC时代中的“ResNet”——LoRA模型，Rocky也进行了深入浅出的全面讲解：

8.6 深入浅出完整解析ControlNet核心基础知识

AI绘画作为AIGC时代的一个核心方向，开源社区已经形成以Stable Difffusion为核心，ConrtolNet和LoRA作为首要AI绘画辅助工具的变化万千的AI绘画工作流。

ControlNet正是让AI绘画社区无比繁荣的关键一环，它让AI绘画生成过程更加的可控，更有助于广泛地将AI绘画应用到各行各业中：

8.7 深入浅出完整解析主流AI绘画框架核心基础知识

AI绘画框架正是AI绘画“工作流”的运行载体，目前主流的AI绘画框架有Stable Diffusion WebUI、ComfyUI以及Fooocus等。在传统深度学习时代，PyTorch、TensorFlow以及Caffe是传统深度学习模型的基础运行框架，到了AIGC时代，Rocky相信Stable Diffusion WebUI就是AI绘画领域的“PyTorch”、ComfyUI就是AI绘画领域的“TensorFlow”、Fooocus就是AI绘画领域的“Caffe”：

8.8 手把手教你成为AIGC算法工程师，斩获AIGC算法offer！

在AIGC时代中，如何快速转身，入局AIGC产业？如何成为AIGC算法工程师？如何在学校中系统性学习AIGC知识，斩获心仪的AIGC算法offer？

Don‘t worry，Rocky为大家总结整理了全面的AIGC算法工程师成长秘籍，为大家答疑解惑，希望能给大家带来帮助：

8.9 AIGC产业的深度思考与分析

2023年3月21日，微软创始人比尔·盖茨在其博客文章《The Age of AI has begun》中表示，自从1980年首次看到图形用户界面（graphical user interface）以来，以OpenAI为代表的科技公司发布的AIGC模型是他所见过的最具革命性的技术进步。

Rocky也认为，AIGC及其生态，会成为AI行业重大变革的主导力量。AIGC会带来一个全新的红利期，未来随着AIGC的全面落地和深度商用，会深刻改变我们的工作、生活、学习以及交流方式，各行各业都将被重新定义，过程会非常有趣。

那么，在此基础上，我们该如何更好的审视AIGC的未来？我们该如何更好地拥抱AIGC引领的革新？Rocky准备从技术、产品、商业模式、长期主义等维度持续分享一些个人的核心思考与观点，希望能帮助各位读者对AIGC有一个全面的了解：

8.10 算法工程师的独孤九剑秘籍

为了方便大家实习、校招以及社招的面试准备，同时帮助大家提升扩展技术基本面，Rocky将符合大厂和AI独角兽价值的算法高频面试知识点撰写总结成《三年面试五年模拟之独孤九剑秘籍》，并制作成pdf版本，大家可在公众号WeThinkIn后台【精华干货】菜单或者回复关键词“三年面试五年模拟”进行取用：

8.11 深入浅出完整解析AIGC时代中GAN系列模型的前世今生与核心知识

GAN网络作为传统深度学习时代的最热门生成式Al模型，在AIGC时代继续繁荣，作为Stable Diffusion系列模型的“得力助手”，广泛活跃于Al绘画的产品与工作流中：

Rocky一直在运营技术交流群（WeThinkIn-技术交流群），这个群的初心主要聚焦于AI行业话题的讨论与研究，包括但不限于算法、开发、竞赛、科研以及工作求职等。群里有很多AI行业的大牛，欢迎大家入群一起交流探讨～（请备注来意，添加小助手微信Jarvis8866，邀请大家进群～）
