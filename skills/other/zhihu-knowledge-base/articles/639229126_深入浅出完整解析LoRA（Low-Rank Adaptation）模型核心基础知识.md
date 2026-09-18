# 深入浅出完整解析LoRA（Low-Rank Adaptation）模型核心基础知识

**作者**: Rocky Ding​北京科技大学 工学硕士

**原文链接**: https://zhuanlan.zhihu.com/p/639229126

---

​
目录
收起
1. LoRA系列模型资源
2. 零基础深入浅出理解LoRA模型核心基础知识（全网最详细讲解）
2.1 零基础深入浅出理解LoRA模型的核心原理
2.2 零基础深入浅出理解LoRA模型的优势
2.3 零基础深入浅出理解LoRA模型的三大特性（易用性、泛化性、还原度）
2.4 零基础深入浅出理解LoRA模型的高阶用法
2.5 零基础深入浅出理解DreamBooth LoRA模型
2.6 零基础深入浅出理解LoRA模型的融合和提取方法（Merge Block Weighted，MBW）
3. 从0到1搭建使用LoRA模型进行AI绘画（全网最详细讲解）
3.1 零基础使用diffusers搭建LoRA推理流程
3.2 零基础使用Stable Diffusion WebUI搭建LoRA推理流程
3.3 零基础使用ComfyUI搭建LoRA推理流程
3.4 零基础使用SD.Next搭建LoRA推理流程
3.5 LoRA生成图像示例
3.6 多个LoRA模型推理高阶融合（Multi-LoRA Composition）策略详解
4. 从0到1上手训练自己的LoRA模型用于AI绘画（全网最详细讲解）
4.1 LoRA训练数据集制作
4.2 使用kohya-trainer框架训练LoRA模型
4.3 kohya-trainer框架中LoRA训练参数全解析（全网最详细）
4.4 使用diffusers框架训练LoRA模型
4.5 LoRA模型的关键训练超参数详解（全网最详细）
4.6 LoRA模型的训练技巧与经验分享
4.7 LoRA模型训练结果测试评估
5. 主流LoRA变体模型深入浅出完整讲解
5.1 LoCon核心基础知识深入浅出完整讲解
5.2 LoHa核心基础知识深入浅出完整讲解
5.3 残差/差异化LoRA模型深入浅出完整讲解
5.4 LCM_LORA模型深入浅出完整解析
5.5 Textual Inversion（embeddings模型）技术深入浅出完整讲解
6. 深入浅出完整讲解MoE-LoRA（Mixture of Experts with LoRA）的核心基础知识
6.1 深入浅出完整讲解MoE框架下LoRA技术的核心原理
6.2 MoE思想与LoRA模型相结合
6.3 MoE-LoRA架构中专家选择(Expert-Choice)与token选择(Token-Choice)的核心基础知识讲解
6.4 深入浅出讲解Token-Choice (TC) 和Expert-Choice (EC) 的具体代码实现
7. 优质LoRA模型推荐（持续更新）
7.1 人物LoRA模型推荐
7.2 风格LoRA模型推荐
7.3 Low-Level功能LoRA模型推荐（美颜、美肤、祛痘、磨皮、精修、画质增强、光影调整等）
8. 推荐阅读
8.1 深入浅出完整解析扩散模型DDPM、DDIM、Classifier/Classifier-Free Guidance、Rectified Flow核心基础知识
8.2 深入浅出完整解析AI Agent（AI智能体）的核心基础知识
8.3 深入浅出完整解析FLUX.1 Kontext和FLUX.1 Krea核心基础知识
8.4 深入浅出完整解析DeepSeek系列核心基础知识
8.5 深入浅出完整解析Stable Diffusion 3（SD 3）和FLUX.1系列核心基础知识
8.6 深入浅出完整解析Stable Diffusion XL（SDXL）核心基础知识
8.7 深入浅出完整解析Stable Diffusion（SD）核心基础知识
8.8 深入浅出完整解析Stable Diffusion中U-Net的前世今生与核心知识
8.9 深入浅出完整解析ControlNet核心基础知识
8.10 深入浅出完整解析Sora等AI视频大模型核心基础知识
8.11 深入浅出完整解析AIGC时代Transformer核心基础知识
8.12 深入浅出完整解析主流AI绘画框架核心基础知识
8.13 手把手教你成为AIGC算法工程师，斩获AIGC算法offer！
8.14 AIGC产业的深度思考与分析
8.15 算法工程师的独孤九剑秘籍
8.16 深入浅出完整解析AIGC时代中GAN系列模型的前世今生与核心知识
本文的专栏：Rocky Ding的AI算法兵器谱
我的公众号：WeThinkIn
更多AI行业干货内容欢迎关注我的知乎，公众号，专栏～

码字不易，希望大家能多多点赞，给我更多坚持写下去的动力，谢谢大家！

Rocky在持续撰写FLUX.1、Stable Diffusion 3、Stable Diffusion XL和Stable Diffusion 1.x-2.x全系列的10万字全方位解析文章，希望大家能多多点赞，让Rocky有更多坚持的动力：

大家好，我是Rocky。

在2022年AIGC时代到来之后，LoRA（Low-Rank Adaptation）成为了AIGC图像生成/AI绘画领域中与Stable Diffusion（简称SD）/FLUX系列大模型组合使用最多的AI技术之一，几乎所有商业化的AIGC工作流（Workflow）中都有LoRA及其变体模型的身影。LoRA属于参数高效微调（Parameter-Efficient Fine-Tuning，PEFT）技术中的一种，PEFT技术的整体思想是冻结AIGC大模型的大部分参数，再引入一小部分可训练的参数作为适配模块进行微调训练，以达到节省AIGC大模型微调时显存开销的目的。

PEFT技术最开始应用于LLM大模型领域，除了LoRA之外，还包含了Prompt Tuning、P-Tuning、Adapter Tuning、Prefix Tuning等。这些技术或多或少都从LLM大模型领域跨界到了AIGC图像生成/AI绘画领域，但是只有LoRA能够无可争议的跨周期持续繁荣，可见LoRA的核心思想足够优雅。

SD/FLUX模型+LoRA模型的组合，不仅可以创造很多脑洞大开的AI绘画风格、人物以及概念，而且还大幅降低了AIGC图像生成/AI绘画的成本，提高了AIGC图像生成/AI绘画的多样性和灵活性，让各行各业的AI爱好者都真真切切地感受到了AIGC图像生成/AI绘画的力量，加速了AIGC领域的破圈式繁荣。

Rocky相信Stable Diffusion 1.x是AIGC时代的“YOLO”，Stable Diffusion 2.x是AIGC时代的“YOLOv2”，Stable Diffusion XL是AIGC时代的“YOLOv3”，Stable Diffusion 3是AIGC时代的“YOLOv4”，FLUX系列是AIGC时代的“YOLOv5”，那么LoRA系列模型就是AIGC时代的“ResNet”。

SD模型+LoRA模型组合生成图像示例

因此在本文中，Rocky主要对LoRA模型的全维度各个方面都做一个深入浅出的分析讲解（LoRA模型核心原理解析、LoRA模型结构解析、LoRA模型从0到1保姆级训练教程、LoRA模型在不同AI绘画框架中从0到1推理运行保姆级教程、主流LoRA变体模型解析、热门LoRA模型资源汇总分享、LoRA经典应用场景介绍、优秀LoRA模型推荐等），和大家一起学习探讨，在AIGC时代中让我们更好地使用LoRA系列模型进行AIGC图像生成/AI绘画创作！

1. LoRA系列模型资源
LoRA模型论文：LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS
LyCORIS项目：LyCORIS
LoCon模型项目：LyCORIS/locon
LoHa模型论文：FEDPARA: LOW-RANK HADAMARD PRODUCT FOR COMMUNICATION-EFFICIENT FEDERATED LEARNING
LCM_LoRA模型论文：LCM-LoRA: A Universal Stable-Diffusion Acceleration ModuleLCM-LoRA: A Universal Stable-Diffusion Acceleration Module
训练代码：kohya-ss/sd-scripts、qaneel/kohya-trainer、qaneel/kohya-trainer、diffusers、bmaltais/kohya_ss
LoRA系列训练资源（SD LoRA训练脚本、SDXL LoRA训练脚本、FLUX LoRA训练脚本、差异化LoRA训练脚本、第四章中宝可梦LoRA模型训练后权重等）百度云网盘：关注Rocky的公众号WeThinkIn，后台回复：LoRA训练资源，即可获得LoRA训练干货资源链接。
LCM_LoRA系列模型权重百度云网盘：关注Rocky的公众号WeThinkIn，后台回复：LCM_LoRA，即可获得资源链接，包含LCM_LoRA SDXL、LCM_LoRA SD1.5和LCM_LoRA SSD-1b三个LCM_LoRA模型权重。
DreamBooth模型第三方资源库：sd-dreambooth-library
Textual Inversion（Embeddings模型）技术论文：Textual Inversion技术
Textual Inversion（Embeddings模型）模型权重百度云网盘：大家关注Rocky的公众号WeThinkIn，后台回复：TextEmbeddings模型，即可获得资源链接。

Rocky会持续把更多LoRA系列模型的干货资源发布到本节中，让大家更加方便的查找LoRA系列模型的最新资讯。

2. 零基础深入浅出理解LoRA模型核心基础知识（全网最详细讲解）

在进入AIGC时代后，以AIGC大模型为核心的AI绘画、AI视频、大模型文本对话、AI多模态、AI音频、数字人等核心领域繁荣发展。AIGC大模型的强大能力让AIGC彻底破圈，可谓是天下谁人不识Stable Diffusion和ChatGPT。

但是AIGC大模型参数量巨大，训练成本较高，当遇到一些细分任务时，对AIGC大模型进行全参微调训练的性价比不高，在这种情况下，本文的主角——LoRA（Low-Rank Adaptation）模型就出场了。

LoRA模型可以与上述提到的所有AIGC核心领域结合使用，具有很强的“万金油”特性，本文主要从AIGC图像生成/AI绘画的角度来全方位解析LoRA模型及其背后的核心技术原理，大家深刻理解文本后，就能举一反三将LoRA模型应用到所有的AIGC核心领域中！

2.1 零基础深入浅出理解LoRA模型的核心原理

LoRA（Low-Rank Adaptation）本质上是对特征矩阵进行低秩分解的一种近似数值分解技术，可以大幅降低特征矩阵的参数量，但是会伴随着一定的有损压缩。从传统深度学习时代走来的读者，可以发现其实LoRA本质上是基于Stable Diffusion/FLUX的一种轻量化技术。

在AIGC图像生成/AI绘画领域，我们可以使用SD/FLUX模型+LoRA模型的组合微调训练方式，只训练参数量很小的LoRA模型，就能在一些AIGC细分领域任务中取得不错的效果。

LoRA模型的训练逻辑是首先冻结SD/FLUX模型的权重，然后在SD模型的U-Net架构或FLUX系列的Transformer架构中注入LoRA权重，主要作用于CrossAttention部分，并只对这部分的参数进行微调训练。

LoRA主要与SD模型的CrossAttention部分结合训练

也就是说，对于SD/FLUX模型权重W_{o} \in \mathbb{R}^{n\times m}，我们不再对其进行全参微调训练，我们对权重加入残差的形式，通过训练\Delta W来完成优化过程：

W' = W_{o} + \Delta W \\

其中\Delta W = AB,A \in \mathbb{R}^{n\times d} ,B \in \mathbb{N}^{d\times m}, d \ll n，d 就是 \Delta W 这个参数矩阵的秩（Rank，lora_dim）， \Delta W 通过低秩分解由两个低秩矩阵的乘积组成。一般来说，尽管SD/FLUX模型的参数量很大，但每个细分任务对应的本征维度（Intrinsic Dimension）并不大，所以我们设置较小的d值就能获得一个参数量远小于SD/FLUX模型的LoRA模型，并在一些细分任务中获得较好的效果。同时如果我们将d设置的越小，LoRA模型的参数量就越小，但是 |W' - AB| 的近似度就越差。

Rocky再为大家举个直观的例子，方便大家深刻地理解。我们假设原来的 \Delta W是100*1024的参数矩阵，那么参数量为102400，LoRA模型将 \Delta W矩阵拆成了两个矩阵相乘，如果我们设置Rank=8，那么就是100*8的A矩阵与8*1024的B矩阵做矩阵乘法，参数量为800+8192=8992，整体参数量下降了约11.39倍。What amazing！非常简洁、高效的思想！

LoRA模型的训练流程示意图

上图是LoRA模型训练的示意图。通常来说，对于矩阵A，我们使用随机高斯分布初始化，并对矩阵B使用全0初始化，使得在训练初始状态下这两个矩阵相乘的结果为0。这样能够保证在训练初始阶段时，SD/FLUX模型的权重完全生效。

虽然矩阵B使用全0初始化能够让SD模型的权重完全生效，但同时也带来了不对称问题（矩阵B全零，矩阵A非全零）。我们可以通过“补权重”法（训练前先在SD/FLUX模型权重中减去矩阵AB的权重）来使矩阵AB都使用随机高斯分布初始化，在效果不变的情况下，增加了对称性： W' = W_{o} - \Delta W + \Delta W \\

其中 \Delta W = AB,A \in \mathbb{R}^{n\times d} ,B \in \mathbb{N}^{d\times m}, d \ll n 。

2.2 零基础深入浅出理解LoRA模型的优势

在本章节中Rocky再和大家一起探讨一下，在加载LoRA模型权重后，会给SD/FLUX模型的微调训练和前向推理带来哪些优势呢？

【1】微调训练阶段

参数量：矩阵乘积AB与SD/FLUX模型的参数有相同的维度，同时分解出来的两个低秩矩阵可以确保参数更新是在低秩情况下的，这样就显著减少训练的参数数量了。同时LoRA模型本身的参数量非常小，最小可至3M左右，这使得LoRA模型在开源社区非常方便传播，也进一步促进了AI绘画领域的爆发式繁荣。
显存占用：训练LoRA模型所需的算力要求很低，我们可以在2080Ti级别的算力设备上进行LoRA模型的训练。因为使用LoRA技术大幅降低了SD/FLUX系列模型训练时的显存占用，整个训练过程中不需要更新SD/FLUX模型的权重，所以SD/FLUX模型对应的优化器参数不需要存储。
计算量：训练过程中的整体计算量没有明显变化，因为LoRA模型是在SD/FLUX模型的全参梯度基础上增加了“残差”梯度，整体上计算量会比SD/FLUX模型的全参微调略大。
训练数据量：LoRA模型能在小数据集上进行训练（1张以上即可，理论上1张图片也能训练）。
训练时长：在其他超参数一致的情况下，与SD/FLUX系列模型全参训练相比，LoRA模型训练速度更快。因为训练过程中只更新LoRA模型对应的参数，无需对SD/FLUX模型权重进行更新；同时由于更新的参数量大幅减少，所以数据传输的通信时间也减少了。
站在“巨人”的肩膀上：LoRA模型能以SD/FLUX模型原有的能力为基础，继续优化学习特定分布特征。

【2】前向推理阶段

参数量：在推理过程中，由于LoRA模型权重与SD/FLUX模型权重进行了合并，同时SD/FLUX模型的结构是不改变的，所以推理时的参数量是不变的。
显存占用：在推理过程中，由于LoRA模型权重与SD/FLUX模型权重进行了合并，同时SD/FLUX模型的结构是不改变的，所以推理时的显存占用和SD模型的显存占用一致。
推理耗时：在推理过程中，由于LoRA模型权重与SD/FLUX模型权重进行了合并，同时SD/FLUX模型的结构是不改变的，所以推理耗时没有增加。
生成效果：针对特定的人物和风格特征，使用LoRA模型+SD/FLUX模型的生成效果会比只用SD/FLUX模型微调训练后的生成效果要好。
高效切换：SD/FLUX模型之间切换需要将所有模型参数加载到内存，从而造成严重的I/O瓶颈。通过对权重更新的有效参数化，不同LoRA模型之间的切换加载既高效又容易。

从上面Rocky对LoRA模型的归纳总结可以看到，SD/FLUX模型与LoRA模型的权重合并后，SD/FLUX模型的结构并没有改变，同时参数量、显存占用、推理耗时也不会改变，整个推理成本是没有增加的，这无疑让LoRA模型有了更多的实用价值。

到这里，可能会有读者疑惑，LoRA模型在训练过程中只对很少的参数更新了权重，为什么能够表现出良好的性能呢？难道不应该更新更多参数的权重来学习更多知识吗？比如说SD/FLUX模型直接微调训练？

大模型的性能与参数维度的关系

上图直观展示了LoRA的核心思想：

高维原始空间：左边的D=3代表预训练大模型的参数空间维度极高（实际中可能是百万/十亿级），θ^(D)是预训练模型的参数，θ₀^(D)是微调的起点。
低秩子空间约束：右边的灰色平面d=2表示LoRA并没有在整个高维空间中更新所有参数，而是将参数更新限制在一个低秩的子空间里。Pθ^(d)是原始参数在低秩子空间上的投影，意味着任务所需的参数更新可以被这个低维子空间很好地捕捉。
当训练步数达到d_int90 = d_int100 = 10左右时，性能突然飙升至接近1.0并保持稳定。
这说明低秩子空间的少量参数更新，已经足以让模型快速收敛到接近最优的性能，不需要在整个高维空间中全量更新参数。

同时，经过预训练的AIGC大模型已经具备“通用知识”，已经学到了语言、图像等领域的通用底层规律（比如语义理解、视觉特征提取）。很多AIGC细分领域的目标需求本质上并不是“重新学习所有知识”，而是在通用知识的基础上，注入任务特定的信息（比如让SD/FLUX生成特定风格的图像）。这些“任务特定信息”往往只需要对原始参数做很小的调整，且这种调整可以被低秩矩阵近似表示（即图中d=2的低维子空间足以覆盖任务所需的更新）。

同时通过LoRA技术的低秩更新避免了全量微调的缺陷，全量微调（如SD/FLUX的直接微调）虽然更新所有参数，但存在明显问题：

计算/存储成本极高：大模型全量微调需要巨大的算力和显存，且每个任务需要单独保存一个完整模型，存储成本爆炸。
容易过拟合与遗忘：在小数据集上全量微调容易过拟合，还可能破坏预训练模型的通用能力（灾难性遗忘）。
效率低下：全量更新中大部分参数的调整对任务收益很小，属于“无效计算”。

而LoRA的低秩更新完美规避了这些问题：

参数效率高：仅更新低秩矩阵（通常是原始参数的1%甚至更少），计算和存储成本大幅降低。
保留预训练能力：原始参数被冻结，预训练的通用知识得以完整保留，仅通过低秩矩阵注入任务信息，避免了灾难性遗忘。
收敛速度快：从右侧性能曲线可以看到，LoRA仅需少量训练步数就达到饱和性能，远快于全量微调。

总的来说，LoRA的核心逻辑是“精准更新关键参数来高效注入特定任务信息”。预训练AIGC大模型已经积累了足够的通用知识，LoRA只需要在低秩子空间中做微小调整，就能让AIGC大模型在特定任务上表现优异，同时还能避免全量微调的各种缺陷。换个更通俗地表达：“不是大模型全参微调训练不起，而是LoRA模型更有性价比！”

2.3 零基础深入浅出理解LoRA模型的三大特性（易用性、泛化性、还原度）

每个LoRA模型都具有三种核心特性：

易用性：在我们加载LoRA模型的权重后，我们需要用多少提示词（Prompt）来使其完全生效。易用性越高，所需的提示词就越少，我们训练的LoRA模型才能在社区更受欢迎，使用量才能快速提升。
泛化性：LoRA模型准确还原其训练素材中主要特征的同时，能否与其他LoRA模型和SD/FLUX模型兼容生效。高泛化性意味着LoRA模型在多种不同的应用场景下都能保持良好的效果。
还原度：在LoRA模型完全生效后，生成的图片与训练素材之间的相似度。高还原性保证了生成图片忠于训练素材，细节和质量上的表现准确无误。
LoRA模型的三大特征之间的协调

这三个核心特性共同定义了LoRA模型的性能和应用范围，但由于资源和技术限制，通常很难同时优化三个特性。我们在选择LoRA模型时，需要根据具体需求考虑哪两个特性最为关键。

下面Rocky以黑魔导女孩LoRA为例，详细为大家讲解一下LoRA的三个核心特性。

黑魔导女孩LoRA在不同SD模型上的效果展示

【一】易用性

一般来说，当我们使用人物/角色LoRA模型时，可以设置一个特殊标签（Trigger Words）来唯一指定人物的主要特征，比如说黑魔导女孩LoRA中可以设置“dark magician girl”作为特殊标签，这时候我们在使用时，不管是与哪种SD模型结合，在输入“dark magician girl”提示词后都能生成不同风格的黑魔导女孩图片。

总的来说，人物/角色LoRA模型的易用性体现在能用特殊标签快速响应数据集中的人物特征。

当我们使用风格/抽象概念LoRA模型时，与人物LoRA正好相反，我们不需要一个特别固化的人物特征，而是需要一个全图像级别的风格渲染，所以风格LoRA要学习的是数据集的整体风格特征，就不需要设置特殊标签。

总的来说，风格/抽象概念LoRA模型的易用性体现在直接的对生成图像的整个风格渲染。

【二】泛化性

AI模型的泛化性能是指模型对未见过的新数据做出准确预测的能力，即模型的“举一反三”能力。一个具有良好泛化性能的模型能够从训练数据中学习到足够的、普遍适用的规律，而不是仅仅记住训练集中的特定案例。这就像一个艺术家在学习和实践过程中不断吸收和整合知识，然后能够运用这些知识和技巧去创作全新的作品。

从美学和艺术的角度来解释AI领域中的模型泛化性能，可以将其比作艺术家创作艺术作品的能力，反映了艺术家创新和适应新挑战的能力。艺术家在面对新的主题或风格时，能够利用其现有的技能和理解创作出新的作品，这需要他们在理解已知元素的基础上进行创新和扩展。

我们在训练LoRA模型时，可能会出现训练好的LoRA在训练时的SD/FLUX底模型上效果很好，但是在其他不同类型的SD/FLUX模型（二次元SD/FLUX模型 -> 真实场景SD/FLUX模型）上效果很差，甚至生成的图片存在特征崩塌的情况。

如果出现这种情况，那就说明LoRA模型的泛化性较弱。而LoRA的主要优势就是轻量且易兼容，如果泛化效果不好，那无疑我们训练的LoRA模型会失去很大的使用价值，在开源社区的传播影响力会大大减弱。

那么，我们该如何优化LoRA模型的泛化性能呢？

首先我们理解一下为什么LoRA模型会出现泛化性能不足的问题。当前的AI绘画开源社区中，所有的热门SD/FLUX系列模型都是从Stable Diffusion 1.x-2.x以及Stable Diffusion XL官方模型上微调而成，同时这些微调训练后的模型还会进行模型融合（Checkpoint Merger）或者进一步微调训练获得新的模型。这就导致了目前的很多开源SD/FLUX系列模型过拟合在了一个风格或者概念上。比如说一个二次元模型，它就只能生成二次元图片，不再具备生成真实场景的能力；比如说一个真实场景模型，它就只能生成真实场景图片，同时丢失了二次元图片的生成能力。

如果我们拿上面提到的过拟合二次元模型作为训练底模型训练一个人物LoRA的话，那么这个LoRA模型大概率也是没有太多泛化性能的，当这个LoRA模型与其他二次元模型配合生成图片时，人物图片的生成效果可能不错。但是当这个LoRA模型与真实场景模型配合时，可能生成的真实人物特征是崩坏的。

所以最好的解决方法是我们直接使用泛化性能最强的官方Stable Diffusion 1.x-2.x、Stable Diffusion XL、Stable Diffusion、FLUX系列模型作为底模型，以此来训练LoRA模型，从而获得泛化性能较强的LoRA模型。

除此之外，我们还可以用正则化技术来降低LoRA模型的过拟合程度，增强LoRA模型的泛化性能。在AI绘画的生成式大模型中，一般有两种正则化方式：

在训练前：设置正则化数据集，正则化数据集会预先将一个概念给锚定下来，使得LoRA模型在训练过程中不会偏离预先设定的概念分布，防止模型过拟合训练数据，提高模型的泛化能力。比如我们想要训练一个美女/帅哥LoRA，那么我们可以使用“1girl”或者“1boy”这个提示词，先在SD底模型上生成一定量的图像，作为正则化集进行先验约束。在本文后面提到的DreamBooth LoRA模型中，就用到了正则化数据集。
在训练中：可以使用梯度截断、Dropout、Normalize、L1和L2正则化等技术在训练过程中不断将想要偏离的梯度进行纠偏，防止LoRA模型跑飞。

与此同时，我们对LoRA的训练数据集进行精细化的标签，也能提升LoRA模型的泛化性。目前AI绘画的开源社区中有一个典型的误区，就是大家总认为训练集中的标签越多会导致LoRA的效果越差。大家觉得效果差的原因是无法用一个触发词获得想要的效果，但其实将图像的全部特征都集中在一个触发词上是一种过拟合的表现。丰富的标签能够降低LoRA模型训练过程中对底模型对应概念的污染，拆解训练集中的各个概念到不同的标签中，避免了训练集特征过度集中在某个提示词中导致的LoRA模型生成图像出现姿势呆板、表情僵硬、着装单一、和其他LoRA模型一起使用时出现大量的非正常色块甚至是噪点的情况。

【三】还原度

还原度是指LoRA生成的图像特征和数据集特征的相似度，是一个比较灵活的特征。

当我们训练的是人物LoRA模型时，通过特定的触发词即可生成高还原度的图像。

当我们训练的是风格LoRA模型时，我们通过丰富的触发词即可生成高还原度的图像。

2.4 零基础深入浅出理解LoRA模型的高阶用法

除了上面我们已经讲到的SD/FLUX模型+LoRA模型的基础使用形式，LoRA模型还有三种高阶用法：

我们可以调整LoRA模型使用时的权重。
使用多个LoRA模型同时作用于一个SD/FLUX模型。
使用LoRA技术微调训练SD/FLUX系列模型的Text Encoder。

【一】调整LoRA模型使用时的权重

首先，我们可以调整LoRA模型的权重：

W' = W + \alpha\Delta W \\

其中 \alpha 代表了LoRA模型的权重。下图展示了将\alpha从0缩放到1的过程中，不同权重LoRA模型对图像生成所产生的影响。

LoRA模型权重从0到1产生的效果

当我们将\alpha设置为0时，与只使用SD模型的效果完全相同；将\alpha设置为1时，与使用W' = W + \Delta W的效果相同。如果出现LoRA模型的效果过于强的情况，我们可以将\alpha设置为较低的值（比如0.2-0.3）。如果使用LoRA的效果不太明显，那我们可以将\alpha设置为略高于1的值（比如1.2-1.5）。

【二】多个LoRA模型同时作用

除了调整单个LoRA的权重，我们还可以使用多个LoRA模型同时作用于一个SD/FLUX模型，并配置他们的各自权重，我们拿两个LoRA模型举例： \Delta W = (\alpha_1 A_1 + \alpha_2 A_2) (\alpha_1 B_1 + \alpha_2 B_2) \\

其中 \alpha_{1} 代表第一个LoRA模型的权重， \alpha_{2} 代表第二个LoRA模型的权重。如果我们设置\alpha_{1} = \alpha_{2} = 0.5，我们就得到两个LoRA模型的权重平均值。如果我们设置\alpha_{1} = 0.8，\alpha_{2} = 0.3，那么第一个LoRA模型的效果会占据主导，如果我们设置\alpha_{1} = 0.1，\alpha_{2} = 0.8，那么第二个LoRA模型的效果会占据主导。

【三】LoRA技术微调训练SD/FLUX系列模型的Text Encoder

我们知道SD/FLUX系列模型中的Text Encoder部分也包含了大量的Attention结构，所以我们也可以使用LoRA技术微调训练SD/FLUX系列模型的Text Encoder，获得Text Encoder LoR模型。

整个训练流程和SD U-Net LoRA模型一致，在前向推理阶段，将Text Encoder LoR模型权重合并到SD/FLUX系列模型的Text Encoder中。

下图是用LoRA技术同时微调训练U-Net和Text Encoder时的训练参数对比图：

LoRA技术同时微调训练U-Net和Text Encoder时的训练参数对比图

在使用时，我们可以更加灵活地在SD/FLUX模型中加载LoRA模型权重：

SD/FLUX模型+ U-Net LoRA/Transformer LoRA模型 + Text Encoder LoRA模型
SD/FLUX模型+ U-Net LoRA/Transformer LoRA模型
SD/FLUX模型+ Text Encoder LoRA模型

三种不同的使用方式，生成图片的效果也会不同，下图是不同LoRA模型的不同权重组合在生成人像的效果：

不同LoRA模型的不同权重组合生成人像的效果
2.5 零基础深入浅出理解DreamBooth LoRA模型

目前在AI绘画开源社区非常繁荣的LoRA训练方法，其实很多都是基于DreamBooth技术的，这一点我们需要注意，其实DreamBooth LoRA训练方式和直接的LoRA Finetune训练方式还是有一定的区别的，为了防止大家的混淆，接下来Rocky将为大家详细的讲解DreamBooth技术以及DreamBooth LoRA的核心基础知识。

DreamBooth是由Google研究团队于2022年发布的一种通过将自定义主题和概念注入扩散模型的微调训练技术，它只借助少量数据集（3-5张图像）微调Stable Diffusion系列模型，让其能够学习稀有或个性化的图像特征。DreamBooth技术使得SD/FLUX系列模型能够在生成图像时，更加精确地反映特定的主题、对象或风格。

这个例子中，将[V]代表“红色书包”这个概念，同时用“书包”作为基础类别[Class]

DreamBooth首先为特定的概念寻找一个特定的描述词[V]作为编码载体，同时设置基础类别[Class]组成全新的数据标签，这个特定的描述词一般需要是稀有的（比如说设置“sks”作为上图中书包的描述词[V]），之所以选择稀有描述词，是希望SD模型没有该描述词的先验知识，否则该描述词容易在模型先验和新注入概念产生混淆。同时基础类别[Class]是对这个特定概念的粗粒度类的描述，通过将稀有描述词[V]和基础类别[Class]绑定（举个例子：上图中的书包可以设定为[V][Class] = "a sks bag"），AI绘画模型可以在基础类别[Class]的基础上再绑定稀有描述词[V]的特征。

Dreambooth技术效果图示

DreamBooth技术可以对SD/FLUX系列模型的U-Net部分进行微调训练，同时DreamBooth技术也可以和LoRA模型结合，用于训练DreamBooth_LoRA模型。

到这里，我们就弄清楚了AI绘画社区中对Dreambooth技术和LoRA技术的混淆与混乱。总的来说，Dreambooth技术一开始是SD/FLUX模型的一种微调技术，但是直接用几张图片微调整个SD/FLUX大模型有点杀鸡用牛刀了，同时LoRA作为SD/LFUX的一个扩展权重，天然的能够与Dreambooth技术适配，并且LoRA也足够轻量，所以才有了使用Dreambooth技术训练LoRA模型，从而诞生了DreamBooth LoRA这个工程优化的概念。

总的来说，本质上Dreambooth是一种训练技术，而LoRA模型是一种轻量化的模型权重。

同时，由于在少量数据上对模型微调容易产生“overfitting and language drift”问题，这里为了防止过拟合，设计了一个class-specific prior preservation loss（基于SD底模型生成相同Class的图像加入batch里面一起训练）来进行正则化。拿上面的书包例子，就是用SD模型生成一些普通书包的图像作为正则集。

正则化在机器学习中是一种用于防止模型过拟合的技术，通过在损失函数中添加惩罚项来约束模型的复杂度。具体到AI绘画领域，正则化有两种理解思路：

1. 锚定概念：在AI绘画模型训练前，先通过正则化预先固定一些关键的特征信息（图像信息、文本信息、条件信息等），使模型在训练过程中以这些先验信息为基础，进行优化学习。

2. 限制偏离：在训练过程中，正则化不断调整模型，防止其偏离预期轨道，限制模型的自由发展，确保生成图像的质量和一致性。

DreamBooth技术的原理示意图

到这里，Rocky可以帮大家总结一下DreamBooth技术的特点：

使用稀有描述词将特定主题注入SD/FLUX系列模型和LoRA系列模型中。
为了防止模型过拟合，使用class-specific prior preservation loss来正则化模型的训练过程。
DreamBooth技术能够在保持模型泛化能力的基础上，让模型学习到特定主题的特征。
如果我们不启用正则集数据和class-specific prior preservation loss，这时训练过程将和fine-tune微调训练一致。

目前diffusers库已经支持DreamBooth_LoRA的训练：diffusers/dreambooth

2.6 零基础深入浅出理解LoRA模型的融合和提取方法（Merge Block Weighted，MBW）

本小节中，Rocky向大家通俗易懂的讲解LoRA模型的融合和提取方式，这主要是为了获得全新的特定效果的LoRA模型。

在2022年AIGC时代元年，Stable Diffusion + LoRA的模型融合方法在开源社区一度火爆非凡。很多开源社区的AI绘画爱好者和AI绘画设计师对Stable Diffusion每一层的权重融合方式都进行了详细的实验与测试，总结了很多广为流传的实验经验，比如SD的哪几层控制主体、哪几层控制颜色、哪几层控制风格、哪几层控制手型等。

Stable Diffusion模型分层与LoRA模型融合生成新模型

一晃多年过去了，SD系列也连续发布了SD 2.x、SDXL、SD 3等大模型，同时FLUX.1、FLUX.1 Kontext、FLUX.2也相继发布，虽然我们不能将SD 1.x 的分层融合经验生搬硬套到最新的模型上，开源社区也没有更多精力对每一款AI绘画大模型进行如此详细的消融实验与测试，但是SD 1.x 的分层融合经验还是给我们提供了很多灵感与认识AIGC大模型的不同视角，以及会让我们想起AIGC元年大家的热情与兴奋，不知不觉也让我们哑然失笑。

到目前为止，LoRA模型的主流融合方式一共有以下两种：

LoRA+LoRA全融合/分层融合来获得新LoRA模型。
SD/SDXL/SD 3/FLUX模型与LoRA全融合/分层融合来获得新LoRA模型。

总的来说，当我们进行模型融合时，通常可以将SD模型的U-Net或者FLUX模型的Transformer进行分层，并对每一层进行特定的权重设置，如下图所示：

MBW模型融合分层示意图

除了LoRA模型的融合，我们还能通过提取的方式，在SD和FLUX.1系列模型中提取LoRA模型：

两个SD/SDXL/SD 3/FLUX模型差分提取新LoRA模型。
两个LoRA差分提取新LoRA模型。
3. 从0到1搭建使用LoRA模型进行AI绘画（全网最详细讲解）

目前能够加载LoRA模型并进行图像生成的主流AI绘画框架有四种：

diffusers框架
Stable Diffusion WebUI框架
ComfyUI框架
SD.Next框架

关于主流AI绘画框架核心基础知识的详细讲解，大家可以研读Rocky持续在撰写的文章：

为了方便大家使用主流AI绘画框架，Rocky这里也总结汇总了相关的资源，方便大家直接部署使用：

Stable Diffusion WebUI资源包可以关注公众号WeThinkIn，后台回复“WebUI资源”获取。
ComfyUI的500+高质量工作流资源包可以关注公众号WeThinkIn，并回复“ComfyUI”获取。
SD.Next资源包可以关注公众号WeThinkIn，后台回复“SD.Next资源”获取。

接下来，为了让大家能够从0到1搭建使用LoRA这个在当前开源生态中与Stable Diffusion/FLUX配合最多的的AI绘画模型，Rocky将详细的讲解如何用这四个AI绘画框架构建LoRA推理流程。那么，跟随着Rocky的脚步，让我们开始吧。

3.1 零基础使用diffusers搭建LoRA推理流程

diffusers是原生支持LoRA系列模型和Stable Diffusion/FLUX系列的推理的，所以在diffusers中能够非常高效的构建基于Stable Diffusion/FLUX的LoRA推理流程。但是由于diffusers目前没有现成的可视化界面，Rocky将在Jupyter Notebook中搭建完整的基于Stable Diffusion/FLUX的LoRA推理工作流，让大家能够快速的掌握。

首先，我们需要安装diffusers库，可以直接安装最新的diffusers版本，我们只需要在命令行中输入以下命令进行安装即可：

pip install diffusers==0.29.2
pip show transformers==4.42.4
# 安装peft库后，diffusers库既可以读取safetensor格式的LoRA模型，也可以读取原生diffusers格式的LoRA模型
pip install peft
pip install torch==2.0.1 torchvision==0.15.2

完成上面的依赖安装后，我们就可以进行LoRA模型的推理生成了。

# 加载diffusers和torch依赖库
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch

# 初始化SD模型，加载预训练权重，选择DPM采样器
pipe = DiffusionPipeline.from_pretrained("/本地路径/stable-diffusion-v1-5", use_safetensors=True, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# 使用GPU加速
pipe.to("cuda")

# 加载diffusers格式的原生LoRA模型
pipe.load_lora_weights("/本地路径/pokemon_LoRA")

# 大家也可以直接加载safetensor格式的LoRA模型
pipe.load_lora_weights("/本地路径/pokemon_LoRA.safetensors")

# 接下来，我们就可以运行pipeline了
prompt = "a girl"
negative_prompt = "(watermark), (signature), (sketch by bad-artist), (signature), (worst quality), (low quality), (bad anatomy), deformed hands, NSFW, nude"
image = pipe(prompt, 
             negative_prompt=negative_prompt,
             num_inference_steps=30,
             width=512,
             height=768,
             guidance_scale=7).images[0]

# 保存生成图像
image.save("LoRA-Test.png")

diffusers格式的原生LoRA模型权重文件夹中有很多子文件，很多朋友可能不太清楚每个文件的含义，Rocky这里再带着大家进行逐一的解读。

首先我们打开下载好的diffusers格式的LoRA模型权重文件夹，可以看到主要由以下几个部分组成：

optimizer.bin、pytorch_model.bin、random_states_0.pkl、scaler.pt、scheduler.bin。

3.2 零基础使用Stable Diffusion WebUI搭建LoRA推理流程

如果你是AI绘画领域的初学者，那么Stable Diffusion WebUI可以说是我们入门首选的最佳AI绘画框架了。

Stable Diffusion WebUI是AI绘画领域最为流行的框架，其生态极其繁荣，非常多的上下游插件能够与Stable Diffusion WebUI一起完成诸如AI绘画生成、AI视频生成、AI写真生成、AI虚拟换装等工作流，可玩性非常强，是非常高效的AIGC时代生产力工具。

接下来，大家就和Rocky一起，从0到1使用这个流行框架搭建基于Stable Diffusion的LoRA模型进行AI绘画吧。

首先，我们需要下载安装Stable Diffusion WebUI框架，我们只需要在命令行输入如下代码即可：

git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git

安装好后，我们可以看到本地的stable-diffusion-webui文件夹。

下面我们需要安装WebUI的依赖库，我们进入stable-diffusion-webui文件夹中，并进行以下的操作：

cd stable-diffusion-webui #进入下载好的stable-diffusion-webui项目中
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple some-package

完成了上面的基础配置，我们还需要配置Stable Diffusion WebUI的repositories插件，我们只需运行下面的代码即可：

sh webui.sh

#主要依赖包括：BLIP CodeFormer generative-models k-diffusion stable-diffusion-stability-ai stable-diffusion-webui-assets taming-transformers

如果发现repositories插件下载速度较慢，出现很多报错，don't worry，大家可以直接使用Rocky已经配置好的资源包，可以快速使用Stable Diffusion WebUI框架。Stable Diffusion WebUI资源包可以关注公众号WeThinkIn，后台回复“WebUI资源”获取。

在完成了依赖库和repositories插件的安装后，我们就可以配置模型了，我们将Stable Diffusion模型放到/stable-diffusion-webui/models/Stable-diffusion/路径下。同时将LoRA模型放到/stable-diffusion-webui/models/Lora/路径下。这样一来，等我们开启可视化界面后，就可以选择基于Stable Diffusion的LoRA模型用于推理生成图片了。

完成上述的步骤后，我们可以启动Stable Diffusion WebUI了！我们到/stable-diffusion-webui/路径下，运行launch.py即可：

python launch.py --listen --port 8888

运行完成后，可以看到命令行中出现的log：

To see the GUI go to: http://0.0.0.0:8888

我们将http://0.0.0.0:8888输入到我们本地的网页中，即可打开如下图所示的Stable Diffusion WebUI可视化界面，就可以愉快的使用基于Stable Diffusion的LoRA模型进行AI绘画了。

3.3 零基础使用ComfyUI搭建LoRA推理流程

ComfyUI是一个基于节点式的Stable Diffusion AI绘画框架。和Stable Diffusion WebUI相比，ComfyUI通过将AI绘画/AI视频生成的pipeline拆分成独立的节点，实现了更加精准的AI绘画/AI视频工作流定制和清晰的可复现性。

目前ComfyUI能够非常成熟的使用基于Stable Diffusion的LoRA模型，下面是Rocky使用ComfyUI来加载基于Stable Diffusion的LoRA模型并生成图片的完整Pipeline：

ComfyUI加载LoRA模型

当然，我们也可以同时加载多个LoRA模型，这时候的工作流如下所示：

ComfyUI加载LoRA模型

大家如果看了感觉复杂，不用担心，Rocky已经为大家保存了这两个工作流，大家只需关注Rocky的公众号WeThinkIn，并回复“ComfyUI”，就能获取这两个工作流以及文生图、图生图、图像Inpainting、ControlNet以及图像超分在内的所有基于Stable Diffusion的LoRA经典工作流json文件，大家只需在ComfyUI界面右侧点击Load按钮选择对应的json文件，即可加载对应的工作流，开始愉快的AI绘画之旅。

话说回来，下面Rocky将带着大家一步一步使用ComfyUI搭建基于Stable Diffusion的LoRA推理流程，从而实现上面两个工作流的生成过程。

首先，我们需要安装ComfyUI框架，这一步非常简单，在命令行输入如下代码即可：

git clone https://github.com/comfyanonymous/ComfyUI.git

安装好后，我们可以看到本地的ComfyUI文件夹。

ComfyUI框架安装到本地后，我们需要安装其依赖库，我们只需以下操作：

cd ComfyUI #进入下载好的ComfyUI文件夹中
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple some-package

完成这些配置工作后，我们就可以配置模型了，我们将Stable Diffusion系列模型放到ComfyUI/models/checkpoints/路径下。同时将LoRA模型放到ComfyUI/models/loras/路径下这样一来，等我们开启可视化界面后，就可以选择基于Stable Diffusion的LoRA模型进行AI绘画了。

接下来，我们就可以启动ComfyUI了！我们到ComfyUI/路径下，运行main.py即可：

python main.py --listen --port 8888

运行完成后，可以看到命令行中出现的log：

To see the GUI go to: http://0.0.0.0:8888

我们将http://0.0.0.0:8888输入到我们本地的网页中，即可打开如上图所示的ComfyUI可视化界面，愉快的使用LoRA模型生成我们想要的图片了。

3.4 零基础使用SD.Next搭建LoRA推理流程

SD.Next原本是Stable Diffusion WebUI的一个分支，再经过不断的迭代优化后，最终成为了一个独立版本。

SD.Next与Stable Diffusion WebUI相比，包含了更多的高级功能，同时兼容Stable Diffusion、 Stable Diffusion XL、Kandinsky,、DeepFloyd IF、PixArt-Σ、HunyuanDiT等20多种AI绘画大模型，是一个功能十分强大的AI绘画框架。

那么我们马上开始SD.Next的搭建与使用吧。

首先，我们需要安装SD.Next框架，这一步非常简单，在命令行输入如下代码即可：

git clone https://github.com/vladmandic/automatic

安装好后，我们可以看到本地的automatic文件夹。

SD.Next框架安装到本地后，我们需要安装其依赖库，我们只需以下操作：

cd automatic #进入下载好的automatic文件夹中
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple some-package

除了安装依赖库之外，还需要配置SD.Next所需的repositories插件，我们需要运行以下代码：

cd automatic #进入下载好的automatic文件夹中
python installer.py

如果发现extensions插件下载速度较慢，出现很多报错，大家可以直接使用Rocky已经配置好的资源包，可以快速启动SD.Next框架。SD.Next资源包可以关注公众号WeThinkIn，后台回复“SD.Next资源”获取。

在完成了依赖库和repositories插件的安装后，我们就可以配置模型了，我们将Stable Diffusion系列模型放到/automatic/models/Stable-diffusion/路径下。同时将LoRA模型放到/stable-diffusion-webui/models/Lora/路径下。这样一来，等我们开启可视化界面后，就可以选择基于Stable Diffusion的LoRA模型用于推理生成图片了。

完成上述的步骤后，我们可以启动SD.Next了！我们到/automatic/路径下，运行launch.py即可：

python launch.py --listen --port 8888

运行完成后，可以看到命令行中出现的log：

To see the GUI go to: http://0.0.0.0:8888

我们将http://0.0.0.0:8888输入到我们本地的网页中，即可打开如下图所示的SD.Next可视化界面，愉快的使用基于Stable Diffusion的LoRA模型进行AI绘画了。

3.5 LoRA生成图像示例

示例一：七龙珠人造人18号

底模型：YesMix

LoRA模型：Android 18

正向提示词：best quality, highres, and18, 1girl, android 18, solo, blonde hair, blue eyes, belt, jeans, pearl_necklace, bracelet, black gloves, white shirt, short hair, short sleeves, earrings, blue pants, open vest, black vest, large breasts, <lora:android_18_v110:0.5>, (ruins:1.3), (torn clothes:1.5), sitting, expressionless, crossed legs,
负向提示词：EasyNegative, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, (worst quality:1.2), low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, lowres graffiti, (low quality lowres simple background:1.1),

其他参数：Guidance：7；Steps：20；Sampler：DPM++ 2M Karras；Seed：515342412；Clip skip：2

示例二：美少女战士

底模型：Meichidark_Mix

LoRA模型：Sailor Venus

正向提示词：((sv1 | sailor senshi uniform | orange skirt | elbow gloves | tiara | orange sailor collar | red bow | orange choker | white gloves | jewelry | long hair | yellow hair)), (white silky skin | oily skin | large breasts | perfect hand | perfect arm), (smile | happy512 | good vibes), (cosmic background, shooting star, strong light beam, venus, venus \(planet\), stars, cloud), ((feh | light magic spell | spelling magic on hand | style-swirlmagic)), looking at viewer, cowboy shot, dynamic background and angle, ray tracing, exposure blend, cinematic bloom, deep contrast, sharp focus, detailed landscape, vivid colours, photorealistic, extremely detailed CG unity 8k wallpaper, high-res, beautiful and aesthetic, masterpiece, best quality, fantasy art, <lora:detailtweaker:0.5> <lora:Sweety_Mix_Girl:0.5> <lora:GoodHands-vanilla:1> <lora:Sailor_Venus:0.7> <lora:FEH_Concept_v2:1>
负向提示词：bad-hands-5, badhandv4, negative_hand-neg, bad-artist, bad-image-v2-39000, bad-picture-chill-75v, bad_pictures, bad_prompt, CyberRealistic_Negative-neg, easynegative, EasyNegativeV2, epiCNegative, FastNegativeEmbedding, BadDream, UnrealisticDream, ng_deepnegative_v1_75t, rmadanegative402_sd15-neg, verybadimagenegative_v1.3, ((abnormal anatomy | mutated hands and fingers | mutated legs | abnormal legs composition | deformed iris | deformed pupils | deformed | distorted | disfigured | stacked torsos | totem | extra limb | missing limb | abnormal neck | long neck | floating limbs | fused fingers | too many fingers | abnormal fingers composition | abnormal hands composition | extra fingers | distorted hands | missing hands | double hands | amputation | bad anatomy | wrong anatomy | gross proportions | everything about abnormal human anatomy composition)), censored, poorly drawn, letter, watermark, signature, greyscale, sepia, vignette, low quality, noise, jpeg artifacts, monochrome

其他参数：Guidance：7；Steps：50；Sampler：DPM++ 2M Karras；Seed：2411861598；Clip skip：2

示例三：新世纪福音战士

底模型：AnyLoraCleanLinearMix-ClearVAE

LoRA模型：Neon Genesis Evangelion

正向提示词：asukalangley, <lora:asuka langley soryu rebuild-lora-nochekaiser:1>, asuka langley soryu, (souryuu asuka langley:1.2), long hair, bangs, blue eyes, brown hair, hair ornament,BREAK bodysuit, pilot suit, plugsuit, (red bodysuit:1.5), interface headset,BREAK outdoors, city, sky, clouds, sun,BREAK looking at viewer, (cowboy shot:1.5),BREAK <lyco:GoodHands-beta2:1>, (masterpiece:1.2), best quality, high resolution, unity 8k wallpaper, (illustration:0.8), (beautiful detailed eyes:1.6), extremely detailed face, perfect lighting, extremely detailed CG, (perfect hands, perfect anatomy),
负向提示词：easynegative, ng_deepnegative_v1_75t, verybadimagenegative_v1.3, hair ornament,

其他参数：Guidance：7；Steps：25；Sampler：Euler a；Seed：1452471501

3.6 多个LoRA模型推理高阶融合（Multi-LoRA Composition）策略详解

我们在使用多个LoRA模型进行组合（如不同的角色、服装、风格、背景等）推理时，除了使用经典的Merge策略外，还可以使用Switch和Composite两种高阶组合策略。

在大量不同功能的LoRA模型组合推理时，通过Merge策略会损失一些LoRA的原本特征细节，甚至完全丢失某个LoRA的特征，使其完全失效。而Switch和Composite策略都会比Merge策略保留更多LoRA的原本特征，同时通过Switch和Composite策略生成的图像中的人物角色特征会比Merge策略要更自然。

多个LoRA的Merge、Switch和Composite推理组合策略示意图

Merge（经典融合方法）方法是最直接的融合方式，它将多个 LoRA 的权重同时激活并加权平均，在整个去噪过程中持续生效。具体代码实现如下所示：

if args.method == "merge":
    pipeline.set_adapters(cur_loras)  # 同时激活所有 LoRA
    switch_callback = None

在每个去噪步骤 t中，噪声预测为：

noise_pred = UNet(latent_t, prompt_embeds, LoRA₁ + LoRA₂ + ... + LoRAₙ)

Merge（经典融合方法）方法的优点是：

实现简单，计算开销小
生成速度快，只需一次前向传播

同时，其缺点是：

多个 LoRA 权重叠加容易产生冲突
难以精确控制每个元素的表现
容易出现某些特征被”淹没”的问题

Switch（轮流切换方法）方法通过在去噪过程中定期切换激活的LoRA模型，让每个LoRA模型轮流发挥作用，避免权重冲突。具体代码实现如下所示：

def make_callback(switch_step, loras):
    def switch_callback(pipeline, step_index, timestep, callback_kwargs):
        callback_outputs = {}
        # 每隔 switch_step 步切换一次 LoRA
        if step_index > 0 and step_index % switch_step == 0:
            for cur_lora_index, lora in enumerate(loras):
                if lora in pipeline.get_active_adapters():
                    # 切换到下一个 LoRA
                    next_lora_index = (cur_lora_index + 1) % len(loras)
                    pipeline.set_adapters(loras[next_lora_index])
                    break
        return callback_outputs
    return switch_callback
# example.py - 使用方式
if args.method == "switch":
    pipeline.set_adapters([cur_loras[0]])  # 先激活第一个 LoRA
    switch_callback = make_callback(switch_step=5, loras=cur_loras)

# 在生成时传入回调
image = pipeline(
    prompt=prompt,
    callback_on_step_end=switch_callback,  # 每步结束时检查是否需要切换
    ...
)

我们假设有 100 个去噪步骤，2 个 LoRA（character 和 clothing），switch_step（控制切换频率）= 5，这时整个推理流程可以如下所示：

步骤 0-4:    使用 LoRA_character
步骤 5-9:    切换到 LoRA_clothing
步骤 10-14:  切换到 LoRA_character
步骤 15-19:  切换到 LoRA_clothing
...循环往复

Switch（轮流切换方法）方法的优点是：

避免了权重叠加冲突
每个 LoRA 都有独立发挥作用的时间
通过调整 switch_step 可以控制融合程度
在真人风格中的稳定性比composite更好

其缺点是：

仍然是一次只用一个 LoRA，可能无法充分体现多个特征的协同效果
切换频率不好设置，可能导致特征不连贯

Composite（组合预测方法）方法则是在每个去噪步骤中分别用每个 LoRA 独立预测噪声，接着将所有预测结果取平均值，最后用平均后的噪声进行去噪。这样既避免了权重冲突，又能充分利用所有 LoRA 的信息。具体的代码实现如下所示：

# 在去噪循环中
if lora_composite:
    adapters = self.get_active_adapters()  # 获取所有激活的 LoRA

# 在每个去噪步骤中
if lora_composite:
    noise_preds = []
    self.enable_lora()
    # 分别用每个 LoRA 预测噪声
    for adapter in adapters:
        self.set_adapters(adapter)  # 切换到当前 LoRA
        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=self.cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        noise_preds.append(noise_pred)
else:
    # 普通方法：只预测一次
    noise_pred = self.unet(...)

# 进行 CFG（Classifier-Free Guidance）
if self.do_classifier_free_guidance:
    if lora_composite:
        noise_preds = torch.stack(noise_preds, dim=0)
        # 分离条件和非条件预测
        noise_pred_uncond, noise_pred_text = noise_preds.chunk(2, dim=1)
        # 关键：对所有 LoRA 的预测取平均
        noise_pred_uncond = noise_pred_uncond.mean(dim=0)
        noise_pred_text = noise_pred_text.mean(dim=0)
        # 应用 CFG
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

工作流程是在每个去噪步骤 t中进行如下操作：

1. 当前潜在变量 latent_t
2. 用 LoRA_character 预测 → noise_pred_1
3. 用 LoRA_clothing 预测 → noise_pred_2
4. 用 LoRA_style 预测 → noise_pred_3
5. 平均噪声 = mean(noise_pred_1, noise_pred_2, noise_pred_3)
6. 用平均噪声更新 latent_t → latent_{t-1}

Composite（组合预测方法）方法的优点是：

避免权重冲突：各 LoRA 独立预测，不会互相干扰
充分融合信息：通过平均综合所有 LoRA 的特征
稳定性好：平均操作具有降噪效果

其缺点是：

计算开销大：需要进行 n 次 UNet 前向传播（n 是 LoRA 数量）
生成速度慢：耗时是 Merge 方法的 n 倍
显存占用高：需要存储多个预测结果：

Rocky最后再对三种方法进行对比总结，让大家能够更好的理解三种LoRA模型推理融合方法的原理：

特性	Merge	Switch	Composite
激活方式	同时激活所有 LoRA	轮流激活单个 LoRA	分别激活每个 LoRA
前向传播次数	1次/步	1次/步	n次/步
计算开销	低	低	高（n倍）
生成速度	最快	速度和Merge相当	最慢
融合质量	一般	较好	最好
特征冲突	严重	较少	无
4. 从0到1上手训练自己的LoRA模型用于AI绘画（全网最详细讲解）

我们从AI绘画开源社区可以看到，除了Stable Diffusion/FLUX大模型之外，数量最多的就是LoRA模型了，每天都有大量的人物LoRA、风格LoRA模型以及功能性LoRA被开发者发布出来，极大繁荣了AI绘画领域的生态，说是“全民炼丹”也不为过。

LoRA模型以其训练速度快、训练数据量级小、训练容易达到好效果、算力资源要求低等特质，成为AIGC时代AI领域从业者、AI绘画师以及各行各业AI绘画学习者必备的AI绘画技术。

接下来，大家就跟着Rocky的脚步，一起来训练自己心仪的AI绘画LoRA模型吧！让我们将自己的心意注入LoRA模型中，一起来推动AIGC时代的发展以及AI绘画领域的持续繁荣！

本章节Rocky主要讲解以下两个主流框架来训练LoRA模型：

kohya-trainer框架
diffusers框架

其中kohya-trainer框架有丰富的参数可供我们调节优化，但是比较复杂，上手需要一定的时间。

与此同时，diffusers框架是支持LoRA训练的原生框架，训练参数简单明了，适合我们入门学习，容易上手。

Rocky将用这两个主流训练框架，详细的讲解LoRA Finetune和LoRA Dreambooth Train两种AI绘画领域热门的训练技术，希望能给大家带来帮助。

本章节中将主要讲解SD、SDXL、FLUX.1模型对应的LoRA训练教程，大家在深入浅出学习之后，可以举一反三对FLUX.1 Kontext LoRA、Z-Image LoRA、Qwen-Image LoRA、Qwen-Edit LoRA、Kolors LoRA、HunYuanDiT LoRA等进行微调训练，在这里Rocky也给出上述这些主流LoRA模型的训练资源与训练脚本，方便大家深入学习：

FLUX.1 Kontext/Z-Image LoRA训练资源：https://github.com/ostris/ai-toolkit
Qwen-Image/Qwen-Edit LoRA训练资源：https://github.com/FlyMyAI/flymyai-lora-trainer
Kolors LoRA训练资源：https://github.com/Kwai-Kolors/Kolors/tree/master/dreambooth
HunYuanDiT LoRA训练资源：https://github.com/Tencent-Hunyuan/HunyuanDiT/blob/main/lora/README.md

Rocky也为大家准备了详细完整的LoRA训练完整资源。LoRA系列训练资源（SD LoRA训练脚本、SDXL LoRA训练脚本、FLUX LoRA训练脚本、差异化LoRA训练脚本、第四章中宝可梦LoRA模型训练后权重等）百度云网盘：关注Rocky的公众号WeThinkIn，后台回复：LoRA训练资源，即可获得LoRA训练干货资源链接。

好的，接下来，就让我们开始学习吧！

4.1 LoRA训练数据集制作

首先，不管使用kohya-trainer框架还是diffusers框架，我们训练LoRA模型的第一步，都是收集数据和对数据进行预处理。

根据我们的训练目标不同，我们需要准备不同数量的训练集。如果我们要训练人物的LoRA模型，我们一般需要准备20-30张的对应角色图片；当我们要训练风格的LoRA模型时，一般需要准备100-300张甚至更多的对应风格数据。总的来说，不管是训练人物LoRA还是风格LoRA，在保证数据质量的前提下，数据量级越多越好。

下面是Rocky总结归纳的人物LoRA和风格LoRA各自所需的数据制作要求与数据预处理过程：

【1】人物数据集制作经验分享

收集整理人物的图片20-30张，数据要符合如下要求：

整体数据质量优质。
图片中能清晰展现出人物的全部特征，要包含人物的正脸、侧脸、全身、站姿、坐姿、动作姿态等。
除了人物特征外，图片中的背景、场景、风格、意境等特征越多样越好。

【2】风格数据集制作经验分享

收集整理风格的图片100-300张，数据要符合如下要求：

整体数据质量优质。
数据集中的风格特征要统一。
在风格特征统一的基础上，图片中的人物、场景等内容越多样越好。

【3】数据标注&标签工程

最好我们能够使用模型现有的标签来作为触发标签，而不是设置一个全新的标签。因为现有的标签经过底模型海量数据的训练，已经形成较好的基础概念，我们将其作为触发标签，能够更好更快地学习相应概念的延展。

举个例子，比如说我们要训练一个3D风格的LoRA模型，我们可以直接将“3D”和“realistic”这些现有的标签作为触发词。

除了触发词外，我们一般还需要使用标注模型（BLIP、WD Tagger、Janus、moondream2等）对数据集进行自动标注，所以我们还需要对这些自动标注进行纠正与纠偏，增删改减自动标签使其能够正确完整的描述图片的内容。

4.2 使用kohya-trainer框架训练LoRA模型

关于使用Kohya-trainer框架训练Stable Diffusion、Stable Diffusion XL、Stable Diffusion 3以及FLUX.1系列的LoRA模型，Rocky在下面的系列文章中已经全方位的详细讲解介绍过了，并给出了保姆级的LoRA训练教程（全网最详细讲解），大家可以直接阅读：

FLUX.1/Stable Diffusion 3 LoRA模型训练保姆级教程请看：

Stable Diffusion XL LoRA模型训练保姆级教程请看：

Stable Diffusion LoRA模型训练保姆级教程请看：

同时，Rocky在下一章节中将对每一个LoRA训练参数进行深入浅出的解析，大家可以将上面的系列文章与本文配合起来研读学习。

4.3 kohya-trainer框架中LoRA训练参数全解析（全网最详细）

下面是kohya-trainer框架中LoRA的全部可调整训练参数：

pretrained_model="/本地路径/SD/SDXL/SD 3/FLUX模型（safetensors格式）" # base model path | 底模路径
is_v2_model=0                                                     # SD2.0 model | SD2.0模型 2.0模型下 clip_skip 默认无效
v_parameterization=0                                              # parameterization | 参数化 v2 非512基础分辨率版本必须使用。
vae=""                                     # 加载单独的VAE模型
train_data_dir="/本地路径/train-datasets"          # train dataset path | 训练数据集路径
reg_data_dir=""              # directory for regularization images | 正则化数据集路径，默认不使用正则化图像。
training_comment="LoRA_model_credit_from_Rocky" # training_comment | 训练介绍，可以写作者名或者使用触发关键词

# Train related params | 训练相关参数
resolution="1024,1024" # image resolution w,h. 图片分辨率，宽,高。支持非正方形，但必须是 64 倍数。
batch_size=1           # batch size
vae_batch_size=4       # vae初始化转换图片批处理大小，2-4。大了可以让一开始处理图片更快
max_train_epoches=8    # max train epoches | 最大训练 epoch
save_every_n_epochs=2  # save every n epochs | 每 N 个 epoch 保存一次

gradient_checkpointing=1      # 梯度检查，开启后可节约显存，但是速度变慢
gradient_accumulation_steps=0 # 梯度累加数量，变相放大batchsize的倍数

network_dim=128   # network dim | 常用 4~128，不是越大越好
network_alpha=64 # network alpha | 常用与 network_dim 相同的值或者采用较小的值，如 network_dim的一半 防止下溢。默认值为 1，使用较小的 alpha 需要提升学习率。

#dropout
network_dropout="0"                # dropout 是机器学习中防止神经网络过拟合的技术，建议0.1~0.3
scale_weight_norms="1.0"           # 配合 dropout 使用，最大范数约束，推荐1.0
rank_dropout="0"                   # lora模型独创，rank级别的dropout，推荐0.1~0.3，未测试过多
module_dropout="0"                 # lora模型独创，module级别的dropout(就是分层模块的)，推荐0.1~0.3，未测试过多
caption_dropout_every_n_epochs="0" # dropout caption
caption_dropout_rate="0"           # 0~1
caption_tag_dropout_rate="0.1"     # 0~1
max_grad_norm="1.0"

train_unet_only=0         # train U-Net only | 仅训练 U-Net，开启这个会牺牲效果大幅减少显存使用。6G显存可以开启
train_text_encoder_only=0 # train Text Encoder only | 仅训练 文本编码器

seed="1026" # reproducable seed | 设置跑测试用的种子，输入一个prompt和这个种子大概率得到训练图。可以用来试触发关键词

#噪声
noise_offset="0"                 # noise offset | 在训练中添加噪声偏移来改良生成非常暗或者非常亮的图像，如果启用，推荐参数为0.1
adaptive_noise_scale="0"         # adaptive noise scale | 自适应噪声偏移范围
noise_offset_random_strength=0   # 0是关，1是开。噪声随机强度
multires_noise_iterations="0"    # 多分辨率噪声扩散次数，推荐6-10,0禁用,和noise_offset冲突，只能开一个
multires_noise_discount="0"      # 多分辨率噪声缩放倍数，推荐0.1-0.3,上面关掉的话禁用。
min_snr_gamma="0"                # 最小信噪比伽马值，减少低step时loss值，让学习效果更好。推荐3-5，5对原模型几乎没有太多影响，3会改变最终结果。修改为0禁用。
weighted_captions=0              # 权重打标，默认识别标签权重，语法同webui基础用法。例如(abc), [abc], (abc:1.23),但是不能再括号内加逗号，否则无法识别。
ip_noise_gamma="0"               # 误差噪声添加，防止误差累计
ip_noise_gamma_random_strength=0 # 0是关，1是开。误差噪声随机强度
debiased_estimation_loss=1       # 0是关，1是开。信噪比噪声修正，minsnr高级版

#标签编辑
shuffle_caption=1           # 随机打乱tokens顺序，默认启用。修改为 0 禁用。
keep_tokens=1               # keep heading N tokens when shuffling caption tokens | 在随机打乱 tokens 时，保留前 N 个不变。
prior_loss_weight="1"       # 正则化权重，0-1
secondary_separator=";;;"   # 次要分隔符。被该分隔符分隔的部分将被视为一个token，并被洗牌和丢弃。然后由 caption_separator 取代。例如，如果指定 aaa;;bbb;;cc，它将被 aaa,bbb,cc 取代或一起丢弃。
keep_tokens_separator="|||" # 批量保留不变，间隔符号
enable_wildcard=0           # 通配符随机抽卡，格式参考 {aaa|bbb|ccc}
caption_prefix=""           # 打标前缀，可以加入质量词如果底模需要，例如masterpiece, best quality,
caption_suffix=""           # 打标后缀，可以加入相机镜头如果需要，例如full body等

# Learning rate | 学习率
lr="2e-6"
unet_lr="8e-4"
text_encoder_lr="1e-5"
lr_scheduler="" # "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
lr_warmup_steps=0                 # warmup steps | 仅在 lr_scheduler 为 constant_with_warmup 时需要填写这个值
lr_scheduler_num_cycles=1                 # cosine_with_restarts restart cycles | 余弦退火重启次数，仅在 lr_scheduler 为 cosine_with_restarts 时起效。

# Output settings | 输出设置
output_name="qinglong" # output model name | 模型保存名称
save_model_as="safetensors"      # model save ext | 模型保存格式 ckpt, pt, safetensors
mixed_precision="bf16"           # 默认fp16,可选 "fp16", "bf16","no"
save_precision="bf16"            # 默认fp16,可选 "fp16", "bf16","fp32"
full_fp16=0                      # 半精度全部使用fp16
full_bf16=1                      # 半精度全部使用bf16
fp8_base=1                       # 实验性功能FP8训练
cache_latents=1                  # 缓存潜变量
cache_latents_to_disk=1          # 开启缓存潜变量保存到磁盘，这样下次训练不用再次缓存转换，速度更快
no_half_vae=0                    # 禁止半精度，防止黑图。无法和mixed_precision混合精度共用。

#保存状态
save_state=0              # save training state | 保存训练状态 名称类似于 <output_name>-??????-state ?????? 表示 epoch 数
resume=""                 # resume from state | 从某个状态文件夹中恢复训练 需配合上方参数同时使用 由于规范文件限制 epoch 数和全局步数不会保存 即使恢复时它们也从 1 开始 与 network_weights 的具体实现操作并不一致
save_state_on_train_end=0 # 只在训练结束最后保存训练状态

# wandb
wandb_api_key="xxxxxxx"
log_tracker_name=$output_name

# Sample output | 出图
enable_sample=1                          # 开启出图
sample_every_n_epochs=2                  # 每n个epoch出一次图
sample_prompts="./toml/qinglong.txt"     # prompt文件路径
sample_sampler="euler_a"                 # 采样器 'ddim', 'pndm', 'heun', 'dpmsolver', 'dpmsolver++', 'dpmsingle', 'k_lms', 'k_euler', 'k_euler_a', 'k_dpm_2', 'k_dpm_2_a'

# 其他设置
network_weights=""               # pretrained weights for LoRA network | 若需要从已有的 LoRA 模型上继续训练，请填写 LoRA 模型路径。
enable_bucket=1                  # arb for diff wh | 分桶
min_bucket_reso=512              # arb min resolution | arb 最小分辨率
max_bucket_reso=1536             # arb max resolution | arb 最大分辨率
persistent_data_loader_workers=1 # persistent dataloader workers | 容易爆内存，保留加载训练集的worker，减少每个 epoch 之间的停顿
clip_skip=1                      # clip skip | 玄学 SD1.5用 2
multi_gpu=0                      # multi gpu | 多显卡训练开关，0关1开， 该参数仅限在显卡数 >= 2 使用
torch_compile=0                  # 使用torch编译功能，需要PyTorch版本大于2.1，训练速度提升10-30%，首次编译会慢，后续加速明显
dynamo_backend="aot_eager"       # aot_eager: 最稳定，推荐；inductor: 最快，可能不稳定；cudagraphs: 适合固定batch size

# 优化器设置
#use_8bit_adam=1                 # use 8bit adam optimizer | 使用 8bit adam 优化器节省显存，默认启用。部分 10 系老显卡无法使用，修改为 0 禁用。
#use_lion=0                      # use lion optimizer | 使用 Lion 优化器
optimizer_type="AdamWScheduleFree" 
# "adafactor","AdamW8bit","Lion","DAdaptation",  推荐新优化器Lion。推荐学习率unetlr=lr=6e-5,tenclr=7e-6
# 新增优化器"Lion8bit"(速度更快，内存消耗更少)、"DAdaptAdaGrad"、"DAdaptAdan"(北大最新算法，效果待测)、"DAdaptSGD"
# 新增优化器 Sophia(2倍速1.7倍显存)、Prodigy天才优化器，可自适应Dylora
# 新增优化器 AdamWScheduleFree、SGDScheduleFree
d0="4e-7"             # d0 | prodigy的初始学习率 4e-7
fused_backward_pass=0 # use fused backward pass | 使用融合后的反向传播,训练大模型float32精度专用节约显存，必须优化器adafactor或者adamw，gradient_accumulation_steps必须为1或者不开。

# lycoris 训练设置
enable_lycoris_train=0 # enable lycoris train | 启用 LoCon 训练 启用后 network_dim 和 network_alpha 应当选择较小的值，比如 2~16
conv_dim=8             # conv dim | 类似于 network_dim，推荐为 4
conv_alpha=1           # conv alpha | 类似于 network_alpha，可以采用与 conv_dim 一致或者更小的值
algo="lokr"            # algo参数，制定训练lycoris模型种类，包括lora(locon)、loha、IA3以及lokr、dylora 。5个可选
dropout="0"            # lycoris专用dropout
preset="attn-mlp"      # 预设训练模块配置

factor=8     # 只适用于lokr的因子，-1~8，8为全维度
block_size=4 # 适用于dylora,分割块数单位，最小1也最慢。一般4、8、12、16这几个选
use_tucker=1 # 适用于除 (IA)^3 和full
use_scalar=1 # 根据不同算法，自动调整初始权重
train_norm=1 # 归一化层

# dylora 训练设置
enable_dylora_train=0 # enable dylora train | 启用 LoCon 训练 启用后 network_dim 和 network_alpha 应当选择较小的值，比如 2~16
unit=4                # block size

#Lora_FA
enable_lora_fa=0 # 开启lora_fa，和lycoris、dylora冲突，只能开一个。

#oft
enable_oft=0 # 开启oft，和已上冲突，只能开一个。

# Merge lora and train | 差异提取法
base_weights=""
base_weights_multiplier="1.0"

# Block weights | 分层训练
enable_block_weights=0                         # 开启分层训练
down_lr_weight="1,0.2,1,1,0.2,1,1,0.2,1,1,1,1" # 12层，需要填写12个数字，0-1.也可以使用函数写法，支持sine, cosine, linear, reverse_linear, zeros，参考写法down_lr_weight=cosine+.25
mid_lr_weight="1"                              # 1层，需要填写1个数字，其他同上。
up_lr_weight="1,1,1,1,1,1,1,1,1,1,1,1"         # 12层，同上上。
block_lr_zero_threshold=0                      # 如果分层权重不超过这个值，那么直接不训练。默认0。

enable_block_dim=0                                                                           # 开启分块dim训练
block_dims="64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64"      # dim分块，25块
block_alphas="1,1,2,1,2,2,4,1,1,4,4,4,1,4,1,4,2,1,1,4,1,1,1,4,1"                             # alpha分块，25块
conv_block_dims="32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32" # convdim分块，25块
conv_block_alphas="1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"                        # convalpha分块，25块

# SDXL
min_timestep="0"                     # 最小时序，默认值0
max_timestep="1000"                  # 最大时序，默认值1000
bucket_reso_steps="32"               # default 64,SDXL can use 32
cache_text_encoder_outputs=0         # 开启缓存文本编码器，开启后减少显存使用。但是无法和shuffle共用
cache_text_encoder_outputs_to_disk=0 # 开启缓存文本编码器到磁盘，开启后减少显存使用。但是无法和shuffle共用

#checkpoint train
no_token_padding=0           # 不进行分词器填充
stop_text_encoder_training=0 # 在N步后停止文本编码器训练
train_text_encoder=1         # 训练文本编码器
learning_rate_te="5e-8"      # 文本编码器学习率 SD1.5/SD2.1

#SDXL_db
diffuser_xformers=0 # 开启diffuser的xformers
learning_rate_te1="5e-8" # 文本编码器1学习率
learning_rate_te2="5e-8" # 文本编码器2学习率

# block lr | SDXL_DB分层训练
enable_block_lr=0
block_lr="0,$lr,$lr,0,$lr,$lr,0,$lr,$lr,0,$lr,$lr,$lr,$lr,$lr,$lr,$lr,$lr,$lr,$lr,$lr,$lr,0"

#GradFilter EMA（快速拟合）
gradfilter_ema_alpha="0.98"  # 推荐0.98
gradfilter_ema_lamb=2.0
# 梯度指数移动平均滤波，减少梯度噪声，加速收敛

下面的章节中，Rocky主要讲解在diffusers框架下的LoRA训练流程和通用的LoRA训练技巧与经验。

4.4 使用diffusers框架训练LoRA模型

我们可以直接使用diffusers库里的代码进行LoRA模型的训练。

我们通过以下命令下载diffusers库代码和安装diffusers库运行所需要的依赖：

git clone https://github.com/huggingface/diffusers.git

pip install --upgrade diffusers[torch] -i https://pypi.tuna.tsinghua.edu.cn/simple some-package

完成diffusers库代码的下载和所需依赖的安装后，我们就可以开始使用diffusers库代码来进行SDXL LoRA模型和SD LoRA模型的训练了！

在这里Rocky选用个人非常喜欢的宝可梦数据集作为训练数据，并让LoRA模型能够学习到所有宝可梦的特征信息，并能生成我们想要的独一无二的“新宝可梦”！

我们首先需要准备数据，包括图片和对应的caption：

数据集格式：图片+caption

由于本次的宝可梦数据集已经封装好了，所以我们可以直接调用数据集路径即可，若是我们想要训练自己的数据集，需要制作diffusers能够读取的数据格式：

#数据格式 metadata.jsonl + 图片
folder/train/metadata.jsonl #存储caption描述
folder/train/0001.png
folder/train/0002.png
folder/train/0003.png

#metadata.json中的内容
{"file_name": "0001.png", "text": "This is a golden retriever playing with a ball"}
{"file_name": "0002.png", "text": "A german shepherd"}
{"file_name": "0003.png", "text": "One chihuahua"}

好的，完成了数据集的制作，我们就可以进行训练了，我们可以使用diffusers库中的diffusers/examples/text_to_image/train_text_to_image_lora.py作为训练代码，并传入相关的预设参数：

export MODEL_NAME="/本地路径/stable-diffusion-v1-5" # 选用的主模型
export OUTPUT_DIR="/本地路径/LoRA_model/pokemon" # LoRA模型保存地址
export DATASET_NAME="/本地路径/pokemon-blip-captions" # 训练数据路径

accelerate launch --mixed_precision="fp16"  train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \ # Batch-Size
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \ # 总的训练步数
  --learning_rate=1e-04 \ # 学习率
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --checkpointing_steps=500 \
  --seed=2048

到这里，我们就完成了使用diffusers库训练LoRA模型的流程。

4.5 LoRA模型的关键训练超参数详解（全网最详细）

【1】train_batch_size对LoRA模型训练的影响

和传统深度学习时代一样，train_batch_size即为训练时的batch size，表示一次性送入LoRA模型进行训练的图片数量。

从AI绘画的角度来解释Batch Size概念，可以把它比喻为一个画家在创作一系列作品时，每次着手处理的画作数量。

想象一下，一位画家接到了一批绘画任务，这些任务代表了机器学习中的整个数据集。如果画家一次只画一幅画，那么这就类似于机器学习中的“Batch Size”为1，这意味着每次只处理一个数据点（即一幅画）。这种方式可以让画家集中精力仔细处理每一幅画，但进度可能会比较慢，而且每幅画之间无法相互借鉴。

另一方面，如果画家选择同时着手处理多幅画作，比如说一次画六幅，这就相当于“Batch Size”为6。这样的话，画家可以在一定程度上提高效率，同时对比和学习不同画作之间的共同点和差异。在机器学习中，较大的“Batch Size”可以加快训练进度，并有助于模型更好地理解数据集的总体特征。

但是，如果“Batch Size”太大，比如一次处理60幅画，虽然看起来效率很高，但画家可能就难以专注于每一幅画的细节，可能会错过一些重要的元素。同样，在机器学习中，过大的“Batch Size”可能导致模型无法充分学习每个数据点的特性，影响最终的学习效果。

因此，选择合适的“Batch Size”就像画家需要找到同时处理多幅画作的最佳数量一样，既要保证质量，又要考虑效率。

考虑到我们训练LoRA模型时一般数据量级不会太大（1-300张为主），所以我们可以设置Batch Size为2-6即可。

【2】学习率对LoRA模型训练的影响

从AI绘画的角度来解释学习率概念，可以将其比喻为一位艺术家在绘画学习过程中调整绘画技巧的速度和程度。

我们想象一位画家正在学习如何绘制风景画。如果学习率很高，这就像是画家在每次尝试后都大幅度地改变他的绘画风格或技术。这样做的好处是，如果画家的当前风格距离理想的风格很远，他可以快速做出调整。但是，风险在于，如果调整过于激烈，可能会导致画作的风格不稳定，或者画家可能会在找到最佳风格之前就越过它错失了。

相反，如果学习率很低，那就像是画家在每次练习后只是微小地调整他的技术。这种方式较为稳妥，可以细致地探索不同的绘画风格，但进步的速度可能会很慢。如果画家的当前方法离理想风格相差甚远，他可能需要很长时间才能达到理想状态。

在机器学习中，学习率决定了模型权重调整的幅度。过高的学习率可能导致模型在最优解周围震荡，甚至发散，而过低的学习率会使训练过程缓慢，并且可能陷入局部最优解。因此，选择合适的学习率就像画家找到调整绘画技巧的合适速度一样重要。

通常来说，在我们训练SD/SDXL LoRA时，将学习率设置为1e-4就够了。

在LoRA的训练中，一共有三种学习率参数，分别是：

learning_rate：学习率，在没有指定U-Net 学习率和text_encoder学习率时生效。
unet_lr：设置U-Net的学习率，默认值是1e-5。
text_encoder_lr：设置Text_Encoder的学习率，一般取定值5e-6，也可以设置成成unet_lr的8-15分之一，调低该学习率参数有助于让Text_Encoder对训练集中的标签更敏感。

同时我们还可以根据LoRA模型的架构，针对A矩阵和B矩阵使用不同学习率。

B矩阵学习率 = A矩阵学习率 \times ratio \\

这样可以达到加速收敛，训练速度提升 1.5-2倍；并且Loss下降更快，提升整体训练效果。

【3】底模型（SD/SDXL/SD 3/FLUX系列模型）对LoRA模型训练的影响

从AI绘画的角度来解释使用预训练模型SD系列模型进行微调的过程，可以将其比喻为一位艺术家在学习绘画时先学习和模仿大师的作品，然后再根据自己的风格和需求进行调整和创新。

想象一位初学者画家想要成为一名肖像画家。他开始时可能没有足够的技能和知识从零开始画出精美的肖像。这时，他可能会先研究并模仿历史上著名的肖像画大师的作品，比如达芬奇或梵高的画作。通过这个过程，他学习了基础的构图、颜色使用、光影处理等技能。这就类似于在机器学习中使用一个预训练的模型，该模型已经被训练能够生成非常多的内容。

然而，这位画家可能会发现，虽然他现在能够模仿这些大师的风格，但这些风格可能并不完全符合他自己的个人表达或他想要描绘的特定主题。因此，他开始对这些技巧进行微调，调整色彩组合以更好地表达他自己的感觉，或改变构图来适应他的主题。这就类似于在机器学习中对预训练模型进行微调，即在特定的数据集或特定任务上进行额外的训练，以使模型更好地适应特定的应用场景。

通过这个过程，画家不仅节省了学习基础技能的时间，还能在大师的基础上发展出自己独特的风格。同样，在机器学习中，使用预训练模型进行微调可以显著减少所需的训练时间和数据，同时提高模型在特定任务上的表现。

当我们在训练LoRA模型时，也需要根据任务场景选择合适的预训练底模型，比如二次元、卡通、真人、风格、概念、国风、科幻、设计、游戏、摄影、风景、建筑、服装、动物、3D、2.5D等。

那我们该如何更好的选择合适的底模型呢？

Rocky总结了以下几点经验：

要选择和训练集的分布/风格接近的底模型，如果一时无法找到合适的底模型，可以使用官方的Stable Diffusion 1.5、Stable Diffusion 2.1、Stable Diffusion XL、Stable Diffusion 3、FLUX系列做为底模型，其泛化性能较好。
目前开源社区有微调训练得到的SD/FLUX系列模型和权重融合得到的合成SD/FLUX系列模型，我们优先使用微调训练得到的SD/FLUX系列模型作为底模型，其泛化性能比合成SD/FLUX系列模型要好，与此同时，合成SD/FLUX系列模型在细节相似度上可能会表现的更好。
LoRA模型在与底模型相似分布的其他底模型上生成图片的效果较好，比如使用2.5D的底模型训练了一个LoRA，在其他2.5D的SD模型上也会有较好的效果，但是在写实风格的SD/FLUX模型上效果可能一般。

【4】训练轮数（Epoch）对LoRA模型训练的影响

从AI绘画的角度来解释epoch概念，可以把它比喻为一位艺术家在创作过程中对作品的一次完整的审视和修正周期。

想象一位画家正在画一幅风景画。在这个过程中，他不是只画一遍就完成了。相反，他可能会多次回顾和修改画作，每一次都在整个画布上增加细节、调整颜色平衡或改善光影效果。每完成一次这样的过程，就相当于完成了一次epoch。

在机器学习中，一个epoch指的是训练算法在整个训练数据集上的一次完整训练迭代。就像画家在每一次审视中都会观察和修改画布上的每一部分，机器学习模型在一个epoch中会评估并调整其参数以更好地学习数据集中的特征和模式。

多个epochs对于机器学习模型至关重要，因为它们允许模型多次从错误中学习，逐渐提高其对数据的理解。这就像画家通过多次修正作品，不断提高作品的质量一样。然而，就像绘画一样，过多的epochs并不总是好事，可能会导致“过度训练”，就像画家过度修饰画作可能会损害其原始美感一样。因此，选择适当数量的epochs是优化机器学习模型性能的关键。

通常来说，我们在训练LoRA模型时，epoch的设置能够保证数据集中的每张图片训练100步即可。

【5】clip_skip参数对LoRA训练的影响

我们都知道，在SD模型和LoRA模型的训练和推理中，CLIP模型的输出作为SD系列中U-Net模型的输入，设置clip_skip=2表示使用CLIP的倒数第二层特征输出作为后续的输入，设置clip_skip=1表示使用CLIP的最后一层输出作为后续的输入。

NovelAI的实验证明，使用CLIP的倒数第二层特征能让LoRA模型更好的学习输入标签的文本信息，LoRA模型能够更快地学习一些概念特征。当使用最后一层的特征时，LoRA模型在训练过程中有可能还是无法理解不同概念和颜色方面的区分，因为CLIP的参数权重值在最后一层突然变化，很多细节特征都丢失了。

所以当我们训练LoRA模型时，可以设置clip_skip=2。

注意：当我们训练SD V2.0/V2.1模型时，默认使用CLIP倒数第二层的输入，这时我们需要省略clip_skip参数。

【6】network_dimension参数对LoRA训练的影响

network_dimension是LoRA模型的特征维度参数，也就是上文讲到的LoRA进行低秩分解中的秩。

那么我们该怎么设置network_dimension这个参数呢？

在这里Rocky也向大家分享一些经验：

当我们使用的训练集内容比较复杂抽象时，我们可以适当设置高network_dimension，提升LoRA模型的学习能力。
network_dimension参数并不是设置的越高越好。设置高network_dimension有助于LoRA模型学习到更多细节特征，但整体上模型的收敛速度变慢，需要的训练时间更长，同时也更容易过拟合。
当我们训练高分辨率（1024x1024）的训练集时，可以设置高network_dimension，比如说128。
训练人物时，可以设置network_dimension = 32-64；训练风格时，可以设置network_dimension = 64-128；训练抽象概念时，可以设置network_dimension = 128-192或者更高。
通常来说，我们设置network_dimension = 128能够兼顾到各个情况，算是一个比较平均的值，有效减少过拟合与欠拟合发生的概率。当network_dimension = 128时，SD LoRA保存下来的文件大小为144MB。

【7】alpha参数

alpha参数将LoRA模型权重进行放缩从而防止下溢并稳定学习， 我们可以将其设置为network_dimension参数的一半。

使用alpha参数后，LoRA模型的参数W = W_{in} \times alpha / dim。alpha参数设置的越大，LoRA模型越倾向于拟合更多的细节，学习速率也越快。在原生LoRA模型中可以取值[1,dim]，在LoCon或者LoHa模型中，推荐取值 [\frac{dim}{2},dim] 。

self.scale = alpha / self.lora_dim
def forward(self, x):
        if self.region is None:
            return self.org_forward(x) + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale

【8】scheduler、cycle、lr_warmup_steps以及optimizer_type参数

scheduler（学习率调度器）：学习率调度器用于调整优化器的学习率。在训练过程中，学习率调度器能够动态地改变学习率。常见的调度策略包括分段常数衰减、指数衰减、余弦退火（"cosine_with_restarts", "cosine", "polynomial", "constant", "constant_with_warmup", "linear"）等，推荐使用 cosine_with_restarts，它会使学习率从高到低下降，变化速度先慢后快再慢。
cycle（周期）：在一些周期性调度器（如余弦退火）中，cycle参数能够指定学习率变化的周期性，一般可以设置为4-8。例如在余弦退火调度器中，学习率会按照一个余弦函数的周期性变化，这种变化可以帮助模型在训练过程中找到更好的局部最优解，避免过早收敛到次优解。
lr_warmup_steps（学习率预热步骤）：学习率预热是一种在训练初期设置一个小学习率并逐渐增大的策略。这样做的好处是可以在训练初期防止模型过早发散，同时让优化过程更平稳地进入主要的学习率调度阶段。
optimizer_type（优化器类型）：优化器是用于更新模型权重以拟合损失函数的算法。常见的优化器包括SGD（随机梯度下降）、Adam（自适应矩估计）、RMSprop等。不同的优化器有不同的优化策略和适用场景，例如SGD通常适用于大型数据集和深层网络，它有助于防止模型过早收敛到局部最小值。而Adam优化器则结合了动量和自适应学习率的特点，通常在小数据集上表现更好。如果我们的显卡显存不够，可以使用AdamW8bit，能一定程度上降低模型训练时的显存占用。
--optimizer_args 选项用于指定优化器选项参数。可以以key=value的格式指定多个值，用逗号分隔，比如说optimizer_args = [ "scale_parameter=False", "relative_step=False", "warmup_init=False",]。

【9】gradient_checkpointing

开启gradient_checkpointing能够减少显存占用，从而能够设置更大的Batch Size，但是会减慢训练速度。开启gradient_checkpointing后，逐步更新模型权重而不是在训练期间一次更新所有模型权重。

【10】--xformers / --mem_eff_attn

开启--xformers参数后，训练时的显存占用大幅下降，非常好用！

【11】prior_loss_weight

当我们使用正则数据集来约束LoRA模型的训练时，设置prior_loss_weight来控制先验知识的正则化强度，默认为1。当使用100张以上正则数据集时，可以设置为0.05-0.1。正则化权重过高比如1时，LoRA模型的收敛难度会增大很多。一般在训练画风时推荐使用正则化。

【12】训练分辨率（training resolution）对LoRA训练的影响

总的来说，如果显卡的显存允许，尽量选择大分辨率进行训练。在较低的分辨率（比如512）下，训练数据中的很多细节特征将会被压缩丢失，甚至人物脸部特征的崩坏。

当我们使用大分辨率（比如768、1024等）训练时，LoRA模型能够学习到数据集中的精细特征和细节信息，图片的生成质感、美感以及精细度都会大大增强。

但是当数据集中人物脸部和手部等特征在图片中的占比本身就非常小时（比如一些人物远景图像），即使使用大分辨率训练，对图像生成的效果提升也不大。这种情况下最好的办法是删除这些不合格的低质量远景图片，能从根源上解决这个问题。

【13】长宽比分桶训练策略（Aspect Ratio Bucketing）详解

目前AI绘画开源社区中很多的LoRA模型和Stable Diffusion模型都是基于单一图像分辨率（比如1:1）进行训练的，这就导致当我们想要生成不同尺寸分辨率的图像（比如1:2、3:4、4:3、9:16、16:9等）时，非常容易生成结构崩坏的图像内容。

如下图所示，为了让所有的数据满足特定的训练分辨率，会进行中心裁剪和随机裁剪等操作，这就导致图像中人物的重要特征缺失：

骑士头戴皇冠的图片，但是由于裁剪丢失了图片黑色部分的重要信息

这上面这种情况下，我们训练的LoRA模型和Stable Diffusion模型在生成骑士图像的时候，就会出现缺失的骑士特征。

与此同时，裁剪后的图像还会导致图像内容与标签内容的不匹配，比如原本描述图像的标签中含有“皇冠”，但是显然裁剪后的图像中已经不包含皇冠的内容了。

长宽比分桶训练策略（Aspect Ratio Bucketing）就是为了解决上面的问题孕育而生。长宽比分桶训练策略的本质是多分辨率训练，就是在LoRA模型的训练过程中采用多分辨率而不是单一分辨率，多分辨率训练技术在传统深度学习时代的目标检测、图像分割、图像分类等领域非常有效，在AIGC时代终于有了新的内涵，在AI绘画领域重新繁荣。

那么在AI绘画领域中是如何使用长宽比分桶训练策略这个技术的呢？主要通过数据分桶+多分辨率训练两者结合来实现。我们设计多个存储桶（Bucket），每个存储桶代表不同的分辨率（比如512x512、768x768、1024x1024等），并将数据存入对应的桶中。在LoRA训练时，随机选择一个桶，从中采样Batch大小的数据用于多分辨率训练。下面Rocky详细介绍一下完整的流程。

我们先介绍如何对训练数据进行分桶，这里包含存储桶设计和数据存储两个部分。

首先我们需要设置存储桶（Bucket）的数量和每个存储桶代表的分辨率。我们定义最大的整体图像像素为1024x1024，最大的单边分辨率为1024。

这时我们以64像素为标准，设置长度为1024不变，宽度以1024为起点，根据数据集中的最小宽度设计存储桶（假设为512），具体流程如下所示：

设置长度为 1024，设置宽度为 1024
设置桶数量为 0
当宽度大于数据集最小宽度 512 时:
    宽度 = 宽度 - 64 （ 960 ）
    那么 （ 960 ， 1024 ）作为一个存储桶的分辨率
    以此类推设计出长度不变，宽度持续自适应的存储桶

按照上面的流程，我们可以获得如下的存储桶：

bucket 0 (512, 1024)
bucket 1 (576, 1024)
bucket 2 (640, 1024)
bucket 3 (704, 1024)
bucket 4 (768, 1024)
bucket 5 (832, 1024)
bucket 6 (896, 1024)
bucket 7 (960, 1024)

接着我们再以64像素为标准，设置宽度为1024不变，长度以1024为起点，根据数据集中的最小长度设计存储桶（假设为512），按照上面相同的规则，设计对应的存储桶：

bucket 8 (1024, 512)
bucket 9 (1024, 576)
bucket 10 (1024, 640)
bucket 11 (1024, 704)
bucket 12 (1024, 768)
bucket 13 (1024, 832)
bucket 14 (1024, 896)
bucket 15 (1024, 960)

最后我们再将1024x1024分辨率作为一个存储桶添加到分桶列表中，从而获得完整的分桶列表：

bucket 0 (512, 1024)
bucket 1 (576, 1024)
bucket 2 (640, 1024)
bucket 3 (704, 1024)
bucket 4 (768, 1024)
bucket 5 (832, 1024)
bucket 6 (896, 1024)
bucket 7 (960, 1024)
bucket 8 (1024, 512)
bucket 9 (1024, 576)
bucket 10 (1024, 640)
bucket 11 (1024, 704)
bucket 12 (1024, 768)
bucket 13 (1024, 832)
bucket 14 (1024, 896)
bucket 15 (1024, 960)
bucket 16 (1024, 1024)

完成了分桶的数量与分辨率设计，我们接下来要做的是将数据集中的图片存储到对应的存储桶中。

那么，具体是如何将不同分辨率的图片放入对应的桶中呢？

我们首先计算存储桶分辨率的长宽比，对于数据集中的每个图像，我们也计算其长宽比。这时我们将长宽比最接近的数据与存储桶进行匹配，并将图像存入对应的存储桶中，下面的计算过程代表寻找与数据长宽比最接近的存储桶： \text{image_bucket} = argmin(abs(\text{bucket_aspects} — \text{image_aspect}))\\如果图像的长宽比与最匹配的存储桶的长宽比差异依然非常大，则从数据集中删除该图像。所以我们最好在数据分桶前将数据进行精细化筛选，增加数据的利用率。

当image_aspect与bucket_aspects完全一致时，可以直接将图片放入对应的存储桶中；当image_aspect与bucket_aspects不一致时，需要对图片进行中心裁剪，获得与存储桶一致的长宽比，再放入存储桶中。中心裁剪的过程如下图所示：

对图片进行中心裁剪后放入对应的存储桶（bucket）

由于我们以经做了精细化的存储桶设计，所以出现长宽比不匹配时的图像裁剪比例一般小于0.033，只去除了小于32像素的实际图像内容，所以对训练影响不大。

在完成数据的分桶存储后，接下来Rocky再讲解一下在训练过程中如何基于存储桶实现多分辨率训练过程。

在LoRA模型的训练过程中，我们需要从刚才设计的16个存储桶中随机采样一个存储桶，并且确保每次能够提供一个完整的Batch数据。当遇到选择的存储桶中数据数量不够Batch大小的情况，需要进行特定的数据补充策略。

为了解决上述的问题，我们需要维护一个公共桶（remaining bucket），其他存储桶中的数据量不足Batch大小时，将剩余的数据全部放到这个公共桶中。在每次迭代的时候，如果是从常规存储桶中取出数据，则训练分辨率调整成存储桶对应的分辨率。如果是从公共桶中取出，则训练分辨率调整成设计分桶时的基础分辨率，也就是1024x1024。

同时我们将所有的存储桶根据桶中数据量进行权重设置，具体的权重计算方式为这个存储桶的数据量除以所有剩余存储桶的数据量总和。如果不通过权重来选择存储存储桶，数据量小的存储桶会在训练过程的早期就被用完，而数据量最大的存储桶会在训练结束时仍然存在，这就会导致存储桶在整个训练周期中采样不均衡问题。通过按数据量加权选择桶可以避免这种情况。

【14】扩展Stable Diffusion Token为原来的3倍

官方的SD系列模型的能够接收的最大提示词长度为75个CLIP Tokens，再加上一个开始和结束Token总共77个。

由于我们训练LoRA模型的时候可能会使用信息密集的长文本标签，因此很容易超过77个Tokens的限制。所以我们需要将模型的最大提示长度进行扩展，更长的提示词将更多信息注入到单次图像的生成中，能够更好的对生成图像进行内容的细粒度控制。

那么我们该如何扩展SD模型和LoRA模型能够接受的输入Tokens最长长度呢？

其实很简单，我们将输入的提示词Tokens沿序列维度拆分城多个75个Tokens的单独子提示词序列，接着将每个子提示词序列都通过CLIP的文本编码器，获得对应的Text Embeddings，最后将生成的Text Embeddings进行连接（concatenate），如下所示：

+---+---+       +---+---+
| 1 | 2 |       | 5 | 6 |
+---+---+  ==>  +---+---+
| 3 | 4 |       | 7 | 8 |
+---+---+       +---+---+

+---+---+---+---+
| 1 | 2 | 5 | 6 |
+---+---+---+---+
| 3 | 4 | 7 | 8 |
+---+---+---+---+

但是直接的拆分与拼接可能会导致输入文本语义被破坏的情况，比如说在对输入提示词进行拆分时，在拆分的边界处有 beautiful girl ，则有可能将beautiful代表的Token拆分到前一组中，girl代表的Token将拆分到后一组中。这种不合理的拆分会导致文本语义信息的bias，从而影响模型的生成效果。

为了解决这个问题，我们可以增加一些约束规则，比如说通过查找输入提示词的逗号来区分文本语义，而不是单独的单词。完整流程如下所示：

输入文本提示词为 `...,WeThinkIn,beautiful girl,AIGCmagic,...`

第 75 个词为 `beautiful`

根据Token数量直接拆分：

集合 1:{..., [74]=WeThinkIn, [75]=beautiful}，集合 2:{[76]=girl, [77]=AIGCmagic, ...}

使用约束规则进行拆分：

集合 1:{..., [74]=WeThinkIn，[75]=“通过padding补全”}，集合 2:{[76]=beautiful，[77]=girl, ...}

【15】优化器选择对LoRA模型训练的影响

AdamWScheduleFree是无调度器的AdamW变体优化器，其优势在于能够自动调整学习率，无需手动配置lr_scheduler。推荐设置参数：weight_decay=0.08, weight_lr_power=0。
SGDScheduleFree是无调度器的SGD变体优化器，其优势是能够更稳定的收敛，减少超参数调优工作。
StableAdamW比标准AdamW更稳定，能够减少训练波动，适合不稳定的数据集中采用。
Ranger (RAdam + LookAhead)结合了RAdam和LookAhead优势，有更平滑的优化路径，适合复杂场景训练。
类别	优化器	特点	推荐场景
传统	AdamW8bit	稳定，通用，推荐新手	日常训练
高效	Lion/Lion8bit	快速，省显存，大学习率稳定	快速实验
自适应	Prodigy	无需调lr	新手友好
无调度	AdamWScheduleFree	自动调度，无需学习率调度器，自动warm-up，自动衰减	简化流程
稳定	StableAdamW	抗扰动，抗噪声	不稳定数据
混合	Ranger	平滑优化，收敛稳定，训练略慢5-10%	复杂场景
极致	adafactor	超省显存，收敛可能不如AdamW，训练需要更多epoch，不适合极小参数模型	大模型

【16】训练精度对LoRA模型训练的影响

如果大家的算力资源非常有限，我们在训练LoRA模型时可以尝试开启FP8精度训练。FP8比FP16能够再节省约30%显存，适用于在消费级24GB显卡上跑更大的batch size，同时精度损失极小。

4.6 LoRA模型的训练技巧与经验分享
一般来说当我们设置Batch Size为2-8时，SD LoRA学习率可以设置为1e-4以及更大，SDXL LoRA学习率可以设置为1e-5以及更大。
不管是SD LoRA还是SDXL LoRA，一般设置dataset_repeats * max_train_epochs 大于等于100时的训练效果会比较好。
训练U-Net LoRA和训练Text Encoder LoRA并不矛盾，他们学习了数据集中不同的特征，互相补充，同时优化能获得更好的效果。
微调训练Text Encoder LoRA的学习率可以设置为训练U-Net LoRA的学习率的0.5倍。
训练时，调整数据集的长宽尺寸都大于512，可以获得比较好的训练结果。
训练时，除了更新LoRA的全部参数，还可以只更新LoRA部分层的参数。
训练人物特征时，需要不同角度、姿态的数据20-40张就可以了，如果要训练风格或者画风，则需要150-200张风格图片，如果是训练一个抽象概念，则数据多多益善。
4.7 LoRA模型训练结果测试评估

完成LoRA模型的训练后，我们可以加载LoRA模型进行推理，生成我们想要的图片，下面是LoRA模型在diffusers库中进行推理的代码：

#读取diffuers库
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

#设置SD模型路径和LoRA模型路径
model_path = "/本地路径/stable-diffusion-v1-5" #修改成本地SD主模型路径
LoRA_Path = "WeThinkIn" #修改成本地LoRA模型路径

#初始化SD模型，加载预训练权重
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# 加载LoRA weights ～3 MB
pipe.unet.load_attn_procs(LoRA_Path)
pipe.to("cuda")

#接下来，我们就可以运行pipeline了
image = pipe("blue pokemon", num_inference_steps=25).images[0]
image.save("test.png")

目前为止，我们已经使用LoRA完成常规的图像生成流程了。

5. 主流LoRA变体模型深入浅出完整讲解
5.1 LoCon核心基础知识深入浅出完整讲解

LoCon（LoRA for Convolution Network）模型是LoRA 技术在卷积神经网络（CNN）中的扩展与适配，核心是将低秩分解思想从Transformer的线性层（如 Attention 的 QKV 变换、全连接层）迁移到卷积层，实现卷积模型的参数高效微调（PEFT）。理论上能够实现更细粒度的生成内容的控制。

下图中红色框部分代表LoCon模型在LoRA模型基础上额外增加的训练部分：

LoCon模型与SD系列模型部分训练示意图

LoRA模型对卷积层是使用1x1卷积进行降维，而LoCon模型将1x1卷积切换成正常尺寸的卷积进行降维，降维到预设的Rank（lora_dim）。

我们先来回顾一下传统深度学习领域中卷积的计算过程：

卷积计算过程的完整图示

接下来我们看看使用LoCon技术后，SD系列模型的卷积层权重的变化：

Conv(in, out, ksize, padding, stride)

\xrightarrow{}Conv(rank, out, 1)\circ Conv(in, rank, ksize, padding, stride)

使用了LoCon技术后，SD系列模型+LoCon模型的FLOPS变化如下所示：

before = \text{out_ch} \times \text{in_ch} \times size^{2} \times \text{out_h} \times \text{out_w}

after = (\text{out_ch} \times \text{LoRA_rank} + \text{LoRA_rank} \times \text{in_ch} \times size^{2}) \times \text{out_h} \times \text{out_w}

同时训练时的参数量也发生了变化：

before = \text{out_ch} \times \text{in_ch} \times size^{2}

after = \text{LoRA_rank} \times \text{in_ch}\times size^{2} + \text{LoRA_rank} \times \text{out_ch}

LoCon在实验中得出可以比LoRA模型在训练中更快地拟合（例如，LoCon模型在训练600步可以达到LoRA模型训练800步的生成性能）。这表明LoCon模型可能在训练角色或特定特征上更为高效。另外，将LoCon模型应用于人物角色的风格化上也表现不错。

LoCon和LoRA效果对比

LoCon推荐训练参数设置：dim <= 64，alpha = 1 (或者更小，比如说0.3)

下面我们看看LoCon模型和LoRA模型在处理卷积层的具体区别：

LoRA模型处理卷积层的代码：

if org_module.__class__.__name__ == 'Conv2d':
      in_dim = org_module.in_channels
      out_dim = org_module.out_channels
      self.lora_down = torch.nn.Conv2d(in_dim, lora_dim, (1, 1), bias=False)
      self.lora_up = torch.nn.Conv2d(lora_dim, out_dim, (1, 1), bias=False)
else:
      in_dim = org_module.in_features
      out_dim = org_module.out_features
      self.lora_down = torch.nn.Linear(in_dim, lora_dim, bias=False)
      self.lora_up = torch.nn.Linear(lora_dim, out_dim, bias=False)

LoCon模型处理卷积层的代码：

if org_module.__class__.__name__ == 'Conv2d':
            # For general LoCon
            in_dim = org_module.in_channels
            k_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            out_dim = org_module.out_channels
            self.lora_down = nn.Conv2d(in_dim, lora_dim, k_size, stride, padding, bias=False)
            self.lora_up = nn.Conv2d(lora_dim, out_dim, (1, 1), bias=False)
else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)
5.2 LoHa核心基础知识深入浅出完整讲解

上面讲到的LoCon主要是对LoRA进行工程应用层面的改造优化（将LoRA的应用扩展到SD/FLUX系列模型的卷积层），接下来我们要讲的LoHa模型主要是针对LoRA的低秩矩阵分解理论层面进行优化。

LoHa (LoRA with Hadamard Product)是在LoRA的基础上，使用了哈达玛积（Hadamard Product）代替原生LoRA中的矩阵点乘，将秩的维度从2R扩展到 R^2，让LoHa理论上在相同的参数配置下能学习到更多的数据分布信息。

左图是原生LoRA模型示意图，右图是LoHa模型示意图

读者朋友可能对哈达玛积不太熟悉，Don't Worry。我们先来了解一下什么是哈达玛积：哈达玛积（Hadamard Product），又称逐元素乘积（element-wise product），是线性代数中的一种矩阵运算。它与标准矩阵乘法不同，哈达玛积是对两个相同大小的矩阵的对应元素进行乘积运算。

给定两个相同大小的矩阵A和B ，它们的哈达玛积C定义如下： C = A \circ B \\

其中C的每个元素c_{ij}计算为：

c_{ij} = a_{ij} \times b_{ij} \\

例如，假设有以下两个矩阵A和 B： A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} \\

它们的哈达玛积C为：

C = A \circ B = \begin{bmatrix} 1 \times 5 & 2 \times 6 \\ 3 \times 7 & 4 \times 8 \end{bmatrix} = \begin{bmatrix} 5 & 12 \\ 21 & 32 \end{bmatrix} \\

秩的维度小于2R从上面的公式中可以看到，哈达玛积通过对两个矩阵的逐元素乘积，能够有效地对矩阵进行特征组合、权重计算和信息传播，增强AI模型的表达能力和计算效率。

在LoHa模型中，应用了哈达玛积后，低秩分解后的形式就转变成如下所示的公式：

\Delta W = (X_1Y_1^T) \odot (X_2Y_2^T) \\

其中需要满足条件 rank(\Delta W) \leq R^2

可以看到比起原生LoRA的秩的维度小于2R，LoHa将秩的维度扩展到R^2，解决了原生LoRA受到低秩的限制。这个思路不仅仅能够用在AIGC图像生成/AI绘画领域，在AIGC其他领域中都可以借鉴与迁移。

LoHa训练经验分享：

LoHa推荐训练参数设置：dim <= 32，alpha = 1 (or lower)
LoHa不适合训练特征不太明确的画风，同时也比较难收敛，LoHa通常需要比LoRA和LoCon更多的训练步数才能达到较好的效果。
5.3 残差/差异化LoRA模型深入浅出完整讲解

残差/差异化LoRA模型可以说是一种巧妙优雅的LoRA训练思想。残差/差异化LoRA模型最早在AIGC开源社区被提出，展现了开源社区的集体智慧。这种LoRA模型的特殊性源自于其训练思想，旨在让LoRA模型学习两类图像之间的差异。因此，在LoRA、LoCon、LoHa等架构以及SD、FLUX等不同的AIGC大模型上都能运用这个训练思想，训练对应配套的残差/差异化LoRA模型。

训练得到的残差/差异化LoRA模型一般用于优化生成图像的整体质量，比如美颜美白、细节增强、质感加强、光影增强等。

那么，残差/差异化LoRA模型是如何训练的呢？

首先我们需要构建两张内容相似的图像：图 A 和图 B。例如下图所示，左图AI感更强，右图质感更强，整体更自然。

在残差/差异化LoRA的训练中，我们分两步进行训练：

以图 A 为训练数据，由于训练数据仅有一张图，过拟合训练得到LoRA A。
以图 B 为训练数据，由于训练数据同样仅有一张图，再次过拟合训练得到LoRA B。

接着我们将两个训练好的LoRA B和LoRA A做差：LoRA B - LoRA A，就最终得到了残差/差异化LoRA C模型。其核心公式如下：

差异LoRA = LoRA_{A} \times ratio_A +LoRA_B \times ratio_B \\

一张训练数据可以保证LoRA模型能够过拟合到训练数据上，但稳定性不足。为了提高稳定性，我们可以用多个图像对（image pairs）进行训练，从而得到效果更稳定的残差/差异化LoRA模型。

到此为止，我们已经了解了残差/差异化LoRA模型的训练过程。我们可以举一反三，比如使用丑陋的和漂亮的图像对，训练提升图像美感的 LoRA；或者使用细节少的和细节丰富的图像对，训练增加图像细节的LoRA。

一般来说，使用残差/差异化LoRA模型时不需要提示词，对生成图像的构图几乎没有影响，可以说是一种“万金油”的LoRA模型系列。

5.4 LCM_LORA模型深入浅出完整解析

【一】LCM_LoRA模型核心基础原理深入浅出讲解

在讲LCM_LoRA之前，Rocky先简单介绍一下LCM模型。

LCM模型的全称是Latent Consistency Models（潜在一致性模型），由清华大学交叉信息研究院发布。在这个模型发布之前，以Stable Diffusion/FLUX等为主的潜在扩散模型（LDM）由于迭代采样过程计算量大，生成速度较慢。而LCM模型通过将原始LDM模型进行一致性蒸馏技术训练，最后得到一个只用少数的几步推理就能生成高分辨率图像的AIGC图像生成大模型。一般来说，LCM模型能将主流文生图模型的效率提高5-10倍，所以能呈现出实时生成的效果。

关于LCM等扩散模型的核心理论知识，大家可以研读Rocky一直在撰写完善的文章：

在AIGC图像生成/AI绘画领域中，如果使用原始LCM进行蒸馏训练，那么每个SD/FLUX模型都需要单独蒸馏，这无疑增加了AI绘画开源社区SD/FLUX模型迭代更新的成本。

这时候，就该LCM_LoRA模型登场了，LCM_LoRA 是LCM与LoRA技术的创新性结合，LCM_LoRA模型的核心思想是将LCM的蒸馏目标浓缩到LoRA模型的少量参数上，而不用对完整SD/FLUX模型进行完整的微调训练，解决了传统LCM 蒸馏成本高、通用性差的核心痛点。在前向推理时，可将训练好的LCM_LoRA模型用于任何一个微调后的SD/FLUX模型，无需再对SD/FLUX模型重新进行蒸馏训练。

通过将LCM_LoRA模型加载到SD/FLUX模型中，可以将SD/FLUX模型的推理步数减少到仅2至8步，而不是常规的25至50步。在使用LCM_LoRA模型的情况下，SDXL模型在3090显卡上运行只需要大约1秒钟。除了文生图任务外，LCM_LoRA模型还支持图生图任务、图像重绘（inpainting）以及其他SD模型与LoRA模型结合使用的任务场景。

目前大家可以直接体验LCM_LoRA的效果：LCM Painter

LCM_LoRA系列模型权重百度云网盘：关注Rocky的公众号WeThinkIn，后台回复：LCM_LoRA，即可获得资源链接，包含LCM_LoRA SDXL，LCM_LoRA SD1.5和LCM_LoRA SSD-1b三个LCM_LoRA模型权重。

【二】LCM_LoRA模型的图像加速生成能力带来的AIGC新可能性

LCM_LoRA模型的加速能力为Stable Diffusion/FLUX在AIGC领域中的新应用和新工作流打开了大门：

普及速度更快：推理速度变快后，AIGC图像生成大模型和配套的AI绘画工具可以被更多人使用，破圈速度更上一层楼。
迭代更快：在同样的时间内生成更多的图像和进行更多的AIGC应用尝试对于AIGC从业者来说非常有价值。
更易部署：可以在各种不同的硬件上进行生产化部署，包括CPU等家用消费级硬件。
更便宜：AIGC图像生成服务会更便宜。

LCM_LoRA模型让AIGC图像生成/AI绘画的整体速度快了一个数量级，我们再也无需等待结果，这带来了颠覆性的体验。如果使用4090，我们几乎可以得到实时响应 (不到 1 秒)。在这种情况下，SD/FLUX都可以用于需要实时响应的AIGC任务场合。

【三】LCM_LoRA的训练过程

那么LCM_LoRA模型是如何训练得到的呢？

我们只需要给Stable Diffusion模型外接一个LoRA模型，然后只用LCM的一致性蒸馏损失优化LoRA模型的权重，在经过蒸馏训练后就得到了LCM_LoRA模型。

【四】使用LCM_LoRA进行AI绘画

在diffusers框架中，我们能够非常方便地使用LCM_LoRA进行图像加速生成：

from diffusers import DiffusionPipeline, LCMScheduler
import torch

model_id = "/本地路径/stable-diffusion-xl-base-1.0"
lcm_lora_id = "/本地路径/lcm-lora-sdxl"

pipe = DiffusionPipeline.from_pretrained(model_id, variant="fp16")

pipe.load_lora_weights(lcm_lora_id)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to(device="cuda", dtype=torch.float16)

prompt = "close-up photography of old man standing in the rain at night, in a street lit by lamps, leica 35mm summilux"
images = pipe(
    prompt=prompt,
    num_inference_steps=4,
    guidance_scale=1,
).images[0]

上述代码所做的事情主要是：

加载SDXL 1.0模型。
加载LCM_LoRA模型。
将调度器改为 LCMScheduler，这是 LCM 模型使用的调度器。
使用SDXL+LCM_LoRA快速生成图像。

我们看一下SDXL+LCM_LoRA经过4步生成的图像效果：

SDXL+LCM_LoRA经过4步生成的图像效果

我们再看一下步数对SDXL+LCM_LoRA生成效果的影响：

images = []
for steps in range(8):
    generator = torch.Generator(device=pipe.device).manual_seed(1337)
    image = pipe(
        prompt=prompt,
        num_inference_steps=steps+1,
        guidance_scale=1,
        generator=generator,
    ).images[0]
    images.append(image)
SDXL+LCM_LoRA经过1-8步的生成效果对比

从上图可以看到，仅使用1步生成的图像细节比较粗略，同时纹理欠缺。随着生成步数的增加，生成图像效果改善迅速，可以看到只需4到6步就可以达到满意的效果。一般来说，8 步生成的图像可能会存在过拟合的情况，而1步生成的图像可能会存在欠拟合的情况。

由于LCM_LoRA模型在训练过程中已经把Guidance Scale集成进去，所以在使用LCM_LoRA模型时一般是不需要再做CFG设置的。但是如果Negative Prompt内容对结果非常重要的话，那么也可以设置Guidance Scale为一个很小的值（0-1.5）。

与此同时，LCM_LoRA模型可以和AI绘画开源社区里的各种LoRA模型组合，共同作用来实现既能加速出图，同时风格又多变的效果。

下面的代码展示了将LCM_LoRA与常规的SDXL LoRA结合起来使用，使其也能够进行4步推理生成图像：

from diffusers import DiffusionPipeline, LCMScheduler
import torch

model_id = "/本丢路径/stable-diffusion-xl-base-1.0"
lcm_lora_id = "/本地路径/lcm-lora-sdxl"
pipe = DiffusionPipeline.from_pretrained(model_id, variant="fp16")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights(lcm_lora_id)
pipe.load_lora_weights("CiroN2022/toy-face", weight_name="toy_face_sdxl.safetensors", adapter_name="toy")

pipe.set_adapters(["lora", "toy"], adapter_weights=[1.0, 0.8])
pipe.to(device="cuda", dtype=torch.float16)

prompt = "a toy_face man"
negative_prompt = "blurry, low quality, render, 3D, oversaturated"
images = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=4,
    guidance_scale=0.5,
).images[0]
images

下面的表格列出了SDXL+LCM_LoRA和单独SDXL在不同硬件上的生成速度对比（batch size均为1）：

可以看到，SDXL+LCM_LoRA的形式在图像整体生成速度上确实有较大提升，如果使用显存容量比较大的显卡(例如A100)，一次生成多张图像，那么性能会有更显著的提高。

目前AIGC图像生成/AI绘画领域快速发展，核心大模型早已从SDXL演进到SD 3、FLUX系列、Z-Iamge、Qwen-Image等，大家可以尝试借鉴LCM思想，在不同的核心大模型上构建对应的LCM_LoRA模型！

5.5 Textual Inversion（embeddings模型）技术深入浅出完整讲解

上面讲到的都是基于Stable Diffusion/FLUX模型架构的fine-tuning训练技术，接下来Rocky再向大家介绍一下基于prompt-tuning的训练技术——Textual Inversion。比起基于fine-tuning训练技术，基于prompt-tuning的训练技术更加轻量化（模型大小几kb-几mb左右），模型存储成本很低。

在详细讲解Textual Inversion技术之前，让我们先回顾一下Text Prompt在SD/FLUX系列模型中的处理流程：

Text Prompt在SD系列模型中的处理流程

如上图所示，我们输入的Text Prompt会先经过Tokenizer转换成Tokens，再经过Text Encoder输出embeddings特征，通过Attention机制注入到SD/FLUX系列模型中，Textual Inversion技术就是作用于上图的Text Prompt过程中。

Textual Inversion技术的核心思路是基于3～5张特定概念（物体或者风格）的示例图像来训练一个特定的Text Embeddings模型，从而将特定概念编码到Text Embedding空间中。Text Embedding空间中的词向量是有足够的表达能力恢复出图像特征，同时Textual Inversion技术不需要对SD/FLUX系列模型中的U-Net/Transformer部分进行微调训练（SD/FLUX模型参数冻结），只需要训练一个新的Token Embedding（下图中的 v_{*} ）就足够了，所以使用Textual Inversion技术不会儿干扰SD/FLUX模型本身已有的先验知识。

我们首先需要定义一个特殊的关键词（下图中的 S_{*} ），这个特殊的关键词与新的Token Embedding对应，在Textual Inversion训练过程中，会不断将包含特殊关键词的Prompt注入SD/FLUX模型，在不改变SD/FLUX模型参数的情况下，在SD/FLUX模型中不断优化来表示特殊的关键字的Embedding向量，最终得到对于特殊的关键词最佳映射的Embedding向量。

Textual Inversion技术的原理示意图

完成训练后，我们就能获得一个包含主题编码或者风格编码的Text Embedding模型。

对风格进行编码训练的Text Embedding模型

除此之外，Textual Inversion技术可以在SD/FLUX模型中同时注入多个概念，如下图所示：

使用多个Text Embedding模型将多个概念注入到SD模型中

目前diffusers库已经支持Textual Inversion技术的训练：diffusers/textual_inversion

在AI绘画开源社区中，有非常多的Text Embedding模型用于优化Stable Diffusion/FLUX模型的图像生成效果，它们用在Negative Prompts和Positive Prompts中，Rocky这边也整理归纳了一些高质量的Text Embedding模型，推荐给大家使用，包括EasyNegative、badhandv4、veryBadImageNegative、bad_prompt Negative Embedding、Negative Embedding for Realistic Vision v2.0、bad-picture negative embedding for ChilloutMix等。

大家关注Rocky的公众号WeThinkIn，后台回复：TextEmbeddings模型，即可获得资源链接。

6. 深入浅出完整讲解MoE-LoRA（Mixture of Experts with LoRA）的核心基础知识

我们在之前章节中讨论过的主流LoRA架构及其变体模型和Stable Diffusion、FLUX等主干大模型一样都是参数稠密的。这意味着每次推理时模型的所有参数都会参与计算，效率上其实仍有不小的瓶颈。

2024年初DeepSeek横空出世，成为全球的热点“明星”大模型。其核心的MoE（Mixture of Experts，混合专家）框架更是成为AIGC领域各个细分方向研究的焦点。

在AIGC图像生成领域也不例外，除了在AIGC图像生成大模型中引入MoE架构外，LoRA技术也天然能够与MoE框架相结合，来提升LoRA技术的发展边界与整体性能，下图是经典LoRA架构和MoEA-LoRA架构差异图解，希望先给大家一个直观的感受，在本章节的后续内容中，Rocky将带着大家深入浅出学习：

经典LoRA架构和MoEA-LoRA架构差异示意图
6.1 深入浅出完整讲解MoE框架下LoRA技术的核心原理

经典MoE模型作为一种基于Transformer架构的大模型，主要由两个关键部分组成：

稀疏MoE层：这些层代替了传统Transformer模型中的前馈网络（FFN）层。MoE层包含若干“专家”（例如 2、4、6、8个等），这些都是独立的神经网络。在实际应用中，这些专家网络通常设置为前馈网络（FFN），但它们也可以是更复杂的网络结构，甚至可以是MoE层本身，从而形成层级式递归的MoE结构。
门控网络/路由（Router）模块：这个部分用于决定哪些Tokens被发送到哪个专家网络中。有时，一个Token甚至可以被发送到多个专家网络。Token的路由方式是MoE框架构建的一个关键点，因为路由器是由可学习的参数组成，并且与大模型的其他参数部分一同进行预训练。

总的来说，我们可以发现，MoE框架的核心就是将传统Transformer模型中的前馈网络（FFN）层替换为MoE层，同时每个MoE层都配备一个门控网络和若干数量的专家网络，从而在AIGC大模型中引入了高价值的“稀疏性”。

因此，稀疏的MoE框架模型与稠密模型相比， 能够实现高效预训练，整体训练速度更快；与具有相同参数数量的稠密模型相比，由于只激活部分专家网络，所以拥有更快的推理速度；但是由于MoE框架模型包含了大量的专家网络，所以需要更大的显存资源，因为所有专家网络都需要加载到显存中作为预备。

在MoE框架下，路由模块（Router）在接收原始输入后，会算出每个专家网络的激活权重，最终输出分两种情况：

如果是软路由（soft routing），就是把所有专家网络的输出做加权求和作为最终的输出结果。
如果是离散路由（discrete routing），也就是Mistral、DeepSeek MoE中使用的稀疏混合专家（Sparse MoE）架构，就会把激活权重排在Top-K之外的专家网络直接置零（K是固定超参数，比如1或2，代表每次只激活这么多数量专家网络），再对剩下的专家网络输出做加权求和作为最终的输出结果。

讲到这里，我们应该已经清楚的明白MoE架构的核心优势：MoE里每个专家网络的参数能不能被激活、激活程度有多高，全看路由模块的选择，这无疑是走了“领导”逻辑。如果路由模块训练的好，就能合理规划专家网络的选择，这就意味着每个专家网络都能专注于自己擅长的那类数据。尤其是在离散路由的场景下，Top-K之外的专家网络根本不用参与计算，既能保住AIGC大模型的总参数容量，又能大幅降低推理时的计算成本，可以说是把大模型的参数计算效率拉满了。

但是这时候也引入了一个问题，那就是在MoE框架模型的训练中，路由模块往往倾向于激活主要的几个专家网络，就如同“二八法则”一样，几个高价值的专家网络获得了越来越多的训练资源，而剩下的大量专家网络连训练于一次都成为了奢望，甚至出现“死亡”的专家网络。这种情况可能会让受欢迎的专家网络训练得更快更好，从而导致了整体训练的不均衡，使得大模型的表征能力严重浪费，整体性能甚至不如同计算量的稠密模型。

为了缓解这个问题，我们可以引入一个负载均衡损失（Load Balancing Loss）作为一种「正则项」，鼓励大模型给予所有专家网络相同的重要性，惩罚“专家使用频率不均”的行为，从而平衡了专家网络之间的选择，实现了不同专家网络之间的负载均衡（Load Balancing）。

Z-Loss是AI工业界最常用的负载均衡损失，由Google在Switch Transformer中首次提出。Z-Loss的核心思想是通过约束「路由模块输出的专家概率分布的对数和」，让每个专家网络被选中的期望频率尽可能相等：

对Top-1路由：理想状态是每个专家被选中的概率 = 1/num_experts（比如8个专家，每个概率1/8）。
对Top-2路由：理想状态是每个专家被选中的概率 = 2/num_experts（比如8个专家，每个概率2/8=1/4）。

Z-Loss的定义如下：

\mathcal{L}_{z} = \sum_{j=1}^{num\_experts} \left( \log\left( \sum_{i=1}^N \exp(g_j(x_i)) \right) - \log(N) - \log(\tau) \right)^2

其中 g(x_i) 表示路由模块对Token x_i 输出的所有专家网络的logits（未做Softmax操作），形状 [num_experts]；

G = softmax(g(x_i)) 表示每个Token对专家网络的概率分布； z_j = (1/N) * Σ_{i=1}^N G_j(x_i) 表示专家网络 j 在所有N个Token上的平均选中概率（实际频率）； τ 表示目标频率（Top-K路由下，τ = K/num_experts）。

可以看到，Z-Loss公式的本质是惩罚「每个专家的实际被选中频率」与「目标频率τ」的偏差——偏差越大，Z-Loss值越高，总损失越大，反向传播时会调整路由模块的参数，让Token分配更均匀。

这时候，MoE框架模型的训练总损失就等于主任务损失（如交叉熵损失） + 权重系数λ × 负载均衡损失：

\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda \times \mathcal{L}_{balance}

λ的选择：关键超参数，通常取0.01~0.1（ λ 太小则负载均衡效果差， λ 太大则主任务性能下降）。
经验值：Top-1路由用λ=0.1，Top-2路由用λ=0.05（因Top-2天然更均衡）。
6.2 MoE思想与LoRA模型相结合

和LoRA技术本身一样，MoE框架也首发于LLM大模型领域，并逐步引入到AIGC图像生成/AI绘画领域中来。相比主流的经典LoRA系列，LoRA + MoE的思想确实能够在参数量日渐庞大的AIGC图像生成大模型上明显提升微调效率。

MoE-LoRA模型架构示意图

为了保持负载平衡和训练效率，除了引入辅助损失外，我们还可以设置以下的策略:

设置随机路由：在专家网络Top-2规则设置中，除了排名最高的专家网络被选择，第二个专家网络根据其权重比例随机选择。
计算专家容量：我们可以设定一个阈值，来定义一个专家网络能处理多少Tokens。如果两个专家网络的容量都达到上限，Tokens就会溢出，可以通过残差连接传递到下一层，或者被完全丢弃。

专家网络容量的具体概念如下所示：

\text{Expert Capacity} = \left(\frac{\text{tokens per batch}}{\text{number of experts}}\right) \times \text{capacity factor} \\

其中capacity factor代表容量因子。合适的专家网络容量能够将输入的Tokens均匀分配到各个专家网络中去。如果我们使用大于 1 的容量因子，我们可以为Tokens分配不完全平衡时提供了一个缓冲。不过增加容量因子会导致更高的设备间通信成本，因此这是一个需要权衡的参数。一般来说，容量因子设置为1 至 1.25时能够展现出色的性能。

同时，稠密模型和稀疏模型在过拟合的表现上存在显著差异。MoE稀疏模型更易于出现过拟合现象，因此在构建MoE-LoRA模型时，我们可以尝试使用更强的内部正则化策略，比如使用更高比例的dropout。具体来说，我们可以为大模型中的稠密层设定一个较低的dropout率，而为稀疏层设置一个更高的dropout率，以此来优化模型整体性能。并且像MoE-LoRA这样的稀疏模型往往更适合使用较小的Batch-Size和较高的学习率，这样可以获得更好的训练效果。

6.3 MoE-LoRA架构中专家选择(Expert-Choice)与token选择(Token-Choice)的核心基础知识讲解

在MoE-LoRA架构中，专家选择(Expert-Choice, EC)和Token选择(Token-Choice, TC) 是两种截然不同的路由模块范式，它们决定了Token与专家网络之间的分配关系，直接影响MoE-LoRA模型的计算效率、负载均衡、训练稳定性和最终性能。

两者的核心区别在于分配主体与方向：Token-Choice是每个Token选择专家网络，而Expert-Choice是每个专家网络选择Token。

【Token-Choice (TC) 路由模式：Token选专家网络】

这是经典MoE架构的主流路由方式，也被称为Top-K路由，其核心运行流程如下：

路由模块对每个Token计算与所有专家网络的适配分数（相似度）。
对每个Token，选择分数最高的Top-K个专家网络（K通常为1或2）。
仅将该Token路由到这K个专家网络进行处理。
最终输出由选中专家网络的输出按路由权重加权组合而成。

Token-Choice (TC) 路由的数学表达式如下所示： y_i = Σ_{j=1}^k G(x_i)_j \cdot E_j(x_i) \\其中 y_i 是Token i的输出， G(x_i) 是路由模块输出的专家网络权重 G(x_i) = softmax(W_g x_i) ， E_j(x_i) 是专家网络 j 对Token i的计算结果。

【Expert-Choice (EC) 路由模式：专家网络选Token】

这是一种反向路由模式，也被称为专家网络主动选择机制，其核心运行流程如下：

路由模块计算所有Token与每个专家网络的适配分数。
对每个专家网络，选择分数最高的固定数量(预先设置的桶大小)的Tokens。
每个专家网络只处理自己选中的Token，确保负载均衡。
最终输出结果同样由对应专家网络的输出加权组合而成。

这种模式的核心特点是专家网络有固定容量限制，每个专家处理的Token数量相同，从根本上解决负载不均衡问题。

Rocky在这里总结了Token-Choice（TC）和Expert-Choice（EC）两者之间的核心差异对比表，让大家能够更加直观的理解与学习：

对比维度	Token-Choice (TC)	Expert-Choice (EC)
分配方向	Token → 专家网络 (每个Token选K个专家网络)	专家网络 → Token (每个专家网络选固定数量Tokens)
负载均衡	天然不均衡，易出现”热门专家”和”冷专家”	天然均衡，每个专家处理相同数量Token
实现复杂度	低，工业界主流实现简单	高，需要全局排序和Token分配协调
计算效率	Top-1时极高，Top-2时略低	略低，因需全局计算和排序
训练稳定性	较低，冷专家网络梯度更新少易欠拟合	高，所有专家网络都有充足训练数据
表征能力	Top-K(K>1)时较强，能融合多专家网络知识	受限于专家网络容量，单个Token可能只被1个专家网络处理
典型应用	Switch Transformer、GLaM、Mixtral 8x7B等模型	Mistral系列(部分变体)、视觉MoE大模型等
适用场景	推理优先、追求极致计算效率	训练优先、需要严格负载均衡
Token-Choice：负载不均衡是固有问题。热门专家网络被大量Token选择，计算压力大；冷专家网络很少被选择，参数利用率低，甚至出现”死亡专家”（完全不被选择）的情况。为缓解此问题，需引入额外的负载均衡损失(例如Z-loss)，来惩罚路由模块对专家使用频率的差异，增加训练复杂度与成熟度。Token-Choice模式计算效率高，尤其是Top-1路由，每个Token仅激活少量专家网络，适合大规模推理部署。
Expert-Choice：从机制上保证负载均衡。每个专家网络处理固定数量的Tokens，所有专家网络都能获得充足的训练数据，梯度更新均衡，彻底避免冷专家问题。同时无需复杂的负载均衡损失，训练更稳定，泛化能力可能更好。由于需要全局排序和分配，推理延迟可能略高，但负载均衡带来的硬件利用率提升可能抵消这一劣势。
Token-Choice和Expert-Choice架构示意图

MoE架构中的Token-Choice与Expert-Choice选择代表了两种截然不同的信息资源分配哲学：Token-Choice追求效率与灵活性，是当前工业界的主流选择；Expert-Choice追求均衡与稳定性，是解决传统MoE负载不均衡问题的创新方案。选择哪种路由范式，我们可以根据具体AIGC任务需求、大模型规模、训练资源和部署场景综合考量，未来更可能出现融合两种优势的混合路由机制。

6.4 深入浅出讲解Token-Choice (TC) 和Expert-Choice (EC) 的具体代码实现

下面我们来看看Token-Choice (TC) 和Expert-Choice (EC) 的具体代码实现。为了让代码更易理解，我们统一设定基础参数：

专家网络数量 num_experts = 8
Token-Choice采用Top-2门控（每个token选2个专家网络）
Expert-Choice中每个专家固定处理 expert_capacity = 1024 个Tokens
输入：tokens_embedding（Embeddings嵌入），形状为 [batch_size, seq_len, hidden_dim] = [4, 2048, 4096]
路由模块输出：gate_scores（每个token对每个专家网络的适配分数），形状为 [batch_size, seq_len, num_experts]

首先是Token-Choice (TC) 的代码实现，其核心逻辑是以每个Token为中心，为它选Top-2专家，仅把该Token发给选中的专家网络处理。

# Token-Choice（TC）核心路由代码（Top-2门控）
import numpy as np

def token_choice_routing(tokens_embedding, gate_scores, num_experts=8, top_k=2):
    """
    Token-Choice路由：每个Token选择Top-K个专家
    Args:
        tokens_embedding: 输入令牌嵌入，shape [batch_size, seq_len, hidden_dim]
        gate_scores: 门控分数，shape [batch_size, seq_len, num_experts]
        num_experts: 专家总数
        top_k: 每个token选择的专家数（这里是Top-2）
    Returns:
        expert_outputs: 各专家处理后的结果，shape [num_experts, batch_size, seq_len, hidden_dim]
        final_output: 融合专家输出后的最终结果，shape [batch_size, seq_len, hidden_dim]
    """
    batch_size, seq_len, hidden_dim = tokens_embedding.shape
    
    # 1. 对每个token，计算Top-2专家的索引和对应的权重（核心：按token维度选专家）
    # top_k_indices: 每个token选中的专家索引，shape [batch_size, seq_len, top_k]
    # top_k_weights: 每个token对选中专家的权重，shape [batch_size, seq_len, top_k]
    top_k_indices = np.argsort(gate_scores, axis=-1)[:, :, -top_k:]  # 按最后一维（专家）排序，取最后2个（分数最高）
    top_k_weights = np.take_along_axis(gate_scores, top_k_indices, axis=-1)
    top_k_weights = top_k_weights / np.sum(top_k_weights, axis=-1, keepdims=True)  # 归一化权重
    
    # 2. 初始化专家输出容器（所有专家初始为0）
    expert_outputs = np.zeros((num_experts, batch_size, seq_len, hidden_dim))
    
    # 3. 遍历每个token，将其发送给选中的Top-2专家处理
    for b in range(batch_size):
        for s in range(seq_len):
            token = tokens_embedding[b, s, :]  # 当前token的嵌入向量
            selected_experts = top_k_indices[b, s, :]  # 该token选中的2个专家
            weights = top_k_weights[b, s, :]          # 对应权重
            
            # 把token发给选中的专家，专家处理（这里用"expert_forward"模拟专家前向计算）
            for i, expert_idx in enumerate(selected_experts):
                expert_output = expert_forward(expert_idx, token)  # 专家处理单个token
                expert_outputs[expert_idx, b, s, :] += weights[i] * expert_output  # 加权累加
    
    # 4. 融合所有专家的输出，得到每个token的最终结果（仅累加被选中的专家贡献）
    final_output = np.sum(expert_outputs, axis=0)  # 按专家维度求和，shape [batch_size, seq_len, hidden_dim]
    
    return expert_outputs, final_output

# 模拟专家前向计算（仅示意，实际是FFN层）
def expert_forward(expert_idx, token):
    """单个专家处理单个token的前向计算"""
    # 这里简化为"专家参数 × token"，实际是FFN的线性变换+激活
    expert_params = np.random.randn(token.shape[0], token.shape[0])  # 模拟专家参数
    return np.dot(expert_params, token)

接着是Expert-Choice（EC）的代码实现，其核心逻辑是以每个专家网络为中心，为它选固定数量的Tokens，仅处理选中的Tokens。

# Expert-Choice（EC）核心路由代码
import numpy as np

def expert_choice_routing(tokens_embedding, gate_scores, num_experts=8, expert_capacity=1024):
    """
    Expert-Choice路由：每个专家选择固定数量的Token
    Args:
        tokens_embedding: 输入令牌嵌入，shape [batch_size, seq_len, hidden_dim]
        gate_scores: 门控分数，shape [batch_size, seq_len, num_experts]
        num_experts: 专家总数
        expert_capacity: 每个专家最多处理的token数
    Returns:
        expert_outputs: 各专家处理后的结果，shape [num_experts, batch_size, seq_len, hidden_dim]
        final_output: 融合专家输出后的最终结果，shape [batch_size, seq_len, hidden_dim]
    """
    batch_size, seq_len, hidden_dim = tokens_embedding.shape
    total_tokens = batch_size * seq_len  # 总token数：4*2048=8192
    
    # 1. 重塑数据：把batch+seq_len展平为一维，方便全局处理，shape [total_tokens, num_experts]
    gate_scores_flat = gate_scores.reshape(total_tokens, num_experts)
    tokens_flat = tokens_embedding.reshape(total_tokens, hidden_dim)
    
    # 2. 初始化专家输出容器
    expert_outputs = np.zeros((num_experts, batch_size, seq_len, hidden_dim))
    final_output = np.zeros((batch_size, seq_len, hidden_dim))
    
    # 3. 对每个专家，选择分数最高的expert_capacity个token（核心：按专家维度选token）
    for expert_idx in range(num_experts):
        # 3.1 取出该专家对所有token的分数，shape [total_tokens]
        expert_scores = gate_scores_flat[:, expert_idx]
        
        # 3.2 选分数最高的expert_capacity个token的索引
        top_token_indices = np.argsort(expert_scores)[-expert_capacity:]  # 取最后1024个（分数最高）
        selected_tokens = tokens_flat[top_token_indices, :]  # 该专家选中的token
        
        # 3.3 专家处理选中的token
        processed_tokens = expert_forward(expert_idx, selected_tokens)  # 批量处理token
        
        # 3.4 把处理结果放回原位置
        for i, token_idx in enumerate(top_token_indices):
            # 把flat的token_idx转回batch和seq_len索引
            b = token_idx // seq_len
            s = token_idx % seq_len
            expert_outputs[expert_idx, b, s, :] = processed_tokens[i]
            
            # 累加该专家对token的贡献（一个token可能被多个专家选中）
            final_output[b, s, :] += processed_tokens[i]
    
    return expert_outputs, final_output

# 模拟专家批量前向计算（适配EC的批量token处理）
def expert_forward(expert_idx, tokens):
    """单个专家批量处理多个token的前向计算"""
    expert_params = np.random.randn(tokens.shape[1], tokens.shape[1])  # 模拟专家参数
    return np.dot(tokens, expert_params)  # 批量计算：[num_tokens, hidden_dim]

到这里，我们就完整讲解好MoE-LoRA架构的核心基础知识了，码字不易，希望大家能给Rocky的辛勤劳动点点赞！！！

7. 优质LoRA模型推荐（持续更新）

目前在AIGC开源社区中，各式各样的前沿LoRA模型层出不穷，促进了AIGC图像生成/AI绘画领域的持续繁荣与创新发展。

Rocky在本章节中为大家梳理了市面上的主流LoRA模型类型，可以从两个层面进行梳理归纳：

使用场景角度：可以分为写实类型、二次元类型、2.5D类型、3D类型、国风类型、游戏类型、插画类型、设计类型、机甲类型、摄影类型、科幻类型、细节增强、Low-Level功能类型等。
模型架构角度：可以分为基于SD 1.x、SD2.x、SDXL、SD 3、FLUX、Z-Image、混元DiT、Qwen-Image等。

上述各个主流类型，Rocky都会持续为大家整理推荐优质的LoRA模型，欢迎大家持续关注！！！

7.1 人物LoRA模型推荐

本小节中，Rocky为大家整理归纳了火影忍者中高人气角色的LoRA模型，包括纲手LoRA、春野樱LoRA、手鞠LoRA、卯月夕顔LoRA、夕日红LoRA、照美冥LoRA、雏田LoRA、红豆LoRA、天天LoRA等。生成效果不错，并且都是可以与不同的AIGC图像生成底模型通用适配的，各位读者可以充分发挥想象进行创造生成。

大家可以关注Rocky的公众号WeThinkIn，后台回复：火影忍者人物LoRA，即可获得资源链接，包含上述提到的全部火影忍者高人气角色LoRA模型资源。

火影忍者人物LoRA推荐
7.2 风格LoRA模型推荐

本小节中，Rocky持续为大家整理归纳风格LoRA模型，包括油画风格LoRA、像素艺术风格LoRA、Nardack风格LoRA、国风LoRA、时尚摄影风格LoRA、电影色彩风格LoRA、涂鸦海报漫画LoRA、精灵感风格LoRA、平面动漫风格LoRA、粘土风格LoRA等。生成效果不错，并且都是可以与不同的AIGC图像生成底模型通用适配的，各位读者可以充分发挥想象进行创造生成。

风格LoRA模型推荐

大家可以关注Rocky的公众号WeThinkIn，后台回复：风格LoRA，即可获得资源链接，包含上述提到的全部风格LoRA模型资源。

7.3 Low-Level功能LoRA模型推荐（美颜、美肤、祛痘、磨皮、精修、画质增强、光影调整等）

本小节中，Rocky持续为大家整理归纳Low-Level功能LoRA模型，包括人像美颜LoRA、人像美肤LoRA、人脸祛痘LoRA、人像磨皮LoRA、人像去油LoRA、人像精修LoRA、图像画质增强LoRA、图像光影调整LoRA等。生成效果不错，并且都是可以与不同的AIGC图像生成底模型通用适配的，各位读者可以充分发挥想象进行创造生成。




Low-Level功能LoRA模型效果示意图
8. 推荐阅读

Rocky会持续分享AIGC的干货文章、实用教程、商业应用/变现案例以及对AIGC行业的深度思考与分析，欢迎大家多多点赞、喜欢、收藏和转发，给Rocky的义务劳动多一些动力吧，谢谢各位！

8.1 深入浅出完整解析扩散模型DDPM、DDIM、Classifier/Classifier-Free Guidance、Rectified Flow核心基础知识

和Rocky一起学习探究扩散模型的本质原理与和核心基础知识，同时不断跟进扩散模型的最新发展。Rocky在本文中对扩散模型的本质做了全面系统的梳理与讲解：

8.2 深入浅出完整解析AI Agent（AI智能体）的核心基础知识

2025年可以说是AI Agent全面落地应用的元年，因此Rocky在持续撰写对AI Agent的全维度解析文章：

8.3 深入浅出完整解析FLUX.1 Kontext和FLUX.1 Krea核心基础知识

Rocky也对FLUX.1 Kontext和FLUX.1 Krea的核心基础知识作了全面系统的梳理与解析：

8.4 深入浅出完整解析DeepSeek系列核心基础知识

Rocky也对DeepSeek系列模型的核心基础知识作了全面系统的梳理与解析：

8.5 深入浅出完整解析Stable Diffusion 3（SD 3）和FLUX.1系列核心基础知识

Rocky也对Stable Diffusion 3和FLUX.1的核心基础知识作了全面系统的梳理与解析：

8.6 深入浅出完整解析Stable Diffusion XL（SDXL）核心基础知识

Rocky也对Stable Diffusion XL的核心基础知识作了全面系统的梳理与解析：

8.7 深入浅出完整解析Stable Diffusion（SD）核心基础知识

Rocky也对Stable Diffusion 1.x-2.x系列模型的核心基础知识做了全面系统的梳理与解析：

8.8 深入浅出完整解析Stable Diffusion中U-Net的前世今生与核心知识

Rocky对Stable Diffusion中最为关键的U-Net结构进行了深入浅出的全面解析，包括其在传统深度学习中的价值和在AIGC中的价值：

8.9 深入浅出完整解析ControlNet核心基础知识

AI绘画作为AIGC时代的一个核心方向，开源社区已经形成以Stable Difffusion为核心，ConrtolNet和LoRA作为首要AI绘画辅助工具的变化万千的AI绘画工作流。

ControlNet正是让AI绘画社区无比繁荣的关键一环，它让AI绘画生成过程更加的可控，更有助于广泛地将AI绘画应用到各行各业中：

8.10 深入浅出完整解析Sora等AI视频大模型核心基础知识

AI绘画和AI视频是两个互相促进、相互交融的领域，2024年无疑是AI视频领域的爆发之年，Rocky也对AI视频领域核心的Sora等大模型进行了全面系统的梳理与解析：

8.11 深入浅出完整解析AIGC时代Transformer核心基础知识

在AIGC时代中，Transformer为AI行业带来了深刻的变革。Transformer架构正在一步一步重构所有的AI技术方向，成为AI技术架构大一统与多模态整合的关键核心基座，大有一统“AI江湖”之势。Rocky也对Transformer模型进行持续的深入浅出梳理与解析：

8.12 深入浅出完整解析主流AI绘画框架核心基础知识

AI绘画框架正是AI绘画“工作流”的运行载体，目前主流的AI绘画框架有Stable Diffusion WebUI、ComfyUI以及Fooocus等。在传统深度学习时代，PyTorch、TensorFlow以及Caffe是传统深度学习模型的基础运行框架，到了AIGC时代，Rocky相信Stable Diffusion WebUI就是AI绘画领域的“PyTorch”、ComfyUI就是AI绘画领域的“TensorFlow”、Fooocus就是AI绘画领域的“Caffe”：

8.13 手把手教你成为AIGC算法工程师，斩获AIGC算法offer！

在AIGC时代中，如何快速转身，入局AIGC产业？如何成为AIGC算法工程师？如何在学校中系统性学习AIGC知识，斩获心仪的AIGC算法offer？

Don‘t worry，Rocky为大家总结整理了全面的AIGC算法工程师成长秘籍，为大家答疑解惑，希望能给大家带来帮助：

8.14 AIGC产业的深度思考与分析

2023年3月21日，微软创始人比尔·盖茨在其博客文章《The Age of AI has begun》中表示，自从1980年首次看到图形用户界面（graphical user interface）以来，以OpenAI为代表的科技公司发布的AIGC模型是他所见过的最具革命性的技术进步。

Rocky也认为，AIGC及其生态，会成为AI行业重大变革的主导力量。AIGC会带来一个全新的红利期，未来随着AIGC的全面落地和深度商用，会深刻改变我们的工作、生活、学习以及交流方式，各行各业都将被重新定义，过程会非常有趣。

那么，在此基础上，我们该如何更好的审视AIGC的未来？我们该如何更好地拥抱AIGC引领的革新？Rocky准备从技术、产品、商业模式、长期主义等维度持续分享一些个人的核心思考与观点，希望能帮助各位读者对AIGC有一个全面的了解：

8.15 算法工程师的独孤九剑秘籍

为了方便大家实习、校招以及社招的面试准备，同时帮助大家提升扩展技术基本面，Rocky将符合大厂和AI独角兽价值的算法高频面试知识点撰写总结成《三年面试五年模拟之独孤九剑秘籍》，并制作成pdf版本，大家可在公众号WeThinkIn后台【精华干货】菜单或者回复关键词“三年面试五年模拟”进行取用：

8.16 深入浅出完整解析AIGC时代中GAN系列模型的前世今生与核心知识

GAN网络作为传统深度学习时代的最热门生成式Al模型，在AIGC时代继续繁荣，作为Stable Diffusion系列模型的“得力助手”，广泛活跃于Al绘画的产品与工作流中：

Rocky一直在运营技术交流群（WeThinkIn-技术交流群），这个群的初心主要聚焦于AI行业话题的讨论与研究，包括但不限于算法、开发、竞赛、科研以及工作求职等。群里有很多AI行业的大牛，欢迎大家入群一起交流探讨～（请备注来意，添加小助手微信Jarvis8866，邀请大家进群～）
