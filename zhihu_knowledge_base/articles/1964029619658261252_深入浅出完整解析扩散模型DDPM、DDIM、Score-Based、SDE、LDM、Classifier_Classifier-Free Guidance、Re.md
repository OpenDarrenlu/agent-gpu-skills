# 深入浅出完整解析扩散模型DDPM、DDIM、Score-Based、SDE、LDM、Classifier/Classifier-Free Guidance、Rectified Flow核心基础知识

**作者**: Rocky Ding​北京科技大学 工学硕士

**原文链接**: https://zhuanlan.zhihu.com/p/1964029619658261252

---

​
目录
收起
1. 主流扩散模型核心资源分享
2. 深入浅出读懂扩散模型DDPM核心基础知识
2.1 零基础深入浅出通俗易懂理解DDPM前向扩散过程
2.2 零基础深入浅出通俗易懂理解DDPM反向去噪过程
2.3 零基础深入浅出通俗易懂理解DDPM的训练优化目标
2.4 零基础深入浅出通俗易懂理解DDPM的通用结构与模块代码原理
3. 深入浅出读懂扩散模型DDIM核心基础知识
3.1 梳理回顾DDPM和DDIM相关核心关联公式
3.2 零基础深入浅出通俗易懂理解DDIM的核心思想
3.3 零基础深入浅出通俗易懂理解DDIM的训练优化目标
3.4 零基础深入浅出通俗易懂理解DDIM的加速采样原理
3.5 零基础深入浅出通俗易懂理解DDIM的反演和插值特性（DDIM Inversion）
3.6 零基础深入浅出通俗易懂理解DDIM的通用结构与模块代码原理
4. 扩散模型在SDE（Stochastic Differential Equations）统一框架视角下的核心基础知识
4.1 SDE统一框架视角下的扩散模型前向扩散过程零基础深入浅出通俗易懂理解
4.2 SDE统一框架视角下扩散模型的反向去噪过程零基础深入浅出通俗易懂理解
4.3 Score-Based扩散模型在SDE统一框架下的本质原理零基础深入浅出通俗易懂理解
4.4 DDPM、DDIM扩散模型在SDE统一框架下的本质原理零基础深入浅出通俗易懂理解
4.5 SDE统一框架中主流求解器&采样算法零基础深入浅出通俗易懂理解
4.6 SDE框架的通用结构与模块代码原理零基础深入浅出通俗易懂理解
5. 深入浅出读懂扩散模型Classifier Guidance和Classifier-Free Guidance核心基础知识
5.1 零基础深入浅出通俗易懂理解Classifier Guidance（分类器引导）的核心基础知识
5.2 零基础深入浅出通俗易懂理解Classifier-Free Guidance（无分类器引导）的核心基础知识
5.3 零基础深入浅出通俗易懂理解Classifier Guidance和Classifier-Free Guidance的通用结构与模块代码原理
6. Rectified Flow扩散模型核心基础知识深入浅出完整讲解
6.1 Flow Matching（FM）、ODE（常微分方程）的核心原理零基础深入浅出通俗易懂理解
6.2 Rectified Flow的核心原理零基础深入浅出通俗易懂理解
6.3 Rectified Flow在Stable Diffusion/FLUX.1中的采样方法优化
6.4 Rectified Flow的通用结构与模块代码原理
7. 扩散模型的未来发展趋势分析（持续更新！）
8. 推荐阅读
8.1 深入浅出完整解析AI Agent（AI智能体）的核心基础知识
8.2 深入浅出完整解析FLUX.1 Kontext和FLUX.1 Krea核心基础知识
8.3 深入浅出完整解析DeepSeek系列核心基础知识
8.4 深入浅出完整解析Stable Diffusion 3（SD 3）和FLUX.1系列核心基础知识
8.5 深入浅出完整解析Stable Diffusion XL（SDXL）核心基础知识
8.6 深入浅出完整解析Stable Diffusion（SD）核心基础知识
8.7 深入浅出完整解析Stable Diffusion中U-Net的前世今生与核心知识
8.8 深入浅出完整解析LoRA（Low-Rank Adaptation）模型核心基础知识
8.9 深入浅出完整解析ControlNet核心基础知识
8.10 深入浅出完整解析Sora等AI视频大模型核心基础知识
8.11 深入浅出完整解析AIGC时代Transformer核心基础知识
8.12 深入浅出完整解析主流AI绘画框架核心基础知识
8.13 手把手教你成为AIGC算法工程师，斩获AIGC算法offer！
8.14 AIGC产业的深度思考与分析
8.15 算法工程师的《三年面试五年模拟》求职秘籍
8.16 深入浅出完整解析AIGC时代中GAN系列模型的前世今生与核心知识
本文的专栏：Rocky Ding的AI算法兵器谱
我的公众号：WeThinkIn
更多AI行业干货内容欢迎关注我的知乎，公众号，专栏～

码字不易！公式深入浅出全面推导确实不易！AIGC技术的本质数学&物理原理生动讲解非常不易！希望大家能多多点赞！

Rocky持续在撰写FLUX.2、Seedream、Z-Image、FLUX.1 Kontext/Krea、FLUX.1、Stable Diffusion 1.x、2.x、XL以及3.x的深入浅出全方位解析文章，希望大家能多多点赞，让Rocky有更多坚持的动力：

大家好，我是Rocky。

“扩散模型是一个优化噪声的蕴含数学&物理本质思想的伟大AI艺术。” —— Rocky Ding

自2022年AIGC时代元年以来，AIGC图像生成/AI绘画领域最引人注目的技术莫过于扩散模型（Diffusion Model）了。

不管是工业界、学术界、竞赛界、应用界乃至投资界，扩散模型已经成为AIGC图像生成领域的绝对热点与跨周期技术基石。

以扩散模型为核心的AIGC产品和AIGC算法解决方案在整个AI业界持续繁荣，结合LoRA、ControlNet、可控生成技术、AI Agent框架等AI配套能力，其长期爆发式发展的势头已经不可逆转。

扩散模型强大的图像生成能力

扩散模型展现的如此深刻的落地影响力也是GAN、VAE等老牌生成式模型所不具备的，也正因此扩散模型天然表现出AI产业级别的“技术核心领导力”与“技术即产品级价值”。

扩散模型、GAN、VAE、Flow四大主流生成模型的对比

但与此相对应的是，扩散模型的数学原理比较深奥难懂，Rocky为了帮助大家更加深入浅出通俗易懂的理解主流扩散模型的本质原理与核心思想，将在本文中为大家娓娓道来扩散模型的本质基础知识。其中包括DDPM核心基础知识讲解、DDPM核心公式深入浅出推导讲解、DDIM核心基础知识讲解、DDIM核心公式深入浅出推导讲解、SDE核心基础知识讲解、SDE核心公式深入浅出推导讲解、Classifier/Classifier-Free Guidance核心基础知识讲解、Classifier/Classifier-Free Guidance核心公式深入浅出推导讲解、Rectified Flow核心基础知识讲解、Rectified Flow核心公式深入浅出推导讲解、扩散模型的未来发展趋势分析以及前沿主流的扩散模型资源分享等干货内容。

让大家能够更好的理解扩散模型的意义与价值，让我们一起繁荣AIGC时代！

1. 主流扩散模型核心资源分享
扩散模型经典入门论文之（扩散模型奠基）：Deep Unsupervised Learning using Nonequilibrium Thermodynamics
扩散模型经典入门论文之（Score-Based奠基）：On Symmetry and Initialization for Neural Networks
扩散模型经典入门论文之（DDPM相关）：Denoising Diffusion Probabilistic Models
扩散模型经典入门论文之（IDDPM相关）：Improved Denoising Diffusion Probabilistic Models
扩散模型经典入门论文之（DDIM相关）：Denoising Diffusion Implicit Models
扩散模型经典入门论文之（Classifier Guidance相关）：Diffusion Models Beat GANs on Image Synthesis
扩散模型经典入门论文之（SDE相关）：Score-Based Generative Modeling through Stochastic Differential Equations
扩散模型经典入门论文之（SDE相关）：Generative Modeling by Estimating Gradients of the Data Distribution
扩散模型经典博客之（SDE相关）：Generative Modeling by Estimating Gradients of the Data Distribution
扩散模型经典入门论文之（CFG相关）：Classifier-Free Diffusion Guidance
扩散模型经典入门论文之（统一解释框架）：Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation
扩散模型经典入门论文之（高价值综述）：Understanding Diffusion Models: A Unified Perspective
扩散模型经典入门论文之（采样器DPM相关）：DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps
扩散模型经典入门论文之（采样器DPM相关）：DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models
扩散模型经典入门论文之（采样器EDM相关）：Elucidating the Design Space of Diffusion-Based Generative Models
扩散模型经典入门论文之（采样器UniPC相关）：UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models
扩散模型经典入门论文之（Prompt控制生成相关）：Prompt-to-Prompt Image Editing with Cross Attention Control
扩散模型经典入门论文之（Img2Img生成相关）：SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations
扩散模型经典入门论文之（Mask控制生成相关）：Blended Latent Diffusion
扩散模型经典入门论文之（LDM和SD相关）：High-Resolution Image Synthesis with Latent Diffusion Models
扩散模型经典入门论文之（Rectified Flow相关）：Neural Ordinary Differential Equations
扩散模型经典入门论文之（Rectified Flow相关）：Flow Matching for Generative Modeling
扩散模型经典入门论文之（Rectified Flow相关）：Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow
扩散模型经典入门论文之（Rectified Flow相关）：Rectified Flow: A Marginal Preserving Approach to Optimal Transport
扩散模型经典入门论文之（高价值综述）：Tutorial on Diffusion Models for Imaging and Vision
扩散模型经典入门论文之（高价值综述）：Step-by-Step Diffusion: An Elementary Tutorial
扩散模型经典入门论文之（SD 3和Rectified Flow相关）：Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
DDPM-OpenAI项目：https://github.com/openai/improved-diffusion
DDIM官方项目：https://github.com/ermongroup/ddim
扩散模型HuggingFace经典博客：https://huggingface.co/blog/annotated-diffusion
扩散模型经典讲解博客：What are Diffusion Models?

Rocky会持续把更多前沿扩散模型的干货资源发布到本节中，让大家更加方便的查找扩散模型的最新资讯。

2. 深入浅出读懂扩散模型DDPM核心基础知识

在最开始，Rocky先从图像生成领域的高观点视角向大家介绍图像生成过程的本质流程。

Rocky认为图像生成本质上就是让神经网络模型学习一个图像数据集所表示的数据分布，之后从尽可能简单的先验分布（比如高斯分布）里随机采样去生成特定的数据分布。比如我们想让神经网络模型生成人像图像，就是要让模型学习一个人像图像集的数据分布。

我们很难直接表示出一个适合采样的复杂数据分布，因此一般情况我们会把学习一个数据分布的问题转换成学习一个简单好采样的分布到复杂分布的高维映射函数。通常来说，一般我们选择的这个简单分布都是标准正态分布（高斯分布，性质足够优良）。

尽管如此，在实际情况中，想要完美学习不同数据分布的映射是十分困难的，特别是当数据集量级扩展到整个世界的所有图像知识范围时。近年来包括扩散模型、GAN、VAE、Flow等主流的生成式模型都用专属于自己的巧妙方法来学习这些映射规则和高维负责函数。而扩散模型则是在众多模型架构中以优异的效果脱引而出，成为AIGC时代的主流图像生成式模型。

图像生成过程示意图

在我们了解了图像生成的本质原理与过程后，Rocky接下来带着大家开始深入浅出全面学习扩散模型的核心基础知识。

当前扩散模型的底层原理大都起源于2020年的《DDPM: Denoising Diffusion Probabilistic Models》工作，DDPM思想可以说是AIGC时代图像生成领域的核心基石之一。

扩散模型是一个隐变量模型（Latent Variable Model），它的核心思想是观测到的数据是由一些我们无法直接观测的、潜在的变量所生成的。这正映射了扩散模型包含的两个核心过程：

前向扩散过程：逐步向数据中添加噪声，直到数据完全变成噪声。
反向去噪过程：从噪声中逐步重建数据。
扩散模型的前向过程和反向过程

我们先对扩散模型有一个基本的初识，扩散模型的前向扩散过程和反向去噪过程的形象动图：

在下面的章节中，Rocky将详细讲解DDPM框架下扩散模型的前向扩散过程和后向去噪过程的原理。

2.1 零基础深入浅出通俗易懂理解DDPM前向扩散过程

扩散模型的前向扩散过程（forward diffusion process）是指对数据逐渐增加高斯噪声直至数据变成随机噪声的过程，是扩散模型中的核心一环。下图中的从右往左路径就代表前向扩散过程：

扩散模型的前向过程和反向过程

对于原始数据\mathbf{x}_0 \sim q(\mathbf{x}_0)，在一共包含 T 步的前向扩散过程中，每一步都是对上一步加噪得到的数据\mathbf{x}_{t-1}按如下方式增加高斯噪声，则前向扩散过程整体上服从多元高斯分布：

q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I}) \\

其中各个符号的含义如下：

\mathbf{x}_{t-1}：第 t-1 步的数据
\mathbf{x}_{t}：第 t 步的数据
\beta_{t}：噪声调度参数，控制每一步添加的噪声量，这里完整的格式为\{\beta_t\}^T_{t=1}，也是每一步所采用的方差，它介于0～1之间（0 < \beta_t < 1）
\mathcal{N}(\mu, \sigma^2)：整个公式是在高斯分布的框架下进行的，均值为 \mu = \sqrt{1 - \beta_t}，方差为 \sigma^2 = \beta_t
\mathbf{I}：单位矩阵，是一个方阵。其主对角线上的元素全为 1，其余元素全为 0。对于一个 t \times t 的单位矩阵 \mathbf{I}_t = \begin{pmatrix} 1 & 0 & 0 & \cdots & 0 \\ 0 & 1 & 0 & \cdots & 0 \\ 0 & 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \cdots & 1 \end{pmatrix} ，表示各维度独立且方差相同

在知道了每个符号的含义后，我们再对前向扩散过程公式进行整体的深入浅出理解，主要分为均值和方差部分。

其中均值这部分\sqrt{1-\beta_{t}}\mathbf{x}_{t-1}保留了前一数据特征的一部分，\sqrt{1-\beta_t} 确保了数据特征逐渐衰减，当 \beta_t 很小时，大部分原始数据特征被保留，当 \beta_t 接近1时，几乎不保留原始数据特征。同时使用 \sqrt{1-\beta_t} 而不是 1-\beta_t 能保证图像特征信息的平滑衰减，防止训练过程中的数值不稳定。

在方差这部分\beta_{t}\mathbf{I}表示每次添加的高斯噪声，\beta_t 控制噪声的强度，\mathbf{I} 表示噪声在各维度上是独立的。

到这里，可能有读者会有疑问，为什么要使用 \beta_t\mathbf{I} 来代表方差 \sigma^2 呢？其中有什么优良特性吗？

我们知道\beta_{t}\mathbf{I} 表示一个对角协方差矩阵，具体形式为： \beta_t \mathbf{I} = \begin{pmatrix} \beta_t & 0 & 0 & \cdots & 0 \\ 0 & \beta_t & 0 & \cdots & 0 \\ 0 & 0 & \beta_t & \cdots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \cdots & \beta_t \end{pmatrix} ，其中对角线元素 = \beta_t，表示每个维度的方差相同；而非对角线元素 = 0，表示不同维度之间没有相关性，即构成独立同分布逻辑。

我们以256×256分辨率图像为例，来分析一下使用单位矩阵的性能优势。256×256的RGB图像其数据维度为256 \times 256 \times 3 = 196,608 维，如果使用全协方差矩阵时实际维度达到了惊人的196,608 \times 196,608，这是一个极其庞大的矩阵，计算非常复杂（存储和计算 O(n^2) 的元素）。同时在图像领域中，不同像素的噪声通常是独立的，没有先验知识表明某个像素的噪声会与其他像素相关。因此在扩散模型中也假设了噪声在各个维度上是独立的且具有相同的方差，所以实际上我们只需要存储对角线元素中的数值来代表每个维度的独立噪声，大大减少了计算量级 (196,608 \times 196,608 \rightarrow 196,608) ，同时只需要 O(1) 的存储空间来用于存储（一个标量 \beta_t）。

Rocky认为这种设置是AI领域中的典型权衡策略，在保持AIGC大模型表达能力的同时，确保计算可行性。正是这种巧妙的设计，使得扩散模型能够处理高维数据（如图像、文本、视频、音频等）而不会遇到计算瓶颈。

除此之外，我们还需要对噪声调度参数 \beta_{t} 有更深的理解。在扩散模型中，我们也将不同采样步（step）对应的方差称作variance schedule或者noise schedule。通常其随着时间步 t 增加而增加（可以预先定义或学习得到），越后面的时间步会采用更大的方差，即满足\beta_1 < \beta_2 < \dots < \beta_T。在一个预先定义好的variance schedule下，如果扩散步数T足够大，那么最终得到的\mathbf{x}_{T}就完全丢失了原始数据特征而变成了一个随机噪声。

同时由于扩散模型的扩散过程采用一个预先定义好的variance schedule，所以扩散过程也是固定的，比如DDPM就采用一个线性的variance schedule。

这样设计的好处是，我们可以通过重复应用这个噪声采样步骤，从初始数据 \mathbf{x}_0 直接计算出任意时刻 t 的 \mathbf{x}_t 的分布，而不需要逐步进行计算。这也让我们得到了扩散模型在扩散过程中的一个重要特性，即我们可以直接基于原始数据\mathbf{x}_{0}来对任意t步的\mathbf{x}_{t}进行采样生成：\mathbf{x}_{t}\sim q(\mathbf{x}_t \vert \mathbf{x}_0)。

接下来，我们站在整个扩散过程的宏观角度看，整个过程定义为一系列的隐变量 \mathbf{x}_{1:T} = \{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T\} ，每个步骤添加一点高斯噪声，生成一个带噪声的隐变量数据\mathbf{x}_{t}，这时整个扩散过程也就形成了一个优美的马尔卡夫链（Markov Chain），完整序列的联合分布如下：

q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = \prod^T_{t=1} q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) \\

这个公式表示的是给定初始数据 \mathbf{x}_0，生成整个噪声序列的概率。\mathbf{x}_{1:T} = \{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T\} 是整个噪声序列，\prod_{t=1}^T 表示马尔可夫链的连乘性质。

其中根据马尔可夫核心性质 p(\mathbf{x}_t | \mathbf{x}_{t-1}, \mathbf{x}_{t-2}, \ldots, \mathbf{x}_0) = p(\mathbf{x}_t | \mathbf{x}_{t-1}) ，即未来状态只依赖于当前状态，与过去状态无关。那么扩散模型中的每个状态 \mathbf{x}_t 只依赖于前一个状态 \mathbf{x}_{t-1}，并且整个过程是固定的（fixed），不包含可学习参数，最终的目标是将数据逐渐转换为纯噪声。

根据马尔可夫核心性质和概率的链式法则，我们可以方便的求得上述完整序列的联合分布的最后结果：

\begin{aligned} q(\mathbf{x}_{1:T}|\mathbf{x}_0) &= q(\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T | \mathbf{x}_0) \\ &= q(\mathbf{x}_1|\mathbf{x}_0) \cdot q(\mathbf{x}_2|\mathbf{x}_1, \mathbf{x}_0) \cdot \cdots \cdot q(\mathbf{x}_T|\mathbf{x}_{T-1}, \ldots, \mathbf{x}_0) \\ &= q(\mathbf{x}_1|\mathbf{x}_0) \cdot q(\mathbf{x}_2|\mathbf{x}_1) \cdot \cdots \cdot q(\mathbf{x}_T|\mathbf{x}_{T-1}) \quad \text{(马尔可夫性质)} \\ &= \prod_{t=1}^T q(\mathbf{x}_t|\mathbf{x}_{t-1}) \end{aligned} \\

至此，Rocky已经深入浅出细致讲解了扩散模型前向扩散过程转化成马尔可夫链的完整推导。

接下来，Rocky给大家梳理制作了扩散模型的前向扩散过程详细图解，方便大家更好的感受其中的奥秘与优美：

扩散模型的前向扩散过程详细图解

好的，接下来，大家再跟着Rocky的脚步，一起研究一下扩散模型在前向扩散过程中每一步具体的采样过程。

我们回到最初这个前向扩散过程的公式：

q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I}) \\

在多元高斯分布 \mathbf{x} \sim \mathcal{N}(\mu, \sigma^2) 中，我们可以使用重参数化技巧（Reparameterization Trick）来进行采样。

重参数化技巧是AI领域的常用技巧，是一种将随机抽样过程重新参数化的方法，使得：

随机性被分离到独立的随机变量中
确定性部分可以进行梯度计算
使得整个过程支持反向传播

对于标准高斯分布来说：

原始采样：\mathbf{x} \sim \mathcal{N}(\mu, \sigma^2)
重参数化：\mathbf{x} = \mu + \sigma \cdot \epsilon ，其中 \epsilon \sim \mathcal{N}(0, \mathbf{I})

也就是说，如果我们要从均值为 μ ，协方差为 \sigma^2 的高斯分布中采样，我们可以先从一个标准高斯分布 ϵ∼N(0,\mathbf{I}) 中采样，然后通过变换 \mathbf{x} = \mu + \sigma\cdot\epsilon \quad \text{其中} \epsilon \sim \mathcal{N}(0, \mathbf{I}) 得到 \mathbf{x} 。

将重参数化技巧应用到扩散模型领域，如下所示：

均值：\mu = \sqrt{1 - \beta_t} x_{t-1}
协方差：\sigma^2 = \beta_t \mathbf{I}
协方差的平方根：\sigma = \sqrt{\beta_t} \mathbf{I}

因此从 \mathbf{x}_{t-1} 到 \mathbf{x}_t 的具体计算采样过程就转化成如下公式：

\mathbf{x}_t = \sqrt{1 - \beta_t} \mathbf{x}_{t-1} + \sqrt{\beta_t} \epsilon,\epsilon \sim \mathcal{N}(0, \mathbf{I}) \\ \mathbf{x}_t = \underbrace{\sqrt{1-\beta_t}\mathbf{x}_{t-1}}_{\text{均值}} + \underbrace{\sqrt{\beta_t}}_{\text{标准差}} \cdot \underbrace{\epsilon}_{\mathcal{N}(0,\mathbf{I})}\\

其中各个符号的含义如下：

\mathbf{x}_{t-1}：第 t-1 步的数据
\sqrt{1 - \beta_t} \cdot \mathbf{x}_{t-1}：前一步数据特征保留部分
\sqrt{\beta_t} \cdot \epsilon：噪声添加部分。系数\sqrt{\beta_t} 控制噪声的强度
\epsilon \sim \mathcal{N}(0, \mathbf{I})：噪声来源，服从标准高斯噪声

这样我们就清楚理解这个公式的具体含义。是扩散模型在前向扩散过程的每一步中，都通过保留上一时刻数据的一部分特征，并添加一部分噪声来得到当前时刻的数据特征。

Rocky在这里举一个通俗易懂的例子，让大家对扩散过程的每一步迭代细节有一个直观的感受：

扩散模型在前向扩散过程中每一步具体的采样过程

到目前为止，我们已经了解了扩散模型在前向扩散中的单步采样过程，我们对全部序列进行递归展开推导，来看看经过累积效应后，多步采样如何进行计算表达。

让我们从 \mathbf{x}_0 开始逐步展开：

\mathbf{x}_1 = \sqrt{1 - \beta_1} \mathbf{x}_0 + \sqrt{\beta_1} \epsilon_1 , \\ \mathbf{x}_2 = \sqrt{1 - \beta_2} \mathbf{x}_1 + \sqrt{\beta_2} \epsilon_2 \\ \mathbf{x}_1代入 \mathbf{x}_2：\\ \mathbf{x}_2 = \sqrt{1 - \beta_2} \left( \sqrt{1 - \beta_1} \mathbf{x}_0 + \sqrt{\beta_1} \epsilon_1 \right) + \sqrt{\beta_2} \epsilon_2 \\ \mathbf{x}_2 = \sqrt{(1 - \beta_2)(1 - \beta_1)} \mathbf{x}_0 + \sqrt{(1 - \beta_2)\beta_1} \epsilon_1 + \sqrt{\beta_2} \epsilon_2 \\

我们令：

\alpha_t = 1 - \beta_t（保留的数据特征比例）
\overline{\alpha}_t = \prod_{i=1}^{t} \alpha_i（累积的数据特征保留比例）

我们就可以重写 \mathbf{x}_2：

\mathbf{x}_2 = \sqrt{\alpha_2 \alpha_1} \mathbf{x}_0 + \sqrt{\alpha_2 \beta_1} \epsilon_1 + \sqrt{\beta_2} \epsilon_2 \\

这时就出现了两个独立的高斯噪声 \epsilon_1, \epsilon_2 \sim \mathcal{N}(0, \mathbf{I}) 的线性组合：

\sqrt{\alpha_2 \beta_1} \epsilon_1 + \sqrt{\beta_2} \epsilon_2 \Rightarrow\\ \sqrt{\alpha_2 \beta_1} \epsilon_1 = \sqrt{\alpha_2 (1 - \alpha_{1})} \epsilon_{1} \sim \mathcal{N}(0, \alpha_2(1 - \alpha_{1})\mathbf{I})\\ \sqrt{\beta_2} \epsilon_2 = \sqrt{1 - \alpha_2} \epsilon_{2} \sim \mathcal{N}(0, (1 - \alpha_2)\mathbf{I}) \\

我们根据高斯分布合并的关键性质：两个独立高斯随机变量之和仍然是高斯分布，且方差相加。从而对这个组合的方差进行合并计算：

\sqrt{\alpha_2 (1 - \alpha_{1})} \epsilon_{1} + \sqrt{1 - \alpha_1} \epsilon_{2} \sim \mathcal{N}(0, [\alpha_2(1 - \alpha_{1}) + (1 - \alpha_1)]\mathbf{I}) \\

\text{Var} = \alpha_2 \beta_1 + \beta_2 = \alpha_2 (1 - \alpha_1) + (1 - \alpha_2) = \alpha_2 - \alpha_1 \alpha_2 + 1 - \alpha_2 = 1 - \alpha_1 \alpha_2 \\

因此，我们可以将两个噪声项合并为一个：

\sqrt{\alpha_2 \beta_1} \epsilon_1 + \sqrt{\beta_2} \epsilon_2 = \sqrt{1 - \alpha_1 \alpha_2} \epsilon ,\epsilon \sim \mathcal{N}(0, \mathbf{I}) \\

我们就可以得到 \mathbf{x}_2采样的最终公式：

\mathbf{x}_2 = \sqrt{\alpha_1 \alpha_2} \mathbf{x}_0 + \sqrt{1 - \alpha_1 \alpha_2} \epsilon = \sqrt{\overline{\alpha}_2} \mathbf{x}_0 + \sqrt{1 - \overline{\alpha}_2} \epsilon \\

最后我们推广到任意 t：

\mathbf{x}_t = \sqrt{\alpha_t \alpha_{t-1} \cdots \alpha_1} \mathbf{x}_0 + \sqrt{1 - \alpha_t \alpha_{t-1} \cdots \alpha_1} \epsilon\\

\mathbf{x}_t = \sqrt{\overline{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \overline{\alpha}_t} \epsilon \\

其中各个符号的含义如下：

\sqrt{\overline{\alpha}_t}：从 x_0 到 x_t 累积保留的数据特征比例
\sqrt{1 - \overline{\alpha}_t}：累积添加的噪声比例
当 t 增大时，\overline{\alpha}_t 减小，x_t 越来越像纯噪声
\epsilon, \epsilon_{t-1}, \epsilon_{t-2} \sim \mathcal{N}(0, \mathbf{I})：独立的标准高斯噪声
当 t \to \infty 时，\overline{\alpha}_t \to 0，x_t \to \mathcal{N}(0, \mathbf{I})，即标准高斯分布

Rocky在这里也举一个生动形象的例子，让我们一起感受扩散模型在前向扩散过程跳中步采样的魅力：

扩散模型在扩散过程跳中步采样案例

到此为止，我们已经完成了扩散模型前向扩散过程的完整推导！

扩散模型的前向过程的完整图解

这也让我们获得了扩散模型中一个极其重要的性质：我们可以直接从原始数据 \mathbf{x}_0 采样任意时间步的噪声数据 \mathbf{x}_t，它建立了从数据空间到噪声空间的直接映射，而无需逐步执行 t 次扩散步骤。

最后，我们再通过反重参数化，就可以最终得到：

q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I}) \\

Rocky在本章节的最后再补充阐述一些本质洞见。

我们可以将\mathbf{x}_{t}理解成是原始数据\mathbf{x}_{0}和随机噪声\mathbf{\epsilon}的线性组合，其中\sqrt{\bar \alpha_t}和\sqrt{1 - \bar{\alpha}_t}为组合系数，我们也可以称两者分别为Signal Rate（信号部分）和Noise Rate（噪声部分），它们的平方和等于1：

(\text{Signal Rate})^2 + (\text{Noise Rate})^2 = \bar{\alpha}_t + (1 - \bar{\alpha}_t) = 1 \\

这也确保了能量守恒，总”能量”在信号和噪声之间分配。

同时在AIGC时代，噪声调度（noise schedule）更倾向于基于\bar \alpha_t而不是\beta_t来定义。因为这样处理更直接，比如我们直接将\bar \alpha_T设定为一个接近0的值，那么就可以保证最终得到的\mathbf{x}_{T}近似为一个随机噪声。

通过精心设计 \bar \alpha_T 序列，我们可以控制收敛速度，在《Improved Denoising Diffusion Probabilistic Models》中提出的cosine schedule如下所示：

\bar{\alpha}_t = \frac{\cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2}{\cos\left(\frac{s}{1 + s} \cdot \frac{\pi}{2}\right)^2}\\

其中 s 是一个小偏移量（如0.008）。其特性是在开始和结束时变化平缓，避免开始阶段噪声添加过快，提供更平滑的训练过程。

2.2 零基础深入浅出通俗易懂理解DDPM反向去噪过程

在本章节中，Rocky将带着大家深入浅出完整讲解扩散模型的反向去噪过程，力求让大家能够深刻理解，让我们开始吧！

我们已经知道扩散模型的核心思想包括前向扩散过程和反向去噪过程，其中：

前向扩散过程：将数据逐步加噪变成纯噪声（已知且固定）
反向去噪过程：从纯噪声逐步去噪生成真实数据（需要训练学习）

下图中从左到右的过程即为反向去噪过程：

扩散模型的前向过程和反向过程

如果我们知道反向去噪过程每一步的真实分布q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)，那么我们就可以从一个随机噪声\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})开始，逐渐去噪来生成一个真实的样本 \mathbf{x}_{0} ，在这种情况下扩散模型的反向去噪过程是经典的数据生成过程。

我们已知前向扩散过程是马尔可夫链，每一步只依赖前一步。和前向扩散过程一样，反向去噪过程也可以定义为马尔可夫链，只不过它是由一系列用神经网络参数化 \theta 的高斯分布来组成：

p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod^T_{t=1} p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) \\ \quad p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))\\

其中各个符号的含义如下：

p(\mathbf{x}_T) = \mathcal{N}(\mathbf{x}_T; 0, \mathbf{I})：起点是标准高斯噪声
p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \Sigma_\theta(\mathbf{x}_t, t))：参数化的高斯分布，每一步都是高斯分布，参数 \theta 由扩散模型学习得到。它们的均值和方差由训练的网络\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)和\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)给出。实际上，扩散模型的目的就是要得到这些训练好的网络，因为它们构成了最终的生成模型。

到这里出现了一个关键问题，那就是真实分布 q(\mathbf{x}_0 \mid \mathbf{x}_t) 难以直接计算，因为这是从噪声推断原始数据的问题，需要知道整个数据集的分布 p_{data}(\mathbf{x}_0)，而这本质上就是生成模型要解决的核心难题。

真实分布 q(x_0∣x_t) 难以直接计算

本质上来说，Rocky认为如果我们能直接计算 q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)，就意味着我们已经知道了从噪声到数据的完美映射，那就不需要训练扩散模型了！

那么我们可以通过什么方法来间接进行计算呢？答案是条件后验分布q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)，这里的关键区别在于多了一个条件 \mathbf{x}_0，我们得到了一个可以解析计算的形式。

这里主要用到了条件概率和贝叶斯定理的基本性质，接下来Rocky为大家进行通俗易懂的详细推导。

P(A|B) = \frac{P(A, B)}{P(B)} , P(A|B,C) = \frac{P(B|A,C) \cdot P(A|C)}{P(B|C)} \\

我们直接应用贝叶斯定理，上述条件后验分布 q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) 即可转化成：

q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0) = \frac{q(\mathbf{x}_t|\mathbf{x}_{t-1},\mathbf{x}_0) \cdot q(\mathbf{x}_{t-1}|\mathbf{x}_0)}{q(\mathbf{x}_t|\mathbf{x}_0)} \\

在上一章节中，我们已经得到了扩散模型前向扩散过程的马尔可夫性质，采样并不依赖 \mathbf{x}_0 （给定 \mathbf{x}_{t-1} 时，\mathbf{x}_0 是冗余信息），因此我们可以进行更多的简化：

q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \frac{q(\mathbf{x}_t | \mathbf{x}_{t-1}, \mathbf{x}_0) \cdot q(\mathbf{x}_{t-1} | \mathbf{x}_0)}{q(\mathbf{x}_t | \mathbf{x}_0)} = \frac{q(\mathbf{x}_t | \mathbf{x}_{t-1}) \cdot q(\mathbf{x}_{t-1} | \mathbf{x}_0)}{q(\mathbf{x}_t | \mathbf{x}_0)} \\

在扩散模型中，我们推导的这个公式非常重要。我们可以发现公式右侧所有组成部分都已知，结合前向扩散过程特性，我们获得了三个完全已知的高斯分布：

q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I}) 表示前向过程的单步转移（已知的加噪过程）
q(\mathbf{x}_{t-1} \mid \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0, (1-\bar{\alpha}_{t-1})\mathbf{I}) 表示从 \mathbf{x}_0 到 \mathbf{x}_{t-1} 的分布
q(\mathbf{x}_t \mid \mathbf{x}_0)= \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I}) 表示从 \mathbf{x}_0 到 \mathbf{x}_t 的分布

因此它们的比值也是一个高斯分布，我们可以解析地计算出 q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0) 这个高斯分布的均值和方差。

接下来让我们一起计算这个高斯分布的概率密度函数。我们易得：

q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I})\\

对于多元高斯分布 \mathcal{N}(x; \mu, \sigma^2)，概率密度函数为：

p(x) \propto \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)\\

将上述三个高斯分布代入 q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) 中：

q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \frac{q(\mathbf{x}_t | \mathbf{x}_{t-1}) \cdot q(\mathbf{x}_{t-1} | \mathbf{x}_0)}{q(\mathbf{x}_t | \mathbf{x}_0)}= \\ \propto \exp\left[-\frac{1}{2}\left( \frac{(\mathbf{x}_t - \sqrt{1-\beta_t}\mathbf{x}_{t-1})^2}{\beta_t} + \frac{(\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0)^2}{1-\bar{\alpha}_{t-1}} - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0)^2}{1-\bar{\alpha}_t} \right)\right] \\ \propto \exp\left(-\frac{1}{2}\left[\frac{(\mathbf{x}_t - \sqrt{\alpha_t} \mathbf{x}_{t-1})^2}{\beta_t} + \frac{(\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0)^2}{1-\bar{\alpha}_{t-1}} - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0)^2}{1-\bar{\alpha}_t}\right]\right) \\ 多项式展开，提取与\mathbf{x}_{t−1}相关的项：\\ \propto\exp\left(-\frac{1}{2}\left[\left(\frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar{\alpha}_{t-1}}\right) \mathbf{x}_{t-1}^2 - \left(\frac{2\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{2\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}} \mathbf{x}_0\right) \mathbf{x}_{t-1} + C(\mathbf{x}_t, \mathbf{x}_0)\right]\right)\\

上述公式中的C(\mathbf{x}_t, \mathbf{x}_0)部分是一个和\mathbf{x}_{t-1}无关的部分，因为会被归一化常数吸收，我们可以不予关注。经过这一番伟大的推导，我们获得的正是一个高斯分布的概率密度函数形式：

\mathcal{N}(\mathbf{x}; \mu, \sigma^2) \propto \exp\left(-\frac{1}{2}\left[\frac{1}{\sigma^2}\mathbf{x}^2 - \frac{2\mu}{\sigma^2}\mathbf{x} + \frac{\mu^2}{\sigma^2}\right]\right)\\

通过对上述两个公式比较系数和配对，我们可以推导得到后验分布q(\mathbf{x}_{t-1} \vert \mathbf{x}_{t}, \mathbf{x}_0)的方差 \tilde{\beta}_t： \frac{1}{\tilde{\beta}_t} = \frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar{\alpha}_{t-1}} \Rightarrow \\ \tilde{\beta}_t = \frac{1}{\frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar{\alpha}_{t-1}}} = \frac{\beta_t(1-\bar{\alpha}_{t-1})}{\alpha_t(1-\bar{\alpha}_{t-1}) + \beta_t}\\ 利用 α_t=1−β_t和 \bar{\alpha}_t=\bar{\alpha}_{t-1}α_t:\\ \begin{aligned} \alpha_t(1-\bar{\alpha}_{t-1}) + \beta_t &= (1-\beta_t)(1-\bar{\alpha}_{t-1}) + \beta_t \\ &= 1 - \bar{\alpha}_{t-1} - \beta_t + \beta_t\bar{\alpha}_{t-1} + \beta_t \\ &= 1 - \bar{\alpha}_{t-1} + \beta_t\bar{\alpha}_{t-1} \\ &= 1 - \bar{\alpha}_{t-1}(1-\beta_t) \\ &= 1 - \bar{\alpha}_{t-1}\alpha_t \\ &= 1 - \bar{\alpha}_t \end{aligned}\\ \Rightarrow \tilde{\beta}_t = \frac{\beta_t(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\\

同样的我们也可以得到后验分布q(\mathbf{x}_{t-1} \vert \mathbf{x}_{t}, \mathbf{x}_0)的均值 \tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0)： \frac{\tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0)}{\tilde{\beta}_t} = \frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}} \mathbf{x}_0 \Rightarrow \\ \tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0) = \tilde{\beta}_t \left(\frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}} \mathbf{x}_0\right) = \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t} \mathbf{x}_0 \\

到此为止，我们已经完整推导了扩散模型的反向去噪过程。我们可以发现方差 \tilde{\beta}_t 是固定值，\tilde{\beta}_t 只依赖于前向扩散过程的参数，不依赖于数据，这简化了学习过程。同时均值 \tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0) 表达式可以重参数化，让神经网络预测噪声而不是直接预测均值。

Rocky在本章节的最后做一个本质总结，我们针对后验分布 q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I}) 推导其均值和方差，并发现其中的一些特性：

方差：\tilde{\beta}_t = \frac{\beta_t(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}（方差是固定值，不依赖数据）
均值：\tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_0（均值是一个依赖\mathbf{x}_0和\mathbf{x}_t的函数）

在下个章节中，Rocky将带着大家用这个分布去推导DDPM扩散模型的优化目标。

扩散模型的反向去噪过程的完整图解
2.3 零基础深入浅出通俗易懂理解DDPM的训练优化目标

在上面的两个章节中，Rocky已经详细讲解了扩散模型的前向扩散过程和反向去噪过程的原理。在本章节中，Rocky将带着大家深入浅出通俗易懂的理解如何将这两种过程与扩散模型的训练过程相结合，从而实现DDPM的训练优化目标。

为了更好的理解如何训练扩散模型，我们可以变换视角将扩散模型视作为一种隐变量模型（Latent Variable Model）：如果我们把扩散模型的前向扩散过程和反向去噪过程中间产生的变量看成隐变量的话，那么扩散模型其实是包含T个隐变量的隐变量模型，同时也是一个特殊的Hierarchical VAEs。

Hierarchical VAEs的生成流程

这时可能会有读者疑惑，什么是Hierarchical VAEs呢？为什么要变换视角与Hierarchical VAEs关联呢？

Don‘t Worry，其中奥妙Rocky将为大家娓娓道来。

我们知道经典VAE（Variational Autoencoder）模型包含了Encoder和Decoder架构，其核心思想是通过Encoder（编码器）将数据映射到低维隐空间，接着通过Decoder（解码器）从隐空间采样并解码重建数据。其中输入只有单一隐变量，同时隐变量维度通常远小于数据维度，而且Encoder和Decoder都可学习的神经网络。

而Hierarchical VAEs则是在经典VAE上进行了扩展，其核心思想是引入多个层次的隐变量，每个层次的隐变量捕捉不同抽象级别的信息。同时隐变量维度依旧小于数据维度，Encoder和Decoder也都可学习的神经网络，并且支持多尺度表示。

传统VAE和Hierarchical VAEs的对比

讲到这里，我们再结合之前对扩散模型的推导理解，可以发现扩散模型的完整流程与Hierarchical VAEs有很多异曲同工之妙。相比Hierarchical VAE来说，扩散模型的隐变量是和原始数据同维度的，而且Encoder（即前向扩散过程）是固定的，将数据映射到层次化隐变量；Decoder则是可学习的神经网络，从隐变量中重建数据。

VAE、Hierarchical VAEs以及扩散模型三者的架构演变关系如下所示：

VAE 
  → 增加隐变量层次 → Hierarchical VAEs
  → 固定Encoder + 同维度隐变量 → 扩散模型

我们已经对扩散模型有了隐变量模型+特殊的Hierarchical VAEs的认知，基于此，我们就可以往下构建扩散模型的优化目标了。

既然扩散模型是隐变量模型，我们就可以基于经典的变分推断来得到变分下界（variational lower bound，VLB，又称ELBO）作为扩散模型这个隐变量模型的最大化优化目标。

下面是Rocky梳理汇总的VAE、Hierarchical VAEs以及扩散模型三者的异同，希望能给读者带来更多直观的理解与感受：

a特征	VAE	Hierarchical VAEs	扩散模型
隐变量结构	单一隐变量 z	层次化隐变量 z1, z2, ...,zn	序列隐变量 x1, x2, ..., xn
隐变量维度	通常远小于数据维度	通常小于数据维度	与数据同维度
Encoder	可学习的神经网络	可学习的神经网络	前向扩散过程（固定的）
Decoder	可学习的神经网络	可学习的神经网络	反向去噪过程（可学习的）
目标函数	ELBO	层次化ELBO	VLB/ELBO
训练稳定性	中等	较难训练	稳定
生成质量	一般	较好	优秀
生成速度	快（单次前向）	快（单次前向）	慢（需要多步迭代）
理论保证	有下界保证	有下界保证	有下界保证
主要应用	数据生成、表示学习	高质量生成、多尺度建模	高质量图像等数据生成
可解释性	中等	较好（层次化结构）	较好（渐进生成过程）
计算复杂度	低	中等	高（训练和生成）
参数数量	相对较少	较多	通常很大
收敛性	容易收敛	可能陷入局部最优	稳定收敛
数据效率	中等	需要更多数据	需要大量数据
模式覆盖	可能模式坍塌	较好的模式覆盖	优秀的模式覆盖
隐空间性质	通常连续、紧凑	层次化、结构化	渐进变化、同维度

好的，Rocky接下来带着大家进行完整的基于变分推断的扩散模型最大化优化目标推导。

我们首先回顾一下上两个章节中推导得到的DDPM扩散模型核心概率分布：

前向扩散过程（固定的Encoder） q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) = \prod_{t=1}^T q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) ，其中 q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t \mathbf{I})

反向去噪过程（可学习的Decoder） p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod_{t=1}^T p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) ， 其中 p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \Sigma_\theta(\mathbf{x}_t, t))

有了上述的基础，我们接着想要最大化观测数据 \mathbf{x}_0 的似然：

\log p_\theta(\mathbf{x}_0)\\

由于包含隐变量 \mathbf{x}_{1:T}，我们需要进行边缘化处理，也是隐变量模型的标准形式：

p_\theta(\mathbf{x}_0) = \int p_\theta(\mathbf{x}_{0:T}) d\mathbf{x}_{1:T}\\

涉及对高维空间 \mathbf{x}_{1:T} 的积分，直接计算这个积分很困难，与其直接优化不可计算的 \log p_\theta(\mathbf{x}_0) ，不如优化它的一个可计算的下界。因此我们引入变分下界（VLB），我们乘以一个等于1的因子：

p_\theta(\mathbf{x}_0) = \int p_\theta(\mathbf{x}_{0:T}) \cdot \frac{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)}{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)} d\mathbf{x}_{1:T}\\

接着我们可以将上式重写为期望形式：

p_\theta(\mathbf{x}_0) = \mathbb{E}_{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)} \left[ \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)} \right]\\

我们应用Jensen不等式，由于对数函数是凹函数，根据Jensen不等式可得：

\log \mathbb{E}[X]\geq \mathbb{E}[\log X]\\

应用到扩散模型中可得：

\log p_\theta(\mathbf{x}_0) = \log \mathbb{E}_{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)} \left[ \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)} \right] \geq \mathbb{E}_{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)} \left[ \log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)} \right]\\

到这里我们就就得到了变分下界（VLB）：

\mathcal{L}_{VLB} = \mathbb{E}_{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)} \left[ \log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)} \right]\\

由于我们通常最小化损失函数，所以我们可以定义扩散模型的训练优化目标为：

\mathcal{L} = -\mathcal{L}_{VLB} = \mathbb{E}_{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)} \left[ \log \frac{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \right]\\

到此为止，我们已经推导获得了扩散模型初步的训练优化目标。

我们接着针对训练目标VLB，还可以进一步分解为很多有意义的项，我们接着进行完整的推导：

\begin{aligned} L &= \mathbb{E}_{q(\mathbf{x}_{1:T}\vert \mathbf{x}_{0})} \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \\ &= \mathbb{E}_{q(\mathbf{x}_{1:T}\vert \mathbf{x}_{0})} \Big[ \log\frac{\prod_{t=1}^T q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{ p_\theta(\mathbf{x}_T) \prod_{t=1}^T p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t) } \Big] \\ &= \mathbb{E}_{q(\mathbf{x}_{1:T}\vert \mathbf{x}_{0})} \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=1}^T \log \frac{q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} \Big] \\ &= \mathbb{E}_{q(\mathbf{x}_{1:T}\vert \mathbf{x}_{0})} \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] \\ &= \mathbb{E}_{q(\mathbf{x}_{1:T}\vert \mathbf{x}_{0})} \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_t\vert\mathbf{x}_{t-1}, \mathbf{x}_{0})}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] & \text{ ;use } q(\mathbf{x}_t \vert \mathbf{x}_{t-1}, \mathbf{x}_0)=q(\mathbf{x}_t \vert \mathbf{x}_{t-1})\\ &= \mathbb{E}_{q(\mathbf{x}_{1:T}\vert \mathbf{x}_{0})} \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \Big( \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)}\cdot \frac{q(\mathbf{x}_t \vert \mathbf{x}_0)}{q(\mathbf{x}_{t-1}\vert\mathbf{x}_0)} \Big) + \log \frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] & \text{ ;use Bayes' Rule }\\ &= \mathbb{E}_{q(\mathbf{x}_{1:T}\vert \mathbf{x}_{0})} \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \sum_{t=2}^T \log \frac{q(\mathbf{x}_t \vert \mathbf{x}_0)}{q(\mathbf{x}_{t-1} \vert \mathbf{x}_0)} + \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] \\ &= \mathbb{E}_{q(\mathbf{x}_{1:T}\vert \mathbf{x}_{0})} \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{q(\mathbf{x}_1 \vert \mathbf{x}_0)} + \log \frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big]\\ &= \mathbb{E}_{q(\mathbf{x}_{1:T}\vert \mathbf{x}_{0})} \Big[ \log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_T)} + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} - \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1) \Big] \\ &= \mathbb{E}_{q(\mathbf{x}_{T}\vert \mathbf{x}_{0})}\Big[\log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_T)}\Big]+\sum_{t=2}^T \mathbb{E}_{q(\mathbf{x}_{t}, \mathbf{x}_{t-1}\vert \mathbf{x}_{0})}\Big[\log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)}\Big] - \mathbb{E}_{q(\mathbf{x}_{1}\vert \mathbf{x}_{0})}\Big[\log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)\Big] \\ &= \mathbb{E}_{q(\mathbf{x}_{T}\vert \mathbf{x}_{0})}\Big[\log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_T)}\Big]+\sum_{t=2}^T \mathbb{E}_{q(\mathbf{x}_{t}\vert \mathbf{x}_{0})}\Big[q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)\log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)}\Big] - \mathbb{E}_{q(\mathbf{x}_{1}\vert \mathbf{x}_{0})}\Big[\log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)\Big] \\ &= \underbrace{D_\text{KL}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T))}_{L_T} + \sum_{t=2}^T \underbrace{\mathbb{E}_{q(\mathbf{x}_{t}\vert \mathbf{x}_{0})}\Big[D_\text{KL}(q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t))\Big]}_{L_{t-1}} -\underbrace{\mathbb{E}_{q(\mathbf{x}_{1}\vert \mathbf{x}_{0})}\log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)}_{L_0} \end{aligned}

可以看到最终的优化目标共包含T+1项，我们可以将其归纳为 L_T,L_t,L_0 三大项。

其中 L_0 = -\mathbb{E}_{q(\mathbf{x}_1 \mid \mathbf{x}_0)} \log p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)代表重构项（Reconstruction Term），衡量从第一个隐变量 \mathbf{x}_1 重建原始数据 \mathbf{x}_0 的能力，优化的是负对数似然，类似于VAE中的重构损失。

L_T = D_{KL}(q(\mathbf{x}_T \mid \mathbf{x}_0) \| p(\mathbf{x}_T)) 代表先验匹配项（Prior Matching Term），计算扩散模型前向扩散过程的最终分布 q(\mathbf{x}_T \mid \mathbf{x}_0) 和标准高斯先验分布 p(\mathbf{x}_T) = \mathcal{N}(0, \mathbf{I})的KL散度，为生成过程提供正确的起点。这个KL散度没有训练参数，近似为0，因为先验p(\mathbf {x}_T)=\mathcal{N}(\mathbf{0}, \mathbf{I})而扩散过程最后得到的随机噪声q(\mathbf{x}_{T}\vert \mathbf{x}_{0})也近似为\mathcal{N}(\mathbf{0}, \mathbf{I})。

L_{t-1} = \mathbb{E}_{q(\mathbf{x}_t \mid \mathbf{x}_0)} \left[ D_{KL}(q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \| p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)) \right], \quad t=2,\dots,T 代表去噪匹配项（Denoising Matching Terms），每个时间步 t，比较真实的后验分布 q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) 与学习的反向过程 p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)的KL散度，确保神经网络学习的去噪步骤近似逼近真实的去噪方向，这是扩散模型训练的核心。

接下来，Rocky将带着大家完整推导我们提炼的扩散模型训练的核心 L_{t-1} 。

我们之前已经推导扩散模型的反向去噪过程的参数化高斯分布，现在可以直接使用了：

真实后验分布：q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)=\mathcal{N}(\mathbf{x}_{t-1}; {\tilde{\boldsymbol{\mu}}} (\mathbf{x}_t, \mathbf{x}_0), {\sigma_t^2} \mathbf{I})
扩散模型的学习分布：p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), {\sigma_t^2} \mathbf{I})

其中上面公式中假设了两个分布的方差相同，针对两个相同方差的高斯分布，我们使用KL散度的通用计算公式如下：

\begin{aligned} &D_{KL}(\mathcal{N}(\mu_1, \sigma^2 \mathbf{I}) \| \mathcal{N}(\mu_2, \sigma^2 \mathbf{I})) \\ &= \frac{1}{2}\left(\text{tr}\left(\frac{\sigma^2 \mathbf{I}}{\sigma^2 \mathbf{I}}\right) + (\mu_2 - \mu_1)^T(\sigma^2 \mathbf{I})^{-1}(\mu_2 - \mu_1) - n + \log\frac{\det(\sigma^2 \mathbf{I})}{\det(\sigma^2 \mathbf{I})}\right) \\ &= \frac{1}{2}\left(n + \frac{1}{\sigma^2}\|\mu_2 - \mu_1\|^2 - n + 0\right) \\ &= \frac{1}{2\sigma^2} \|\mu_2 - \mu_1\|^2 \end{aligned}\\

将上面的通用计算公式带入到 L_{t-1} 中（由于我们需要在整个数据集上优化，需要加上对\mathbf{x}_0的数学期望），我们能够得到最新的优化目标，核心是希望神经网络学模型习到的均值\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)和后验分布的均值{\tilde{\boldsymbol{\mu}}} (\mathbf{x}_t, \mathbf{x}_0)一致：

L_{t-1} = \mathbb{E}_{\mathbf{x}_{0}}\Big(\mathbb{E}_{q(\mathbf{x}_t \mid \mathbf{x}_0)} \left[ D_{KL}(q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \| p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)) \right]\Big) = \mathbb{E}_{\mathbf{x}_{0},\mathbf{\epsilon}\sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left[ \frac{1}{2\sigma_t^2} \|\tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0) - \mu_\theta(\mathbf{x}_t, t)\|^2 \right]\\

同时DDPM论文中发现直接预测均值并不是最好的选择，这样的训练过程并不优雅。从上一章节的的推导中我们已经得到扩散模型反向去噪过程中的后验分布均值 \tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0) 表达式：

\tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0\\

我们再根据扩散模型前向扩散过程中推导得到的特性，得到 \mathbf{x}_0 ：

\mathbf{x_t}(\mathbf{x_0},\mathbf{\epsilon}) = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})\\ \Rightarrow \mathbf{x}_0 = \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \epsilon}{\sqrt{\bar{\alpha}_t}}\\

接着我们将 \mathbf{x}_0 表达式代入 \tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0)，再结合由于 \alpha_t + \beta_t = 1 和 \bar{\alpha}_t = \bar{\alpha}_{t-1} \alpha_t，可得：

\begin{aligned} \tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0) &= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \cdot \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \epsilon}{\sqrt{\bar{\alpha}_t}} \\ &= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\beta_t}{1 - \bar{\alpha}_t} \cdot \frac{\sqrt{\bar{\alpha}_{t-1}}}{\sqrt{\bar{\alpha}_t}} (\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \epsilon)\\ &= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\beta_t}{\sqrt{\alpha_t}(1 - \bar{\alpha}_t)} (\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \epsilon) \\ &= \frac{1}{\sqrt{\alpha_t}} \left( \frac{\alpha_t(1 - \bar{\alpha}_{t-1}) + \beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon \right)\\ &= \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon \right)\\ \end{aligned}\\

然后，我们再将神经网络也进行重参数化，这里的\mathbf{\epsilon}_\theta是一个基于神经网络的拟合函数，表示扩散模型将预测噪声而不是直接预测均值：

\mu_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(\mathbf{x}_t, t) \right)\\

最后，我们将 \tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0) 和 \mu_\theta(\mathbf{x}_t, t) 的最新表达式代入损失函数中，结合 \mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon 可得：

\begin{aligned} L_{t-1} &= \mathbb{E}_{\mathbf{x}_0, \mathbf{\epsilon}\sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left[ \frac{1}{2\sigma_t^2} \left\| \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon \right) - \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(\mathbf{x}_t, t) \right) \right\|^2 \right] \\ &= \mathbb{E}_{\mathbf{x}_0, \mathbf{\epsilon}\sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left[ \frac{1}{2\sigma_t^2} \cdot \frac{1}{\alpha_t} \left\| \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} (\epsilon - \epsilon_\theta(\mathbf{x}_t, t)) \right\|^2 \right] \\ &= \mathbb{E}_{\mathbf{x}_0, \mathbf{\epsilon}\sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left[ \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)} \| \epsilon - \epsilon_\theta(\mathbf{x}_t, t) \|^2 \right]\\ &=\mathbb{E}_{\mathbf{x}_0, \mathbf{\epsilon}\sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left[ \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)} \left\| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t) \right\|^2 \right]\end{aligned}\\

DDPM发现预测噪声比预测均值效果更好，同时去掉权重系数可以简化训练并提高性能。最终我们能够得到简化后的扩散模型训练的损失函数：

L_{t-1}^{\text{simple}} = \mathbb{E}_{\mathbf{x}_0, \mathbf{\epsilon}\sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left[ \left\| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t) \right\|^2 \right]\\

这里的t在 [1, T] 范围内取值（其中取1时对应L_0）。由于去掉了不同t的权重系数，所以这个简化的目标其实是VLB优化目标进行了reweight。

到此为止，我们终于完成了扩散模型训练优化目标函数的伟大推导过程！从最后的推到结果我们可以看到，虽然扩散模型的底层原理推导十分复杂，但我们最终得到的优化目标非常简洁，本质上是让神经网络模型预测的噪声和真实的噪声一致。

在DDPM论文中，通过对比实验也证明了使用扩散模型预测噪声会比预测均值的训练效果要好，同时采用简化版本的优化目标比训练优化VLB目标效果要好：

DDPM官方论文中的训练对比结果

在实际训练中，DDPM扩散模型的训练流程设计得直观而高效，主要包括以下步骤：

从训练集中随机选取一个数据样本
在 1 到 T 的范围内随机选择一个时间步 t ，生成随机高斯噪声
据此计算当前时刻的含噪声数据（如下图中Training部分红色框标注）
将含噪声数据输入扩散模型神经网络中以预测噪声
计算预测噪声与真实噪声之间的L2损失
最后基于该损失计算梯度并更新神经网络参数
扩散模型DDPM的训练和采样流程图

等DDPM扩散模型训练完成后，推理生成新样本的采样过程同样清晰明了：

从一个符合高斯分布的随机噪声出发
利用已训练的扩散模型神经网络预测噪声，进而计算条件概率分布的均值（上图中Sampling部分红色框标注）
随后使用该均值加上标准差与随机噪声的乘积得到迭代中数据
逐步迭代直至时刻 t=0 ，完成新样本的生成（最后一步不再添加噪声）

总的来说，Rocky认为扩散模型虽然理论基础涉及复杂的概率推导，但最终落地为简洁高效的训练与生成流程，兼具理论深度与工程实用性。

2.4 零基础深入浅出通俗易懂理解DDPM的通用结构与模块代码原理

未完待续，大家敬请期待！！！

码字确实不易，希望大家能多多点赞！！！

3. 深入浅出读懂扩散模型DDIM核心基础知识

在之前章节中，Rocky已经深入浅出详细讲解了DDPM的核心基础知识，相信大家对DDPM已经有一个深刻的理解。

那么，在本章节中，Rocky将带着大家在DDPM的基础上，全面讲解DDIM（Denoising Diffusion Implicit Models）的核心基础知识。DDIM是对DDPM的一个关键性改进，它从根本上重新思考了扩散模型的采样过程。

在我们深入学习DDIM之前，我们首先要理解DDPM存在的瓶颈：

采样速度极慢：DDPM的反向去噪过程需要完整地模拟训练时的前向扩散过程。如果扩散步数用了1000步，这时生成数据也需要1000步。每一步都需要神经网络的推理，这导致生成一张图片需要几十秒到几分钟，无法满足实时应用的需求。
生成过程是随机的：由于反向去噪过程的每一步都注入了新的随机噪声，所以数据生成过程是随机的。这虽然有利于多样性，但使得对生成结果的精确控制和复现（如插值）变得困难。

DDIM的核心目标就是解决这两个问题，其最大的突破是：在保持训练方式与DDPM一致的前提下，实现更快速、更确定的生成过程。

接下来，就让我们一起开始研究DDIM的思想吧！

3.1 梳理回顾DDPM和DDIM相关核心关联公式

因为DDIM是在DDPM基础上的扩展，因此在本章节中，Rocky首先带着大家回顾我们推导得到的DDPM关键公式与结论，让我们能够在学习DDIM思想时更加从容。

在DDPM中，前向扩散过程通过一系列步骤向数据中添加高斯噪声，定义为一个马尔卡夫链：

q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = \prod^T_{t=1} q(\mathbf{x}_t \vert \mathbf{x}_{t-1})\\ \quad q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{\frac{\bar\alpha_t}{\bar\alpha_{t-1}}} \mathbf{x}_{t-1}, \Big(1-\frac{\bar\alpha_t}{\bar\alpha_{t-1}}\Big)\mathbf{I}) \\

同时DDPM前向扩散过程中的噪声方差调度参数\beta_t为：

\beta_t = \Big(1-\frac{\bar\alpha_t}{\bar\alpha_{t-1}}\Big) \\

DDPM前向扩散过程的一个重要特性是可以直接用\mathbf{x}_0来对任意的\mathbf{x}_t进行采样：

q(\mathbf{x}_t \vert \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{{\bar\alpha}_t} \mathbf{x}_0, (1 - {\bar\alpha}_t)\mathbf{I}) \\

在DDPM的反向去噪过程中，DDPM通过神经网络p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t)训练来近似拟合真实的后验分布 q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)，也定义为一个马尔卡夫链：

p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod^T_{t=1} p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t)\\ \quad p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))\\

同时我们推导发现后验分布可以表示为一个高斯分布：

q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I}) \\

其中后验分布的方差\tilde{\beta}_t = \frac{\beta_t(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}（方差是固定值，不依赖数据）；后验分布的均值\tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_0（均值是一个依赖\mathbf{x}_0和\mathbf{x}_t的函数）

DDPM使用神经网络 \epsilon_\theta(\mathbf{x}_t, t) 来预测噪声，并通过预测的噪声来估计 \mathbf{x}_0：

\hat{x}_0 = \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}} ，在生成过程中，从 \mathbf{x}_t 生成 \mathbf{x}_{t-1} 的计算就是基于这个估计。

最后，我们推导得到DDPM的简化损失函数是： L_{t-1}^{\text{simple}} = \mathbb{E}_{\mathbf{x}_0, \mathbf{\epsilon}\sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left[ \left\| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t) \right\|^2 \right]\\

到此为止，我们已经回顾了第二章节中我们推导过的DDPM核心公式与认知，这些关键内容将在后续的章节中使用。

3.2 零基础深入浅出通俗易懂理解DDIM的核心思想

首先，在之前章节中我们本质推导后获得的DDPM简化损失函数如下所示： L_{t-1}^{\text{simple}} = \mathbb{E}_{\mathbf{x}_0, \mathbf{\epsilon}\sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left[ \left\| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t) \right\|^2 \right]\\

而DDIM的核心正是在此基础上重新思考了扩散模型的扩散过程。

DDIM通过仔细研究DDPM的优化目标，发现DDPM实际上只依赖边缘分布 q(\mathbf{x}_t \mid \mathbf{x}_0) （代表单个时间步的分布，不考虑中间路径，比如只描述在步骤500时，半图像半噪声的状态），而不是整个联合分布 q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) （代表整个扩散路径的概率，比如描述了图像变成噪声的所有可能路径）。

我们对之前推导得到的DDPM完整版优化目标的每一项进行观察： \begin{aligned} L &= \mathbb{E}_{q(\mathbf{x}_{T}\vert \mathbf{x}_{0})}\Big[\log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_T)}\Big]+\sum_{t=2}^T \mathbb{E}_{q(\mathbf{x}_{t}\vert \mathbf{x}_{0})}\Big[q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)\log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)}\Big] - \mathbb{E}_{q(\mathbf{x}_{1}\vert \mathbf{x}_{0})}\Big[\log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)\Big] \\ &= \underbrace{D_\text{KL}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T))}_{L_T} + \sum_{t=2}^T \underbrace{\mathbb{E}_{q(\mathbf{x}_{t}\vert \mathbf{x}_{0})}\Big[D_\text{KL}(q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t))\Big]}_{L_{t-1}} -\underbrace{\mathbb{E}_{q(\mathbf{x}_{1}\vert \mathbf{x}_{0})}\log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)}_{L_0} \end{aligned}

我们可以发现：

L_0 = \mathbb{E}_{q(\mathbf{x}_1 \mid \mathbf{x}_0)}[\log p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)] 只依赖边缘分布
L_{t-1} = \mathbb{E}_{q(_t \mid \mathbf{x}_0)}[D_{KL}(q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \| p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t))] 依赖于边缘分布 q(\mathbf{x}_t \mid \mathbf{x}_0) 和条件分布 q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)
L_T = D_{KL}(q(\mathbf{x}_T \mid \mathbf{x}_0) \| p(\mathbf{x}_T)) 也只依赖于边缘分布 q(\mathbf{x}_T \mid \mathbf{x}_0)

在所有这些项中，联合分布 q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) 从未直接出现！这也是符合我们的直观认知的，在DDPM中，我们从未需要生成整个扩散路径 \mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_T 。我们只需要在任意时间步 t 从 \mathbf{x}_0 直接采样 \mathbf{x}_t 。

这给我们一个深刻的启发：DDPM并不只依赖于其马尔可夫链的前向扩散过程，而是依赖于其学习到的真实数据分布与噪声分布之间的关联。马尔可夫框架确实提供了计算 q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) 的一种方法，但不是唯一方法！我们可以定义一个非马尔可夫的前向扩散过程，从而解除马尔可夫约束。也就是说，可以有多种不同的前向扩散过程（不同的 q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) ）都能产生相同的边缘分布，只要边缘分布 q(\mathbf{x}_t \mid \mathbf{x}_0) 保持一致，我们可以自由设计联合分布。

但值得注意的一个点是，我们在之前推导获得DDPM的优化目标时，需要知道条件分布q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)，之前我们根据贝叶斯公式推导这个分布时是通过分布q(\mathbf{x}_t \vert \mathbf{x}_{t-1})和依赖扩散模型前向扩散过程的马尔卡夫链特性的。如果要解除对前向扩散过程马尔可夫特性的约束，那么我们就需要直接定义这个条件分布q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)。

基于上述的分析，DDIM论文中“异想天开”将前向扩散过程的完整分布直接重新定义为：

q_{\sigma}(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = q_{\sigma}(\mathbf{x}_{T} \vert \mathbf{x}_0)\prod^T_{t=2} q_{\sigma}(\mathbf{x}_{t-1} \vert \mathbf{x}_{t},\mathbf{x}_{0}) \\

其中要满足终点分布q_{\sigma}(\mathbf{x}_{T} \vert \mathbf{x}_0)=\mathcal{N}(\sqrt{\bar\alpha_T}\mathbf{x}_{0},(1-{\bar\alpha_T})\mathbf{I})，并且DDIM重新定义了对于所有的t\ge2时要满足条件分布：

q_\sigma(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \sqrt{\bar\alpha_{t-1}}\mathbf{x}_0 + \sqrt{1 - \bar\alpha_{t-1} - \sigma_t^2} \frac{\mathbf{x}_t - \sqrt{\bar\alpha_t}\mathbf{x}_0}{\sqrt{1 - \bar\alpha_t}}, \sigma_t^2 \mathbf{I})\\

上述公式本质上是DDIM将后验分布一般化了。其中 \epsilon = \frac{\mathbf{x}_t - \sqrt{\bar\alpha_t} \mathbf{x}_0}{\sqrt{1 - \bar\alpha_t}} ，这里的方差\sigma_t^2是一个实数参数，控制着生成过程中的随机性程度，设置不同的数值就代表了不一样的分布。所以q_{\sigma}(\mathbf{x}_{1:T} \vert \mathbf{x}_0)总的来说其实是一系列的推理分布总和。

当 \sigma_t = 0：确定性生成，条件分布退化为狄拉克\delta函数（Dirac delta function），在概率论中它表示一个确定性的分布，q_\sigma(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \delta\left(\mathbf{x}_{t-1} - \left[\sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1}} \cdot \frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0}{\sqrt{1 - \bar{\alpha}_t}}\right]\right)
当 \sigma_t > 0：随机生成
特定选择：当选择 \sigma_t^2 = \frac{(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \beta_t 时，DDIM就退化为DDPM
大方差 (\sigma_t^2 大)：更多的随机探索，多样性好但可能质量下降
小方差 (\sigma_t^2 小)：更确定的生成，质量稳定但多样性受限

同时均值 \mu_\sigma(\mathbf{x}_t, \mathbf{x}_0) 被构造为：

\mu_\sigma(\mathbf{x}_t, \mathbf{x}_0) = \sqrt{\bar\alpha_{t-1}} \mathbf{x}_0 + \sqrt{1 - \bar\alpha_{t-1} - \sigma_t^2} \cdot \frac{\mathbf{x}_t - \sqrt{\bar\alpha_t} \mathbf{x}_0}{\sqrt{1 - \bar\alpha_t}}\\

其中\bar\alpha_t 和 \bar\alpha_{t-1} 是累积乘积参数，\sigma_t 是一个可调参数。从上面公式中我们可以得知条件分布q_\sigma(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)的均值也被DDIM定义为一个依赖\mathbf{x}_0和\mathbf{x}_t的组合函数。 \sqrt{\bar{\alpha}_{t-1}} x_0 表示我们希望 x_{t-1} 中包含的原始数据特征部分。 \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{\sqrt{1 - \bar{\alpha}_t}} 中的\frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{\sqrt{1 - \bar{\alpha}_t}} 是代表从 x_t 估计的噪声方向，\sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} 控制这个方向项的强度。

之所以DDIM定义出上述的整体公式框架形式，是因为根据q_{\sigma}(\mathbf{x}_{T} \vert \mathbf{x}_0)，我们可以通过数学归纳法证明（这部分的证明可以直接参考DDIM论文），对于所有的t依旧均满足：

q_{\sigma}(\mathbf{x}_t \vert \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{{\bar\alpha}_t} \mathbf{x}_0, (1 - {\bar\alpha}_t)\mathbf{I}) \\

到此为止，我们可以看到DDIM论文中定义的联合分布q_{\sigma}(\mathbf{x}_{1:T} \vert \mathbf{x}_0)并没有直接依赖DDPM原有的前向扩散过程，同时也满足了我们前面要讨论的两个条件：边缘分布q_{\sigma}(\mathbf{x}_t \vert \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{{\alpha}_t} \mathbf{x}_0, (1 - {\alpha}_t)\mathbf{I})，同时已知后验分布q_\sigma(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)。

在此基础上，我们可以按照和DDPM的一样的方式去推导优化目标，最终也会得到同样的L^{\text{simple}}（只是VLB的系数不同）。 论文也给出了一个前向扩散过程是非马尔可夫链的示例，如下图所示，这里前向扩散过程是q_\sigma(\mathbf{x}_{t} \vert \mathbf{x}_{t-1}, \mathbf{x}_0)，由于生成\mathbf{x}_t不仅依赖\mathbf{x}_{t-1}，而且依赖\mathbf{x}_0，所以是一个非马尔可夫链：

DDPM和DDIM的前向扩散过程和反向去噪过程对比
3.3 零基础深入浅出通俗易懂理解DDIM的训练优化目标

有了上面几节的讲解，Rocky在本章节中将带着大家进行DDIM训练优化目标的完整推导。

在DDIM的反向去噪过程中，我们不知道真实的 \mathbf{x}_0，因此使用神经网络 \epsilon_\theta(\mathbf{x}_t, t) 预测的 \hat{\mathbf{x}}_0 进行估计：

\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t - \sqrt{1 - \bar\alpha_t} \epsilon_\theta(\mathbf{x}_t, t)}{\sqrt{\bar\alpha_t}}\\

将上式代入DDIM后验分布的重参数化表达式：

\begin{aligned}\\ \mathbf{x}_{t-1} &= \sqrt{\bar\alpha_{t-1}} \hat{\mathbf{x}}_0 + \sqrt{1 - \bar\alpha_{t-1} - \sigma_t^2} \cdot \frac{\mathbf{x}_t - \sqrt{\bar\alpha_t} \hat{\mathbf{x}}_0}{\sqrt{1 - \bar\alpha_t}} + \sigma_t \epsilon_t\\ &= \sqrt{\bar\alpha_{t-1}}\Big(\underbrace{\frac{\mathbf{\mathbf{x}}_t-\sqrt{1-\bar\alpha_{t}}\mathbf{\epsilon}_\theta(\mathbf{\mathbf{x}}_t, t)}{\sqrt{\bar\alpha_{t}}}}_{\text{predicted}\ \mathbf{\mathbf{x}}_0}\Big) + \underbrace{\sqrt{1 - \bar\alpha_{t-1} - \sigma_t^2} \cdot \mathbf{\epsilon}_\theta(\mathbf{\mathbf{x}}_t, t)}_{\text{direction pointing to }\ \mathbf{\mathbf{x}}_t} + \underbrace{\sigma_t\epsilon_t}_{\text {random noise}}\\ \end{aligned}\\

这里将DDIM的反向去噪过程分成三个部分：一是由预测的\mathbf{x}_0来产生的，我们不知道真实的 \mathbf{x}_0，因此使用神经网络预测的 \epsilon_\theta(\mathbf{x}_t, t) 代替；二是由指向\mathbf{x}_t的部分；三是随机噪声 \sigma_t \epsilon_t （这里\epsilon_t是与\mathbf{x}_t无关的噪声，用于注入随机性）。

同时DDIM论文中重新定义了方差：

\sigma_t^2 = \eta \cdot \hat{\beta}_t\\

其中 \hat{\beta}_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t} \beta_t是DDPM反向去噪过程的方差，我们可以推导得到：

\hat{\beta}_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t} \left(1 - \frac{\bar\alpha_t}{\bar\alpha_{t-1}}\right) \Rightarrow \sqrt{\hat{\beta}_t} = \sqrt{(1 - \bar\alpha_{t-1})/(1 - \bar\alpha_t)} \sqrt{(1 - \bar\alpha_t/\bar\alpha_{t-1})} \\

这时候我们需要考虑两种情况：

当 \eta = 1时：\sigma_t^2 = \hat{\beta}_t，生成过程与DDPM相同，包含随机噪声。
当 \eta = 0时：\sigma_t^2 = 0，生成过程没有随机噪声，成为一个确定性过程。一旦初始噪声 x_T 确定，整个生成过程就固定了，从而可以为后续的加速采样做铺垫。

DDIM通过引入后验分布 q_\sigma(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) 和参数 \sigma_t，推广了DDPM的反向去噪过程。当 \sigma_t^2 = 0 时，生成过程变为确定性，允许更高效的采样。公式的推导基于对前向扩散过程的线性假设和噪声预测，确保了与DDPM相同的训练目标。

我们再梳理一下DDIM的完整反向去噪流程，从 \mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I}) 开始，对于 t = T, T-1, \ldots, 1：

DDIM扩散模型预测噪声：\epsilon_\theta = \epsilon_\theta(\mathbf{x}_t, t)
估计原始数据，利用当前时刻 t 的带噪数据 \mathbf{x}_t 和预测出的噪声 \epsilon_\theta(\mathbf{x}_t, t) ，来估计原始数据 \hat{\mathbf{x}}_0 ：\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}}
计算下一步，根据估计的 \hat{\mathbf{x}}_0 和当前的 \mathbf{x}_t ，朝着反向去噪过程中 \mathbf{x}_{t-1} 的方向“迈进一步”：\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{\mathbf{x}}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \epsilon_\theta + \sigma_t z,z \sim \mathcal{N}(0, \mathbf{I})
3.4 零基础深入浅出通俗易懂理解DDIM的加速采样原理

经过上面几章节的“天才”假设和伟大推导过程，我们终于获得了DDIM扩散模型的优化目标。

那么，DDIM扩散模型是如何实现反向去噪过程的加速采样呢？

我们回顾一下，在DDPM扩散模型中，前向扩散和反向去噪过程都需要 T 步（ T=1000 ），这导致生成速度很慢。而我们已经知道DDIM虽然训练过程和DDPM一致，但只要满足边缘分布 q(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\alpha_t}\mathbf{x}_0, (1-\alpha_t)\mathbf{I}) ，就不存在明确的前向扩散过程规则约束（不一定是马尔可夫链），可以构造各种不同的前向扩散分布，这代表我们可以定义一个更短步数的前向扩散过程，进行跳步采样。

DDIM论文中的跳步采样示意图

具体来说，针对原始的序列[1, ..., T]，DDPM必须按顺序 0 \rightarrow 1 \rightarrow 2 \rightarrow ... \rightarrow T进行采样。DDIM则可以采样一个长度为 S 的子序列 \tau =[\tau_1,...,\tau_S]，例如 \tau = [0, ..., T-200, T-100, T] ，其中 S \ll T （例如 S=20, T=1000 ）。我们将[\mathbf{x}_{\tau _{1}},...,\mathbf{x}_{\tau_{S}}]的前向扩散过程定义为一个马尔卡夫链，并且它们保持边缘分布q(\mathbf{x}_{\tau_{i}} \vert \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{{\bar\alpha}_{\tau_{i}}} \mathbf{x}_0, (1 - {\bar\alpha}_{\tau_{i}})\mathbf{I})。

然后，DDIM的反向去噪过程也可以用这个子序列的反向马尔卡夫链来替代，我们直接将之前推导获得的DDIM反向去噪过程采样公式中的 t 和 t-1 替换为子序列中的 \tau_i 和 \tau_{i-1} 。这时对于子序列，DDIM的反向去噪过程转变为：

\mathbf{x}_{\tau_{i-1}} = \underbrace{\sqrt{\bar\alpha_{\tau_{i-1}}}\Big(\frac{\mathbf{x}_{\tau_{i}}-\sqrt{1-\bar\alpha_{\tau_{i}}}\mathbf{\epsilon}_\theta(\mathbf{x}_{\tau_{i}}, \tau_{i})}{\sqrt{\bar\alpha_{\tau_{i}}}}\Big)}_{\text {预测的}\mathbf{x}_{0}{部分}} + \underbrace{\sqrt{1 - \bar\alpha_{\tau_{i-1}} - \sigma_{\tau_{i}}^2} \cdot \mathbf{\epsilon}_\theta(\mathbf{x}_{\tau_{i}}, \tau_{i})}_{\text {确定性的}\tau_{i}{部分}}+\underbrace{\sigma_{\tau_{i}}\epsilon}_{\text {随机噪声部分}}\\

上述公式与标准DDIM公式形式相同，只是时间步从连续的 t 变成了子序列的\tau_i。为了实现上述的加速生成，本质上是对DDIM的前向扩散过程进行了有效的拆解。

q_{\sigma,\tau}(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = q_{\sigma,\tau}(\mathbf{x}_{T} \vert \mathbf{x}_0)\prod^S_{i=1} q_{\sigma}(\mathbf{x}_{\tau_{i-1}} \vert \mathbf{x}_{\tau_{i}},\mathbf{x}_{0})\prod_{t\in \bar\tau}q_{\sigma,\tau}(\mathbf{x}_{t}\vert \mathbf{x}_{0}) \\

其中\tau代表子序列时间点集合，\bar{\tau}代表不在子序列中的时间点集合（\bar{\tau} = \{1,\ldots,T\} \setminus \tau）

q_{\sigma,\tau}(\mathbf{x}_T|\mathbf{x}_0)代表最终时间步T的分布，服从分布q(\mathbf{x}_T|\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_T; \sqrt{\bar\alpha_T}\mathbf{x}_0, (1-\bar\alpha_T)\mathbf{I})
\prod_{i=1}^{S} q_{\sigma}(\mathbf{x}_{\tau_{i-1}}|\mathbf{x}_{\tau_i}, \mathbf{x}_0)代表子序列上的条件分布，形成马尔可夫链，这是实际用于生成的部分，服从分布q_{\sigma}(\mathbf{x}_{\tau_{i-1}}|\mathbf{x}_{\tau_i}, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{\tau_{i-1}}; \mu_\sigma(\mathbf{x}_{\tau_i}, \mathbf{x}_0), \sigma_{\tau_i}^2 \mathbf{I})
\prod_{t\in\bar{\tau}} q_{\sigma,\tau}(\mathbf{x}_t|\mathbf{x}_0)代表非子序列时间点的分布，形成”星状图”，每个\mathbf{x}_t只依赖于\mathbf{x}_0，彼此独立，主要用于变分下界(VLB)计算，不用于实际生成

接下来Rocky带着大家详细推导其中的奥妙。有了上面的前向扩散过程，我们可以得到DDIM反向去噪过程的对应公式。同时反向去噪过程，我们也只用马尔可夫链的那部分来生成：

p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \underbrace {\prod^S_{i=1} p_\theta(\mathbf{x}_{\tau_{i-1}} \vert \mathbf{x}_{\tau_{i}})}_{\text{use to produce sample}} \times \underbrace {\prod_{t\in \bar\tau}p_\theta(\mathbf{x}_{0} \vert \mathbf{x}_{t})}_{\text {only for VLB}} \\

其中前向扩散过程和反向去噪过程关键对应关系：

q(\mathbf{x}_T|\mathbf{x}_0) ↔ p(\mathbf{x}_T)
q_{\sigma}(\mathbf{x}_{\tau_{i-1}}|\mathbf{x}_{\tau_i}, \mathbf{x}_0) ↔ p_\theta(\mathbf{x}_{\tau_{i-1}}|\mathbf{x}_{\tau_i})
q_{\sigma,\tau}(\mathbf{x}_t|\mathbf{x}_0) ↔ p_\theta(\mathbf{x}_0|\mathbf{x}_t)

最后，DDIM论文中给出了两种子序列采样方法，分别为：

Linear：均匀采样，适用于大多数情况。比如 \tau_i = \lfloor c \cdot i \rfloor，其中\tau_S = T，c = \frac{T}{S}。如果T=1000，S=50，则 \tau_i = \lfloor 20 \cdot i \rfloor = [20, 40, 60, \ldots, 1000]
Quadratic：在早期时间步采样更密集，适用于需要精细控制初始生成阶段的数据集。 \tau_i = \lfloor c \cdot i^2 \rfloor 其中\tau_S = T，c = \frac{T}{S^2}。如果T=1000，S=20，则 c = \frac{1000}{400} = 2.5, \quad \tau_i = \lfloor 2.5 \cdot i^2 \rfloor 。 \tau = [2, 10, 22, 40, 62, 90, 122, 160, 202, 250, 302, 360, 422, 490, 562, 640, 722, 810, 902, 1000]

在DDIM论文实验中，使用DDIM仅需50-100步就能生成与DDPM1000步相媲美甚至更好的质量，实现了10-20倍的加速。

Rocky在最后总结归纳了DDIM和DDPM的差异，希望能给大家打来更加直观的理解：

特性	DDPM	DDIM
训练目标	噪声预测损失	与 DDPM 完全相同，预测噪声损失
前向过程	马尔可夫链	非马尔可夫过程
采样过程	随机，必须逐步进行	可选择的随机性，可以是确定的
采样速度	慢（需满步数，如1000步）	快（可跳步，如20-50步）
结果确定性	随机，每次生成都不同	可确定性，结果可复现
应用优势	生成多样性高	快速生成、图像编辑、插值

DDIM解耦了扩散模型的训练和推理采样，揭示了扩散模型的本质能力蕴含在已训练好的噪声预测模型中。它提供了一种强大的“采样器”，让我们能更灵活、高效地利用预训练的扩散模型，是推动扩散模型走向实用化的关键一步。它不是一个新的训练模型，而是同一个训练好的分数模型的另一种采样方式。它的出现第一次揭示了：同一个扩散模型可以同时支持随机采样和确定性采样，并为后来的所有快速采样算法奠定了理论基础。

3.5 零基础深入浅出通俗易懂理解DDIM的反演和插值特性（DDIM Inversion）

DDIM反演是扩散模型领域最具革命性的技术之一，它首次实现了真实图像到扩散模型隐空间的精确、确定性编码，使得我们可以对任意真实图像进行隐空间编辑、插值、修复和风格迁移。这项技术完全基于 DDIM 的确定性采样特性，不需要重新训练任何模型，是所有现代扩散模型图像编辑功能（如img2img、ControlNet条件控制）的理论基础。

DDIM反演是DDIM生成过程的逆过程：

生成过程（正向）：从初始高斯噪声\mathbf{x}_T出发，通过确定性迭代得到真实图像\mathbf{x}_0：\mathbf{x}_T \xrightarrow{\text{DDIM采样}} \mathbf{x}_0
反演过程（逆向）：从真实图像\mathbf{x}_0出发，通过反向确定性迭代得到对应的初始噪声\mathbf{x}_T：\mathbf{x}_0 \xrightarrow{\text{DDIM反演}} \mathbf{x}_T

其中DDIM反演最关键的性质，由于标准DDIM（\sigma_t=0）的生成过程是完全确定性的双射（bijection），因此反演过程满足：

用反演得到的\mathbf{x}_T进行DDIM正向采样，将精确重建出原始图像\mathbf{x}_0

这是DDPM永远无法做到的——DDPM的生成过程是随机的，同一个\mathbf{x}_T会生成不同的\mathbf{x}_0，同一个\mathbf{x}_0也对应无数个可能的\mathbf{x}_T，不存在唯一的逆映射。

我们已经知道DDIM确定性采样（\sigma_t=0）从带噪图像x_t生成上一步的去噪图像x_{t-1}的公式为：

\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \hat{\mathbf{x}}_0(\mathbf{x}_t, t) + \sqrt{1-\bar{\alpha}_{t-1}} \cdot \epsilon_\theta(\mathbf{x}_t, t)

其中：

\bar{\alpha}：预定义的噪声调度参数
\hat{\mathbf{x}}_0(\mathbf{x}_t, t)：模型从x_t预测的干净原始图像
\epsilon_\theta(\mathbf{x}_t, t)：模型从x_t预测的添加的高斯噪声

同时我们发现DDIM的两个预测目标可以互相转换：

\hat{\mathbf{x}}_0(\mathbf{x}_t, t) = \frac{\mathbf{x}_t - \sqrt{1-\alpha_t} \cdot \epsilon_\theta(\mathbf{x}_t, t)}{\sqrt{\alpha_t}}

\epsilon_\theta(\mathbf{x}_t, t) = \frac{\mathbf{x}_t - \sqrt{\alpha_t} \cdot \hat{\mathbf{x}}_0(\mathbf{x}_t, t)}{\sqrt{1-\alpha_t}}

接下来，Rocky就带着大家进行DDIM反演的核心推导。DDIM反演的目标是：已知x_{t-1}和真实的x_0，求对应的x_t。

我们从正向采样公式出发，通过代数变形解出x_t。将\hat{x}_0(x_t, t) = \frac{x_t - \sqrt{1-\alpha_t} \epsilon_\theta}{\sqrt{\alpha_t}}代入正向采样公式：

x_{t-1} = \sqrt{\alpha_{t-1}} \cdot \frac{x_t - \sqrt{1-\alpha_t} \epsilon_\theta}{\sqrt{\alpha_t}} + \sqrt{1-\alpha_{t-1}} \cdot \epsilon_\theta

将含x_t的项和含\epsilon_\theta的项分开：

x_{t-1} = \frac{\sqrt{\alpha_{t-1}}}{\sqrt{\alpha_t}} x_t + \left( \sqrt{1-\alpha_{t-1}} - \frac{\sqrt{\alpha_{t-1}} \sqrt{1-\alpha_t}}{\sqrt{\alpha_t}} \right) \epsilon_\theta

将上式变形，把x_t单独放在左边：

x_t = \frac{\sqrt{\alpha_t}}{\sqrt{\alpha_{t-1}}} x_{t-1} + \left( \sqrt{1-\alpha_t} - \frac{\sqrt{\alpha_t} \sqrt{1-\alpha_{t-1}}}{\sqrt{\alpha_{t-1}}} \right) \epsilon_\theta

在反演过程中，我们知道最终的干净图像是真实的x_0，因此模型预测的噪声\epsilon_\theta可以用真实值代替：

\epsilon_\theta = \frac{x_t - \sqrt{\alpha_t} x_0}{\sqrt{1-\alpha_t}}

将其代入上式并化简，最终得到DDIM反演的核心递推公式：

\boldsymbol{x_t = \sqrt{\alpha_t} x_0 + \sqrt{1-\alpha_t} \cdot \frac{x_{t-1} - \sqrt{\alpha_{t-1}} x_0}{\sqrt{1-\alpha_{t-1}}}}

DDIM论文证明了，标准DDIM的采样过程等价于求解以下ODE：

\frac{d\bar{x}(t)}{dt} = \frac{d\sigma(t)}{dt} \cdot \epsilon_\theta\left( \frac{\bar{x}(t)}{\sqrt{\sigma^2(t)+1}} \right)

其中：

\bar{x}(t) = \frac{x(t)}{\sqrt{\alpha(t)}}是重参数化后的隐变量
\sigma(t) = \sqrt{\frac{1-\alpha(t)}{\alpha(t)}}是信噪比的平方根

反演的本质：

生成过程：从t=T（\sigma(T)\to\infty，对应纯噪声）积分到t=0（\sigma(0)=0，对应干净图像）
反演过程：从t=0积分到t=T，得到对应的初始隐变量

这一视角将DDIM反演与神经ODE（Neural ODE）统一起来，为后续更高效的反演算法（如基于欧拉法、龙格-库塔法的快速反演）奠定了基础。

DDIM反演彻底改变了扩散模型的应用范式，从”只能从噪声生成图像”变成了”可以编辑任意真实图像”。

图像隐空间编辑：

属性编辑：将真实图像反演到隐空间，在隐空间中沿着特定属性方向（如”微笑”、”年龄”）移动，再采样得到编辑后的图像
风格迁移：将内容图和风格图分别反演，在隐空间中融合两者的特征
图像插值：将两张图像反演得到x_{T1}和x_{T2}，在隐空间中进行球面线性插值（slerp），得到平滑的过渡动画

基于参考图的生成：

img2img（图生图）：这是Stable Diffusion最常用的功能之一。其核心思想是：
将输入参考图反演到第k步的噪声x_k
从x_k开始，使用文本引导进行后续的采样过程
这样生成的图像既保留了参考图的结构和布局，又符合文本描述的内容

图像修复与补全：

将待修复的图像反演到隐空间，在隐空间中对缺失区域进行约束，再采样得到修复后的图像
相比传统修复方法，扩散模型修复的结果更自然、语义更连贯

视频生成与编辑：

对视频的每一帧进行DDIM反演，得到隐空间序列
在隐空间中进行平滑、插值或编辑，再逐帧采样生成编辑后的视频

总的来说，Rocky认为DDIM反演的核心本质就是利用DDIM生成过程的确定性双射性质，通过反向迭代DDIM采样公式，将真实图像精确编码为扩散模型隐空间中的初始噪声，实现了”真实图像→隐噪声→重建图像”的完美闭环。

这项技术的伟大之处在于，它没有提出任何新的模型或训练方法，仅仅通过改变采样的顺序，就赋予了扩散模型编辑真实世界的能力。没有DDIM反演，就没有今天的Stable Diffusion、FLUX、MidJourney、Nano Banana、GPT-Image、Seedream、Z-Image等所有实用的图像创作技术工具。

码字确实不易，希望大家能多多点赞！！！

3.6 零基础深入浅出通俗易懂理解DDIM的通用结构与模块代码原理

未完待续，大家敬请期待！！！

码字确实不易，希望大家能多多点赞！！！

4. 扩散模型在SDE（Stochastic Differential Equations）统一框架视角下的核心基础知识

2021年发表的高价值论文《Score-Based Generative Modeling through Stochastic Differential Equations》中，首次将所有主流扩散模型（DDPM、DDIM、Score-Based）统一到连续时间随机微分方程（SDE，Stochastic Differential Equations）框架下。这不仅揭示了各类扩散模型具有共同的数学本质，还为设计新模型架构和新采样算法提供了通用的理论基础，可以说是为扩散模型领域提供了“大道至简”式的技术思想突破与整合。

作为扩散模型理论中的重要进展和技术基石，SDE框架也为后续的基于常微分方程（ODE）的扩散模型（如Stable Diffusion 3、FLUX、Seedream、Nano Banana、GPT-Image-2等）提供了理论基础。

扩散模型在随机微分方程（SDE，Stochastic Differential Equations）中的统一框架

接下来，在本章节中Rocky将带着大家深入浅出讲解在SDE框架下扩散模型统一的本质原理与核心基础知识，希望能给大家带来醍醐灌顶的帮助与灵感。话不多说，直接让我们开始吧！

4.1 SDE统一框架视角下的扩散模型前向扩散过程零基础深入浅出通俗易懂理解

经过之前章节的深入学习，我们已经知道扩散模型本质上是一个随机过程，分为前向扩散过程和反向去噪过程：

前向扩散过程：逐渐向原始数据中加入噪声，直到变成纯噪声（高斯分布）
反向去噪过程：从纯噪声中逐步去噪，恢复出原始数据分布

经典扩散模型DDPM使用 \mathbf{x}_0 \to \mathbf{x}_T,\mathbf{x}_T \to \mathbf{x}_0 的离散时间步描述前向扩散过程和反向去噪过程。

除了上述表达随机过程的方式，最经典的用于刻画随机过程的数学工具，我们知道还有随机微分方程（SDE，Stochastic Differential Equations），并且SDE框架提供了一种连续时间的表达方式，使得随机过程的建模方式呈现出不一样的优雅。

在本章节中，我们先探讨SDE框架视角下的前向扩散过程。在连续时间 t \in [0,1] 下，前向扩散过程可用如下正向SDE方程进行建模（数据→噪声）：

d\mathbf{x} = \boldsymbol{f}_t(\mathbf{x}) \, dt + g_t \, d\boldsymbol{w} = \lim_{\Delta t \rightarrow 0}{(\mathbf{x}_{t+\Delta t} - \mathbf{x}_t)}\\

其中各个符号含义如下：

\boldsymbol{f}_t(\mathbf{x}) 是漂移系数，描述确定性演化，是向量函数
g_t 是扩散系数，描述随机噪声强度，是标量函数
\mathbf{w} 是标准维纳过程（标准布朗运动），布朗运动的增量可以表示为\Delta \mathbf{w} = \sqrt{\Delta t} \cdot \boldsymbol{\epsilon}，其中\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})是标准高斯噪声
d\boldsymbol{w} 是标准维纳过程（布朗运动）的微分，表示极小的高斯噪声
t \in [0,1] 是连续时间变量，不是物理时间，而是噪声强度的归一化度量，可以任意缩放区间（如[0,T]）。t=0对应原始干净图像\mathbf{x}_0（无噪声），t=1：对应纯高斯噪声\mathbf{x}_1（完全被噪声覆盖），t越大，图像中的噪声占比越高

正向SDE方程中 \boldsymbol{f}_t(\mathbf{x}) , g_t 并不是固定的，而是可以通过人为设计的。在后面的章节中我们会知道，其实DDPM本质上就是SDE框架下设计了特定的 \mathbf{f},g 得到的，在这里我们先按下不表。

先前DDPM的离散前向扩散过程为： \mathbf{x}_t \to \mathbf{x}_{t+1} ，现在SDE框架下的连续形式为 \mathbf{x}_t\to \mathbf{x}_{t+\Delta t} ，当我们取一个很小的时间步长时（当 \Delta t \to 0），SDE框架下的连续前向扩散过程可以离散近似为：

\mathbf{x}_{t+\Delta t} \approx \mathbf{x}_t + \underbrace{\boldsymbol{f}_t(\mathbf{x}_t) \Delta t }_{确定部分} + \underbrace{g_t \sqrt{\Delta t} \, \boldsymbol{\varepsilon}}_{随机部分}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(0, \mathbf{I})\\

确定性部分：\boldsymbol{f}_t(\mathbf{x}_t) \Delta t，表示按照漂移方向移动。
随机性部分：g_t \sqrt{\Delta t} \, \boldsymbol{\varepsilon}，表示加入高斯噪声，其方差为 g_t^2 \Delta t。

之前DDPM的离散序列为 1 \to 2 \to ... \to T ，现在SDE框架离散近似为 0 \to \Delta t \to 2\Delta t \to ...\to 1 。

我们知道SDE框架的解是一系列随机变量的连续集合 \{\mathbf{x}(t)\}_{t\in[0,T]}，即每一个时刻都对应着一个随机变量，如果我们沿着时间维度去观察，这些随机变量的所有采样结果会构成一条随机轨迹（如下图所示）。随机变量 \mathbf{x}(t) 的边缘概率分布可以记为 p_t(\mathbf{x})，在前向扩散过程中，初始的 p_0(\mathbf{x}) 和最终的 p_T(\mathbf{x}) 分别对应原始数据分布和趋向于纯高斯噪声的分布。

SDE框架下扩散模型基于一个连续时间随机过程的前向扩散过程

到此为止，我们已经构建出了SDE框架下的扩散模型连续前向扩散过程，同时在很小的时间步长（当 \Delta t \to 0）下，SDE框架的连续前向扩散过程可以进行离散近似。接下来我们就需要构建一个SDE框架下的连续反向去噪过程。

4.2 SDE统一框架视角下扩散模型的反向去噪过程零基础深入浅出通俗易懂理解

在本章节中，Rocky接着带着大家详细讲解SDE框架下扩散模型的反向去噪过程。在反向去噪过程中，我们想从高斯噪声分布 p(\mathbf{x}_1) 中恢复出真实数据分布 p(\mathbf{x}_0) ，这时就需要求解反向SDE（Reverse SDE）。

那么反向SDE的解析表达式是什么样的呢？Don‘t Worry，Rocky将带着大家娓娓推导逐步领会。

通过前一章节的推导，根据连续正向SDE的离散化结果，我们已经知道给定 \mathbf{x}_t 时，\mathbf{x}_{t+\Delta t} 的条件分布服从高斯分布：

p(\mathbf{x}_{t+\Delta t} | \mathbf{x}_t) = \mathcal{N}\big(\mathbf{x}_{t+\Delta t}; \mathbf{x}_t + \boldsymbol{f}_t(\mathbf{x}_t) \Delta t, \; g_t^2 \Delta t \mathbf{I}\big)\\

我们可以获得高斯分布的完整概率密度函数（包含一个复杂的归一化常数）：

p(\mathbf{x}_{t+\Delta t} \mid \mathbf{x}_t) = \frac{1}{(2\pi g_t^2 \Delta t)^{d/2}} \exp\left( -\frac{\|\mathbf{x}_{t+\Delta t} - \mathbf{x}_t - \boldsymbol{f}_t(\mathbf{x}_t)\Delta t\|^2}{2g_t^2 \Delta t} \right)

但在后续推导中，我们只关心概率的相对大小和对\mathbf{x}_t的梯度，而归一化常数是与\mathbf{x}_t无关的常数，因此可以忽略，用\propto表示”正比于”得到：

p(\mathbf{x}_{t+\Delta t} \mid \mathbf{x}_t) \propto \exp\left( -\frac{\|\mathbf{x}_{t+\Delta t} - \mathbf{x}_t - \boldsymbol{f}_t(\mathbf{x}_t)\Delta t\|^2}{2g_t^2 \Delta t} \right)

我们希望得到的是反向去噪过程中的条件概率分布p(\mathbf{x}_t | \mathbf{x}_{t+\Delta t})，即已知未来状态 \mathbf{x}_{t+\Delta t}，推测当前状态 \mathbf{x}_t 的数据分布。我们可以通过贝叶斯公式和取对数策略对条件概率分布 p(\mathbf{x}_t | \mathbf{x}_{t+\Delta t}) 进行展开推导，得到其概率密度函数：

\begin{aligned} p(\mathbf{x}_t | \mathbf{x}_{t+\Delta t}) &= \frac{p(\mathbf{x}_{t+\Delta t} | \mathbf{x}_t) p(\mathbf{x}_t)}{p(\mathbf{x}_{t+\Delta t})} \\ &\Rightarrow_{取对数} \log p(\mathbf{x}_{t+\Delta t} | \mathbf{x}_t) + \log p(\mathbf{x}_t) - \log p(\mathbf{x}_{t+\Delta t})\\ &\propto -\frac{\|\mathbf{x}_{t+\Delta t} - \mathbf{x}_t - \boldsymbol{f}_t(\mathbf{x}_t)\Delta t\|^2}{2g_t^2 \Delta t} + \log p(\mathbf{x}_t) - \log p(\mathbf{x}_{t+\Delta t})\\ \end{aligned}\\

其中 p(\mathbf{x}_{t+\Delta t} \mid \mathbf{x}_t) 代表正向转移概率，已知是高斯分布；p(\mathbf{x}_t)代表 t 时刻所有带噪声数据的边缘分布；p(\mathbf{x}_{t+\Delta t})代表 t+Δt 时刻所有带噪声数据的边缘分布。

接着我们再对不好分析的 \log p(\mathbf{x}_{t+\Delta t}) 进行泰勒展开，不好分析是因为p(\mathbf{x}_t) 和 p(\mathbf{x}_{t+\Delta t}) 都是边缘分布，我们只有有限样本，不知道其解析式。但在 \Delta t 很小时，\mathbf{x}_{t+\Delta t} 与 \mathbf{x}_t 非常接近，因此我们可以用 \mathbf{x}_t 处的信息来近似 \mathbf{x}_{t+\Delta t} 处的对数概率密度，这时候就需要使用泰勒展开策略。由于 p(\mathbf{x}_t) 依赖于 (\mathbf{x}, t) 双变量，利用\Delta t很小的特性，可以在(\mathbf{x}_t, t)处进行泰勒展开。我们容易知道对于二元函数f(t, \mathbf{x})，在点(t, \mathbf{x}_t)处的一阶泰勒展开为：

\begin{aligned} &f(t+\Delta t, \mathbf{x}_{t+\Delta t}) \approx f(t, \mathbf{x}_t) + \frac{\partial f}{\partial t}(t, \mathbf{x}_t)\Delta t + \nabla_{\mathbf{x}} f(t, \mathbf{x}_t) \cdot (\mathbf{x}_{t+\Delta t} - \mathbf{x}_t)\\ &\Rightarrow \log p_{t+\Delta t}(\mathbf{x}_{t+\Delta t}) = \log p_t(\mathbf{x}_t) + (\mathbf{x}_{t+\Delta t} - \mathbf{x}_t) \cdot \nabla_{\mathbf{x}} \log p_t(\mathbf{x}_t) + \Delta t \cdot \frac{\partial}{\partial t} \log p_t(\mathbf{x}_t) + O((\Delta t)^2, \|\Delta\mathbf{x}\|^2)\\ \end{aligned}\\

其中各个符号含义如下：

\frac{\partial f}{\partial t}： f 对时间t的偏导数
\nabla_{\mathbf{x}} f： f 对空间变量\mathbf{x}的梯度（向量）
O(\cdot)表示高阶项，当\Delta t很小时可以忽略

将把泰勒展开的结果带入SDE框架中，并进行化简可以得到：

\log p(\mathbf{x}_{t+\Delta t}) \approx \log p(\mathbf{x}_t) + (\mathbf{x}_{t+\Delta t} - \mathbf{x}_t) \cdot \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) + \Delta t \frac{\partial}{\partial t} \log p(\mathbf{x}_t) \\

第 0 项：\log p_t(\mathbf{x}_t)表示当前 t 时刻的对数概率密度。
第 1 项：(\mathbf{x}_{t+\Delta t} - \mathbf{x}_t) \cdot \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)，空间变化的一阶近似，其中 \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) 就是分数函数（Score Function），SDE框架中非常关键的一项，也是后面要讲到的Score-Based扩散模型的核心，表示对数概率密度的梯度方向，概率密度增加最快的方向，也就是 "去噪方向"。
第 2 项：\Delta t \cdot \partial_t \log p_t(\mathbf{x}_t)，时间变化的一阶近似，表示概率分布本身随时间的变化率。

接着我们将上式代入 p(\mathbf{x}_t | \mathbf{x}_{t+\Delta t}) 后进行化简，并利用 \Delta t \to 0 忽略高阶项，推导得到：

\begin{aligned} p(\mathbf{x}_t | \mathbf{x}_{t+\Delta t}) &\approx -\frac{\|\mathbf{x}_{t+\Delta t} - \mathbf{x}_t - \boldsymbol{f}_t(\mathbf{x}_t)\Delta t\|^2}{2 g_t^2 \Delta t} \quad + \log p_t(\mathbf{x}_t) \quad - \left[ \log p_t(\mathbf{x}_t) + (\mathbf{x}_{t+\Delta t} - \mathbf{x}_t) \cdot \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) + \Delta t \cdot \partial_t \log p_t(\mathbf{x}_t) \right]\\ &\approx -\frac{\|\mathbf{x}_{t+\Delta t} - \mathbf{x}_t - \boldsymbol{f}_t(\mathbf{x}_t)\Delta t\|^2}{2 g_t^2 \Delta t} - (\mathbf{x}_{t+\Delta t} - \mathbf{x}_t) \cdot \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) - \Delta t \cdot \partial_t \log p_t(\mathbf{x}_t)\\ &\approx -\frac{\|\mathbf{x}_{t+\Delta t} - \mathbf{x}_t\|^2}{2g_t^2\Delta t} + \frac{(\mathbf{x}_{t+\Delta t} - \mathbf{x}_t) \cdot \boldsymbol{f}_t(\mathbf{x}_t)}{g_t^2} - (\mathbf{x}_{t+\Delta t} - \mathbf{x}_t) \cdot \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) - \Delta t \cdot \partial_t \log p_t(\mathbf{x}_t)\\ &\approx -\frac{\|\mathbf{x}_{t+\Delta t} - \mathbf{x}_t\|^2}{2g_t^2\Delta t} + (\mathbf{x}_{t+\Delta t} - \mathbf{x}_t) \cdot \left[ \frac{\boldsymbol{f}_t(\mathbf{x}_t)}{g_t^2} - \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) \right] - \Delta t \cdot \partial_t \log p_t(\mathbf{x}_t)\\ &\approx -\frac{\left\| \mathbf{x}_{t+\Delta t} - \mathbf{x}_t - \left[ \boldsymbol{f}_{t+\Delta t}(\mathbf{x}_{t+\Delta t}) - g_{t+\Delta t}^2 \nabla_{\mathbf{x}_{t+\Delta t}} \log p_{t+\Delta t}(\mathbf{x}_{t+\Delta t}) \right] \Delta t \right\|^2}{2g_{t+\Delta t}^2\Delta t}\\ &\approx \mathcal{N}\left( \mathbf{x}_t; \mathbf{x}_{t+\Delta t} - \left[ \boldsymbol{f}_{t+\Delta t}(\mathbf{x}_{t+\Delta t}) - g_{t+\Delta t}^2 \nabla_{\mathbf{x}_{t+\Delta t}} \log p(\mathbf{x}_{t+\Delta t}) \right] \Delta t, \; g_{t+\Delta t}^2 \Delta t \mathbf{I} \right)\\ \end{aligned}

我们可以发现 \log p(\mathbf{x}_t) 项完全抵消了 ！这是整个推导中最巧妙的一步，通过泰勒展开，我们同时消去了两个未知的边缘分布项。由于 \Delta t 很小，上式中 \Delta t \cdot \partial_t \log p_t(\mathbf{x}_t) 是 O(\Delta t) 量级，而其他项都是O(1)或O(1/\Delta t)量级的，因此相对于其他项可以忽略。并且我们注意到 \mathbf{x}_{t+\Delta t} 已知，因此我们将上式中的函数 \boldsymbol{f} 在 \mathbf{x}_{t+\Delta t} 处进行取值（因为在极限下 \mathbf{x}_t \approx \mathbf{x}_{t+\Delta t}，这在\Delta t \to 0的极限下是精确的），而不是x_t处。

由上述我们推导得到的高斯分布，可以获到反向SDE的离散采样公式：

\mathbf{x}_t = \mathbf{x}_{t+\Delta t} - \left[ \boldsymbol{f}_{t+\Delta t}(\mathbf{x}_{t+\Delta t}) - g_{t+\Delta t}^2 \nabla_{\mathbf{x}_{t+\Delta t}} \log p_{t+\Delta t}(\mathbf{x}_{t+\Delta t}) \right] \Delta t + g_{t+\Delta t} \sqrt{\Delta t} \, \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(0, \mathbf{I})

我们接着进行移项操作：

\mathbf{x}_{t+\Delta t} - \mathbf{x}_t = \left[ \boldsymbol{f}_{t+\Delta t}(\mathbf{x}_{t+\Delta t}) - g_{t+\Delta t}^2 \nabla_{\mathbf{x}_{t+\Delta t}} \log p_{t+\Delta t}(\mathbf{x}_{t+\Delta t}) \right] \Delta t - g_{t+\Delta t} \sqrt{\Delta t} \, \boldsymbol{\varepsilon}

再令 \Delta t \to 0，我们可以得到如下的一些观察发现：

左边的有限差分 \mathbf{x}_{t+\Delta t} - \mathbf{x}_t 变为微分 d\mathbf{x}。虽然我们在推导中使用了正向时间的符号t \to t+\Delta t，但在实际采样时，我们是从t=T开始，逐步减小t到0。不过由于SDE的形式只依赖于当前时刻t，所以无论时间方向如何，方程的形式都是一样的
右边第一项变为 \left[ \boldsymbol{f}_{t}(\mathbf{x}) - g_{t}^2 \nabla_{\mathbf{x}} \log p_{t}(\mathbf{x}) \right] dt，其中离散时间差 \Delta t 变为微分 dt
离散噪声 g_t \sqrt{\Delta t} \boldsymbol{\varepsilon} 变为 g_t d\boldsymbol{w}，其中 d\boldsymbol{w} 是维纳过程的微分。注意到高斯噪声是对称的，-\epsilon和\epsilon服从完全相同的分布，所以负号可以去掉

基于上述观察，当我们取极限 \Delta t \to 0时，最终得到反向SDE的连续形式为：

d\mathbf{x} = \left[ \boldsymbol{f}_t(\mathbf{x}) - g_t^2 \nabla_{\boldsymbol{x}} \log p_t(\mathbf{x}) \right] dt + g_t \, d\boldsymbol{w} \\

反向SDE由三部分组成：

反向漂移项：\boldsymbol{f}_t(\mathbf{x}) dt，这是前向漂移的反向，类似于”回推”确定性的变化。
分数函数项：-g_t^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) dt是核心项，指向数据分布的高概率区域，系数 g_t^2 反映了噪声强度，这一项实现了”去噪”和”概率质量集中”，是SDE框架下最重要的Score Function项，我们需要通过训练神经网络来估计 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})。
随机扩散项：g_t d\boldsymbol{w} 保留了随机性，防止模式坍塌到单个模式，保证了生成样本的多样性。
SDE框架下扩散模型基于一个连续时间随机过程的反向去噪过程

到此为止，经过我们的伟大推导，终于得到了SDE框架下扩散模型通用的前向扩散过程和反向去噪过程！并且正向扩散和反向去噪的一步条件分布都严格服从高斯分布！

同时也严格证明了如果正向过程是一个高斯扩散SDE，那么反向过程也是一个高斯扩散SDE。

在前向 SDE 和反向 SDE 的形式背后，有一个严格的数学定理保证了它们的正确性，这就是 Anderson 时间反转定理。这个定理证明了，任何正向扩散过程的时间反转仍然是一个扩散过程，并且给出了反向过程漂移项和扩散项的显式表达式。正是这个定理，让我们可以从第一性原理出发，去推导和总结SDE框架。

总的来说，我们从前向SDE出发，通过离散化得到前向转移概率，利用贝叶斯公式和泰勒展开推导出反向条件概率的高斯近似，最终取连续极限得到反向SDE。反向SDE的关键在于漂移项中的分数函数 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})，它指向数据分布的高概率区域（概率密度增加的方向），从而实现去噪生成。同时也揭示了分数函数的核心地位，它是反向SDE中唯一未知的、需要学习的参数变量。这意味着：

SDE框架下的扩散模型的本质就是学习高维空间中任意一点的分数函数，然后通过求解反向SDE，沿着分数函数的方向从纯噪声走回真实数据。
SDE框架下正向SDE和反向SDE的示意图

由于分数函数 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})是不可知的，我们需要用神经网络 s_\theta(\mathbf{x}, t) 来拟合近似它，并通过最小化费舍尔散度（Fisher divergence）作为SDE框架下扩散模型的优化目标。

而这也正是Score-Based扩散模型的核心本质，因此在下一章节中，Rocky先带着大家深入浅出理解Score-Based扩散模型的原理，再对其与SDE框架的本质统一进行推导与讲解。

4.3 Score-Based扩散模型在SDE统一框架下的本质原理零基础深入浅出通俗易懂理解

首先，Rocky带着大家清晰的学习领会什么是Score-Based扩散模型。

Score-Based扩散模型的核心思想是：不直接建模数据的概率密度函数，而是建模其对数概率密度的梯度（即分数函数），然后通过朗之万动力学从噪声中逐步生成数据。

Rocky认为我们需要搞清楚设计概率密度→对数概率→对数梯度（分数函数）这一链条的数学原理和物理意义，这是Scored-Based扩散模型的核心价值。

对于数据分布p_{\text{data}}(x)，我们可以将其分数函数（Score Function）定义为：

s(x) = \nabla_x \log p_{\text{data}}(x)

从微积分的基本原理易知，梯度指向函数值增长最快的方向。因此，在数据空间中的任意一个带噪声的图像 \(x_t\) ，分数函数 \(s_\theta(x_t,t)\) 本质上指向的是从 \(x_t\) 到原始干净图像 \(x_0\) 的平均方向。Score-Based扩散模型中，整个反向去噪过程，本质上就是沿着分数函数的方向，一步步从噪声走回数据的"导航系统"。

为什么必须使用对数概率呢？对数变换\log p(x)不是一个随意的数学技巧，而是解决高维数值问题和简化计算的必然选择，有三个不可替代的优势：

解决高维下的数值下溢问题。高维空间中，单个点的概率密度p(x)会极其微小。例如，一个d维标准高斯分布在原点的密度为：

p(0)=\frac{1}{(2\pi)^{d/2}}

当d=3072时，这个值约为10^{-3500}，远小于计算机浮点数能表示的最小值（约10^{-308}），会直接下溢为0。而对数变换将极小的正数转化为中等大小的负数：\log p(0)\approx-8000，完全在计算机的数值范围内。

将乘法转化为加法，简化计算。概率的链式法则是乘法：p(x_1,x_2,...,x_n)=p(x_1)p(x_2|x_1)...p(x_n|x_1,...,x_{n-1})，对数变换后变为加法：

\log p(x_1,...,x_n)=\log p(x_1)+\log p(x_2|x_1)+...+\log p(x_n|x_1,...,x_{n-1})

这不仅计算更快，还能避免多个小数相乘导致的精度损失。

不改变极值点，等价于原优化目标。对数函数是严格单调递增函数，因此：

argmax_x p(x)=argmax_x \log p(x)

这意味着最大化对数似然和最大化原始似然是完全等价的，不会改变模型的最优解。

可能有读者会问，为什么使用分数函数用于扩散模型的去噪呢？接下来Rocky带着大家进行详细讲解。

所有基于概率密度的生成模型，最终都要面对归一化这个问题。对于任意一个灵活的模型（比如用神经网络表示的能量函数E_\theta(x)），我们只能写出未归一化的概率密度：

\tilde{p}_\theta(x) = e^{-E_\theta(x)}

而真实的概率密度必须满足归一化条件\int_{\mathbb{R}^d} p_\theta(x)dx=1，因此需要除以配分函数(Partition Function)Z_\theta：

p_\theta(x) = \frac{\tilde{p}_\theta(x)}{Z_\theta}, \quad Z_\theta = \int_{\mathbb{R}^d} e^{-E_\theta(x')} dx'

而问题的核心是在高维空间中，Z_\theta是完全、绝对、永远无法精确计算的。比如说对于d=3072的CIFAR-10图像，这个积分是在3072维空间上进行的，没有任何解析解。

而在分数函数中，我们把p_\theta(x)=\frac{\tilde{p}_\theta(x)}{Z_\theta}代入：

\begin{align*} s_\theta(x) &= \nabla_x \log\left(\frac{\tilde{p}_\theta(x)}{Z_\theta}\right) \\ &= \nabla_x \log \tilde{p}_\theta(x) - \nabla_x \log Z_\theta \\ &= \nabla_x \log \tilde{p}_\theta(x) \end{align*}

关键结论：\log Z_\theta是一个与x无关的常数（配分函数 Z_\theta 的积分变量 x' 是一个哑变量 (dummy variable)），它的梯度为0！这意味着分数函数s_\theta(x)完全不依赖于配分函数Z_\theta，我们可以用任意灵活的神经网络直接建模s_\theta(x)，完全不需要考虑归一化问题。

我们有了上述的数学技术认知后，还会发现直接估计原始数据的分数函数在高维下会遇到严重的流形问题。因为所有真实世界的高维数据（如图像、音频、文本）都严格服从流形假设(Manifold Hypothesis)，尽管数据点存在于一个d维的高维空间中，但它们实际上只分布在一个维度远小于d的低维光滑流形上。低维光滑流形外的特征分数函数无法准确估计。为了解决这个问题，主流的策略是增加多尺度噪声扰动：

向原始数据中添加一系列不同强度的高斯噪声：x_\sigma = x_0 + \sigma\epsilon, \epsilon \sim \mathcal{N}(0,I)
噪声强度\sigma从小到大变化，覆盖从接近原始数据到接近纯噪声的整个范围
训练一个噪声条件分数网络 s_\theta(x, \sigma) ，估计每个噪声尺度下的分数函数\nabla_x \log p_\sigma(x)

当我们向原始数据x_0中加入强度为\sigma的高斯噪声时：

x_\sigma = x_0 + \sigma\epsilon, \quad \epsilon\sim\mathcal{N}(0,I)

得到的新分布p_\sigma(x)是原始分布p_{\text{data}}(x)与高斯核的卷积：

p_\sigma(x) = \int p_{\text{data}}(x_0)\mathcal{N}(x; x_0, \sigma^2I)dx_0

这个卷积操作有一个极其重要的几何效应：它会把低维流形”吹胀”成一个高维的厚流形。

当\sigma=0时：p_\sigma(x)=p_{\text{data}}(x)，分布集中在那条细线上
当\sigma很小时：分布变成了一条宽度为\sigma的”窄公路”，覆盖了流形附近的区域
当\sigma增大时：公路变得越来越宽，逐渐覆盖更大的区域
当\sigma足够大时：公路变得和整个沙漠一样宽，分布变成了近似的高斯分布，充满整个高维空间

总的来说，通过加入高斯噪声，我们把原来只有在低维流形上才有非零值的分布，变成了在整个高维空间都有非零值的分布。

接下来，我们讲解Score-Based扩散模型的目标函数。在上个章节中我们已经知道，分数匹配的原始目标是最小化Fisher散度，也就是让我们的分数网络s_\theta(x_\sigma, \sigma)尽可能接近真实的边缘分数：

L_{\text{SM}} = \mathbb{E}_{x_\sigma \sim p_\sigma} \left[ \left\| s_\theta(x_\sigma, \sigma) - \nabla_{x_\sigma} \log p_\sigma(x_\sigma) \right\|_2^2 \right]

这个目标看起来很合理，但现实情况是完全无法直接计算，因为我们不知道真实的边缘分数\nabla_{x_\sigma} \log p_\sigma(x_\sigma)，这个是我们想要学习的目标分布。

这时我们需要使用去噪分数匹配（Denoising Score Matching）方法利用下面的恒等式，把这个不可计算的目标转化为了可计算的形式。对于高斯扰动x_\sigma = x_0 + \sigma\epsilon，有一个完美的数学恒等式：

\nabla_x \log p_\sigma(x_\sigma) = -\frac{\epsilon}{\sigma}

我们把边缘分数的期望表达式代入Fisher散度：

L_{\text{SM}} = \mathbb{E}_{x_\sigma \sim p_\sigma} \left[ \left\| s_\theta(x_\sigma, \sigma) - \mathbb{E}_{x_0 | x_\sigma} \left[ -\frac{\epsilon}{\sigma} \right] \right\|_2^2 \right]

根据期望的塔式法则（Law of Total Expectation），我们可以把对x_\sigma的期望和对x_0 | x_\sigma的期望合并为对x_0和\epsilon的联合期望：

L_{\text{DSM}} = \mathbb{E}_{x_0 \sim p_{\text{data}}, \epsilon \sim \mathcal{N}(0,I)} \left[ \left\| s_\theta(x_0 + \sigma \epsilon, \sigma) + \frac{\epsilon}{\sigma} \right\|_2^2 \right]

这就是去噪分数匹配的基本损失函数！它不需要知道真实的边缘分数，只需要我们采样干净数据x_0和噪声\epsilon，生成带噪声数据x_\sigma。它是一个简单的均方误差损失，可以用标准的梯度下降优化。

最后再在损失函数上乘上一个\sigma^2的权重：

L_{\text{DSM}} = \mathbb{E}_{x_0, \epsilon, \sigma} \left[ \sigma^2 \left\| s_\theta(x_0 + \sigma \epsilon, \sigma) + \frac{\epsilon}{\sigma} \right\|_2^2 \right]

这个权重不是随意加的，它有两个至关重要的作用：

数值稳定性：当\sigma很小时，\epsilon/\sigma的幅值会非常大，导致损失值爆炸。乘以\sigma^2后，损失变为\|\sigma s_\theta + \epsilon\|_2^2，所有噪声尺度下的损失值都在同一个数量级。
平衡不同噪声尺度的梯度：大噪声下，分数的幅值小（\|\nabla\log p_\sigma\| \approx 1/\sigma），梯度也小；小噪声下，分数的幅值大，梯度也大。如果不加权重，训练会被小噪声下的大梯度主导，导致大噪声下的分数估计不准确。\sigma^2正好抵消了这个差异，让模型在所有噪声尺度上都能得到同等程度的训练。

在训练好分数网络后，我们就可以使用退火朗之万动力学（Annealed Langevin Dynamics）进行反向去噪过程生成数据样本：

从最大噪声尺度\sigma_{\text{max}}对应的分布中采样初始点x \sim \mathcal{N}(0, \sigma_{\text{max}}^2 I)
从大到小依次遍历每个噪声尺度\sigma_i
在每个噪声尺度下运行M步朗之万动力学去噪过程：
x \leftarrow x + \epsilon \cdot s_\theta(x, \sigma_i) + \sqrt{2\epsilon} \cdot z, \quad z \sim \mathcal{N}(0,I)
最终得到的x就是近似服从p_{\text{data}}(x)的样本

值得一提的是，当 \boldsymbol{f}_t(\mathbf{x}) = 0 时，反向SDE退化为：

d\mathbf{x} = -g_t^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) dt + g_t d\boldsymbol{w}\\

这正是朗之万动力学方程。

到这里，Rocky就带着大家完整讲解了Score-Based扩散模型的本质基础原理。

在SDE框架下，将离散的多尺度噪声过程推广到连续时间极限，证明原始Score-Based模型（SMLD/NCSN等）本质上是方差爆炸SDE（Variance Exploding SDE, VE-SDE）的离散化实现。

原始Score-Based模型的离散加噪扩散过程可以表示为马尔可夫链：

x_i = x_{i-1} + \sqrt{\sigma_i^2 - \sigma_{i-1}^2} \cdot z_{i-1}, \quad z_{i-1} \sim \mathcal{N}(0,I)

当噪声尺度的数量N \to \infty，步长\Delta t = \frac{1}{N} \to 0时，离散序列\{\sigma_i\}变成连续函数\sigma(t)，离散马尔可夫链收敛到连续时间SDE：

dx_t = \sqrt{\frac{d\sigma^2(t)}{dt}} dW_t

这就是VE-SDE的精确形式，其中dW_t是标准布朗运动增量。

VE-SDE的核心性质：

正向过程解析解：x_t = x_0 + \sigma(t)\epsilon, \epsilon \sim \mathcal{N}(0,I)
边缘分布：p_t(x_t|x_0) = \mathcal{N}(x_0, \sigma^2(t)I)
方差演化：Var(x_t) = Var(x_0) + \sigma^2(t) \to \infty（当t \to T时），这也是”方差爆炸”名称的由来

根据Anderson时间反转定理，任何前向加噪扩散过程都有一个对应的反向去噪过程，其形式为：

dx_t = \left[ f(x_t,t) - g^2(t)\nabla_x \log p_t(x_t) \right] dt + g(t)d\bar{W}_t

对于VE-SDE，漂移项f(x_t,t)=0，扩散项g(t)=\sqrt{\frac{d\sigma^2(t)}{dt}}，代入得反向SDE：

dx_t = -\frac{d\sigma^2(t)}{dt} \nabla_x \log p_t(x_t) dt + \sqrt{\frac{d\sigma^2(t)}{dt}} d\bar{W}_t

由此我们就得到了关键结论：原始Score-Based模型的退火朗之万动力学采样，本质上就是这个反向VE-SDE的Predictor-Corrector离散化（其中Predictor是恒等函数，Corrector是朗之万动力学）。

SDE框架下的通用训练目标是连续时间去噪分数匹配：

\mathcal{L}_{\text{SDE}} = \mathbb{E}_{t \sim U(0,T), x_0 \sim p_{\text{data}}, x_t \sim p_t(x_t|x_0)} \left[ \lambda(t) \left\| s_\theta(x_t,t) - \nabla_x \log p_t(x_t|x_0) \right\|_2^2 \right]

对于VE-SDE，\nabla_x \log p_t(x_t|x_0) = -\frac{x_t - x_0}{\sigma^2(t)} = -\frac{\epsilon}{\sigma(t)}，权重\lambda(t) = \sigma^2(t)，代入后得到：

\mathcal{L}_{\text{VE}} = \mathbb{E}_{t,x_0,\epsilon} \left[ \sigma^2(t) \left\| s_\theta(x_t,t) + \frac{\epsilon}{\sigma(t)} \right\|_2^2 \right]

这与原始离散的去噪分数匹配目标完全一致，证明了两者的训练过程等价。

我们已经知道，对于任意高斯扰动x_t = \mu_t(x_0) + \sigma_t\epsilon，分数函数与噪声满足恒等式：

\nabla_x \log p_t(x_t) = -\frac{\epsilon}{\sigma_t}

对于Score-Based模型（VE-SDE）：\sigma_t = \sigma(t)，因此s_\theta(x_t,t) = -\frac{\epsilon_\theta(x_t,t)}{\sigma(t)}
对于DDPM（VP-SDE）：\sigma_t = \sqrt{1-\bar{\alpha}_t}，因此s_\theta(x_t,t) = -\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar{\alpha}_t}}

革命性结论：同一个神经网络可以同时作为噪声预测器和分数预测器，只需乘以一个简单的系数转换。DDPM预测噪声\epsilon_\theta和Score-Based模型预测分数s_\theta，本质上是在学习同一个东西。

总的来说，SDE框架不仅统一了训练过程，还统一了采样过程，证明Score-Based模型和DDPM的采样方法都是反向SDE的不同离散化实现。

4.4 DDPM、DDIM扩散模型在SDE统一框架下的本质原理零基础深入浅出通俗易懂理解

在之前的章节中，我们已经推导得到了SDE统一框架，并且证明了Score-Based扩散模型可以归入这个框架，我们接下来看看能否将DDPM归入这个框架呢？

我们首先需要对DDPM的前向扩散过程和正向SDE的一致性进行证明。

我们已经知道DDPM离散的前向扩散过程为：

\mathbf{x}_{i+1} = \sqrt{1-\beta_{i+1}} \mathbf{x}_i + \sqrt{\beta_{i+1}} \epsilon, \quad \epsilon \sim \mathcal{N}(0,\mathbf{I})

其中i=0,1,\dots,T-1是离散时间步，\beta_i \in (0,1)是噪声调度参数。

为了将其推广到连续时间，我们将离散时间步i映射到连续时间区间t \in [0,1]：

t = \frac{i}{T}, \quad \Delta t = \frac{1}{T}

当T \to \infty时，\Delta t \to 0，离散过程收敛到连续过程。

接着，我们再对DDPM的噪声调度参数连续化，我们定义连续噪声调度函数 \beta(t)，满足\beta\left(\frac{i}{T}\right) = T \cdot \beta_i。

为什么这样定义呢？因为在DDPM中，\beta_i 通常很小（比如 10^{-4} 量级）；当 T \to \infty 时，\beta_i \to 0，但 T \cdot \beta_i 保持有限，\int_0^1 \beta(t) dt = \lim_{T \to \infty} \sum_{i=1}^T \beta_i是一个有限值，与 T 无关。

于是，DDPM的离散前向扩散过程可以重写为连续形式：

\mathbf{x}_{t+\Delta t} = \sqrt{1 - \beta(t+\Delta t) \Delta t} \, \mathbf{x}_t + \sqrt{\beta(t+\Delta t) \Delta t} \, \boldsymbol{\varepsilon}\\

其中 t = i/T，\Delta t = 1/T，\beta(t+\Delta t) = T \cdot \beta_{i+1}。

由于 \Delta t 很小，我们可以对系数 \sqrt{1 - \beta(t+\Delta t) \Delta t} 进行一阶泰勒展开。我们可以直接利用近似公式 \sqrt{1 - u} \approx 1 - \frac{u}{2}（当u \ll 1时），并注意到\beta(t+\Delta t) \approx \beta(t)（函数在微小区间内变化不大），得到：

\sqrt{1 - \beta(t+\Delta t) \Delta t} \approx 1 - \frac{1}{2} \beta(t) \Delta t\\

接着我们再将上述近似代入DDPM前向扩散过程得到连续化后的前向扩散过程：

\mathbf{x}_{t+\Delta t} \approx \left(1 - \frac{1}{2} \beta(t) \Delta t\right) \mathbf{x}_t + \sqrt{\beta(t)} \sqrt{\Delta t} \, \boldsymbol{\varepsilon} = \mathbf{x}_t - \frac{1}{2} \beta(t) \mathbf{x}_t \Delta t + \sqrt{\beta(t)} \sqrt{\Delta t} \, \boldsymbol{\varepsilon}\\

接着我们将其整理为增量形式：

\mathbf{x}_{t+\Delta t} - \mathbf{x}_t \approx -\frac{1}{2} \beta(t) \mathbf{x}_t \Delta t + \sqrt{\beta(t)} \sqrt{\Delta t} \, \epsilon

当T \to \infty，\Delta t \to 0时，近似变为严格相等，我们得到DDPM在连续时间下的SDE方程：

d\mathbf{x} = -\frac{1}{2} \beta(t) \mathbf{x} dt + \sqrt{\beta(t)} d\boldsymbol{w}

其中dw是标准布朗运动的增量。

我们已经知道SDE框架中前向扩散过程的连续形式为：

d\mathbf{x} = \boldsymbol{f}_t(\mathbf{x}) \, dt + g_t \, d\boldsymbol{w}

我们对比DDPM和SDE框架的两个前向扩散过程公式，我们易得：

\boldsymbol{f}_t(\mathbf{x}) = -\frac{1}{2} \beta(t) \mathbf{x}, \quad g_t = \sqrt{\beta(t)}\\

到此为止，我们终于得到DDPM对应的SDE框架中的连续前向扩散过程。我们经过上面的坚强推导可以知道，DDPM对应的前向扩散过程是SDE框架的一种特例，对应着特定的f,g。

值得注意的是，在SDE框架中，DDPM的前向扩散过程设计为保持方差不爆炸（”Variance Preserving” SDE，VP-SDE）：

离散形式：\text{Var}(\mathbf{x}_t) = 1 - \bar{\alpha}_t + \bar{\alpha}_t \text{Var}(\mathbf{x}_0)
当 \text{Var}(\boldsymbol{x}_0) \approx 1 时，\text{Var}(\boldsymbol{x}_t) \approx 1
在连续极限下，\text{Var}(\boldsymbol{x}_t) 保持有界（通常接近1）

相比之下，Score-Based扩散模型对应的是”Variance Exploding” SDE，即VE-SDE，其方差随时间指数增长，我们在之前的文章节中已经详细讲解。

我们已经证明了DDPM对应的前向扩散过程是SDE框架的一种特例，接下来就可以基于SDE框架推导得到SDE框架下DDPM的反向去噪过程。

我们已经知道SDE框架的反向去噪过程为：

d\mathbf{x} = \left[ \boldsymbol{f}_t(\mathbf{x}) - g_t^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) \right] dt + g_t d{\boldsymbol{w}}\\

我们接着将已经推导获得的 \boldsymbol{f}_t(\mathbf{x}) = -\frac{1}{2} \beta(t) \mathbf{x}，g_t = \sqrt{\beta(t)}代入上式：

d\mathbf{x} = \left[ -\frac{1}{2} \beta(t) \mathbf{x} - \beta(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) \right] dt + \sqrt{\beta(t)} d{\boldsymbol{w}}\\

这样，我们就得到了DDPM在SDE框架下的反向去噪过程的公式。

到此为止，我们可以得出结论：DDPM可以归入SDE框架中，DDPM是SDE框架中的一个特定例子，DDPM 本质上就是连续VP-SDE的一个特定离散化实现，只需要设置合适的f,g即可实现！这揭示了 DDPM 所有设计选择背后的连续时间数学本质，证明了离散只是连续过程的一种数值近似。

我们已经知道，DDIM是DDPM的一种加速优化形态，那么DDIM在SDE框架下是什么样的形态呢？

DDPM的随机采样对应反向SDE的求解，而去掉噪声项的DDIM采样对应概率流ODE（Probability Flow ODE）的求解。

这就是DDIM在SED框架中的本质：

DDIM是VP-SDE对应的概率流ODE的欧拉离散化。

它和DDPM共享完全相同的训练目标、完全相同的模型参数，仅在采样时对噪声项的系数做了调整。

接下来，我们从已经讲过的VP SDE出发，一步步推导出DDIM的更新公式，这个过程将清晰地展示DDIM的数学本质。

我们已经知道VP SDE的正向过程：

dx = -\frac{1}{2}\beta(t)x dt + \sqrt{\beta(t)} d\boldsymbol{w}

对应的概率流ODE（确定性过程，与SDE具有相同的边缘分布）：

dx = \left[ f(x,t) - \frac{1}{2}g(t)^2 s_\theta(x,t) \right] dt

对于VP SDE，漂移系数\boldsymbol{f}(x,t)=-\frac{1}{2}\beta(t)x，扩散系数g(t)=\sqrt{\beta(t)}，代入得：

dx = \left[ -\frac{1}{2}\beta(t)x - \frac{1}{2}\beta(t) s_\theta(x,t) \right] dt

这就是VP SDE对应的概率流ODE，我们的目标是对这个ODE进行欧拉离散化，得到DDIM的更新公式。

为了和DDPM的离散符号保持一致，我们引入VP SDE的标准参数化：

\alpha(t) = e^{-\int_0^t \beta(s) ds}, \quad \bar{\alpha}_t = \alpha(t)

\sigma(t) = \sqrt{1-\alpha(t)}

其中\bar{\alpha}_t就是DDPM中常用的累积乘积项，\sigma(t)是t时刻的噪声标准差。

根据之前我们的推导结果，分数函数和DDPM预测的噪声之间的关系为：

s_\theta(x_t,t) = -\frac{\epsilon_\theta(x_t,t)}{\sigma(t)} = -\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar{\alpha}_t}}

将其代入概率流ODE：

dx = \left[ -\frac{1}{2}\beta(t)x - \frac{1}{2}\beta(t) \cdot \left(-\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar{\alpha}_t}}\right) \right] dt

dx = -\frac{1}{2}\beta(t)x dt + \frac{\beta(t)}{2\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t,t) dt

现在我们对这个连续时间的ODE进行欧拉离散化。将时间区间[0,1]划分为N个离散步：0 = t_0 < t_1 < \dots < t_N = 1，步长\Delta t_i = t_i - t_{i-1}。

对于反向过程（从t_i到t_{i-1}），欧拉离散化的更新公式为：

x_{i-1} = x_i + \left[ -\frac{1}{2}\beta(t_i)x_i + \frac{\beta(t_i)}{2\sqrt{1-\bar{\alpha}_{t_i}}} \epsilon_\theta(x_i,t_i) \right] \cdot (-\Delta t_i)

注意这里的-\Delta t_i是因为时间在反向流动。

我们知道\bar{\alpha}_t = e^{-\int_0^t \beta(s) ds}，对其求导得：

\frac{d\bar{\alpha}_t}{dt} = -\beta(t) \bar{\alpha}_t

\beta(t) = -\frac{1}{\bar{\alpha}_t} \frac{d\bar{\alpha}_t}{dt}

当步长\Delta t_i很小时，我们可以用差分近似导数：

\frac{d\bar{\alpha}_t}{dt} \approx \frac{\bar{\alpha}_{i-1} - \bar{\alpha}_i}{\Delta t_i}

代入\beta(t_i)的表达式：

\beta(t_i) \approx -\frac{1}{\bar{\alpha}_i} \cdot \frac{\bar{\alpha}_{i-1} - \bar{\alpha}_i}{\Delta t_i} = \frac{\bar{\alpha}_i - \bar{\alpha}_{i-1}}{\bar{\alpha}_i \Delta t_i}

将\beta(t_i)的近似代入欧拉离散化的更新公式：

x_{i-1} = x_i + \left[ -\frac{1}{2} \cdot \frac{\bar{\alpha}_i - \bar{\alpha}_{i-1}}{\bar{\alpha}_i \Delta t_i} \cdot x_i + \frac{1}{2} \cdot \frac{\bar{\alpha}_i - \bar{\alpha}_{i-1}}{\bar{\alpha}_i \Delta t_i} \cdot \frac{\epsilon_\theta(x_i,i)}{\sqrt{1-\bar{\alpha}_i}} \right] \cdot (-\Delta t_i)

\Delta t_i项抵消，整理得：

x_{i-1} = x_i + \frac{\bar{\alpha}_i - \bar{\alpha}_{i-1}}{2\bar{\alpha}_i} x_i - \frac{\bar{\alpha}_i - \bar{\alpha}_{i-1}}{2\bar{\alpha}_i \sqrt{1-\bar{\alpha}_i}} \epsilon_\theta(x_i,i)

合并同类项：

x_{i-1} = \frac{\bar{\alpha}_i + \bar{\alpha}_{i-1}}{2\bar{\alpha}_i} x_i - \frac{\bar{\alpha}_i - \bar{\alpha}_{i-1}}{2\bar{\alpha}_i \sqrt{1-\bar{\alpha}_i}} \epsilon_\theta(x_i,i)

当步长\Delta t_i足够小时，\bar{\alpha}_i \approx \bar{\alpha}_{i-1}，我们可以做进一步的近似：

\frac{\bar{\alpha}_i + \bar{\alpha}_{i-1}}{2\bar{\alpha}_i} \approx \frac{\sqrt{\bar{\alpha}_{i-1}}}{\sqrt{\bar{\alpha}_i}}

\frac{\bar{\alpha}_i - \bar{\alpha}_{i-1}}{2\bar{\alpha}_i \sqrt{1-\bar{\alpha}_i}} \approx \frac{\sqrt{1-\bar{\alpha}_{i-1}} - \sqrt{1-\bar{\alpha}_i}}{\sqrt{\bar{\alpha}_i}}

代入后得到：

x_{i-1} = \frac{\sqrt{\bar{\alpha}_{i-1}}}{\sqrt{\bar{\alpha}_i}} x_i - \frac{\sqrt{1-\bar{\alpha}_{i-1}} - \sqrt{1-\bar{\alpha}_i}}{\sqrt{\bar{\alpha}_i}} \epsilon_\theta(x_i,i)

整理一下：

x_{i-1} = \sqrt{\bar{\alpha}_{i-1}} \left( \frac{x_i - \sqrt{1-\bar{\alpha}_i} \epsilon_\theta(x_i,i)}{\sqrt{\bar{\alpha}_i}} \right) + \sqrt{1-\bar{\alpha}_{i-1}} \epsilon_\theta(x_i,i)

✅ 这就是DDIM的确定性采样公式！

我们通过严格的数学推导，从VP SDE的概率流ODE出发，得到了DDIM的更新公式。这无可辩驳地证明了：

DDIM就是VP SDE对应的概率流ODE的欧拉离散化。

最后，我们再学习领会一下SDE框架下的扩散模型的训练流程具体是什么样的呢？

我们之前在DDPM中还没有估计过Score Function，不过我们已经知道DDPM通过训练一个神经网络 \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t) 来预测加入的真实噪声 \boldsymbol{\varepsilon} = \frac{\mathbf{x}_t - \sqrt{\bar\alpha_t} \mathbf{x}_0}{\sqrt{1 - \bar\alpha_t}} 。

同时我们也已经知道SDE框架下 \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) 就是分数函数（Score Function），它表示在时间 t、位置 \mathbf{x} 处，真实数据分布的对数概率密度的梯度方向，指向概率密度增加最快的方向。在反向去噪过程中，用于引导噪声样本“上坡”走向高概率区域（即真实数据分布）。

但问题是，我们不知道真实数据分布 p_t(\mathbf{x})，因此无法直接计算分数函数。所以我们需要像经典DDPM扩散模型一样，训练一个神经网络 s_\theta(\mathbf{x}_t, t) 来近似它。

最直接最理想的形式是让神经网络预测的分数与真实分数尽可能接近，我们需要进行分数匹配（Score Matching）：

\mathcal{L}_{\text{ideal}} = \mathbb{E}_{t, \mathbf{x}_t} \left[ \| s_\theta(\mathbf{x}_t, t) - \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) \|^2 \right]\\

这里t 从 [0, T] 均匀采样，\mathbf{x}_t \sim p_t(\mathbf{x}_t)即前向扩散过程在时间 t 的边际分布。

但是这个损失函数同样无法直接计算，因为我们不知道真实分数 \nabla \log p_t(\mathbf{x}_t)，也难以直接从 p_t(\mathbf{x}_t) 采样。因此我们需要利用前向扩散过程的条件分布 p(\mathbf{x}_t | \mathbf{x}_0) 来构造可行的训练目标。我们已经知道在DDPM扩散模型中，前向扩散过程服从高斯分布：

p(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}\left( \mathbf{x}_t; \sqrt{\bar\alpha_t} \mathbf{x}_0, (1 - \bar\alpha_t) \mathbf{I} \right) \\

这个公式表示从数据 \mathbf{x}_0 出发，经过时间 t 后，得到的高斯噪声混合。其中\bar\alpha_t 是噪声调度参数，满足 \alpha_0 = 1，\alpha_T \approx 0；通常 \bar\alpha_t = \prod_{s=1}^t (1 - \beta_s)，\beta_s 是噪声方差。

由于 p(\mathbf{x}_t | \mathbf{x}_0) 是高斯分布，因此我们可以直接计算其对数梯度：

\log p(\mathbf{x}_t | \mathbf{x}_0) = -\frac{d}{2} \log(2\pi(1-\bar\alpha_t)) - \frac{\|\mathbf{x}_t - \sqrt{\bar\alpha_t} \mathbf{x}_0\|^2}{2(1-\bar\alpha_t)}\\

我们接着将上式对 \mathbf{x}_t 求梯度：

\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t | \mathbf{x}_0) = -\frac{\mathbf{x}_t - \sqrt{\bar\alpha_t} \mathbf{x}_0}{1 - \bar\alpha_t}\\

我们可以发现，对于给定的原始数据 \mathbf{x}_0，在 \mathbf{x}_t 处的条件分布指向从 \mathbf{x}_t 到 \sqrt{\bar\alpha_t} \mathbf{x}_0 的方向（但取反）。得到上述的表达式后，我们暂且将其放到一边，后续将会使用到。

在给定的前向扩散过程，我们知道时间 t 的边缘分布是通过对所有可能的 \mathbf{x}_0 积分得到的：

p_t(\mathbf{x}_t) = \int p(\mathbf{x}_t | \mathbf{x}_0) p(\mathbf{x}_0) d\mathbf{x}_0\\

其中 p(\mathbf{x}_0) 是真实数据分布， p(\mathbf{x}_t | \mathbf{x}_0) 是前向扩散过程的条件分布（通常是高斯分布）。上述公式表达了 p_t(\mathbf{x}_t) 是在时间 t 时，不考虑具体是从哪个 \mathbf{x}_0 演化而来的所有可能 \mathbf{x}_t 的分布。

接着我们计算边缘分布的分数函数 \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) ，这个分数函数将用于反向SDE的漂移项修正。

我们根据微分的链式法则，对任何概率密度函数 p(\mathbf{x}) ，可以推导得到：

\frac{\partial}{\partial x_i} \log p(\mathbf{x}) = \frac{1}{p(\mathbf{x})} \frac{\partial p(\mathbf{x})}{\partial x_i} \Rightarrow\nabla_{\mathbf{x}} \log p(\mathbf{x}) = \frac{\nabla_{\mathbf{x}} p(\mathbf{x})}{p(\mathbf{x})} \\

因此我们首先计算 \nabla_{\mathbf{x}_t} p_t(\mathbf{x}_t) ：

\nabla_{\mathbf{x}_t} p_t(\mathbf{x}_t) = \nabla_{\mathbf{x}_t} \int p(\mathbf{x}_t | \mathbf{x}_0) p(\mathbf{x}_0) d\mathbf{x}_0 \\

在很一般的条件下（如积分收敛、被积函数足够光滑），我们可以将梯度算子移入积分号内，同时由于 p(\mathbf{x}_0) 不依赖于 \mathbf{x}_t ，可以提到梯度外面：

\nabla_{\mathbf{x}_t} p_t(\mathbf{x}_t) = \nabla_{\mathbf{x}_t} \int p(\mathbf{x}_t | \mathbf{x}_0) p(\mathbf{x}_0) d\mathbf{x}_0 = \int \nabla_{\mathbf{x}_t} \left[ p(\mathbf{x}_t | \mathbf{x}_0) p(\mathbf{x}_0) \right] d\mathbf{x}_0 = \int \left[ \nabla_{\mathbf{x}_t} p(\mathbf{x}_t | \mathbf{x}_0) \right] p(\mathbf{x}_0) d\mathbf{x}_0\\

这里我们再使用一个关键技巧，将梯度表示为原函数乘以对数梯度：

\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t | \mathbf{x}_0) = \frac{\nabla_{\mathbf{x}_t} p(\mathbf{x}_t | \mathbf{x}_0)}{p(\mathbf{x}_t | \mathbf{x}_0)} \Rightarrow\nabla_{\mathbf{x}_t} p(\mathbf{x}_t | \mathbf{x}_0) = p(\mathbf{x}_t | \mathbf{x}_0) \cdot \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t | \mathbf{x}_0) \\

因此我们可以推导得到：

\nabla_{\mathbf{x}_t} p_t(\mathbf{x}_t) = \int p(\mathbf{x}_t | \mathbf{x}_0) p(\mathbf{x}_0) \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t | \mathbf{x}_0) d\mathbf{x}_0\\

接着我们将其代入对数梯度公式：

\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) = \frac{\nabla_{\mathbf{x}_t} p_t(\mathbf{x}_t)}{p_t(\mathbf{x}_t)} = \frac{\int p(\mathbf{x}_t | \mathbf{x}_0) p(\mathbf{x}_0) \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t | \mathbf{x}_0) d\mathbf{x}_0}{p_t(\mathbf{x}_t)} \\

接着我们再根据贝叶斯定理 p(\mathbf{x}_0 | \mathbf{x}_t) = \frac{p(\mathbf{x}_t | \mathbf{x}_0) p(\mathbf{x}_0)}{p_t(\mathbf{x}_t)} ，我们可以推导得到：

\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) = \int \frac{p(\mathbf{x}_t | \mathbf{x}_0) p(\mathbf{x}_0)}{p_t(\mathbf{x}_t)} \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t | \mathbf{x}_0) d\mathbf{x}_0 = \int p(\mathbf{x}_0 | \mathbf{x}_t) \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t | \mathbf{x}_0) d\mathbf{x}_0 \\

到此为止，经过我们的伟大推导，我们可以发现上面的积分正是条件期望：

\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) = \mathbb{E}_{p(\mathbf{x}_0 | \mathbf{x}_t)} \left[ \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t | \mathbf{x}_0) \right]\\

到这里，我们可以得出一个重要结论：边缘分布的分数函数是条件分数函数在后验分布 p(\mathbf{x}_0 | \mathbf{x}_t) 下的期望。

之前我们已经推导得到\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t | \mathbf{x}_0) = -\frac{\mathbf{x}_t - \sqrt{\bar\alpha_t} \mathbf{x}_0}{1 - \bar\alpha_t} ，将其代入期望中：

\begin{aligned} \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) &= \mathbb{E}_{p(\mathbf{x}_0 | \mathbf{x}_t)} \left[ -\frac{\mathbf{x}_t - \sqrt{\bar\alpha_t} \mathbf{x}_0}{1 - \alpha_t} \right] \\ &= -\frac{\mathbf{x}_t - \sqrt{\bar\alpha_t} \mathbb{E}_{p(\mathbf{x}_0 | \mathbf{x}_t)}[\mathbf{x}_0]}{1 - \bar\alpha_t} \end{aligned}\\

其中 \mathbb{E}[\mathbf{x}_0 | \mathbf{x}_t] 是给定噪声观测 \mathbf{x}_t 时，原始数据 \mathbf{x}_0 的后验均值。这也是 Tweedie公式 在扩散模型中的具体形式。

接着我们将DDPM的噪声预测与分数函数联系起来。在DDPM中，从预测的噪声可以对 \mathbf{x}_0 进行估计：

\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t - \sqrt{1 - \bar\alpha_t} \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{\bar\alpha_t}}\\

这个 \hat{\mathbf{x}}_0 实际上近似于后验期望 \mathbb{E}_{p(\mathbf{x}_0 | \mathbf{x}_t)}[\mathbf{x}_0]。代入分数函数公式中，可推导得到：

\begin{aligned} \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) &= -\frac{\mathbf{x}_t - \sqrt{\bar\alpha_t} \hat{\mathbf{x}}_0}{1 - \bar\alpha_t} \\ &= -\frac{\mathbf{x}_t - \sqrt{\bar\alpha_t} \left( \frac{\mathbf{x}_t - \sqrt{1 - \bar\alpha_t} \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{\bar\alpha_t}} \right)}{1 - \bar\alpha_t} \\ &= -\frac{\mathbf{x}_t - (\mathbf{x}_t - \sqrt{1 - \bar\alpha_t} \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t))}{1 - \bar\alpha_t} \\ &= -\frac{\sqrt{1 - \bar\alpha_t} \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t)}{1 - \bar\alpha_t} \\ &= -\frac{\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar\alpha_t}} \end{aligned} \\

我们终于得到了关键关系式，即我们可以用DDPM的神经网络来估计SDE分数函数，两者只是幅度因子 1/\sqrt{1 - \bar\alpha_t}存在差异，本质思想是一致的：

\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) = -\frac{1}{\sqrt{1 - \bar\alpha_t}} \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t) \\

当 t 较大时，\bar\alpha_t \to 0，\sqrt{1 - \bar\alpha_t} \to 1，幅度因子接近1。
当 t 较小时，\bar\alpha_t \to 1，\sqrt{1 - \bar\alpha_t} \to 0，幅度因子很大。

这反映了在早期去噪时需要更大的调整幅度，因为早期噪声更大。

后续Rocky将持续更新补充，大家敬请期待！！！

4.5 SDE统一框架中主流求解器&采样算法零基础深入浅出通俗易懂理解

SDE框架之所以能统一看似独立的 DDPM 和 Score-Based Model，核心在于其将离散时间的扩散过程推广到连续时间极限时，揭示了两者本质上是同一类随机过程的不同离散化实现。

SDE框架下的不同扩散模型三大统一基石：

共同的范式内核：两者都遵循 "逐步注入噪声破坏数据 → 学习逆转噪声过程生成数据" 的生成逻辑
连续时间极限收敛：当离散时间步长 Δt→0 时，DDPM 的马尔可夫链和 Score-Based Model 的多尺度噪声扰动都收敛到形式一致的伊藤 SDE
核心优化目标等价：DDPM 预测噪声 ε_θ 与 Score-Based Model 预测分数∇log p_t (x) 存在精确的数学转换关系，这意味着同一个神经网络可以同时作为噪声预测器和分数预测器，只需进行简单数学系数转换。

总的来说，SDE 框架提供了统一的数学语言，任何扩散过程都可以用一对正向 - 反向 SDE 描述。证明了这两类扩散模型只是在噪声调度方式和采样离散化策略上有所不同，而核心生成机制完全一致。

接下来，Rocky带着大家详细讲解SDE框架下主流的求解器&采样算法。

在SDE统一框架中，所有扩散采样算法本质上都是反向SDE或概率流ODE的数值求解器。不同算法的核心差异在于：

求解的是随机SDE还是确定性ODE
采用的数值离散化方案（Euler、Runge-Kutta、线性多步等）
是否引入MCMC校正步骤来修正离散化误差
是否利用SDE的解析性质来加速收敛

我们首先明确需要求解的两个核心方程，所有采样算法都是围绕这两个方程展开的：

【反向SDE（随机采样）】

正向SDE：dx = f(x,t)dt + g(t)dw（数据→噪声） 反向SDE（时间从T到0）：

dx = \left[ f(x,t) - g(t)^2 s_\theta(x,t) \right] dt + g(t) d\bar{w}

其中s_\theta(x,t) \approx \nabla_x \log p_t(x)是我们训练好的分数网络，d\bar{w}是反向时间的布朗运动增量。

特点：包含随机噪声项，生成的样本具有多样性，是最通用的采样方式。

【概率流ODE（确定性采样）】

对于任意扩散SDE，存在一个对应的确定性ODE，其轨迹与SDE具有完全相同的边缘分布\{p_t(x)\}：

dx = \left[ f(x,t) - \frac{1}{2} g(t)^2 s_\theta(x,t) \right] dt

特点：无噪声项，生成过程完全确定，采样速度更快，支持精确似然计算和潜在空间操作（如插值、编辑）。

我们先来讲解第一类SDE求解器（随机采样算法）

SDE求解器通过数值离散化反向SDE来生成样本，每一步都会注入高斯噪声，因此生成的样本具有随机性和多样性。

【Euler-Maruyama求解器：最基础的SDE求解器】

Euler-Maruyama是最简单、最常用的SDE数值求解方法，它将连续时间的SDE离散化为有限步的更新。

离散化过程将时间区间[0,T]划分为N个等距步长：0 = t_0 < t_1 < \dots < t_N = T，步长\Delta t_i = t_i - t_{i-1}。

对于反向SDE，我们从t=T开始，逐步向t=0推进。在每一步t_i \to t_{i-1}，我们用当前时刻的导数近似整个步长内的导数：

x_{i-1} = x_i + \left[ f(x_i, t_i) - g(t_i)^2 s_\theta(x_i, t_i) \right] \cdot (-\Delta t_i) + g(t_i) \cdot \sqrt{\Delta t_i} \cdot z_i

其中z_i \sim \mathcal{N}(0,I)是标准高斯噪声，-\Delta t_i是因为时间在反向流动。

我们再来看看Euler-Maruyama求解器与DDPM祖先采样的关系。DDPM的祖先采样公式：

x_{i-1} = \frac{1}{\sqrt{1-\beta_i}} \left( x_i + \beta_i s_\theta(x_i, i) \right) + \sqrt{\beta_i} z_i

Euler-Maruyama求解器本质上是VP SDE下Euler-Maruyama求解器的近似。当步长\Delta t_i很小时，两者几乎完全一致。

Euler-Maruyama求解器优缺点：

✅ 优点：实现简单，计算量小，是所有其他SDE求解器的基础
❌ 缺点：一阶精度，离散化误差较大，需要较多的步数（通常1000步）才能获得高质量样本

【反向扩散采样器：SDE框架原生的求解器】

反向扩散采样器是SDE论文中提出的专门针对反向SDE的求解器，它比Euler-Maruyama更精确，也更通用。

正向SDE的离散化形式为：

x_{i} = x_{i-1} + f_{i-1}(x_{i-1}) + g_{i-1} z_{i-1}

其中f_{i-1}和g_{i-1}是正向SDE在t_{i-1}时刻的漂移和扩散系数的离散化。

反向扩散采样器直接对正向离散化进行”反转”，得到反向更新公式：

x_{i-1} = x_i - f_i(x_i) + g_i^2 s_\theta(x_i, i) + g_i z_i

论文实验表明，在相同步数下，反向扩散采样器的FID比DDPM祖先采样低5%-10%。例如，在CIFAR-10上，VP SDE的P1000采样：

祖先采样：FID=3.24
反向扩散采样：FID=3.19

反向扩散采样器优缺点：

✅ 优点：比Euler-Maruyama更精确，适用于任意SDE（不仅是VP SDE）
❌ 缺点：仍然是一阶精度，需要较多步数

【Milstein求解器：二阶精度的SDE求解器】

Milstein方法是SDE的二阶数值求解方法，它通过引入额外的导数项来减小离散化误差。

对于一般的SDE dx = f(x,t)dt + g(x,t)dw，Milstein更新为：

x_{i+1} = x_i + f(x_i,t_i)\Delta t + g(x_i,t_i)\Delta w_i + \frac{1}{2} g(x_i,t_i) g'(x_i,t_i) \left( (\Delta w_i)^2 - \Delta t \right)

其中\Delta w_i = \sqrt{\Delta t} z_i是布朗运动增量。

对于扩散模型中常用的标量扩散系数g(t)（与x无关），g'(x,t)=0，因此Milstein方法退化为Euler-Maruyama方法。这就是为什么在扩散模型中很少看到Milstein求解器的原因——我们常用的VE/VP/sub-VP SDE的扩散系数都与x无关，二阶项消失了。

我们再来讲解第二类ODE求解器（确定性采样算法）

ODE求解器通过数值离散化概率流ODE来生成样本，整个过程没有随机噪声，因此生成的样本是确定的。

【Euler求解器：最基础的ODE求解器】

Euler方法是最简单的ODE数值求解方法，它将连续时间的ODE离散化为有限步的更新。

对于概率流ODE：

dx = \left[ f(x,t) - \frac{1}{2} g(t)^2 s_\theta(x,t) \right] dt

Euler离散化的更新公式为：

x_{i-1} = x_i + \left[ f(x_i, t_i) - \frac{1}{2} g(t_i)^2 s_\theta(x_i, t_i) \right] \cdot (-\Delta t_i)

我们看看Euler求解器与DDIM的关系。DDIM的确定性采样公式：

x_{i-1} = \sqrt{\alpha_{i-1}} \left( \frac{x_i - \sqrt{1-\alpha_i} \epsilon_\theta(x_i,i)}{\sqrt{\alpha_i}} \right) + \sqrt{1-\alpha_{i-1}} \epsilon_\theta(x_i,i)

本质上是VP SDE下概率流ODE的Euler离散化。

当我们在DDIM中加入噪声项\sigma_i z_i时，就得到了一个介于SDE和ODE之间的采样器：

\sigma_i = \sqrt{\beta_i}：恢复为DDPM采样（SDE）
\sigma_i = 0：退化为DDIM采样（ODE）

Euler求解器优缺点：

✅ 优点：实现简单，无随机噪声，支持潜在空间操作
❌ 缺点：一阶精度，离散化误差较大，样本多样性略低于SDE采样

【自适应步长ODE求解器：速度-精度的灵活权衡】

概率流ODE是一个普通的常微分方程，我们可以使用任何成熟的ODE求解器来求解它，包括自适应步长求解器。

常用的自适应ODE求解器：

RK45（Dormand-Prince）：最常用的自适应Runge-Kutta求解器，四阶精度，五阶误差估计
RK23：二阶精度，三阶误差估计，适合精度要求不高的场景
BDF（Backward Differentiation Formula）：多步求解器，适合刚性ODE

自适应步长求解器可以自动调整步长大小，在误差允许的范围内尽可能使用大步长，从而显著减少采样步数。

论文实验表明，使用RK45求解概率流ODE，可以在不影响视觉质量的前提下，将分数函数评估次数（NFE）减少90%以上。例如，在256×256 CelebA-HQ上：

1000步Euler采样：NFE=1000，FID≈3.5
RK45自适应采样：NFE≈50，FID≈3.6（视觉上几乎无差异）

自适应步长ODE求解器优缺点：

✅ 优点：采样速度极快，支持精确似然计算，可灵活调整精度-速度trade-off
❌ 缺点：实现相对复杂，样本多样性略低，在VE SDE上的表现不如VP SDE

【高阶ODE求解器：少步高精度采样】

为了进一步减少采样步数，研究者们提出了许多针对扩散模型优化的高阶ODE求解器，其中最著名的是DPM-Solver和DPM-Solver++。

DPM-Solver的核心思想是利用了扩散SDE的半线性结构和分数函数的解析性质，设计了专门的高阶数值求解器，而不是使用通用的ODE求解器。

对于VP SDE，概率流ODE可以改写为：

\frac{d}{dt} \left( \frac{x(t)}{\sqrt{\alpha(t)}} \right) = -\frac{\beta(t)}{2\sqrt{\alpha(t)}} \epsilon_\theta(x(t), t)

其中\alpha(t) = e^{-\int_0^t \beta(s)ds}，\epsilon_\theta(x,t) = -\sqrt{1-\alpha(t)} s_\theta(x,t)是DDPM预测的噪声。

DPM-Solver通过指数积分器和线性多步法，直接求解这个半线性ODE，实现了在20步内生成与1000步DDPM相当质量的样本。

DPM-Solver++进一步优化了求解器的阶数和稳定性，支持10-20步生成高质量样本，并且适用于所有类型的SDE（VE/VP/sub-VP）。

DPM-Solver优缺点：

✅ 优点：采样速度极快（10-20步），精度高，是目前工业界最常用的采样器
❌ 缺点：理论相对复杂，实现难度较高

最后，我们再学习第三类Predictor-Corrector（PC）混合采样器。

Predictor-Corrector（预测 - 校正）是 SDE 扩散框架中最经典的质量提升技术，也是现代所有高质量采样器的核心思想。它通过 "先粗步预测轨迹，再细步修正分布" 的两步策略，深度解决了大步长采样时的数值误差问题，在几乎不增加总步数的情况下大幅提升生成质量。

在SDE框架下，任何采样过程本质上都是求解反向SDE：

dx_t = \left[f(x_t, t) - g^2(t)s_\theta(x_t,t)\right]dt + g(t)d\bar{W}_t

纯Predictor采样（如欧拉法、DDPM、DDIM）只使用单步数值积分来近似这个SDE，而Predictor-Corrector采样在每一步都增加了一个朗之万动力学校正步：

PC采样将每一步采样分为两个阶段：

Predictor（预测）阶段：使用任意数值SDE/ODE求解器，从x(t)预测x(t-\Delta t)的初步估计，得到预测值\hat{x}_{t-\Delta t}。
Corrector（校正）阶段：在\hat{x}_{t-\Delta t}上运行M步未调整朗之万动力学（ULA），修正到更接近真实分布p_{t-\Delta t}(x)的位置：
x_{t-\Delta t}^{(k+1)} = x_{t-\Delta t}^{(k)} + \epsilon \cdot s_\theta(x_{t-\Delta t}^{(k)}, t-\Delta t) + \sqrt{2\epsilon} \cdot \epsilon_k
其中\epsilon是朗之万步长，k=1,2,...,M

核心优势：Predictor负责快速穿越时间轴，Corrector负责在每个时间点修正分布误差。两者分工明确，协同工作。朗之万动力学这种MCMC校正可以有效消除数值离散化带来的分布误差，从而在较少的步数下获得更高质量的样本。

最常用的PC组合是反向扩散采样器（Predictor）+ 退火Langevin动力学（Corrector）。完整的PC采样算法流程如下所示：

输入：训练好的分数网络s_θ(x,t)，步数N，校正步数M
1. 初始化x_T ~ p_T(x)（标准高斯分布）
2. 对于i从N-1到0：
   a. Predictor步：x_i' = Predictor(x_{i+1}, t_{i+1}, t_i)
   b. 对于j从1到M：
      i. Corrector步：x_i' = x_i' + ε_i * s_θ(x_i', t_i) + sqrt(2ε_i) * z_j
   c. x_i = x_i'
3. 最后一步去噪：x_0 = Tweedie(x_0, t_0)
4. 返回x_0

校正步的步长\epsilon_i是一个关键超参数，通常根据信噪比（SNR）来选择：

\epsilon_i = 2 \cdot \left( r \cdot \frac{\|z\|_2}{\|s_\theta(x_i', t_i)\|_2} \right)^2

其中r是信噪比参数，通常在0.01-0.2之间取值。论文实验表明，对于CIFAR-10，最佳的r值约为0.16。

所有传统的扩散采样方法都是PC采样的特例：

NCSN的退火Langevin采样：Predictor=恒等映射，Corrector=多步Langevin动力学
DDPM的祖先采样：Predictor=祖先采样，Corrector=恒等映射
DDIM的确定性采样：Predictor=概率流ODE Euler，Corrector=恒等映射

论文实验表明，PC采样在相同计算量下，能显著提升样本质量。例如，在CIFAR-10上，VE SDE的采样结果：

P1000（纯Predictor，1000步）：FID=4.79
PC1000（1000步Predictor+1000步Corrector）：FID=3.60
P2000（纯Predictor，2000步）：FID=4.74

可以看到，PC1000的计算量与P2000相同，但FID低了约25%。

【最后一步去噪：Tweedie公式】

所有扩散采样器在最后一步（t \to 0）都应该使用Tweedie公式进行去噪，这能显著提升样本质量，降低FID。

Tweedie公式：

\hat{x}_0 = x_t + \sigma(t)^2 s_\theta(x_t, t)

对于DDPM，这等价于：

\hat{x}_0 = \frac{x_t - \sqrt{1-\alpha_t} \epsilon_\theta(x_t,t)}{\sqrt{\alpha_t}}

为什么需要最后一步去噪？ 因为在t \to 0时，分数函数的幅值很小，采样过程中引入的噪声会变得明显。Tweedie公式利用分数函数的估计，直接从带噪声的x_t预测干净的x_0，从而消除最后一步的噪声。

【噪声调度的选择】

噪声调度（即\sigma(t)或\beta(t)的选择）对采样质量有很大影响。

VE SDE：通常使用几何级数的噪声调度，\sigma(t) = \sigma_{\text{min}} (\sigma_{\text{max}}/\sigma_{\text{min}})^t，\sigma_{\text{min}}=0.01，\sigma_{\text{max}}=50
VP SDE：通常使用线性的\beta(t)调度，\beta(t) = \beta_{\text{min}} + t(\beta_{\text{max}} - \beta_{\text{min}})，\beta_{\text{min}}=0.1，\beta_{\text{max}}=20
sub-VP SDE：使用与VP SDE相同的\beta(t)调度

【步数与质量的权衡】

1000步：传统DDPM/DDIM的标准步数，样本质量高，但速度慢
50-100步：DPM-Solver++的常用步数，质量接近1000步，速度快20倍
10-20步：DPM-Solver++的快速模式，质量略有下降，但速度快50-100倍
1-4步：需要使用蒸馏技术（如LCM、SDXL Turbo），质量下降明显，但速度极快

SDE、DDPM、Score-Based扩散模型三者关系总结与理论意义的完整关系图：

Score-Based模型 (SMLD/NCSN) ←→ VE-SDE ←→ 反向SDE + 纯Corrector采样
                                  ↑
                                  |
通用SDE框架 (正向-反向SDE对 + 分数匹配)
                                  |
                                  ↓
DDPM ←→ VP-SDE ←→ 反向SDE + 纯Predictor采样
            ↓
            概率流ODE ←→ DDIM确定性采样

后续Rocky将持续更新补充，大家敬请期待！！！

4.6 SDE框架的通用结构与模块代码原理零基础深入浅出通俗易懂理解

后续Rocky将持续更新补充，大家敬请期待！！！

5. 深入浅出读懂扩散模型Classifier Guidance和Classifier-Free Guidance核心基础知识

在之前的章节中，Rocky已经详细讲解了DDPM、DDIM扩散模型的核心原理。就像GAN、VAE一样，扩散模型也是先研究无条件生成，后续逐步扩展出有条件控制生成能力。

而Classifier Guidance和Classifier-Free Guidance两大技术正是给予扩散模型原生有条件控制生成能力的关键基石。

在基础的无条件扩散模型中，模型从一个随机高斯噪声开始，生成一张“看起来真实”的图片，但内容是完全随机的。这就像让一个画家“随便画点什么”，结果可能很棒，但无法满足我们想要“画一只宇航员猫”的特定需求。

无条件生成前置探索了扩散生成式模型的效果上限，接着有条件生成则更多是探索扩散生成式模型落地应用层面的内容。因为我们可以“引导”扩散模型根据我们给定的条件来控制生成的结果（比如类别标签“猫”，或文本描述“一只戴着宇航员头盔的猫”）。

扩散模型通过条件引导生成各种可控的图像内容

接下来，Rocky将带着大家详细讲解学习Classifier Guidance和Classifier-Free Guidance在扩散模型中的作用与意义。

5.1 零基础深入浅出通俗易懂理解Classifier Guidance（分类器引导）的核心基础知识

Classifier Guidance方案最早由《Diffusion Models Beat GANs on Image Synthesis》提出，使得扩散模型开始能够按类别进行简单的控制生成。后来《More Control for Free! Image Synthesis with Semantic Diffusion Guidance》扩展推广了“Classifier”的概念，使得扩散模型可以按图像、文本和多模态信息等各种条件来生成。

我们可以通过Score Function来直观的对Classifier Guidance进行表达，在进行深入讲解前，我们先前置学习了解一下Score Function的核心基础知识（关于Score-based Models的相关知识，本文暂时不做讲解，大家可自行学习，后续Rocky有时间也会补充更新！）。

什么是Score Function？Score Function在扩散模型中的意义是什么？Don‘t Worry，Rocky向大家娓娓道来。

在扩散模型中，Score Function定义为数据分布的对数概率密度的梯度：

s(\mathbf{x}) = \nabla_\mathbf{x} \log p(\mathbf{x})\\

Score Function指向概率密度增长最快的方向，它的模长表示概率变化的剧烈程度。

我们以高斯分布为例，进行通用的Score Function推导，让我们有一个通识的理解。

对于高斯分布 \mathbf{x} \sim \mathcal{N}(\mu, \sigma^2 \mathbf{I})，其概率密度函数为：

p(\mathbf{x}) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{\left(-\frac{\|\mathbf{x} - \mu\|^2}{2\sigma^2}\right)}\Rightarrow s(\mathbf{x})=\nabla_\mathbf{x} \log p(\mathbf{x}) = -\frac{1}{\sigma^2}(\mathbf{x} - \mu) \\

我们可以发现，对于高斯分布，Score Function与”去均值方向”成正比。

接着在扩散模型的框架下，Score Function与噪声预测器有直接关系。我们已经知道DDPM的前向扩散过程是一个马尔可夫链，每一步都向数据添加高斯噪声，同时我们也得到一个重要的性质是，我们可以直接从 \mathbf{x}_0 采样得到任意时刻 t 的 \mathbf{x}_t，所以给定 x_0 时，x_t 的条件分布是高斯分布：

q(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})\\ \mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, \epsilon \sim \mathcal{N}(0, \mathbf{I})\\

我们再根据高斯分布的Score Function公式带入即可推导得到：

\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t|\mathbf{x}_0) = -\frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0}{1-\bar{\alpha}_t}\\

我们接着推导，将DDPM的噪声预测 \epsilon = \frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0}{\sqrt{1-\bar{\alpha}_t}} 引入到Score Function中，从而获得最终的Score Function表达式：

\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t|\mathbf{x}_0) = -\frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0}{1-\bar{\alpha}_t} = -\frac{1}{\sqrt{1-\bar{\alpha}_t}} \cdot \frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0}{\sqrt{1-\bar{\alpha}_t}} = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}} \\

有了Score Function的公式，接下来我们就加入类别标签 y 对扩散模型的反向去噪过程进行引导，可以通过贝叶斯定理进行进一步推导：

\begin{aligned} \nabla \log p\left(\mathbf{x}_{t} \mid y\right) & =\nabla \log \left(\frac{p\left(\mathbf{x}_{t}\right) p\left(y \mid \mathbf{x}_{t}\right)}{p(y)}\right) \\ & =\nabla \log p\left(\mathbf{x}_{t}\right)+\nabla \log p\left(y \mid \mathbf{x}_{t}\right)-\nabla \log p(y) \\ & =\underbrace{\nabla \log p\left(\mathbf{x}_{t}\right)}_{\text {unconditional score }}+\underbrace{\nabla \log p\left(y \mid \mathbf{x}_{t}\right)}_{\text {classifier gradient }} \end{aligned}\\

在上边的推导过程中，由于 p(y) 与 \mathbf{x}_t 无关，因此\nabla \log p(y) = 0。第一项unconditional score部分是扩散模型本身的梯度引导，引导样本走向数据分布的高概率区域；新增的是第二项classifier gradient部分，额外添加一个classifier的梯度来引导，引导样本走向特定类别的高概率区域。

到这里，我们可以发现，Classifier Guidance本质上是引入了一个额外的分类器，并使用分类器的梯度对扩散模型的生成过程进行引导。

接下来我们看看Classifier Guidance与DDPM是如何结合应用的。

在之前的章节中，我们已经推导出DDPM反向去噪过程的均值 \mu_\theta(\mathbf{x}_t) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(\mathbf{x}_t)\right) 和方差 \Sigma_\theta(\mathbf{x}_t) = \sigma_t^2 \mathbf{I} ，可以得到DDPM的方向去噪采样过程：

\mathbf{x}_{t-1} \sim \mathcal{N}(\mu_\theta(\mathbf{x}_t), \Sigma_\theta(\mathbf{x}_t)) \\

在引入Classifier Guidance后的DDPM增加了分类器梯度计算：\nabla_{\mathbf{x}_t} \log p_\phi(y|\mathbf{x}_t)，同时均值调整（完整推导可查看Classifier Guidance论文）为：\mu_\theta(\mathbf{x}_t)' = \mu_\theta(\mathbf{x}_t) + s \cdot \Sigma \cdot \nabla_{\mathbf{x}_t} \log p_\phi(y|\mathbf{x}_t)\\

可以看到Classifier分类器的梯度对采样时的均值进行了引导。其中s 是引导尺度，也被称为guidance scale，控制分类器影响的强度；\Sigma来我们作为缩放因子，与噪声水平相关。

这时整个DDPM反向去噪的采样过程转换成：

\mathbf{x}_{t-1} = \mu_\theta(\mathbf{x}_t) + s \cdot \sigma_t^2 \nabla_{\mathbf{x}_t} \log p(y|\mathbf{x}_t) + \sigma_t \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})\\

上式中各项含义：

\mathbf{x}_{t-1}：下一时间步的图像
\mathbf{x}_t：当前时间步的噪声图像
y：条件信息（如类别标签、文本描述、图像特征等）
\mu(\mathbf{x}_t)：无条件生成的均值函数
\sigma_t^2：方差参数
\nabla_{\mathbf{x}_t} \log p(y|\mathbf{x}_t)：分类器梯度，指导生成符合条件 y
\epsilon：随机噪声项

上述公式通过在DDPM中添加分类器梯度项 \sigma_t^2 \nabla_{\mathbf{x}_t} \log p(y|\mathbf{x}_t) 来调整无条件生成的方向，使其朝着符合条件 y 的方向演化，从而实现条件生成。

DDPM+Classifier Guidance

接下来我们再来看看Classifier Guidance与DDIM是如何结合的。之前我们已经推导出DDIM反向去噪过程的确定性采样公式：

\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\left(\frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(\mathbf{x}_t,t)}{\sqrt{\bar{\alpha}_t}}\right) + \sqrt{1-\bar{\alpha}_{t-1}}\epsilon_\theta(\mathbf{x}_t,t) \\

在DDIM中，反向去噪过程完全由噪声预测器 \epsilon_\theta(\mathbf{x}_t,t) 驱动，Classifier Guidance的核心是通过分类器梯度来调整这个噪声预测。在引入了Classifier Guidance之后，DDIM的噪声预测调整为： \hat{\epsilon} = \epsilon_\theta(\mathbf{x}_t,t) - s \cdot\sqrt{1-\bar{\alpha}_t} \nabla_{\mathbf{x}_t} \log p_\phi(y|\mathbf{x}_t) \Rightarrow \\ \mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\left(\frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\hat{\epsilon}}{\sqrt{\bar{\alpha}_t}}\right) + \sqrt{1-\bar{\alpha}_{t-1}}\hat{\epsilon} \\

DDIM+Classifier Guidance

总的来说，在Classifier Guidance和DDPM、DDIM结合应用的过程中，引导尺度 s 都控制着条件生成的强度：

s = 0：退化为无条件生成
s > 0：条件控制生成，生成结果更符合类别 y，在实际应用中s 通常取1.0-10.0之间的值（开源社区一般取0.75为宜）
s 过大：可能导致过拟合，生成不自然的图像
使用CG和不使用CG的效果对比图

总的来说，Classifier Guidance通过将分类器的梯度信息融入扩散模型的反向去噪过程中，实现了精确的原生条件控制生成。其在DDPM和DDIM中的具体实现虽然形式不同，但都基于相同的数学原理。

5.2 零基础深入浅出通俗易懂理解Classifier-Free Guidance（无分类器引导）的核心基础知识

Classifier Guidance采用显式分类器来引导扩散模型的条件图像生成，虽然思路直观，但在实际场景中存在若干局限性：

Classifier Guidance需要额外训练一个能够在不同噪声水平下准确工作的分类器，这增加了训练成本和工程复杂度。
最终生成图像的质量在很大程度上依赖于Classifier Guidance分类器的性能，若分类器训练不充分，会直接影响条件生成的效果。
由于Classifier Guidance通过梯度更新图像内容，容易引入对抗样本效应，即生成图像可能包含人眼难以察觉的细微扰动，虽能误导分类器做出特定判断，但并未真正实现符合语义的条件生成。

针对上述问题，谷歌在2022年于《Classifier-Free Diffusion Guidance》中提出了Classifier-Free Guidance方法。该方案不仅有效克服了原有技术的缺陷，还通过引入可调节的引导权重，使扩散模型能够在图像真实性与多样性之间实现灵活权衡。它是条件扩散模型最朴素的方案，但在数据和算力都比较充裕的前提下，Classifier-Free方案表现出了令人惊叹的细节控制能力。已经成为Stable Diffusion、FLUX.1等基于扩散模型架构的主流文生图大模型的基础基石，广泛应用于其训练与推理流程中。

Classifier-Free Guidance的核心创新点在于采用隐式分类机制替代了原有Classifier Guidance的显式分类器，从而避免直接计算分类器梯度。该方法基于贝叶斯理论，将分类器梯度项重新表述为条件生成概率与无条件生成概率之间的差值，从而在不需要显式分类器的情况下实现有效的条件引导。

Classifier-Free Guidance直接从条件概率分布出发，我们可以直接在DDPM的基础上引入条件 y 进行定义：

p(\mathbf{x}_{t-1}|\mathbf{x}_t, y) = \mathcal{N}(\mathbf{x}_{t-1}; \mu(\mathbf{x}_t, y), \sigma_t^2 \mathbf{I})\\上述公式表达了条件扩散模型的条件反向去噪过程，给定当前噪声图像 \mathbf{x}_t 和条件 y，预测下一步的去噪图像 \mathbf{x}_{t-1}。\mu(\mathbf{x}_t, y) 是条件均值函数，依赖于条件 y；\sigma_t^2 \mathbf{I} 是固定的方差，与无条件DDPM保持一致。

根据之前我们对DDPM的推导结果，其中条件均值函数可以被重参数化为：

\mu(\mathbf{x}_t) = \frac{1}{\sqrt\alpha_t} \left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(\mathbf{x}_t, t) \right) \Rightarrow\mu(\mathbf{x}_t, y) = \frac{1}{\sqrt\alpha_t} \left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(\mathbf{x}_t, y, t) \right)\\

在DDPM扩散模型中，我们已经知道Score Function与噪声预测器之间的关系，其中无条件生成和有条件生成的情况分别为：

\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) = -\frac{1}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta (\mathbf{x}_t, t) \\ \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t | y) = -\frac{1}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta (\mathbf{x}_t, t, y)\\

将上述两个结果代入之前我们推导的Score Function公式中：

\nabla_{\mathbf{x}_t} \log p(y | \mathbf{x}_t) = -\frac{1}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta (\mathbf{x}_t, t, y) + \frac{1}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta (\mathbf{x}_t, t) = -\frac{1}{\sqrt{1-\bar{\alpha}_t}} (\epsilon_\theta (\mathbf{x}_t, t, y) - \epsilon_\theta (\mathbf{x}_t, t))\\

我们回顾Classifier Guidance的噪声预测公式，在此基础上代入隐式分类器梯度，最终得到Classifier-Free Guidance的核心噪声预测公式：

\begin{aligned} \bar{\epsilon}_\theta (\mathbf{x}_t, t, y) &= \epsilon_\theta (\mathbf{x}_t, t, y) - \sqrt{1-\bar{\alpha}_t} w \nabla_{\mathbf{x}_t} \log p(y | \mathbf{x}_t)\\ &= \epsilon_\theta (\mathbf{x}_t, t, y) - \sqrt{1-\bar{\alpha}_t} w \left[ -\frac{1}{\sqrt{1-\bar{\alpha}_t}} (\epsilon_\theta (\mathbf{x}_t, t, y) - \epsilon_\theta (\mathbf{x}_t, t)) \right]\\ &= \epsilon_\theta (\mathbf{x}_t, t, y) + w (\epsilon_\theta (\mathbf{x}_t, t, y) - \epsilon_\theta (\mathbf{x}_t, t))\\ &= (w+1) \epsilon_\theta (\mathbf{x}_t, t, y) - w \epsilon_\theta (\mathbf{x}_t, t)\\ \end{aligned} \\

我们只需将DDPM中的\epsilon_\theta(\mathbf{x}_t,t)替换为\bar{\epsilon}(\mathbf{x}_t,t,y)就得到了基于条件的反向去噪过程。上式中各项的含义：

\epsilon_\theta (\mathbf{x}_t, t, y)：条件生成的噪声预测，专注于生成符合条件 y 的图像，(w+1) \epsilon_\theta (\mathbf{x}_t, t, y) 放大了条件生成的方向
\epsilon_\theta (\mathbf{x}_t, t)：无条件生成的噪声预测，专注于生成逼真但不指定内容的图像，- w \epsilon_\theta (\mathbf{x}_t, t) 减弱了无条件生成的随机性
w：引导权重，控制条件生成的强度，通过调节 w 在真实性和条件符合度之间取得平衡。当w = 0时退化为普通的条件生成模型；当w = 1时适度的条件引导；当w > 1时强条件引导，生成结果更符合条件但可能牺牲多样性；当w \to \infty时极端条件引导，可能产生过拟合
不同CFGscale大小的效果对比图

这时有条件扩散模型的训练目标函数为：

\mathcal{L}=\mathbb{E}_{\mathbf{x}_0, y \sim \tilde{p}(\mathbf{x}_0, y), \epsilon \sim \mathcal{N}(0, \mathbf{I})} \left[ \| \epsilon - \epsilon_\theta(\tilde{\alpha}_t \mathbf{x}_0 + \tilde{\beta}_t \epsilon, y, t) \|^2 \right]\\

上式中各项含义如下：

\mathbf{x}_0, y \sim \tilde{p}(\mathbf{x}_0, y)：从数据分布中采样真实图像和对应条件
\epsilon \sim \mathcal{N}(0, \mathbf{I})：采样随机噪声
\tilde{\alpha}_t \mathbf{x}_0 + \tilde{\beta}_t \epsilon：前向扩散过程，得到带噪图像 x_t
\epsilon_\theta(\mathbf{x}_t, y, t)：条件噪声预测器
\|\epsilon - \epsilon_\theta(\cdots)\|^2：噪声预测的均方误差

在训练过程中，基于Classifier-Free Guidance的单一条件扩散模型实现双重功能。训练时以一定概率 p_{\text{drop}} 将条件 y 置为空（通常 p_{\text{drop}} \approx 0.1 \sim 0.2 ），当 y = \phi（空条件）时，模型学习无条件生成；当 y 有具体值时，模型学习条件生成。这样同一个模型同时具备 \epsilon_\theta(\mathbf{x}_t, y, t) 和 \epsilon_\theta(\mathbf{x}_t, t) 两种能力。

Classifier-Free Guidance核心思想可以概括为通过训练时随机丢弃条件（以一定概率将条件置空即可）来获得无条件生成能力，在推理时通过权重调节来平衡条件引导的强度，实现了无需额外分类器的高质量条件生成。

CFGscale过大导致的效果失真

推理时，扩散模型最终生成结果可以由条件生成和无条件生成的线性外推获得，生成效果可以由引导系数调节，控制生成样本的逼真性和多样性的平衡。

def training_step(x_0, y, t):
    # 以概率 p_drop 随机丢弃条件
    if random() < p_drop:

        y = None  # 无条件训练
    
    # 添加噪声
    x_t = add_noise(x_0, t)
    
    # 预测噪声
    if y is None:
        epsilon_pred = model(x_t, t)      # 无条件预测
    else:
        epsilon_pred = model(x_t, t, y)   # 条件预测
    
    # 计算损失
    loss = mse_loss(epsilon, epsilon_pred)
    return loss

Rocky在这里汇总对比一下Classifier Guidance和Classifier-Free Guidance的异同：

特性	Classifier Guidance	Classifier-Free Guidance
核心机制	使用外部分类器的梯度来引导生成过程。	使用同一个模型的条件与无条件预测之差进行引导。
训练需求	需要训练一个在噪声图像上工作的额外分类器。	只需训练一个扩散模型，但需随机丢弃条件。
条件形式	通常是简单的类别标签。	可以是任意条件（文本、图像、类别等）。
稳定性	相对不稳定，易产生伪影。	非常稳定，生成质量高。
控制能力	通过引导尺度 s 控制，但效果有限。	通过引导尺度 s 提供强大且平滑的控制， s>1 能显著提升质量。
当前地位	多为早期和历史性方法。	现代主流方法（Stable Diffusion, FLUX.1等均采用）。
5.3 零基础深入浅出通俗易懂理解Classifier Guidance和Classifier-Free Guidance的通用结构与模块代码原理

未完待续，大家敬请期待！！！

码字确实不易，希望大家能多多点赞！！！

6. Rectified Flow扩散模型核心基础知识深入浅出完整讲解

在2024年AIGC图像创作/AI绘画领域进入FLUX时代后，Rectified Flow（RF，直线整流）就成为了扩散模型的主流核心数学思想。

而我们想要理解Rectified Flow，就需要前置的理解Flow Matching（FM，流匹配）框架。Flow Matching是一种定义图像生成目标的方法，将生成过程视为一个连续的流动，它可以兼容当前扩散模型的训练目标，扩展丰富了扩散模型的思想与边界。而Flow Matching框架中一个非常有代表性的特例分支就是我们接下来要讲解的Rectified Flow，它也正是Stable Diffusion 3以及后续FLUX系列、Seedream系列、OpenAI Sora、Midjourney v6、Google Nano Banana、Seedrance系列等主流AIGC图像创作/AIGC视频生成大模型用到的核心训练优化目标。

为什么Flow Matching中的Rectified Flow分支会成为扩散模型乃至AIGC图像生成领域的核心主流数学思想，我们首先要明白传统扩散模型一直饱受困扰的核心问题：采样速度慢和采样效率低。

经典扩散模型（如DDPM、DDIM、SDE等）通过一个前向扩散过程将数据（如图像、视频、文本、音频等）逐步加噪变成纯噪声，然后再学习一个反向去噪过程。这个前向扩散过程通常被设计成一条复杂的、非线性的路径（通常遵循一个预先定义好的方差调度表（variance schedule）），这就好比我们想从杭州大运河点到杭州西湖，却走了一条蜿蜒曲折的复杂路程。因为路径是弯曲复杂的，经典扩散模型在每个采样点上的“步伐”和“方向”都很复杂，导致反向去噪过程（采样）中需要很多步（例如1000步，即使DDIM跳步优化后也需要30-50步）才能保证生成质量，所以生成速度与效率一直存在瓶颈。

而基于Rectified Flow框架的扩散模型，与经典扩散模型相比，Rocky认为两者本质区别在于数据到噪声路径的定义方式：其核心思想极其简单却威力无穷，直接学习从噪声分布到数据分布的直线传输映射。在实际应用中甚至可以一步生成图像。

Rectified Flow框架与传统扩散模型的区别示意图

为了让大家更好的理解Flow Matching和Rectified Flow，在接下来的章节中，Rocky将逐步深入浅出讲解它们的核心基础知识。

6.1 Flow Matching（FM）、ODE（常微分方程）的核心原理零基础深入浅出通俗易懂理解

在本章节中，Rocky先带着大家学习理解Rectified Flow必备的一些前置基础知识与概念，以便大家在后续的章节中能更好的理解Rectified Flow的本质原理。

我们已经知道Rectified Flow是Flow Matching框架中的一个特例，Flow Matching（FM）是一个连续时间生成模型的通用框架。它的核心思想及其直观：与其像经典扩散模型那样一步步地去噪，不如直接学习一个连续的、平滑的变换运动轨迹（即一个“连续变换流”），将高斯噪声分布直接“流动”转变成数据分布，即将生成过程可以看作是数据概率分布在高维空间中的连续流动。

Flow Matching的思想最早可追溯到continuous normalizing flows，本质上是将任何连续时间生成模型都定义成一个ODE（Ordinary Differential Equation，常微分方程）：

dz_t = v_\theta(z_t, t) \, dt \Leftrightarrow \frac{dz_t}{dt} = v_\theta(z_t, t) \\

z_t ： 在时间 t 时，数据分布的状态。在AIGC图像创作领域中， z_t 可以理解为一张“正在生成过程中的图像”。当 t=1 时， z_1 是纯高斯噪声；当 t=0 时， z_0 是清晰的图像。
t \in [0, 1] ： 归一化后的时间变量。它描述了从纯噪声（起点， t=1 ）到清晰数据（终点， t=0 ）的整个分布变换过程。
v_\theta(z_t, t) ： 代表一个可学习的向量场（神经网络模型）。它是整个Flow Matching框架的核心。我们可以把它想象成一个“变化规则”。这个神经网络模型学习的就是数据分布状态 z_t 在时间 t 的变化方向和速度。它的输出是一个特征分布，其维度和 z_t 相同。
dz_t = v_\theta(z_t, t) \, dt ： 代表一个ODE方程。它描述了数据分布状态 z_t 随时间 t 的变化规律。简单来说，表示“数据分布状态 z_t 随时间 t 的变化，由速度场 v_\theta 决定。”。

到这里，读者可能会有第一个疑问？什么是ODE（常微分方程）？Don't Worry，Rocky接下来为大家娓娓道来其本质含义。

ODE方程的本质是描述一个数据分布状态如何随时间演化的数学方程。 它建立了数据分布状态与其变化率之间的关系，本质上就是描述一个未知函数与其导数之间的关系。而“常”这个字，指的是我们关心的函数只有一个自变量，在AIGC图像生成/视频生成领域中，这个自变量通常是时间 t 。通用ODE方程的基本形式如下：

\frac{dy}{dt} = f(t, y) \\

其中 y 代表数据分布状态（未知函数，可以是标量、向量、张量等数据分布）； t 代表时间变量； \frac{dy}{dt} 代表数据分布状态随时间的变化率； f(t, y) 代表变化规律函数，描述了在时间 t 和数据分布状态 y 下，未来数据分布状态应该如何变化。

Rocky在这里举一个简单的例子方便大家理解，比如 \frac{dy}{dt} = 2t 这个ODE告诉我们：“函数 y(t) 的变化率（导数）等于 2t ”。接着我们以物体运动为例讲解，想象一辆汽车在路上直线行驶，我们知道它在任意时刻 t 的瞬时速度 v(t) 。

我们已知什么？ 速度，也就是位置 y(t) 的导数： v(t) = \frac{dy}{dt} 。
我们想要知道什么？ 汽车在特定时间点的位置 y(t) 。

那么，我们就可以将它们的关系写成一个ODE： \frac{dy}{dt} = v(t) 。它描述了“位置函数的变化率等于速度函数”这个“变化规则”。

现在，我们求解这个ODE，也就是根据这个“变化规则”，找出位置函数 y(t) 本身的表达式。

假设速度是常数，比如 v(t) = 5 （米/秒），这时ODE是 \frac{dy}{dt} = 5 。
接着求解它，我们得到 y(t) = 5t + C 。这里的 C 是一个常数，代表汽车的初始位置（即当 t=0 时， y(0) = C ）。

到此为止，我们可以把ODE的解转换成一条在时间-状态空间中的运动轨迹。 求解ODE，就是从初始点 (t_0, y_0) 出发，画一条曲线，使得这条曲线在每一个点的切线方向都严格符合ODE的“变化规则”规定的方向。

上述只是一个简单的例子，但是现实情况一般都比较复杂，大多数ODE没有简单的解析解，我们需要通过数值方法（比如欧拉法）来近似求解：

y_{n+1} = y_n + f(t_n, y_n) \cdot \Delta t \\

其中的物理意义可以表示为：”下一步的数据分布状态 = 当前数据分布状态 + 变化率 × 时间步长”。就像用一系列短直线来近似一条曲线，步长 \Delta t 越小，近似越精确，但同时会导致采样步数增加，采样速度变慢，反之采样速度快但是近似误差大。

那么，什么样的ODE形式，或者说什么样的数据分布运动轨迹，能够让我们更好地兼顾精度和速度呢？Rocky认为答案其实很直觉：那就是尽可能走直线，而这也是Rectified Flow的本质思想。如果轨迹特别弯曲，那我们就必须使用尽量多的步数进行尽可能高的近似精度拟合。而如果轨迹足够”直“，理想状态下数据初次移动的方向就是指向终点的，那么我们甚至可以一步实现的高精度近似。关于Rectified Flow的内容，在本章节中我们先按下不表。

到目前为止，Rocky通过上述生动例子已经讲解了ODE方程的本质概念与意义。在AIGC图像创作领域的Flow Matching框架中，我们关心的不是一个简单系统的运动轨迹，而是整个数据概率分布如何随时间演化。经典的扩散模型如DDPM、DDIM使用离散时间步，而Flow Matching则将离散过程视为连续过程的离散化，转变成ODE方程：

\frac{dz_t}{dt} = v_\theta(z_t, t) \\

到此为止，我们就完整讲解常微分方程的数学本质含义，重新推导出我们在本章开头的Flow Matching框架下AIGC图像创作领域的ODE方程。

总的来说，ODE在Flow Matching框架中提供了一个连续的、平滑的变换视角，描述了数据（如图像、视频、文本、音频等）如何从噪声“流动”到清晰图片的连续、确定的轨迹。图像数据分布不是被“一步步去噪”，而是沿着一条连续的轨迹“流动”成最终形态。

在此基础上，我们就可以在ODE方程上建立概率路径（probability path）与分布变换。构建一个概率路径 p_t ，它可以实现从噪声分布 p_1 到数据分布 p_0 的变换：

p_1 ： 起点分布，一般来说是标准高斯分布 \mathcal{N}(0, \mathbf{I}) 。
p_0 ： 终点分布，即我们想要建模的复杂数据分布，比如“所有动物图像数据的分布”。
概率路径 p_t ： 这不是一条单一数据点的运动轨迹，而是作用在整个高维数据分布上。它推动着整个数据概率分布从 p_1 流向 p_0 。在 t=1 时， p_t 就是 p_1 （一堆噪声分布）；在 t=0 时， p_t 就是 p_0 （一堆清晰的动物图像）。在中间时刻 t ， p_t 描述了所有数据处于图像和噪声混合的状态。

而在模型推理阶段，我们就需要反向求解ODE：

从一个噪声样本 z_1 \sim p_1 = \mathcal{N}(0, \mathbf{I}) 开始。
以这个ODE为规则，从 t=1 积分到 t=0 ： z_0 = z_1 + \int_{1}^{0} v_\theta(z_t, t) \, dt
这个积分过程可以通过数值ODE求解器（如欧拉法）来近似完成。最终得到的 z_0 就是从数据分布 p_0 中采样的一个新数据点（一张新的动物图像）

下面是使用欧拉法求解ODE的简单代码示例，初始条件为 z_1 是随机高斯噪声。目标是通过数值方法（如欧拉法等）反向求解这个ODE，从 t=1 积分到 t=0 ，最终得到 z_0 （清晰的图像）：

z = torch.randn(batch_size, 3, 1024, 1024)  # 初始噪声
for i in range(num_steps):
    t = 1.0 - i / num_steps  # 从1到0
    dz = v_theta(z, t) * (-dt)  # 反向时间
    z = z + dz
# 最终 z 就是生成的图像

Rocky推导讲解到这里，聪明的读者可能已经发现，现在我们遇到了核心问题：我们并不知道上述数值求解方法中向量场 v(z_t, t) 的具体表达式。我们需要用一个神经网络 v_\theta(z_t, t) 来学习拟合它。

那么，我们学习训练的目标是什么？我们如何定义Flow Matching框架的损失函数呢？

Flow Matching框架的目标是学习训练得到一个向量场v_\theta(z,t)，使得当初始分布p_1(x)是简单的高斯分布时，经过时间t=1的流动后，最终分布p_0(x)恰好等于真实数据分布。我们可以将这个优化目标建模为：

\displaystyle\mathcal{L}_{FM}=\mathbb{E}_{t,p_{t}(z)}||v_{\theta}(z,t)-u_{t}(z)||_{2}^{2} \\

这个优化目标是Flow Matching框架的灵魂。其中各个符号的含义如下：

t ：时间变量，在 [0,1] 区间内均匀采样
p_t(z) ：在时间 t 的边际概率分布，描述了从噪声到数据的整个概率路径
v_\theta(z, t) ： 我们需要训练的Flow Matching框架下的扩散模型，一个参数为 \theta 的神经网络，它的目标是去拟合真实向量场
u_t(z) ： 代表目标向量场/真实向量场。它是唯一能恰好将分布 p_1 变到 p_0 的那个“完美”的向量场
|| \cdot ||_2^2 ：L2范数的平方，代表MSE损失，即均方误差。用来衡量神经网络模型拟合的向量场和目标向量场之间的差异

我们从直观上理解， 这个损失函数要求神经网络 v_\theta 在整个概率路径 p_t(z) 上，尽可能准确地预测目标向量场 u_t(z) 。如果能够完美匹配，那么沿着 v_\theta 定义的ODE进行数值积分，就能准确地将噪声分布 p_1 转变为数据分布 p_0 。

但是现实往往没有这么理想，那就是目标向量场的不可知性，这是Flow Matching框架面临的根本性挑战。

那么目标向量场为什么不可知呢？因为从噪声分布 p_1 到数据分布 p_0 的概率路径 p_t(z) 本身存在无穷多种，每个不同的概率路径都对应一个不同的目标向量场 u_t(z) ，如果没有额外的约束或前置的假设，我们无法确定应该学习哪一个 u_t(z) 。就像我们要从杭州到北京，有无数条路径（高铁、高速、高架、国道、乡间小路等）。其中体现的数学本质是给定两个边缘分布p_0（数据）和p_1（噪声），存在无穷多个联合分布p(x_0, x_1)满足边缘条件。每个联合分布都对应一个不同的传输计划(Transport Plan)，进而对应一个不同的向量场u_t(z)。

为了解决上述问题，Flow Matching采用了一个极其巧妙的策略。不直接学习全局的目标向量场，而是先为每个数据点构造一个局部的条件概率路径（Conditional Probability Path）：

p_t(z|\mathbf{x}_0) = \mathcal{N} (z|a_t \mathbf{x}_0, b_t^2 \mathbf{I}) \\

上述公式的含义是对于每一个真实数据点\mathbf{x}_0，我们构造一个以a_t \mathbf{x}_0为均值、b_t为标准差的高斯分布，作为时刻t的中间分布。其中各个符号的含义如下：

\mathbf{x}_0 ：真实数据样本（条件）
a_t ：与时间 t 相关的缩放函数
b_t ：与时间 t 相关的标准差函数
\mathbf{I} ：单位矩阵

为了让这些局部路径能够共同构成一个从数据分布到噪声分布的全局变换，Flow Matching同时定义了边界条件的约束，对于每个数据点 \mathbf{x}_0 \sim q(\mathbf{x}_0) ，我们构造一个条件概率路径 p_t(z|\mathbf{x}_0) ，满足：

当 t = 0 时，取 a_0 = 1, b_0 = 0 。此时 p_0(z|\mathbf{x}_0) = \mathcal{N}(z|\mathbf{x}_0, 0) = \delta(z - \mathbf{x}_0) （狄拉克函数），这确保在终点时刻，分布完全集中在真实数据 \mathbf{x}_0 上。
当 t = 1 时，取 a_1 = 0, b_1 = 1 。此时 p_1(z|\mathbf{x}_0) = \mathcal{N}(z|0, \mathbf{I}) （标准高斯分布），这确保在起点时刻，分布是标准高斯噪声。

有了上述的约定限制，这里定义的条件概率路径p_t(z|\mathbf{x}_0)能够保证噪声分布p_1到真实数据分布q(\mathbf{x}_0)的转变。

引入条件概率路径p_t(z|\mathbf{x}_0)，Rocky认为其本质上是一种“分而治之”的机器学习策略。它把”学习两个高维分布之间的全局映射”这个极其困难的问题，分解成了”为每个数据点学习一个从数据到噪声的局部路径”这个相对简单的问题，然后通过对所有数据点的局部路径取期望，就学习得到了全局的向量场。

接着我们使用重参数化技巧，可以将条件概率路径p_t(z|\mathbf{x}_0)转化成采样公式：

z_t = a_t \mathbf{x}_0 + b_t \epsilon \quad \text{where } \epsilon \sim \mathcal{N}(0, \mathbf{I}) \\

这是整个生成模型领域最重要的公式之一。它的本质是任何时刻的中间样本z_t，都可以表示为真实数据\mathbf{x}_0和高斯噪声\epsilon的线性组合。

这个形式与经典扩散模型的前向扩散过程完全相同，只是在DDPM中取 a_t = \sqrt{\bar{\alpha}_t} ， b_t = \sqrt{1 - \bar{\alpha}_t} 。这给了我们一个伟大的结论，所有的传统扩散模型，本质上都是Flow Matching框架的一个特殊实例。同时无论是传统扩散模型还是Flow Matching模型，它们的前向扩散过程本质上都是在真实数据和高斯噪声之间进行不同形式的线性插值，我们可以通过认为设定 a_t 和 b_t 的取值来确定传输路径。

有了上面的基础，Flow Matching框架就从学习目标向量场 u_t(z) 转变为学习条件向量场 u_t(z|x_0) ，从而就构建了一个全新的优化目标Conditional Flow Matching（CFM）：

\mathcal{L}_{CFM} = \mathbb{E}_{t,q(\mathbf{x}_0),p_t(z|\mathbf{x}_0)} || v_\theta(z,t) - u_t(z|\mathbf{x}_0)||^2_2 \\

其中各个符号的含义如下：

q(\mathbf{x}_0) ：真实数据分布
p_t(z|\mathbf{x}_0) ：条件概率路径
\mathbb{E}_{t, q(\mathbf{x}_0), p_t(z|\mathbf{x}_0)} ：对时间 t 、数据 x_0 、数据分布状态 z 的期望
t ：在 [0,1] 上均匀采样的时间变量
v_\theta(z, t) ：神经网络训练学习的向量场
u_t(z|\mathbf{x}_0) ：真实的条件目标向量场

这个期望可以具体等价写为：

\mathcal{L}_{CFM} = \int_0^1 dt \int q(\mathbf{x}_0) d\mathbf{x}_0 \int p_t(z|\mathbf{x}_0) \| v_\theta(z, t) - u_t(z|\mathbf{x}_0) \|^2 dz \\

在实际训练中，我们依旧过通过深度学习领域的训练基石技术蒙特卡洛采样来近似这个积分：先从 U[0,1] 采样时间 t ，再从数据集采样数据 \mathbf{x}_0 \sim q(\mathbf{x}_0) ，接着从条件分布采样 z \sim p_t(z|\mathbf{x}_0) ，最后计算损失并求平均。

到这里大家可能会有一个小疑问，为什么可以用 \mathcal{L}_{CFM} 作为新的优化目标， \mathcal{L}_{CFM} 和 \mathcal{L}_{FM} 是等价的吗？Flow Matching论文中主要通过证明两者的梯度等价性，来推导其作为优化目标的一致性。虽然 \mathcal{L}_{FM} 和 \mathcal{L}_{CFM} 看起来不同，但可以证明：

\nabla_\theta \mathcal{L}_{FM}(\theta) = \nabla_\theta \mathcal{L}_{CFM}(\theta) \\

主要是利用重参数化技巧和期望的线性性质证明（感兴趣的可以看Flow Matching论文中的证明）了：

\mathcal{L}_{CFM} = \mathcal{L}_{FM} + C \\

其中 C 是与 \theta 无关的常数，因此两个损失的梯度相同。这个结论这意味着虽然我们无法直接优化Flow Matching的初始目标（因为 u_t(z) 未知），但我们可以通过优化 \mathcal{L}_{CFM} 来间接实现对 \mathcal{L}_{FM} 的优化！

到此为止，我们知道虽然u_{t}(z)是不可知的，但是引入条件后的u_{t}(z|\mathbf{x}_0)是可以计算的：

\begin{aligned}\\u_t(z|\mathbf{x}_0) &= z'_t = \frac{dz_t}{dt} = \frac{da_t}{dt} \mathbf{x}_0 + \frac{db_t}{dt} \epsilon\\ &= a'_t \mathbf{x}_0 + b'_t \epsilon\\ &= a'_t \cdot \frac{z_t - b_t \epsilon}{a_t} + b'_t \epsilon \\ &= \frac{a'_t}{a_t} z_t - \frac{a'_t}{a_t} b_t \epsilon + b'_t \epsilon \\ &= \frac{a'_t}{a_t} z_t + \epsilon \left( b'_t - \frac{a'_t}{a_t} b_t \right) \\ \end{aligned}\\

在实际训练中，我们可以按照如下步骤进行：

从数据集中采样 \mathbf{x}_0 \sim q(\mathbf{x}_0)
从噪声分布采样 \epsilon \sim \mathcal{N}(0, \mathbf{I})
在时间上均匀采样 t \sim U[0,1]
计算 z_t = a_t \mathbf{x}_0 + b_t \epsilon
计算目标向量场 u_t(z|\mathbf{x}_0) = a'_t \mathbf{x}_0 + b'_t \epsilon
让神经网络 v_\theta(z_t, t) 去拟合 u_t(z|\mathbf{x}_0)

如果我们能训练神经网络 v_\theta 让它非常接近这个真实的 u_t(z) ，那么我们就得到了一个高质量的Flow Matching框架扩散模型。

接着Flow Matching框架中还“天才”的全新引入信噪比（Signal-to-Noise Ratio, SNR） \lambda_t 概念：

\lambda_t = \log \frac{a^2_t}{b^2_t} = 2(\log a_t - \log b_t) \Rightarrow \lambda'_t = 2 \left( \frac{a'_t}{a_t} - \frac{b'_t}{b_t} \right) \\

将信噪比带入条件目标向量场 u_{t}(z|\mathbf{x}_0) 重新整理括号内的项：

b'_t - \frac{a'_t}{a_t} b_t = -b_t \left( \frac{a'_t}{a_t} - \frac{b'_t}{b_t} \right) = -\frac{b_t}{2} \lambda'_t \\

最终得到：

u_t(z|\mathbf{x}_0) = \frac{a'_t}{a_t} z_t - \frac{b_t}{2} \lambda'_t \epsilon \\

我们再将 u_{t}(z|\mathbf{x}_0) 代入CFM目标函数可以推导得到：

\mathcal{L}_{CFM} = \mathbb{E}_{t,q(\mathbf{x}_0),p_t(z|\mathbf{x}_0),\epsilon \sim \mathcal{N}(0,\mathbf{I})} \left\| v_\theta(z,t) - \left( \frac{a'_t}{a_t} z + \frac{b_t}{2} \lambda'_t \epsilon \right) \right\|^2_2 \\

接着我们再将 v_\theta(z,t) 重参数化定义为：

v_\theta(z,t) = \frac{a_t'}{a_t} z_t - \frac{b_t}{2} \lambda_t' \epsilon_\theta(z,t) \\

这时神经网络不再直接预测整个向量场，而是让神经网络 \epsilon_\theta(z,t) 预测噪声 \epsilon ，这与DDPM等传统扩散模型的思路一致。

将重参数化的 v_\theta 代入CFM目标，并进行化简：

\begin{aligned}\mathcal{L}_{CFM} &= \mathbb{E}_{t, q(\mathbf{x}_0), p_t(z|\mathbf{x}_0), \epsilon \sim \mathcal{N}(0,\mathbf{I})} \left\| \left( \frac{a_t'}{a_t} z_t - \frac{b_t}{2} \lambda_t' \epsilon_\theta(z,t) \right) - \left( \frac{a_t'}{a_t} z_t - \frac{b_t}{2} \lambda_t' \epsilon \right) \right\|^2_2\\ &= \mathbb{E}_{t, q(\mathbf{x}_0), p_t(z|\mathbf{x}_0), \epsilon \sim \mathcal{N}(0,\mathbf{I})} \left\| - \frac{b_t}{2} \lambda_t' \epsilon_\theta(z,t) + \frac{b_t}{2} \lambda_t' \epsilon \right\|^2_2 \\ &= \mathbb{E}_{t, q(\mathbf{x}_0), p_t(z|\mathbf{x}_0), \epsilon \sim \mathcal{N}(0,\mathbf{I})} \left\| -\frac{b_t}{2} \lambda_t' (\epsilon - \epsilon_\theta(z,t)) \right\|^2_2\\ &= \mathbb{E}_{t, q(\mathbf{x}_0), p_t(z|\mathbf{x}_0), \epsilon \sim \mathcal{N}(0,\mathbf{I})} \left( -\frac{b_t}{2} \lambda_t' \right)^2 \| \epsilon_\theta(z,t) - \epsilon \|^2\\ \end{aligned}\\

推导到这里，我们可以发现Flow Matching框架统一了扩散模型的目标函数：

\mathcal{L}_w(\mathbf{x}_0) = -\frac{1}{2} \mathbb{E}_{t \sim \mathcal{U}(0,1), \epsilon \sim \mathcal{N}(0,\mathbf{I})} \left[ w_t \lambda_t' \| \epsilon_\theta(z_t,t) - \epsilon \|^2 \right] \\

当我们设置权重 w_t = -\frac{2}{\lambda_t'} ，则目标函数与DDPM ( \mathcal{L}_{simple} )一致：

\mathcal{L}_w(\mathbf{x}_0) = -\frac{1}{2} \mathbb{E} \left[ \left( -\frac{2}{\lambda_t'} \right) \lambda_t' \| \epsilon_\theta - \epsilon \|^2 \right] = \mathbb{E} \left[ \| \epsilon_\theta - \epsilon \|^2 \right] \\当我们设置权重 w_t = -\frac{1}{2} \lambda_t' b_t^2 ，则目标函数与Flow Matching ( \mathcal{L}_{CFM} )一致：

\mathcal{L}_w(\mathbf{x}_0) = -\frac{1}{2} \mathbb{E} \left[ \left( -\frac{1}{2} \lambda_t' b_t^2 \right) \lambda_t' \| \epsilon_\theta - \epsilon \|^2 \right] = \mathbb{E} \left[ \frac{1}{4} (\lambda_t')^2 b_t^2 \| \epsilon_\theta - \epsilon \|^2 \right] \\

（注意 \left( \frac{b_t}{2} \lambda_t' \right)^2 = \frac{1}{4} (\lambda_t')^2 b_t^2 ）。

这个发现的意义非常重大，结合我们之前推导的采样方法的大一统，Flow Matching框架为AIGC图像创作领域提出了一个统一的视角，不同形态的扩散模型都可以用统一的采样方法和优化目标表示。这揭示了不同扩散模型（DDPM、DDIM、SDE、Flow Matching等）本质上是Flow Matching框架的不同特例。通过选择不同的 a_t 、 b_t 和 w_t ，可以设计出更适合特定任务的扩散模型。

总的来说，Rocky认为扩散模型的多样性主要源于两方面。其一，是数据到噪声演变路径的前向过程的定义与设计，这由 a_t 和 b_t 的选取来决定不同的条件概率路径；其二，是模型的学习目标的不同定义，例如优化目标可以为预测噪声\epsilon（DDPM、DDIM），预测分数s（SDE），以及预测向量场v（Flow Matching）等。

同时这些差异仅是表象。Rocky认为一个统一的本质观点是，它们最终都可被归结为对噪声\epsilon的预测，不同扩散模型的特性则由损失函数中特定的权重系数 w_t 来体现。

下一章节要详细讲解的Rectified Flow正是Flow Matching框架的一个具体、高效且强大的特例实现，它直接采用了直线作为条件流，并证明了其卓越的性能。

6.2 Rectified Flow的核心原理零基础深入浅出通俗易懂理解

我们在之前的章节中，我们已经理解了Flow Matching的核心原理。在本章节中，Rocky将带着大家学习Rectified Flow这个目前扩散模型领域的主流核心数学思想。

到这里，很多读者应该对Flow Matching和Rectified Flow之间的关系有初步的理解。Rocky在这里再给大家一个前置的总结归纳，Flow Matching和Rectified Flow之间的关系可以精辟地概括为：Flow Matching 是一个强大而通用的理论框架，而Rectified Flow是这个框架下一个极其简单、高效且强大的伟大特例。

我们已经知道Rectified Flow是Flow Matching框架中的一个特例，Rectified Flow提出了一个革命性的数学思想：为什么不直接从数据分布到噪声分布构建一条直线路径呢？这就是Rectified Flow提出的“两点之间，直线最短”的本质哲学。

为了建模这个数学思想，Rectified Flow引入了一个常微分方程（ODE），来描述从真实数据分布（ t=0 ）到纯高斯噪声（ t=1 ）的连续变换流。

我们可以用一个比喻来理解：Flow Matching 就像是提出了“制造一辆交通工具”的宏伟蓝图。而Rectified Flow 则是根据这个蓝图，选择了一条最直接、最高效的路径，制造出了一辆“超级跑车”。

特性	Flow Matching (FM)	Rectified Flow (RF)
定位	通用理论框架	具体算法实例
核心贡献	提出了Conditional FM，解决了目标向量场不可知的问题。	指出直线路径是最优选择，并提出了拉直路径的整流方法。
路径选择	不指定，可以是任何路径（如扩散路径）。	明确指定为直线路径。
向量场	可能很复杂。	非常简单且恒定。
采样效率	取决于路径，如果路径弯曲，则采样慢。	路径为直线，采样效率极高（可少至1-2步）。
关键概念	条件流匹配（CFM）	整流（Reflow）

Rectified Flow的核心贡献是在FM框架下，提出最简单的直线路径往往是最优路径。为了得到最优路径的“直线” 轨迹，论文中将这个移动轨迹直接建模为起点和终点的插值，将Rectified Flow的前向扩散过程定义为：

z_t = a_t x_0 + b_t \epsilon = (1 - t)\mathbf{x}_0 + t\epsilon \quad \text{where } \epsilon \sim \mathcal{N}(0, \mathbf{I})\\

这里设置 a_t = 1-t ， b_t = t 。其中的各符号含义如下：

z_t ：在时间 t 的状态（部分噪声的数据）
\mathbf{x}_0 ：原始数据样本
\epsilon ：标准高斯噪声， \epsilon \sim \mathcal{N}(0, \mathbf{I})
t ：归一化时间， t \in [0,1]

从物理角度来看，这是一个线性插值过程，它用一条直线直接将数据点 \mathbf{x}_0 和噪声点 \epsilon 连接起来：

当 t=0 时： z_0 = \mathbf{x}_0 （纯数据）
当 t=1 时： z_1 = \epsilon （纯噪声）
在中间时刻：数据与噪声的线性混合

其实Rectified Flow的前向过程与最优传输有着千丝万缕的关系。基于最优传输（ Optimal Transport）更一般的前向过程如下所示：

z_t = (1 - t)\mathbf{x}_0 + ((1 - t)\sigma_{min} + t)\epsilon \\

最优传输研究如何以最小成本将一个分布转移到另一个分布。当 \sigma_{min} = 0 时，我们就获得了Rectified Flow的前向过程 z_t = (1 - t)\mathbf{x}_0 + t\epsilon ，可以说Rectified Flow的前向过程是最优传输思想中充满价值的一个分支。

接下来，Rocky在此基础上带着大家对Rectified Flow的优化目标进行完整的推导。

我们首先计算时间导数得到条件向量场的表达式：

u_t(z|\mathbf{x}_0) = z_t' = \frac{d}{dt}[(1-t)\mathbf{x}_0 + t\epsilon] = -\mathbf{x}_0 + \epsilon \\

因此，RF的训练目标就是让神经网络 v_\theta(z_t, t) 在所有点和所有时间，都预测出这个恒定的方向 (\mathbf{x}_0 - \epsilon) 。 接着我们将其代入CFM目标函数中，就得到了Rectified Flow的最终优化目标：

\mathcal{L}_{RF} = \mathbb{E}_{t,q(\mathbf{x}_0),p_t(z|\mathbf{x}_0),\epsilon \sim \mathcal{N}(0,\mathbf{I})} ||v_\theta(z,t) - (\epsilon - \mathbf{x}_0)||_2^2 \\

总的来说，Rectified Flow就是把图像生成问题，巧妙地转化成了一个求解常微分方程的数学问题，从而实现了极致的加速。其构建过程非常巧妙：

配对：对于数据集中的每一个真实数据样本 \mathbf{x}_0 （例如一张猫的图片），我们从一个简单的分布（如标准高斯分布）中随机采样一个对应的噪声 \mathbf{x}_1 。这样我们就得到了一个数据-噪声对 (\mathbf{x}_0, \mathbf{x}_1) 。
线性插值：我们定义一条连接 \mathbf{x}_0 和 \mathbf{x}_1 的直线。在任意时刻 t （ 0 \leq t \leq 1 ），插值点 z_t 为： z_t = (1 - t) \cdot \mathbf{x}_0 + t \cdot \mathbf{x}_1 这可以理解为在时间 t ，数据 \mathbf{x}_0 和噪声 \mathbf{x}_1 的线性混合。
学习速度场：我们的目标是学习一个速度场 v_\theta ，它是一个神经网络，输入是当前点 z_t 和时间 t ，输出是一个速度向量。这个速度场应该满足：沿着这条直线路径，它在任意点 z_t 处的速度，正好等于这条直线的方向向量，即 \mathbf{x}_1 - \mathbf{x}_0 。训练目标（损失函数） 被设计得非常简单和直接： \mathcal{L}(\theta) = \mathbb{E}_{\mathbf{x}_0, \mathbf{x}_1, t \sim U[0,1]} \left[ \| (\mathbf{x}_1 - \mathbf{x}_0) - v_\theta(z_t, t) \|^2 \right] 直观理解：我们要求神经网络 v_\theta 在路径上任意一点 z_t 预测的速度，都应该指向它对应的终点 \mathbf{x}_1 。如果模型能完美学会这一点，那么整个生成过程就变成了一条直线。
生成（采样）过程： 一旦模型训练完成，我们就可以进行生成。生成过程就是求解一个ODE，从噪声 \mathbf{x}_1 出发，“倒着走”回数据 \mathbf{x}_0 。 \frac{dz_t}{dt} = v_\theta(z_t, t) 初始条件： z_1 \sim \mathcal{N}(0, \mathbf{I}) （从高斯分布采样一个起点）。 目标：通过数值ODE求解器（如欧拉法）从 t=1 积分到 t=0 ，得到 z_0 ，即生成的图片。

同时，如果第一次训练学到的路径还不够直，论文中还提出了Reflow方法，用第一代模型生成的数据-噪声对来训练第二代模型。通过迭代，路径会被持续“拉直”：

用上述方法训练第一个模型 \pi_1 。
使用 \pi_1 从噪声 x_1 生成新的数据 \tilde{x}_0 。
用这些新生成的数据-噪声对 (\tilde{\mathbf{x}}_0, \mathbf{x}_1) 作为训练集，去训练第二个模型 \pi_2 。
重复此过程。

经过一轮或多轮整流后，路径会变得越来越直。一个极端的例子是，经过整流后的“Rectified Flow-2”模型，甚至可以用1步就生成出高质量的图片，这是经典扩散模型无法想象的。

可以说，Flow Matching为生成模型打开了新的大门，而Rectified Flow则第一个高效冲过这扇门，并发现了一片广阔的新天地。当前的主流扩散模型都建立在RF的思想之上，充分体现了其作为FM框架“杀手级应用”的价值。

6.3 Rectified Flow在Stable Diffusion/FLUX.1中的采样方法优化

在前几个章节中，Rocky已经详细讲解了Rectified Flow的核心思想与本质原理。在AIGC图像创作领域的核心大模型Stable Diffuison/FLUX中，还针对Rectified Flow的采样方法进行了定制化的优化，使其更加适合扩散模型的训练与推理。

接下来，Rocky将带着大家详细讲解整个优化逻辑。

首先我们需要知道的是，为什么要改进Rectified Flow的采样方法？原始Rectified Flow的采样方法有什么缺陷？

原始Rectified Flow采用的是均匀分布的时间步采样：时间步t \sim \mathcal{U}(0,1)，即采样密度 \pi(t)=1。在AIGC图像创作领域中，采样密度 \pi(t) 代表训练阶段连续时间步 t∈[0,1] 被随机选中的概率密度函数 。而概率密度函数的核心思想是描述连续随机变量（这里是时间步 t）在某一取值附近极小区间内被采样到的概率相对大小。

Rocky在这里举一个通俗易懂的例子，比如 p(0.5)=2.0 、 p(0.75)=1.5 ，仅表示“在 t=0.5 附近采样的概率密度比 t=0.75 附近更高，也就是说更容易被采样到”。简单来说，概率密度函数 p(t) 就是模型训练时的“注意力分配器”，告诉模型“该把更多精力放在哪个噪声阶段的学习上”。

当原始Rectified Flow均匀分布的时间步采样时， 最直白的物理意义就是0到1之间的任意等长区间，被选中的概率完全相等。这意味着所有时间步被同等对待，给所有时间步相同的训练机会，这看似是公平的，但实际上会导致中间复杂难例学习的不充分，同时浪费大量计算资源在简单的两端时间步上。而这里挖掘出的本质问题则是不同时间步表示了不同的图像状态，对应的训练任务难度存在差异：

t < 0.2 时：此时是低噪声的数据状态，接近完全清晰的纯图像数据，噪声很少，模型容易重建。主要学习像素级细节、边缘锐化、纹理修正等。
0.2 < t < 0.8 时：此时是复杂中噪声的数据状态，图像特征和噪声混合，模型需要精确平衡去噪和保真。主要学习物体结构、空间关系、语义理解、文字生成等。
t > 0.8 时：此时是高噪声的数据状态，接近纯噪声，几乎没有图像特征，模型主要预测噪声。主要学习整体构图、颜色分布、场景基调等。

其中，中噪声阶段是模型能力的瓶颈。如果这个阶段学不好，即使低噪声阶段再精细，生成的图像也会出现”结构崩坏、语义不符、文字乱码”等致命问题。

Rectified Flow框架下，不同时间步的图像状态

由于在Rectified Flow中 t 与信噪比\text{SNR} = \frac{a_t^2}{b_t^2} = \frac{(1-t)^2}{t^2} 正相关，我们还可以从图像信噪比的角度出发，也能理解采样 t 本质上就是采样不同的SNR水平：

t→0：SNR→∞ （高信噪比，纯图像特征）
t→0.5：SNR=1 （信噪比平衡，图像特征和噪声混合）
t→1：SNR→0 （低信噪比，纯高斯噪声）
时间步区间	预测难度	采样占比（π(t)=1）	资源匹配度
t∈[0,0.2]	极低（只需修正细节）	20%	严重浪费
t∈[0.2,0.8]	极高（学习语义、结构、布局）	60%	严重不足
t∈[0.8,1]	极低（只需预测整体基调）	20%	极度浪费

Rocky认为，上面的图表也揭示了采样密度 \pi(t)的本质就是基于Rectified Flow架构的扩散模型训练资源的分配器。

因此，后续Stable Diffusion/FLUX所有针对采样方法的改进，本质上都是重新分配训练资源，让更多样本被用来训练最难、最重要的中间时间步（即图像特征和噪声混合的状态）。解决原始均匀采样训练效率低、中间难例学习不充分的问题。

同时，所有优化采样方法的设计都基于一个核心等价关系：改变时间步t的采样分布\pi(t)，等价于对原始CFM损失施加权重\pi(t)，即：

w_t^\pi = \frac{t}{1-t} \cdot \pi(t)

接下来，Rocky带着大家进行本质的推导，让大家更新清楚的了解这个等价关系是如何得到的。

我们容易知道，对于任意损失函数项 f(t) （比如单个时间步的噪声预测MSE损失 \left\| \epsilon_\Theta - \epsilon \right\|^2 ），当我们用分布 \pi(t) 采样 t 时，总的期望损失为：

\mathcal{L}_{\text{new}} = \mathbb{E}_{t \sim \pi(t), \epsilon} \left[ \text{损失项}(t, \epsilon) \right] = \int_0^1 \pi(t) \cdot \mathbb{E}_{\epsilon|t} \left[ \text{损失项}(t, \epsilon) \right] dt

而使用均匀采样时的期望损失为：

\mathcal{L}_{\text{original}} = \mathbb{E}_{t \sim u(t), \epsilon} \left[ \text{损失项}(t, \epsilon) \right] = \int_0^1 1 \cdot \mathbb{E}_{\epsilon|t} \left[ \text{损失项}(t, \epsilon) \right] dt

同时由于均匀分布的概率密度 u(t)=1 ，我们可以将 \pi(t) 写成 u(t) \cdot \pi(t) ，代入 \mathcal{L}_{\text{new}} 的表达式：

\mathcal{L}_{\text{new}} = \int_0^1 u(t) \cdot \pi(t) \cdot \mathbb{E}_{\epsilon|t} \left[ \text{损失项}(t, \epsilon) \right] dt

我们根据期望的定义，上式就等价于：

\mathcal{L}_{\text{new}} = \mathbb{E}_{t \sim u(t), \epsilon} \left[ \pi(t) \cdot \text{损失项}(t, \epsilon) \right]

我们再将这个结论代入Rectified Flow的统一损失框架，原始均匀采样的损失为：

\mathcal{L}_{\text{RF}} = -\frac{1}{2} \mathbb{E}_{t \sim u(t), \epsilon} \left[ w_t^{\text{RF}} \lambda_t' \left\| \epsilon_\Theta - \epsilon \right\|^2 \right]

当采用自定义采样分布 \pi(t) 时，新的损失为：

\mathcal{L}_{\text{RF-}\pi} = -\frac{1}{2} \mathbb{E}_{t \sim u(t), \epsilon} \left[ \pi(t) \cdot w_t^{\text{RF}} \lambda_t' \left\| \epsilon_\Theta - \epsilon \right\|^2 \right]

因此，总的时间步权重为原始内置权重与采样分布密度的乘积：

\boldsymbol{w_t^\pi = w_t^{\text{RF}} \cdot \pi(t) = \frac{t}{1-t} \cdot \pi(t)}

总结来说，原始Rectified Flow框架的权重： w_t = \frac{t}{1-t} （本身的固有属性，与采样方式无关），当使用均匀分布 \pi(t) = 1 时： w_t^{\pi} = \frac{t}{1-t} ，当使用非均匀分布 \pi(t) 时，需要在原始权重基础上乘以 \pi(t) 即可。

Rocky认为这个等价关系是伟大的，这样我们就无需修改损失函数，仅通过调整t的采样方式 \pi(t) 即可有重点的关注训练过程中比较复杂难学的时间步，从而优化Rectified Flow架构中扩散模型的整体训练效果。

下面Rocky带着大家深入浅出详细讲解SD 3论文中给出了几种实验的Rectified Flow优化采样方法。

【第一种方法：Logit-Normal Sampling（对数正态采样）】

为什么要设计这个采样器呢？Rocky认为其底层思想是「用最成熟的正态分布，解决 (0,1) 区间的时间步采样的精准加权问题」。同时整个采样器公式推导的核心是概率论中连续随机变量的变换法则，

我们已经知道Rectified Flow的时间步 t \in (0,1) ，其中：

t \to 0 （接近纯数据）、 t \to 1 （接近纯噪声）：预测难度极低，模型几步就能学会，属于“简单样本”
t \approx 0.5 （数据与噪声混合最充分）：预测难度最大，是决定模型最终性能、尤其是少步采样质量的“难例样本”

原始Rectified Flow采用均匀采样（ t \sim \mathcal{U}(0,1) ），给所有时间步相同的训练机会，导致40%的算力浪费在简单样本上，难例学习不充分。我们急需要一个采样分布，满足3个核心要求：

取值严格限制在 (0,1) 区间，匹配时间步的物理意义
能把大部分采样概率集中在中间难例区域，两端简单区域采样概率极低
调参直观、工程实现简单、数值稳定，同时有完备的数学理论支撑

而我们接下来要讲解的Logit-Normal分布就是为了完美满足这3个需求而设计的。

我们先来看一下Logit-Normal分布的本质含义： Logit-Normal分布是指变量的logit服从正态分布。

这句话是整个采样器设计的逻辑起点，我们先拆解两个核心概念：

第一个概念是 logit 函数，它是搭建 (0,1) 与全实数轴的核心桥梁，定义如下：

\text{logit}(t) = \log \frac{t}{1-t}

它具备以下的核心特性：

定义域： t \in (0,1) ，完美匹配Rectified Flow时间步的取值范围
值域： (-\infty, +\infty) ，正好和正态分布的取值范围完全匹配
单调性：严格单调递增，是可逆双射（这是后续变量变换的前提）
物理意义：事件发生概率的「对数几率」。比如 t=0.5 时， \text{logit}(0.5)=\log1=0 ； t\to0 时 \text{logit}(t)\to-\infty ； t\to1 时 \text{logit}(t)\to+\infty ，完美的把 (0,1) 区间拉伸到整个实数轴。

第二个概念则是正态分布的引入，我们定义：

u = \text{logit}(t) = \log \frac{t}{1-t}, \quad u \sim \mathcal{N}(m, s^2)

位置参数 m ：控制分布的中心位置。 m=0 时，分布对称集中在 t=0.5 （SD 3最终选用的最优参数）； m>0 偏向噪声端， m<0 偏向数据端。
尺度参数 s ：控制分布的集中程度。 s 越小，分布越集中在中心； s 越大，分布越分散。

这里的核心设计思想是： 正态分布是统计学中性质最完备、调参最直观、数值最稳定的分布，但它的取值是全实数轴，无法直接用于 (0,1) 区间的时间步。我们通过 logit 变换，让时间步 t 的 logit 值服从正态分布，就可以把正态分布的所有优良特性，平移到了 (0,1) 区间的时间步上。

我们通过逆变换从正态变量 u 得到时间步 t 。我们从 u = \log \frac{t}{1-t} 解出 t 的表达式，这一步是工程实现的核心：

两边取指数： e^u = \frac{t}{1-t}
交叉相乘移项： e^u (1-t) = t \implies e^u = t(1+e^u)
最终解得：
t = \frac{e^u}{1+e^u} = \text{sigmoid}(u)

这一步的核心意义：

工程极简性：我们不需要直接计算复杂的Logit-Normal分布，只需要先从正态分布采样 u ，再通过sigmoid函数映射到 (0,1) 区间得到 t ，无复杂数值计算，无溢出风险，数值极其稳定。
可逆闭环：logit和sigmoid是一对互逆变换，完美实现了「全实数轴正态分布」和「 (0,1) 区间时间步」的双向映射，极具数学之美。

接着我们需要在上面的推导基础上，构建Logit-Normal Sampling采样器的核心数学工具：连续随机变量的概率密度变换公式。

我们先进行导数计算，来推导logit函数的微分。我们对 u = \text{logit}(t) = \log t - \log(1-t) 求导：

\frac{du}{dt} = \frac{d}{dt}\left( \log t - \log(1-t) \right) = \frac{1}{t} + \frac{1}{1-t} = \frac{1}{t(1-t)}

推导细节：

基础求导公式： \frac{d}{dx}\log x = \frac{1}{x}
复合函数求导： \frac{d}{dt}\log(1-t) = \frac{-1}{1-t} ，因此负负得正，得到 +\frac{1}{1-t}
通分化简： \frac{1}{t} + \frac{1}{1-t} = \frac{(1-t)+t}{t(1-t)} = \frac{1}{t(1-t)}

在这一步，我们可以得到一个关键洞察，即这个导数 \frac{1}{t(1-t)} 是实现「难例加权」的核心：

当 t \to 0 或 t \to 1 时，分母 t(1-t) \to 0 ，导数趋近于 +\infty
当 t=0.5 时，分母取最大值0.25，导数取最小值4

它会和后续的正态分布核配合，天然实现「两端概率密度趋近于0，中间概率密度最高」的效果，完美匹配难样本的重点采样需求。

接着我们再根据概率密度函数的变量变换公式 \pi(t) = p(u) \cdot \left| \frac{du}{dt} \right| 推导出最终的PDF：

\pi(t) = p(u) \cdot \left| \frac{du}{dt} \right|

这是整个PDF推导的数学基石，我们先讲清楚它的来源和物理意义。对于连续型随机变量，概率不会因为变量的变换而消失或新增：

随机变量u落在微小区间 du 内的概率是 p(u)du
这个区间对应到t的微小区间 dt ，概率是 \pi(t)dt
概率守恒要求： p(u)du = \pi(t)dt
变形后得到： \pi(t) = p(u) \cdot \left| \frac{du}{dt} \right|

这一步的核心作用是我们已经知道正态变量u的PDF p(u) ，现在只需要求出导数 \frac{du}{dt} ，就能代入公式得到t的PDF \pi(t) ，也就是Logit-Normal分布的完整概率密度。

到这里，我们就可以把两部分代入变量变换公式，得到最终的Logit-Normal分布PDF：

正态分布u的PDF： p(u) = \frac{1}{s\sqrt{2\pi}} \exp\left( -\frac{(u-m)^2}{2s^2} \right)
导数项： \left| \frac{du}{dt} \right| = \frac{1}{t(1-t)}
代入变量变换公式，同时把 u = \text{logit}(t) 代回，得到：
\pi_{\text{ln}}(t; m, s) = \frac{1}{s\sqrt{2\pi}} \frac{1}{t(1-t)} \exp\left( -\frac{(\text{logit}(t) - m)^2}{2s^2} \right)

最终的PDF由两部分相乘，形成了完美的“难例聚焦”效果：

正态分布核： \exp\left( -\frac{(\text{logit}(t) - m)^2}{2s^2} \right) ，当t偏离中心时，指数项快速衰减到0
导数项： \frac{1}{t(1-t)} ，在两端快速增长，但增长速度远慢于指数项的衰减速度

两者相乘的最终效果是：t在0和1两端的概率密度趋近于0，几乎不采样；90%以上的采样概率集中在中间难例区域。

更巧妙的是，它通过「改变采样频率」实现了损失加权，而非直接给损失函数乘权重：

直接加权会导致中间时间步梯度过大，出现梯度爆炸
而调整采样频率，只是让模型更多地看到难例样本，梯度幅值始终稳定，完全不改变损失函数的最优解，只会加速收敛，没有任何副作用。

也就是说，Logit-Normal采样没有修改模型的优化目标，只是调整了优化的优先级，让模型先学最难的、最影响生成质量的部分，在相同的训练算力下，实现了性能的最大化。

在实际中具体的采样实现则如下：

从正态分布采样： u \sim \mathcal{N}(m, s)
变换回 t： t = \frac{e^u}{1 + e^u}
Logit-Normal（对数正态）分布的概率密度函数（PDF）示意图

Logit-Normal（对数正态）分布的概率密度函数（PDF），它是 SD 3 论文中性能全局最优的时间步采样器，能够实现解决原始 Rectified Flow 均匀采样的算力浪费问题，把训练资源精准集中在难度最高的中间时间步这个核心目标。

【第二种方法：Mode Sampling with Heavy Tails】

为什么要设计Mode Sampling with Heavy Tails这个采样器呢？

Logit-Normal分布在两端（ t\approx0 和 t\approx1 ）概率密度极低，几乎采样不到（尾部缺失问题）。但在实际训练过程中两端的时间步仍然很重要，我们需要适当关注。因此，Mode Sampling with Heavy Tails采样器要解决的核心痛点如下：

Logit-Normal的固有缺陷：Logit-Normal分布在t\to0和t\to1时，概率密度会快速趋近于0，训练中几乎采样不到两端的时间步。
两端时间步的不可替代性：
t\approx1（接近纯噪声）：决定了生成图像的全局语义、整体布局，是生成的“起点”；
t\approx0（接近纯数据）：决定了生成图像的细节还原、纹理保真，是生成的“终点”。 完全不采样两端，会导致模型在这两个区间的拟合不足，推理时出现分布偏移：少步采样时全局语义崩坏，多步采样时细节出现伪影。

总的来说，Rocky认为Mode Sampling with Heavy Tails采样器核心目标就是在保留「中间难例优先采样」核心需求的同时，给两端（t\approx0和t\approx1）的时间步保留非零的采样概率（重尾特性），同时把原始均匀采样作为特例。

采样器的核心是定义了一个严格单调的映射函数：

f_{\text{mode}}(u; s) = 1 - u - s \cdot \left( \cos^2\left( \frac{\pi}{2}u \right) - 1 + u \right)

其中：

u \sim \mathcal{U}[0,1]：先从均匀分布采样基础随机数u；
t = f_{\text{mode}}(u; s)：通过这个函数，把均匀的u映射为我们需要的时间步t；
s \in \left[-1, \frac{2}{\pi-2}\right]：唯一的控制参数，范围由函数的严格单调性约束。

为什么参数s有严格范围呢？

因为映射函数必须是严格单调可导的，才能保证存在逆函数，才能用概率变量变换定理推导合法的概率分布。我们可以通过求导来推导参数s的范围。对f_{\text{mode}}(u;s)关于u求导：

逐项求导：
\frac{d}{du}f_{\text{mode}} = \frac{d}{du}(1-u) - s \cdot \frac{d}{du}\left( \cos^2\left( \frac{\pi}{2}u \right) -1 +u \right)
基础项求导：\frac{d}{du}(1-u) = -1
复合项求导（二倍角公式简化）：
\frac{d}{du}\cos^2\left( \frac{\pi}{2}u \right) = 2\cos\left( \frac{\pi}{2}u \right) \cdot \left(-\sin\left( \frac{\pi}{2}u \right)\right) \cdot \frac{\pi}{2} = -\frac{\pi}{2}\sin(\pi u)
剩余项求导：\frac{d}{du}(-1+u) = 1

最终得到导数：

f'_{\text{mode}}(u;s) = -1 - s \cdot \left( 1 - \frac{\pi}{2}\sin(\pi u) \right) = -(1+s) + s \cdot \frac{\pi}{2} \sin(\pi u)

我们想要保证函数严格单调递减（u从0到1时，t从1到0平滑过渡），必须满足所有u\in[0,1]下，导数恒小于0。结合\sin(\pi u)在u\in[0,1]的取值范围是[0,1]，我们可以得到导数的极值：

最大值（最接近0）：在u=0.5时，\sin(\pi u)=1，f'_{\text{max}} = -1 + s\left( \frac{\pi}{2}-1 \right)
最小值：在u=0/1时，\sin(\pi u)=0，f'_{\text{min}} = -(1+s)

要让所有导数小于0，需要同时满足两个条件：

f'_{\text{min}} < 0 \implies -(1+s) < 0 \implies s > -1（下界）
f'_{\text{max}} < 0 \implies -1 + s\left( \frac{\pi}{2}-1 \right) < 0 \implies s < \frac{1}{\frac{\pi}{2}-1} = \frac{2}{\pi-2} \approx 1.752（上界）

到此为止，我们就推导出参数范围s \in \left[-1, \frac{2}{\pi-2}\right]的严格数学来源，保证了采样器的合法性。

这个采样器的精髓，就是通过单参数s，实现了从「偏向两端」到「均匀采样」到「偏向中间」的全场景覆盖，我们逐一讲解 s=0、s>0、s<0 三种场景：

首先是基准场景：s=0，代入函数得：

f_{\text{mode}}(u;0) = 1-u

物理意义：u是均匀分布，t=1-u也服从\mathcal{U}[0,1]均匀分布，完全等价于原始Rectified Flow的均匀采样。
核心价值：把论文的基准方法作为自己的一个特例，对比实验时无需修改代码，仅调整参数即可实现公平对比，保证了实验的严谨性和可复现性。

接着是核心场景：s>0，函数在中间部分“凹陷”，采样更偏向中间时间步。这是SD 3论文中实际使用的场景，我们拆解它的工作原理：

函数形状：s>0时，函数在u<0.5时的取值小于1-u，在u>0.5时的取值大于1-u，整体向中间点(0.5,0.5)凹陷，形成S形曲线。
导数特性：中间u=0.5处导数绝对值更小（曲线更平缓），两端u=0/1处导数绝对值更大（曲线更陡峭）。
采样效果：根据逆变换采样原理，函数越平缓的区域，对应t的采样概率越高；函数越陡峭的区域，采样概率越低。因此中间时间步的采样概率大幅提升，两端概率降低但始终非零。
实验表现：SD 3论文中最优参数s=1.29，全局排名第4，5步采样排名3.25，性能远超原始均匀采样和大部分扩散基线。

最后是对照场景：s<0，函数在中间部分“凸起”，采样更偏向两端时间步。这个场景用于验证两端时间步的重要性，原理与s>0完全相反：

函数形状：中间凸起，两端更平缓；
采样效果：两端时间步的采样概率提升，中间概率降低；
核心价值：用于消融实验，验证“中间难例优先”的合理性——当s<0时，模型训练效果会显著下降，反向证明了中间时间步的核心地位。

最后我们在讲解一下Mode Sampling with Heavy Tails采样器基于变量变换定理的PDF公式：

\pi_{\text{mode}}(t; s) = \left| \frac{d}{dt} f_{\text{mode}}^{-1}(t) \right|

我们结合概率守恒原理，完整推导这个公式，和之前Logit-Normal的推导形成连贯的逻辑闭环。

对于连续型随机变量，概率不会因为变量变换而消失或新增：

基础变量u \sim \mathcal{U}[0,1]，它的概率密度p_u(u)=1，因此u落在微小区间du的概率为p_u(u)du = 1\cdot du；
映射关系t=f(u)，u=f^{-1}(t)（逆函数），u的区间du对应t的区间dt，两者的概率完全相等：
du = \pi_{\text{mode}}(t) dt

我们通过逆函数求导法则，对u=f^{-1}(t)两边关于t求导，得：

\frac{du}{dt} = \frac{d}{dt}f^{-1}(t) = \frac{1}{f'(u)}

其中f'(u)是原函数的导数，我们之前已经推导完成。我们再结合概率守恒公式，变形得：

\pi_{\text{mode}}(t) = \left| \frac{du}{dt} \right| = \left| \frac{d}{dt}f^{-1}(t) \right| = \frac{1}{\left| f'(u) \right|}

其中u=f^{-1}(t)。当s=0时，f'(u)=-1，代入得：

\pi_{\text{mode}}(t) = \frac{1}{|-1|} = 1

完美对应均匀分布的概率密度，验证了推导的正确性。

【第三种方法：CosMap采样器】

CosMap是SD 3论文中第三个核心采样器，它的设计本质是把扩散模型领域经过工业级验证的「余弦噪声调度（cosine schedule）」，通过 SNR（信噪比）对齐的方式，无缝迁移到 Rectified Flow（RF）框架中，既继承了余弦调度训练稳定、泛化性强的成熟优势，又保留了Rectified Flow直线路径的采样效率优势。

下面Rocky带着大家从这个采样器的设计动机出发，逐行拆解公式推导的每一步含义，再深入讲解背后的深层设计思想。

在正式推导前，我们先明确这个采样器要解决的核心问题：

余弦调度是扩散模型的工业级标准：由Nichol & Dhariwal 2021年提出的余弦调度（cosine schedule），是扩散模型领域最成熟、最稳定的噪声调度方案。它的 SNR 变化全程平滑，避免了线性调度在 t\to0 和 t\to1 时的 SNR 突变问题，训练稳定性、泛化性都远超其他调度，是 Stable Diffusion、FLUX、Midjourney 等工业级模型的默认选择。
Rectified Flow框架需要复用成熟经验：Rectified Flow作为新的扩散模型框架，需要和传统扩散模型做公平的性能对比，同时也需要经过验证的稳定调度方案来支撑大规模训练。直接从零设计新调度，不如把已经被无数实践验证的余弦调度，平移到Rectified Flow框架中。
核心设计目标：让Rectified Flow的 SNR 变化曲线和余弦调度完全一致，让Rectified Flow继承余弦调度的所有优良特性，同时不破坏Rectified Flow直线路径的核心优势。

我们已经知道，对于生成模型的前向加噪过程 z_t = a_t x_0 + b_t \epsilon （a_t 是信号项系数，b_t 是噪声项系数），信噪比 SNR 的定义是信号功率与噪声功率的比值：

\text{SNR} = \frac{a_t^2}{b_t^2}

取对数后得到对数信噪比（log-SNR），它是生成模型训练中最核心的指标，直接决定了不同时间步的训练难度和权重：

\log\text{SNR} = 2\log\frac{a_t}{b_t}

第一步：SNR 对齐，建立 RF 与余弦调度的映射关系

我们先分别写出 RF 和余弦调度的 SNR 表达式，再令两者相等，建立时间步的映射关系。

RF 的前向过程是直线插值：

z_t = (1-t)x_0 + t\epsilon

其中信号项系数 a_t=1-t，噪声项系数 b_t=t，代入 SNR 公式得：

\text{SNR}_\text{RF} = \frac{(1-t)^2}{t^2} \implies \log\text{SNR}_\text{RF} = 2\log\frac{1-t}{t}

传统扩散模型的余弦调度前向过程为：

z_u = \cos\left(\frac{\pi}{2}u\right) x_0 + \sin\left(\frac{\pi}{2}u\right) \epsilon, \quad u\sim\mathcal{U}[0,1]

其中信号项系数 a_u=\cos(\frac{\pi}{2}u)，噪声项系数 b_u=\sin(\frac{\pi}{2}u)，代入 SNR 公式得：

\text{SNR}_\text{cos} = \frac{\cos^2(\frac{\pi}{2}u)}{\sin^2(\frac{\pi}{2}u)} \implies \log\text{SNR}_\text{cos} = 2\log\frac{\cos(\frac{\pi}{2}u)}{\sin(\frac{\pi}{2}u)}

我们的核心目标是：让 RF 的 log-SNR 和余弦调度的 log-SNR 完全相等，这样 RF 就能继承余弦调度的 SNR 变化特性。因此令两者相等：

2\log\frac{1-t}{t} = 2\log\frac{\cos(\frac{\pi}{2}u)}{\sin(\frac{\pi}{2}u)}

推导步骤拆解：

两边同时除以2，去掉对数（对数函数单调递增，等式两边取指数后仍相等），得到：
\frac{1-t}{t} = \frac{\cos(\frac{\pi}{2}u)}{\sin(\frac{\pi}{2}u)}
利用三角恒等式 \cot(x)=\frac{\cos(x)}{\sin(x)}（余切函数），简化右边：
\frac{1-t}{t} = \cot\left(\frac{\pi}{2}u\right)
代数变形解出 t： 先把左边拆分为 \frac{1}{t} - 1 = \cot\left(\frac{\pi}{2}u\right)，移项得 \frac{1}{t} = 1 + \cot\left(\frac{\pi}{2}u\right)，最终得到：
t = \frac{1}{1+\cot\left(\frac{\pi}{2}u\right)}
把余切换回正弦/余弦，通分得到更简洁的形式：
t = \frac{1}{1+\frac{\cos(\frac{\pi}{2}u)}{\sin(\frac{\pi}{2}u)}} = \frac{\sin\left(\frac{\pi}{2}u\right)}{\sin\left(\frac{\pi}{2}u\right) + \cos\left(\frac{\pi}{2}u\right)}

到此为止，我们就得到了从均匀分布的 u 到 RF 时间步 t 的映射函数。当 u 从0到1均匀变化时，t 的变化会让 RF 的 SNR 完全复刻余弦调度的平滑特性。

我们想要计算 t 的概率密度函数（PDF），我们需要再次使用连续随机变量的变换定理，因此必须先得到 u 关于 t 的逆函数。

推导步骤拆解：

为了简化三角函数运算，做变量替换：令 \theta = \frac{\pi}{2}u，则 u = \frac{2}{\pi}\theta，代入 t 的表达式得：
t = \frac{\sin\theta}{\sin\theta + \cos\theta}
利用三角恒等式变形，解出 \tan\theta： 把右边的分子分母同时除以 \cos\theta（\cos\theta\neq0），得到：
t = \frac{\frac{\sin\theta}{\cos\theta}}{\frac{\sin\theta}{\cos\theta} + 1} = \frac{\tan\theta}{\tan\theta + 1}
代数变形解出 \tan\theta： 交叉相乘得 t(\tan\theta + 1) = \tan\theta，展开移项：
t\tan\theta + t = \tan\theta \implies t = \tan\theta(1-t) \implies \tan\theta = \frac{t}{1-t}
反解出 \theta 和 u： 对两边取反正切，得 \theta = \arctan\left(\frac{t}{1-t}\right)，代回 \theta = \frac{\pi}{2}u，最终得到逆函数：
u = \frac{2}{\pi}\arctan\left(\frac{t}{1-t}\right)

到这一步，我们就得到了 u 关于 t 的显式表达式，为后续求导计算 PDF 奠定了基础。

最后，我们开始求导计算最终的概率密度函数。根据我们之前反复用到的连续随机变量变换定理：

基础变量 u \sim \mathcal{U}[0,1]，它的概率密度 p_u(u)=1；
映射关系 t=f(u)，逆函数 u=f^{-1}(t)；
则 t 的概率密度为：\pi(t) = \left| \frac{du}{dt} \right|（绝对值保证密度非负）。

我们对逆函数 u = \frac{2}{\pi}\arctan\left(\frac{t}{1-t}\right) 求导，用复合函数求导法则分步计算：

推导步骤拆解：

基础求导公式：\frac{d}{dx}\arctan(x) = \frac{1}{1+x^2}
令中间变量 x = \frac{t}{1-t}，则复合函数求导展开为：
\frac{du}{dt} = \frac{2}{\pi} \cdot \frac{1}{1+x^2} \cdot \frac{dx}{dt}
计算 \frac{dx}{dt}（商的求导法则）： 商的求导公式：\frac{d}{dt}\left(\frac{分子}{分母}\right) = \frac{分子导 \cdot 分母 - 分子 \cdot 分母导}{分母^2} 这里分子是 t（导数为1），分母是 1-t（导数为-1），代入得：
\frac{dx}{dt} = \frac{1\cdot(1-t) - t\cdot(-1)}{(1-t)^2} = \frac{1-t+t}{(1-t)^2} = \frac{1}{(1-t)^2}
计算 \frac{1}{1+x^2}： 代入 x=\frac{t}{1-t}，得 x^2=\frac{t^2}{(1-t)^2}，因此：
1+x^2 = 1 + \frac{t^2}{(1-t)^2} = \frac{(1-t)^2 + t^2}{(1-t)^2} \implies \frac{1}{1+x^2} = \frac{(1-t)^2}{(1-t)^2 + t^2}
代入化简，得到最终导数： 把两部分结果代入求导公式，(1-t)^2 直接约掉：
\frac{du}{dt} = \frac{2}{\pi} \cdot \frac{(1-t)^2}{(1-t)^2 + t^2} \cdot \frac{1}{(1-t)^2} = \frac{2}{\pi} \cdot \frac{1}{t^2 + (1-t)^2}
最终概率密度函数： 导数恒为正，绝对值可以直接去掉，得到 CosMap 的 PDF：
\pi_{\text{CosMap}}(t) = \frac{2}{\pi} \cdot \frac{1}{t^2 + (1-t)^2}

我们可以通过代入端点和中点，直观看到它的分布特性：

中点（t=0.5）：分母为 0.25+0.25=0.5，\pi(0.5)=\frac{2}{\pi \cdot 0.5} = \frac{4}{\pi} \approx 1.27，是分布的最大值，采样概率最高，符合「中间难例优先」的核心需求。
两端（t=0 和 t=1）：分母为 0+1=1，\pi(0)=\pi(1)=\frac{2}{\pi} \approx 0.636，是正的常数，两端有非零的采样概率（重尾特性），不会像 Logit-Normal 一样趋近于0，保证了训练的鲁棒性。
全程平滑：PDF 是一个对称、光滑的钟形曲线，没有突变、没有拐点，保证了训练时梯度的稳定性。
CosMap采样器的示意图

Rocky在这里再总结上述三种采样方法的特点：

Logit-Normal：理论优雅，数学性质好；但两端采样不足，可能丢失端点信息。
Mode Sampling：通过重尾设计，保证两端也能被采样；参数调节直观。
CosMap：直接模拟余弦调度的SNR特性；有明确的物理意义。

使用不同采样方法进行对比实验揭示了扩散模型训练中的一个深刻洞见：不仅模型架构和损失函数重要，训练数据的”呈现方式”（采样策略）同样关键。通过精心设计的时间步采样策略，我们可以在不增加模型复杂度的情况下，显著提升生成质量，这为后续扩散模型的研究提供了新的ideas灵感。

6.4 Rectified Flow的通用结构与模块代码原理

未完待续，大家敬请期待！！！

码字确实不易，希望大家能多多点赞！！！

7. 扩散模型的未来发展趋势分析（持续更新！）

未完待续，大家敬请期待！！！

码字确实不易，希望大家能多多点赞！！！

8. 推荐阅读

Rocky会持续分享AIGC的干货文章、实用教程、商业应用/变现案例以及对AIGC行业的深度思考与分析，欢迎大家多多点赞、喜欢、收藏和转发，给Rocky的义务劳动多一些动力吧，谢谢各位！

Rocky一直在运营技术交流群（WeThinkIn-技术交流群），这个群的初心主要聚焦于AI行业话题的讨论与研究，包括但不限于算法、开发、竞赛、科研以及工作求职等。群里有很多AI行业的大牛，欢迎大家入群一起交流探讨～（请备注来意，添加小助手微信Jarvis8866，邀请大家进群～）

8.1 深入浅出完整解析AI Agent（AI智能体）的核心基础知识

2025年可以说是AI Agent全面落地应用的元年，因此Rocky在持续撰写对AI Agent的全维度解析文章：

8.2 深入浅出完整解析FLUX.1 Kontext和FLUX.1 Krea核心基础知识

Rocky也对FLUX.1 Kontext和FLUX.1 Krea的核心基础知识作了全面系统的梳理与解析：

8.3 深入浅出完整解析DeepSeek系列核心基础知识

Rocky也对DeepSeek系列模型的核心基础知识作了全面系统的梳理与解析：

8.4 深入浅出完整解析Stable Diffusion 3（SD 3）和FLUX.1系列核心基础知识

Rocky也对Stable Diffusion 3和FLUX.1的核心基础知识作了全面系统的梳理与解析：

8.5 深入浅出完整解析Stable Diffusion XL（SDXL）核心基础知识

Rocky也对Stable Diffusion XL的核心基础知识作了全面系统的梳理与解析：

8.6 深入浅出完整解析Stable Diffusion（SD）核心基础知识

Rocky也对Stable Diffusion 1.x-2.x系列模型的核心基础知识做了全面系统的梳理与解析：

8.7 深入浅出完整解析Stable Diffusion中U-Net的前世今生与核心知识

Rocky对Stable Diffusion中最为关键的U-Net结构进行了深入浅出的全面解析，包括其在传统深度学习中的价值和在AIGC中的价值：

8.8 深入浅出完整解析LoRA（Low-Rank Adaptation）模型核心基础知识

对于AIGC时代中的“ResNet”——LoRA模型，Rocky也进行了深入浅出的全面讲解：

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

8.15 算法工程师的《三年面试五年模拟》求职秘籍

为了方便大家实习、校招以及社招的面试准备，同时帮助大家提升扩展技术基本面，Rocky将符合大厂和AI独角兽价值的算法高频面试知识点撰写总结成《三年面试五年模拟之独孤九剑秘籍》，并制作成pdf版本，大家可在公众号WeThinkIn后台【精华干货】菜单或者回复关键词“三年面试五年模拟”进行取用：

8.16 深入浅出完整解析AIGC时代中GAN系列模型的前世今生与核心知识

GAN网络作为传统深度学习时代的最热门生成式Al模型，在AIGC时代继续繁荣，作为Stable Diffusion系列模型的“得力助手”，广泛活跃于Al绘画的产品与工作流中：
