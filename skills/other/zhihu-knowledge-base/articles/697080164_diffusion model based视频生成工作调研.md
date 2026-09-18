# diffusion model based视频生成工作调研

**作者**: meton​北京大学 计算机硕士

**原文链接**: https://zhuanlan.zhihu.com/p/697080164

---

​
目录
收起
背景介绍
数据集与数据生成
文本视频数据集
数据清洗
视频数据构造
评估
图像级别指标
视频级别指标
评估维度
评估方法
自动化评测
前置知识：图片生成的Diffusion理论
扩散理论推导
设定：图片加噪
正向扩散过程推导
逆向去噪过程推导
总结
DDPM算法 （论文）
DDPM训练
DDPM预测
Diffusion基础模型架构
视频生成论文工作
文生图：Stable Diffusion
LDM 框架
stable diffusion
stable diffusion XL
stable diffusion3
带图文条件的文生图：SD+ControlNet
视频生成总体工作概述
首次尝试: Video Diffusion Model
多阶段视频生成：Make-a-Video
多阶段视频生成2：Imagen
隐空间插帧超分：show-1
英伟达：Align your Latents
复用文生图模型，可插拔的视频生成模块：AnimateDiff
带图文条件的视频生成：MicroCinema
局部指令控制的视频生成：Follow your click
一致性保持的视频生成：StoryDiffusion
Stbale Video Diffusion
OpenAI Sora
总体技术原理
视频编解码器的细节（重要）
DiT：diffusion transformer
视频生成的scaling law
训练数据
Sora作为世界模拟器的局限性
参考文献
背景介绍

最新一波的人工智能生成内容（AIGC）在计算机视觉领域取得了显著的成功，扩散模型在这一成就中扮演了关键角色。由于其印象深刻的生成能力，扩散模型逐渐取代了基于GAN和自回归变换器的方法，不仅在图像生成和编辑方面表现出色，还在与视频相关的研究领域表现出卓越性能。自2022年以来，基于扩散模型的视频研究论文数量显著增加，可以分为三个主要类别：视频生成、视频编辑和视频理解。

视频生成主要分为以下：

基于文本的视频生成（Text-to-Video Generation, T2V）：

- 目标：根据文本描述自动生成相应的视频内容。

- 方法：早期方法主要基于生成对抗网络（GANs）和自回归Transformer框架。随着扩散模型的兴起，T2V方法开始利用这些模型的强大生成能力和对文本条件的敏感性。

- 应用：T2V在自动生成电影、动画、虚拟现实内容、教育演示视频等领域有广泛应用。

- 挑战：需要理解文本描述中的场景、对象和动作，并将其转化为连贯的视觉帧序列，同时保持逻辑和视觉一致性。

无条件视频生成（Unconditional Video Generation）：

- 目标：从随机噪声或固定初始状态开始，生成连续且视觉上一致的视频序列，不依赖特定输入条件。

- 方法：生成模型需要自主学习如何捕捉时间动态、动作和视觉一致性，以产生真实且多样化的视频内容。

- 应用：探索生成模型从无监督数据中学习视频内容的能力，展示多样性。

- 挑战：生成模型需要在没有明确输入的情况下，自主学习视频内容的复杂动态。

视频补全（Video Completion）：

- 目标：在视频序列中填补缺失的部分，或者增强视频的质量和细节。

- 方法：视频完成通常涉及到视频帧的插值、超分辨率、去噪等技术，以提高视频的整体质量和连贯性。

- 应用：视频增强和恢复，如提高视频分辨率、修复损坏的视频片段、预测视频的未来帧等。

- 挑战：需要在保持原始视频内容和风格的同时，生成高质量的新视频帧，同时确保时间上的连贯性和视觉上的自然性。

文生视频工作发展历史轴
数据集与数据生成
文本视频数据集

文本视频对数据集，一般可以用于视频检索任务、也被用来进行视频生成训练。

数据清洗

使用数据处理数据，从过去开源的视频数据集进行重新整理，切分和caption，清洗出更高质量的视频数据集。

可以看到这个数据集从分类角度也只覆盖比较常见的一些大的类目，视频时长都是比较短的。

视频数据构造

从多个来源构造视频数据：

图片：通过cut and crop制造出图片zoom in 和 zoom out 的视频效果。
短视频：利用BLIP2将文本描述扩写
长视频：按照内容实体进行视频切分，然后使用多模态模型进行文本描述生成。
评估
图像级别指标
FID 通过比较合成的视频帧与真实视频帧来评估生成视频的质量。它涉及将图像进行归一化以获得一致的尺度，利用 InceptionV3 从真实和合成视频中提取特征，并计算均值和协方差矩阵。然后将这些统计数据结合起来计算 FID 分数。
SSIM 评估原始和生成图像的亮度、对比度和结构特征。
PSNR 是代表峰值信号与均方误差（MSE）之比的系数。
CLIPSIM 是一种用于测量图像文本相关性的方法。基于 CLIP 模型，它提取图像和文本特征，然后计算它们之间的相似性。这个指标通常用于文本条件的视频生成或编辑任务。
视频级别指标

虽然图像级别的评估指标代表了生成视频帧的质量，但它们主要关注单个帧，忽略了视频的时间一致性。另一方面，视频级别的指标会提供更全面的视频生成评估。

Fréchet Video Distance (FVD) 是一种基于 FID 的视频质量评估指标。FVD 利用在 Kinetics 上预训练的 Inflated-3D Convnets (I3D) 从视频片段中提取特征。随后，通过均值和协方差矩阵的组合来计算 FVD 分数。
Kernel Video Distance (KVD) 也基于 I3D 特征，但它通过利用基于核的方法最大均值差异 (MMD) 来评估生成视频的质量。
Video IS (Inception Score) 使用由 3D-Convnets (C3D) 提取的特征来计算生成视频的 Inception 分数，通常用于 UCF-101 上的评估。
Frame Consistency CLIP Score 通常用于视频编辑任务，用于测量编辑后视频的一致性。其计算包括为所有编辑后视频的帧计算 CLIP 图像嵌入，并报告所有视频帧对之间的平均余弦相似度。
评估维度

视频质量维度 、视频-条件对齐

评估方法
自动化评测
前置知识：图片生成的Diffusion理论

当前主流的视频生成模型主要是在图片生成领域取得落地的difusion model的基础上进行拓展适配而来，视频生成模型也使用图片生成模型的权重进行预训练和初始化。而图片生成模型的目前主流的理论基础是diffusion理论。

扩散理论推导
设定：图片加噪

扩散过程逐步加快，β越来越接近1，如果扩散步数足够大，那么最终得到的图片就完全丢失了原始数据而变成了一个随机噪音图片。

在DDPM中，会将原始图像的像素值从[0, 255]范围归一化到[-1, 1]，像素值属于离散化值，这样不同的像素值之间的间隔其实就是2/255。

正向扩散过程推导
从递推等式 来看 如何从 x(t-2) 直接到x(t)
如下图推导：等价于从新的高斯分布N(0，1-αtαt-1）采样噪声，然后和x(t-2)叠加，就可以获得x(t)
使用重参数技巧，继续化简公式

只需要仍从一个标准高斯分布采样，并且乘上系数。这等价于上述的高斯分布N(0，1-αtαt-1）。

使用数学归纳法，得到递推等式

可以得知，从t-k步到t步的扩散过程，可以从一个标准高斯分布给图片x(t-k)直接获得。

得到最终前向扩散过程的化简式子：

逆向去噪过程推导

由贝叶斯公式推导可以得到逆向过程（省略推导过程）：

可以看出，要想从x(t)得到x(t-1), 需要先得到 x0到x(t)的加噪噪声（可以通过模型预测），然后再从上面的正态分布N中采样出一个图片，也就是x(t-1)。如下图所示：

总结
利用扩散过程的正向过程，可以训练一个预测t时刻图片相对于0时刻的加噪噪声的模型。
有了这个模型，根据上文中的逆向过程公式。可以逆向从t时刻的噪声，逐步恢复出0时刻的原图。
DDPM算法 （论文）
DDPM训练

DDPM的训练过程也非常简单，如下所示：

随机选择一个训练样本。
从1~T随机抽样一个t
随机产生 0-1正态分布噪音
计算t时刻所产生的带噪音图像数据（红色框所示）
输入网络预测t0到达t时刻的噪音
计算产生的噪音和预测的噪音的L2损失（或者用更复杂的loss函数）
计算梯度并更新网络
DDPM预测

一旦训练完成，其采样过程也非常简单，如上算法图所示：

从一个随机噪音（可以看作是t时刻噪音图片）开始
用训练好的网络预测噪音（代表从t0到t时刻的加噪声）
然后计算条件分布的均值（红色框部分），然后用均值加标准差乘以一个随机噪音，最终得到x（t-1）的去噪图片。
重复上述过程，直至t=0完成新样本x0的生成（最后一步不加噪音）。
Diffusion基础模型架构

unet中嵌入cross-attention模块，用于图片和其他condition的信息融合。U-Net属于encoder-decoder架构，其中encoder分成不同的stages，每个stage都包含下采样模块来降低特征的空间大小（H和W），然后decoder和encoder相反，是将encoder压缩的特征逐渐恢复。U-Net在decoder模块中还引入了skip connection，即concat了encoder中间得到的同维度特征，这有利于网络优化。

DDPM所采用的U-Net每个stage包含2个residual block，而且部分stage还加入了self-attention模块增加网络的全局建模能力。 另外，扩散模型其实需要的是TT个噪音预测模型，实际处理时，我们可以增加一个time embedding（类似transformer中的position embedding）来将timestep编码到网络中，从而只需要训练一个共享的U-Net模型。具体地，DDPM在各个residual block都引入了time embedding。

视频生成论文工作
文生图：Stable Diffusion
LDM 框架

原本的difussion论文中difussion的训练是直接在原图上进行加噪声和去噪声。而LDM是先通过VAE编码器将原图编码到隐空间的一个更小的固定尺寸的小图（latent space feature），然后在上面进行diffusion训练，这样可以减少计算代价。

两阶段训练
先训练一个图像的编解码模型（VAE， VQ-VAE）
再进行latent空间的difusion model 训练




图片编解码器：
VAE
编码器输出一个向量，向量每个位置的值是从正态分布中采样的，也就是说，编码器实际上预测的是分布。

VQ-VAE




编码器的输出是从学习的emb space中挑取距离相近的emb来替代，由于emb space的数目有限的，所以是离散的。

stable diffusion
stable diffusion XL
总体

SDXL使用了扩大模型参数结构，使用更强的图文特征提取器，使用更多训练数据，支持更多的condition调制信号（crop参数等。）

在sd的pipeline中插入一个refine阶段（也是sd的结构）来提高图片细节。

模型架构：
增加refiner
CLIP编码器增加
模型宽度深度增加




Stable Diffusion XL是一个二阶段的级联扩散模型，包括Base模型和Refiner模型。其中Base模型的主要工作和Stable Diffusion一致，具备文生图，图生图，图像inpainting等能力。在Base模型之后，级联了Refiner模型，对Base模型生成的图像Latent特征进行精细化，其本质上是在做图生图的工作。

Base模型由U-Net，VAE，CLIP Text Encoder（两个）三个模块组成，在FP16精度下Base模型大小6.94G（FP32：13.88G），其中U-Net大小5.14G，VAE模型大小167M以及两个CLIP Text Encoder一大一小分别是1.39G和246M。

Refiner模型同样由U-Net，VAE，Text Encoder（一个）三个模块组成，在FP16精度下Refiner模型大小6.08G，其中U-Net大小4.52G，VAE模型大小167M（与Base模型共用）以及CLIP Text Encoder模型大小1.39G（与Base模型共用）。

Refiner模型和Base模型一样是基于Latent的扩散模型，也采用了Encoder-Decoder结构，和U-Net兼容同一个VAE模型，不过Refiner模型的Text Encoder只使用了OpenCLIP ViT-bigG。

可以看到，Stable Diffusion XL无论是对整体工作流还是对不同模块（U-Net，VAE，CLIP Text Encoder）都做了大幅的改进，能够在1024x1024分辨率上从容生成图片。同时这些改进无论是对生成式模型还是判别式模型，都有非常大的迁移应用价值。

更多的condition调制

除了文本和timestep作为条件输入，还增加 分辨率参数、crop参数的condition输入。

stable diffusion3
文本编码：
除了拼接CLIP-G和CLIP-L，还拼接了T5 XXL的embedding，对于处理复杂长prompt的图片生成。
训练时候使用dropout的方式，让模型学会可以不依赖依赖全部的text-embeding，例如可以不使用t5 emb。




图片编码:
使用了更多通道的vae，提高模型容量，需要更多的训练计算成本。




MM-DiT：
图文特征的融合使用双通道，图和文有各自的权重去学习。
timestep和类信息使用linear层进行映射然后去调制layer-norm的输出（类似DiT做法，而不是使用corss-att将标量信息混入unet）




训练：
使用rectified flow理论去训练，而不是基于diffusion model理论，前者在数学理解和推导上更加简洁。




数据：
构造图片的文本描述使用使用了多模态大语言模型来recaptioning，产生更加丰富的文本描述。
带图文条件的文生图：SD+ControlNet

ControlNet的模型结构如下所示，这里是直接复制一份SD的上半部分：Encoder和中间的Middle Block。

ControlNet复制UNet结构的同时继承权重来初始化，此外ControlNet还采用了zero初始化，这里在condition的特征输出后加了一个zero conv，同时13个skip connection特征输出上分别也加上了一个zero conv。zero初始化使得整个网络在训练开始时的输出和原始UNet是一样的，这样可以尽量避免初始训练的噪音对ControlNet复制的结构和权重的破坏。

ControlNet的训练是将SD原始UNet和ControlNet一起训练，但SD的UNet是冻结的，只训练ControlNet部分的权重。训练的损失函数还是采用原始SD所用的拟合噪音的Lsimple

视频生成总体工作概述
首次尝试: Video Diffusion Model

扩展图像SD框架

conv2D -> 3D （1x3x3）
空间注意力 --> 分离的 时空注意力：
先对视频每一张latent图片做空间attention
再将做完空间attention的所有输出concat，并在通道维度（视角维度）进行attention操作。
联合的图片和视频训练
图片就是只有1帧的视频
其他训练原理类似前面讲的sd训练




缺点是只能生成很短而且分辨率很低的视频。

多阶段视频生成：Make-a-Video
多阶段pipeline来生成更多视频帧
伪3D卷积 + 分离的时空注意力

空间注意力权重从文生图模型初始化而来，时序注意力结构的参数需要从头训。

插帧模型：

输入视频的前后两帧，模型补出两帧之间的帧。

多阶段视频生成2：Imagen

插帧模型 + 超分模型 pipeline：

先获得一个base的小分辨率视频，然后插帧，然后扩大分辨率，然后再扩大分辨率，接着继续插帧... 以此类推。
隐空间插帧超分：show-1

直接在latent 空间进行插帧 和 超分，以减少计算代价。

英伟达：Align your Latents

改造卷积 和 时空注意力，张量计算的维度变化：

大部分时候，帧维度被放到了batch维度，所以张量形状是[B∗T, C,H,W]，跟2D图片时的4维tensor一样，这时候可以用2D卷积等常见算子做2D操作。
当要做时序上的注意力时，张量形状变成3维张量[B∗H∗W, T ,C]，这时候注意力机制关注的是帧与帧之间的关系。
而要应用3D卷积时，形状就得变成5维张量[B,C,T,H,W]。

多stage pipeline：

英伟达特色：会详细介绍训练时候的计算代价。

复用文生图模型，可插拔的视频生成模块：AnimateDiff

有一类工作研究如何复用现有的大量的文生图模型权重资源，插入一个训练好的运动模块，就能让图片动起来成为视频！

训练阶段

基于文生图模型（参数冻结），插入运动建模模块进行参数微调以获得视频运动先验。

所有帧的latent tensor是一起初始化、一起去噪的，不是一帧接着一帧生成的，运动模块就是在计算这些帧与帧之间的注意力：

因为视频比图片多了时间这一个维度，所以原始输入是5维的，分别是[batch,channels,frames,height,width]而为了与生成2D图像的T2I模型兼容，作者将形状变成[batch×frames,channels,height,width]，这时候张量就是4D的了。

而张量来到motion module后，形状又会变成[batch×height×width, frames, channels]，这时候张量就是3D的，这样是为了便于运动模块对每个批次中的各帧做注意力，以实现视频的运动平滑性和内容一致性。这个操作其实和之前讲的英伟达的aligin your latent的tensor变换操作类似。

推理阶段

选择一个风格的开源文生图模型，将训练好的运动模型插入模型中。

输入是形状为 [batch, frame, height, width, channel]的噪音tensor， 和文本prompt， 输入就是一段运动的视频。

带图文条件的视频生成：MicroCinema

其中的AppearNet的思想类似前面提到的controlnet，可以理解为是controlnet在视频领域的扩展。

局部指令控制的视频生成：Follow your click

图文生视频中，用户往往希望输入一张参考图片，prompt文本指示如何运动（指令控制），最后希望生成的视频是用户希望的运动效果。

但是目前的现实： pika、runway等生成的视频很难遵循用户指令控制，这和训练预料中，prompt大部分是背景和整体描述，缺少运动控制的描述有关。

在latent输入中增加了 主体区域mask，输入prompt是对这个区域的控制要求，这两个condition可以让生成的视频对局部运动控制要求有更好的遵循。

在latent输入中还对加入的首帧图片在latent空间进行随机masking， 作者发现这样的masking最后学出来的视频质量更优，一致性更好。

一致性保持的视频生成：StoryDiffusion

无论是图片生成还是视频生成，内容一致性保持都是一个巨大的挑战。例如要根据文本描述生成漫画，那么我们总是希望漫画分镜图之间的主体特征、背景、主题属性可以保持一致。从漫画生成进一步延伸，如果希望将生成的漫画进一步插帧生成关于某段故事的视频，我们也希望视频帧之间有很高的一致性。

论文中作者提出一种基于diffusion model的两阶段视频生成方法，可以生成高度一致性保持的视频结果。

第一阶段 是提出一种免训练的热拔插attention模块CAB，可以直接将开源文生图基座模型（例如stable diffusion xl）的attention模块直接替换，然后来生成一批一致性保持度高的图片。以漫画生成场景为例：

选择一个开源文生图基座模型，例如SDXL，将网络结构中的attention替换成CAB模块，这个过程不需要重新训练微调。
对于一段较长的漫画故事文本描述，分拆成一批（几句）用于不同漫画分镜生成的prompt描述，对于这批prompt描述我们希望SDXL可以生成对应的一批主体一致性高度保持的漫画图。
SDXL的模型输入是一个batch的prompt描述，对应的batch内的噪声图输入通过编码器变成batch的image latent（维度NxC）后，在这个batch中，某个image_latent作为Q输入attention模块时，这个image_latent会和batch内随机采样的其他image_latent进行concat后转化为attention操作的K和V（维度（S+N)xC ），然后再进行attention操作。这个过程的意义在于对于batch内某一个图片的生成中，也会考虑当前batch内其他图片的信息，这样可以更好的保持主体和属性的一致性，这个过程attention的权重没有改变，也没有重新训练。
通过CAB模块，可以生成一批主体特性/属性高度一致的漫画分镜图

第二阶段 对于前面生成的一批漫画分镜图，想进一步生成一致性高度保持的视频，作者在video diffusion model的基础上去集成一个semantic space motion predictor模块来实现，具体的操作是：

对于漫画的某两帧，利用image encoder转换到latent空间后（维度2xNxC），在latent空间上进行插帧变成FxNxC的latent特征，之后通过一个参数可训的semantic space motion 模块（其实就是一堆tansformer bock），之后后变成一批具有一致性语义的latent 特征（维度FxNxC）。
之后这些latent特征可以作为一种condition信息 拼接 prompt文本嵌入信息然后通过cross attention注入到常规的video diffusion训练中。

训练之后的视频生成模型，可以将前一阶段的每两个漫画帧作为输入，生成一段短视频。之后将多段短视频拼接起来后可以得到更长的视频，作者展示的是10s左右的漫画视频。

Stbale Video Diffusion

可以称作是视频生成的resnet时刻？

论文强调了数据治理的重要性

训练：
分三步走： 文生图预训 --> 低分辨率视频预训练 --> 下游高质量高分辨率视频微调




数据：
一个关键：数据规模+数据治理
cut detection pipeline : 视频分段, 提高视频内容连贯性、一致性，这样训练出来的模型生成视频的跳变情况减缓。
多种打标签方法综合：
利用各种caption的方法： CoCA软件、V-BLIP、LLM多帧caption总结
质量筛选指标：CLIP 分数 & aesthtic 分数 & OCR detection & optic flow score。




质量高的小量视频数据集： （LVD）～577M clips （LVD- F）～2.3M clips




OpenAI Sora

SORA可以称作是视频生成的gpt-2,-3时刻？

总体技术原理

在足量的数据，优质的标注（recaption），灵活的编码（时空token）下，scaling law 在 diffusion model 的架构上继续成立。

SORA使用 difussion transformer model来进行视频生成。

视频编解码器的细节（重要）

可以将很长的视频在时序和空间上进行大量的压缩，但是还能保持这么好的质量，可能不是常见的VAE结构，而是openai自己的架构。（未知）

计算代价估计很大？ 可能是在大量小视频预训练，然后少量高质量长视频再训练？（未知）

DiT：diffusion transformer

Patch化：DiT的输入是通过VAE后的一个稀疏的表示z（256×256×3的图片，z为32×32×4），类似其他ViTs的方式，首先要将输入转成patch，文章采用超参p=2，4，8进行对比实验。

DiT模块设计：

In-context条件：in-context条件是将t和c作为额外的token拼接到DiT的token输入中；
Cross-attention模块：DiT结构与Condition交互的方式，与原来U-Net结构类似；
Adaptive layer norm（adaLN）模块：使用adaLN替换原生LayerNorm（NeurIPS2019的文章，LN 模块中的某些参数不起作用，甚至会增加过拟合的风险。所以提出一种没有可学习参数的归一化技术）；
adaLN-zero模块：之前的工作发现ResNets中每一个残差模块使用相同的初始化函数是有益的。文章提出对DiT中的残差模块的参数γ、β、α进行衰减，以达到类似的目的。

模型大小：与ViT大小相似，分别使用DiT-S、DiT-B、DiT-L和DiT-XL，Gflops从0.3dao118.6。

Transformer Decoder：在Transformer最上层需要预测噪音，因为Transformer可以保证大小与输入一致，所以在最上层使用一层线性进行decoder。

视频生成的scaling law

transformer架构有很好的scaling law特性。

在Sora的技术报告中可以看出，OpenAI实现scaling law的想法其实很大程度上沿袭了大语言模型的经验。

随着网络的FLOPs增加（采用网络变大，或者patch变小序列变长两种方式实现），FID会越来越小，也就是Scaling Law。

训练数据
Sora作为世界模拟器的局限性

1.概率无法模拟物理规律

soar的token预测本质上还是基于概率，概率不是物理，现在物理规律和表现是通过偏微分方程去表征和捕捉的。用概率去模拟物理，不够真实。

2.局部合理，整体荒谬

Transformer学会了Token间局部的连接概率，但是缺乏时空上下文的大范围整体观念。e.g.老奶奶吹蜡烛

3. 物理临界状态缺失

临界点数据缺少

自然界的绝大多数物理过程都是稳恒态与临界态的交替变化。在稳恒态中，系统参数缓慢变化，容易获取观察数据；在临界态中（灾变态），系统骤然突变，令人猝不及防，很难抓拍到观察数据。因此，临界态的数据样本非常稀少。e.g. 杯子落下，杯子没有碎。

分布边界的平滑模糊

sora采用的目前最为热门的扩散模型，在计算传输映射的时候，倾向于光滑化数据流形的边界，从而混淆不同的模式，直接跳过临界态图像的生成。因此视频看上去从一个状态突然跳跃到另外一个状态，中间最为关键的倾倒过程缺少，导致物理上的荒谬。 e.g. 三小狗嘻戏变四只。

参考文献

DDPM https://arxiv.org/pdf/2010.02502.pdf

DDPM解读怎么理解今年 CV 比较火的扩散模型（DDPM）？ - 知乎

sora技术报告 Video generation models as world simulators

surveys：GitHub - ChenHsing/Awesome-Video-Diffusion-Models: [Arxiv] A Survey on Video Diffusion Models

https://aicarrier.feishu.cn/file/Ds0BbCAo6oTazdxxo3Zciw1Nnne

https://www.bilibili.com/video/BV1cJ4m1e7sQ/?spm_id_from=333.999.0.0&vd_source=7b6e123d488e2a5d43fb972abef382a2

https://proceedings.neurips.cc/paper_files/paper/2023/file/c481049f7410f38e788f67c171c64ad5-Paper-Datasets_and_Benchmarks.pdf
