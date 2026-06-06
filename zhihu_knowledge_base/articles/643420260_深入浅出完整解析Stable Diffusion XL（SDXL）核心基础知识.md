# 深入浅出完整解析Stable Diffusion XL（SDXL）核心基础知识

**作者**: Rocky Ding​北京科技大学 工学硕士

**原文链接**: https://zhuanlan.zhihu.com/p/643420260

---

​
目录
收起
1. Stable Diffusion XL系列资源
2. Stable Diffusion XL核心基础内容
2.1 SDXL整体架构初识
2.2 VAE模型（包含详细图解）
2.3 U-Net模型（Base部分，包含详细图解）
2.4 Text Encoder模型（包含详细图解）
2.5 Refiner模型（包含详细图解）
2.6 SDXL官方训练技巧&细节
3. 从0到1搭建使用Stable Diffusion XL进行AI绘画（全网最详细讲解）
3.1 零基础使用diffusers搭建Stable Diffusion XL推理流程
3.2 零基础使用Stable Diffusion WebUI搭建Stable Diffusion XL推理流程
3.3 零基础使用ComfyUI搭建Stable Diffusion XL推理流程
3.4 零基础使用SD.Next搭建Stable Diffusion XL推理流程
3.5 SDXL生成图像示例
4. 从0到1上手使用Stable Diffusion XL训练自己的AI绘画模型（全网最详细讲解）
4.0 SDXL训练资源分享
4.1 SDXL训练脉络初识
4.2 配置训练环境与训练文件
4.3 SDXL训练数据集制作
4.4 SDXL微调（finetune）训练
4.5 基于SDXL训练LoRA模型
4.6 SDXL训练结果测试评估
4.7 SDXL训练经验分享（持续更新！）
5. 生成式模型的性能测评
5.1 FID（Fréchet inception distance）
5.2 CLIP score
5.3 Aesthetics Scorer（美学评分）
5.4 与Midjourney系列进行对比
5.5 不同实际场景的务实评估
6. SDXL Turbo模型核心基础内容完整讲解
6.1 SDXL Turbo整体架构初识
6.2 SDXL Turbo核心原理详解
6.3 SDXL Turbo效果测试
7. Playground v2.5核心基础内容完整讲解
7.1 Playground v2.5模型整体架构讲解
7.2 Playground v2.5模型效果测试
8. AI绘画领域的未来发展
8.1 AI绘画的“数据工厂”
8.2 AI绘画的“工作流”产品
8.3 AI绘画模型未来需要突破的瓶颈
8.4 构建AI绘画产品的开发流程
8.5 AI绘画的多模态发展
8.6 AI绘画的轻量化与端侧部署
9. 推荐阅读
9.1 深入浅出完整解析Stable Diffusion（SD）核心基础知识
9.2 深入浅出完整解析Stable Diffusion中U-Net的前世今生与核心知识
9.3 深入浅出完整解析LoRA(Low-Rank Adaptation)模型核心基础知识
9.4 深入浅出完整解析ControlNet核心基础知识
9.5 深入浅出完整解析主流AI绘画框架核心基础知识
9.6 手把手教你成为AIGC算法工程师，斩获AIGC算法offer！
9.7 AIGC产业的深度思考与分析
9.8 算法工程师的独孤九剑秘籍

本文的专栏：算法兵器谱
我的公众号：WeThinkIn
更多AI行业干货内容欢迎关注我的知乎，公众号，专栏～

码字不易，希望大家能多多点赞，给我更多坚持写下去的动力，谢谢大家！

2024.03.04最新消息，本文已经增加对Playground v2.5模型的解读。

2023.11.29最新消息，本文已经增加对SDXL Turbo模型的解读。

2023.09.26最新消息，由于Stable Diffusion XL模型的网络结构比较复杂，不好可视化，导致大家看的云里雾里。因此本文中已经发布Stable Diffusion XL中VAE，U-Net，Refiner，OpenCLIP ViT-bigG和OpenAI CLIP ViT-L五大模型的可视化网络结构图，大家可以下载用于学习！

2023.08.26最新消息，本文已经撰写Stable Diffusion XL以及对应LoRA的训练全流程与详细解读内容，同时发布对应的保姆级训练资源，大家可以愉快地训练属于自己的SDXL和LoRA模型了！

大家好，我是Rocky。

2022年作为AIGC（Artificial Intelligence Generated Content）时代的元年，各个领域的AIGC模型与技术都有一个迅猛的发展（比如Stable Diffusion、ChatGPT、Midjourney等），未来15年的科技新浪潮已然来临，AIGC无疑给工业界、投资界、学术界以及竞赛界都注入了新的“AI活力”与“AI势能”。

其中在AI绘画领域，Stable Diffusion模型当仁不让地成为了开源社区中持续繁荣的AI绘画核心模型，并且快速破圈让AIGC的ToC可能性比肩移动互联网时代的产品，每个人都能感受到AI带来的力量与影响。Rocky之前撰写过深入浅出解析Stable Diffusion模型的文章（依旧在持续补充完善中，欢迎大家点赞，给我更多坚持写下去的动力）：

本文中介绍的Stable Diffusion XL系列模型（简称SDXL）是Stable Diffusion的最新优化版本，由Stability AI发布。比起Stable Diffusion，Stable Diffusion XL做了全方位的优化，Rocky相信，Stable Diffusion是AI绘画领域的“YOLO”，而Stable Diffusion XL就是“YOLOv3”。

Stable Diffusion XL生成图片示例

因此在本文中，Rocky主要对Stable Diffusion XL系列模型（Stable Diffusion XL 1.0、Stable Diffusion XL 0.9、Stable Diffusion XL Turbo等）的全维度各个方面都做一个深入浅出的分析总结（SDXL模型结构解析、SDXL模型从0到1保姆级训练教程、SDXL模型不同AI绘画框架从0到1推理运行保姆级教程、最新SDXL资源汇总分享、AI绘画模型的性能测评、AI绘画领域未来发展、SDXL相关配套工具使用等），和大家一些探讨学习，让我们在AIGC时代能够更好地融入和从容。

1. Stable Diffusion XL系列资源
官方项目：Stability-AI/generative-models（包括SDXL，SDXL Turbo等）
diffusers库中的SDXL代码pipelines：diffusers/pipelines/stable_diffusion_xl
训练代码：sd-scripts、kohya_ss、qaneel/kohya-trainer、Linaqruf/kohya-trainer、diffusers_sdxl_train
SDXL技术报告：SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis
SDXL Turbo技术报告：Adversarial Diffusion Distillation
Playground v2.5技术报告：Playground v2.5
SDXL模型权重百度云网盘：关注Rocky的公众号WeThinkIn，后台回复：SDXL模型，即可获得资源链接，包含Stable Diffusion XL 1.0和Stable Diffusion XL 0.9（Base模型+Refiner模型）模型权重、Stable Diffusion XL Turbo模型权重、Playground v2.5模型权重以及Stable Diffusion XL VAE模型权重。不同格式的模型权重比如safetensors格式、diffusers格式、FP16精度格式、ONNX格式、flax/jax格式以及openvino格式等均已包含。
SDXL保姆级训练资源百度云网盘：关注Rocky的公众号WeThinkIn，后台回复：SDXL-Train，即可获得资源链接，包含数据处理、SDXL模型微调训练以及基于SDXL的LoRA模型训练代码全套资源，帮助大家从0到1快速上手训练属于自己的SDXL AI绘画模型。更多SDXL训练资源使用教程，请看本文第四章内容。
Stable Diffusion XL中VAE，U-Net，Refiner，OpenCLIP ViT-bigG和OpenAI CLIP ViT-L五大模型的可视化网络结构图下载：关注Rocky的公众号WeThinkIn，后台回复：SDXL网络结构，即可获得网络结构图资源链接。

Rocky会持续把更多Stable Diffusion XL的资源更新发布到本节中，让大家更加方便的查找SDXL系列模型的最新资讯。

2. Stable Diffusion XL核心基础内容

与Stable Diffusion 1.x-2.x相比，Stable Diffusion XL主要进行如下的优化：

对Stable Diffusion 1.x-2.x的U-Net，VAE，CLIP Text Encoder三大核心模型都做了改进。
增加一个独立的基于Latent的Refiner模型，也是一个扩散模型，用来提升生成图像的精细化程度。
设计了很多训练Tricks，包括图像尺寸条件化策略、图像裁剪参数条件化策略以及多尺度训练策略等。
先发布Stable Diffusion XL 0.9测试版本，基于用户的使用体验和图片生成的反馈情况，针对性增加数据集和使用RLHF（Reinforcement Learning from Human Feedback，基于人类反馈的强化学习）技术优化训练后，推出了Stable Diffusion XL 1.0正式版。
2.1 SDXL整体架构初识

Stable Diffusion XL是一个二阶段的级联扩散模型（Latent Diffusion Model），包括Base模型和Refiner模型。其中Base模型的主要工作和Stable Diffusion 1.x-2.x一致，具备文生图（txt2img）、图生图（img2img）、图像inpainting等能力。在Base模型之后，级联了Refiner模型，对Base模型生成的图像Latent特征进行精细化提升，其本质上是在做图生图的工作。

SDXL Base模型由U-Net、VAE以及CLIP Text Encoder（两个）三个模块组成，在FP16精度下Base模型大小6.94G（FP32：13.88G），其中U-Net占5.14G、VAE模型占167M以及两个CLIP Text Encoder一大一小（OpenCLIP ViT-bigG和OpenAI CLIP ViT-L）分别是1.39G和246M。

SDXL Refiner模型同样由U-Net、VAE和CLIP Text Encoder（一个）三个模块组成，在FP16精度下Refiner模型大小6.08G，其中U-Net占4.52G、VAE模型占167M（与Base模型共用）以及CLIP Text Encoder模型（OpenCLIP ViT-bigG）大小1.39G（与Base模型共用）。

从下图可以看到，Stable Diffusion XL无论是对模型的整体工作流还是对不同子模块（U-Net、VAE、CLIP Text Encoder）都做了大幅的改进，能够生成1024x1024分辨率及以上的高质量图片。同时这些改进思想无论是对AIGC时代的模型还是传统深度学习时代的模型，都有非常大的迁移应用价值。

Stable Diffusion XL整体结构

比起Stable Diffusion 1.x-2.x，Stable Diffusion XL的参数量增加到了66亿（Base模型35亿+Refiner模型31亿），并且先后发布了模型结构完全相同的0.9和1.0两个版本。Stable Diffusion XL 1.0在0.9版本上使用更多训练集+RLHF来优化生成图像的色彩、对比度、光线以及阴影方面，使得生成图像的构图比0.9版本更加鲜明准确。

Rocky相信，以Stable Diffusion XL 1.0版本为核心的AI绘画和AI视频生态将会持续繁荣。

Stable Diffusion XL Base模型参数：SDXL-Base

Stable Diffusion XL Refiner模型参数：SDXL-Refiner

下图中展示了Stability AI用户对SDXL 1.0、SDXL 0.9、SD 1.5以及SD 2.1的性能评估结果。可以看到单独的SDXL 1.0 Base模型的表现明显优于之前的所有SD版本，而完整的SDXL 1.0模型（Base模型 + Refiner模型）则实现了最佳的图像生成整体性能。

SDXL 1.0、SDXL 0.9、SD 1.5、SD 2.1之间的整体性能对比
2.2 VAE模型（包含详细图解）

VAE模型（变分自编码器，Variational Auto-Encoder）是一个经典的生成式模型，其基本原理就不过多介绍了。在传统深度学习时代，GAN的风头完全盖过了VAE，但VAE简洁稳定的Encoder-Decoder架构，以及能够高效提取数据Latent特征和Latent特征像素级重建的关键能力，让其跨过了周期，在AIGC时代重新繁荣。

Stable Diffusion XL依旧是基于Latent的扩散模型，所以VAE的Encoder和Decoder结构依旧是Stable Diffusion XL提取图像Latent特征和图像像素级重建的关键一招。

当输入是图片时，Stable Diffusion XL和Stable Diffusion一样，首先会使用VAE的Encoder结构将输入图像转换为Latent特征，然后U-Net不断对Latent特征进行优化，最后使用VAE的Decoder结构将Latent特征重建出像素级图像。除了提取Latent特征和图像的像素级重建外，VAE还可以改进生成图像中的高频细节，小物体特征和整体图像色彩。

当Stable Diffusion XL的输入是文字时，这时我们不需要VAE的Encoder结构，只需要Decoder进行图像重建。VAE的灵活运用，让Stable Diffusion系列增添了几分优雅。

Stable Diffusion XL使用了和之前Stable Diffusion系列一样的VAE结构（KL-f8），但在训练中选择了更大的Batch-Size（256 vs 9），并且对模型进行指数滑动平均操作（EMA，exponential moving average），EMA对模型的参数做平均，从而提高性能并增加模型鲁棒性。

下面是Rocky梳理的Stable Diffusion XL的VAE完整结构图，希望能让大家对这个在Stable DIffusion系列中未曾改变架构的模型有一个更直观的认识，在学习时也更加的得心应手：

Stable Diffusion XL Base VAE完整结构图

SDXL VAE模型中有三个基础组件：

GSC组件：GroupNorm+SiLU+Conv
Downsample组件：Padding+Conv
Upsample组件：Interpolate+Conv

同时SDXL VAE模型还有两个核心组件：ResNetBlock模块和SelfAttention模型，两个模块的结构如上图所示。

SDXL VAE Encoder部分包含了三个DownBlock模块、一个ResNetBlock模块以及一个MidBlock模块，将输入图像压缩到Latent空间，转换成为Gaussian Distribution。

而VAE Decoder部分正好相反，其输入Latent空间特征，并重建成为像素级图像作为输出。其包含了三个UpBlock模块、一个ResNetBlock模块以及一个MidBlock模块。

在损失函数方面，使用了久经考验的生成领域“交叉熵”—感知损失（perceptual loss）以及L1回归损失来约束VAE的训练过程。

下表是Stable Diffusion XL的VAE在COCO2017 验证集上，图像大小为256×256像素的情况下的性能。

（注：Stable Diffusion XL的VAE是从头开始训练的）

Stable Diffusion XL中优化VAE带来的性能提升

上面的表中的三个VAE模型结构是一样的，不同点在于SD 2.x VAE是基于SD 1.x VAE微调训练了Decoder部分，同时保持Encoder部分权重不变，使他们有相同的Latent特征分布，所以SD 1.x和SD 2.x的VAE模型是互相兼容的。而SDXL VAE是重新从头开始训练的，所以其Latent特征分布与之前的两者不同。

由于Latent特征分布产生了变化，SDXL VAE的缩放系数也产生了变化。VAE在将Latent特征送入U-Net之前，需要对Latent特征进行缩放让其标准差尽量为1，之前的Stable Diffusion系列采用的缩放系数为0.18215，由于Stable Diffusion XL的VAE进行了全面的重训练，所以缩放系数重新设置为0.13025。

注意：由于缩放系数的改变，Stable Diffusion XL VAE模型与之前的Stable Diffusion系列并不兼容。如果在SDXL上使用之前系列的VAE，会生成充满噪声的图片。

与此同时，与Stable Diffusion一样，VAE模型在Stable Diffusion XL中除了能进行图像压缩和图像重建的工作外，通过切换不同微调训练版本的VAE模型，能够改变生成图片的细节与整体颜色（更改生成图像的颜色表现，类似于色彩滤镜）。

目前在开源社区常用的SDXL VAE模型有：sdxl_vae.safetensors、lastpiecexlVAE_baseonA0897.safetensors、fixFP16ErrorsSDXLLowerMemoryUse_v10.safetensors、xlVAEC_f1.safetensors、flatpiecexlVAE_baseonA1579.safetensors等。

这里Rocky使用了6种不同的SDXL VAE模型，在其他参数保持不变的情况下，对比了SDXL模型的出图效果，如下所示：

Stable Diffusion XL中6种不同VAE模型的效果对比

可以看到，我们在SDXL中切换VAE模型进行出图时，均不会对构图进行大幅改变，只对生成图像的细节与颜色表现进行调整。

Rocky目前也在整理汇总高价值的SDXL VAE模型（持续更新），方便大家获取使用。大家可以关注Rocky的公众号WeThinkIn，后台回复：SDXLVAE，即可获得资源链接，包含上述的全部SDXL VAE模型权重和更多高价值SDXL VAE模型权重。

官方的Stable Diffusion XL VAE的权重已经开源：sdxl-vae

需要注意的是，原生Stable Diffusion XL VAE采用FP16精度时会出现数值溢出成NaNs的情况，导致重建的图像是一个黑图，所以必须使用FP32精度进行推理重建。如果大家想要FP16精度进行推理，可以使用sdxl-vae-fp16-fix版本的SDXL VAE模型，其对FP16出现的NANs的情况进行了修复。

在官网如果遇到网络问题或者下载速度很慢的问题，可以关注Rocky的公众号WeThinkIn，后台回复：SDXL模型，即可获得Stable Diffusion XL VAE模型权重（包含原生SDXL VAE与FP16修复版本）资源链接。

接下来Rocky将用diffusers库来快速加载Stable Diffusion XL中的VAE模型，并通过可视化的效果直观展示SDXL VAE的压缩与重建效果，完整代码如下所示：

import cv2
import torch
import numpy as np
from diffusers import AutoencoderKL

# 加载SDXL VAE模型: SDXL VAE模型可以通过指定subfolder文件来单独加载。
# SDXL VAE模型权重百度云网盘：关注Rocky的公众号WeThinkIn，后台回复：SDXL模型，即可获得资源链接
VAE = AutoencoderKL.from_pretrained("/本地路径/sdxl-vae")
VAE.to("cuda") 

# 用OpenCV读取和调整图像大小
raw_image = cv2.imread("test_vae.png")
raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
raw_image = cv2.resize(raw_image, (1024, 1024))

# 将图像数据转换为浮点数并归一化
image = raw_image.astype(np.float32) / 127.5 - 1.0

# 调整数组维度以匹配PyTorch的格式 (N, C, H, W)
image = image.transpose(2, 0, 1)
image = image[None, :, :, :]

# 转换为PyTorch张量
image = torch.from_numpy(image).to("cuda")

# 压缩图像为Latent特征并重建
with torch.inference_mode():
    # 使用SDXL VAE进行压缩和重建
    latent = VAE.encode(image).latent_dist.sample()
    rec_image = VAE.decode(latent).sample

    # 后处理
    rec_image = (rec_image / 2 + 0.5).clamp(0, 1)
    rec_image = rec_image.cpu().permute(0, 2, 3, 1).numpy()

    # 反归一化
    rec_image = (rec_image * 255).round().astype("uint8")
    rec_image = rec_image[0]

    # 保存重建后图像
    cv2.imwrite("reconstructed_sdxl.png", cv2.cvtColor(rec_image, cv2.COLOR_RGB2BGR))

接下来，我们分别使用1024x1024分辨率的真实场景图片和二次元图片，使用SDXL VAE模型进行四种尺寸下的压缩与重建，重建效果如下所示：

SDXL VAE模型对真实场景图片和二次元图片的压缩与重建效果

从对比结果中可以看到，SDXL VAE在对图像进行压缩和重建时，虽然依然存在一定的精度损失，但只在256x256分辨率下会明显出现，比如说人脸特征丢失的情况。同时比起SD 1.5 VAE模型，SDXL VAE模型在图像压缩与重建时的精度损失大幅降低。并且不管是二次元图片还是真实场景图片，在不同尺寸下重建时，图片的主要特征都能保留下来，局部特征畸变的情况较少，损失程度较低。

2.3 U-Net模型（Base部分，包含详细图解）
Stable Diffusion全系列对比图

上表是Stable Diffusion XL与之前的Stable Diffusion系列的对比，从中可以看出，Stable Diffusion 1.x的U-Net参数量只有860M，就算是Stable Diffusion 2.x，其参数量也不过865M。但等到Stable Diffusion XL，U-Net模型（Base部分）参数量就增加到2.6B，参数量增加幅度达到了3倍左右。

下图是Rocky梳理的Stable Diffusion XL Base U-Net的完整结构图，大家可以感受一下其魅力，看着这个完整结构图学习Stable Diffusion XL Base U-Net部分，相信大家脑海中的思路也会更加清晰：

Stable Diffusion XL Base U-Net完整结构图

上图中包含Stable Diffusion XL Base U-Net的十四个基本模块：

GSC模块：Stable Diffusion Base XL U-Net中的最小组件之一，由GroupNorm+SiLU+Conv三者组成。
DownSample模块：Stable Diffusion Base XL U-Net中的下采样组件，使用了Conv（kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)）进行采下采样。
UpSample模块：Stable Diffusion Base XL U-Net中的上采样组件，由插值算法（nearest）+Conv组成。
ResNetBlock模块：借鉴ResNet模型的“残差结构”，让网络能够构建的更深的同时，将Time Embedding信息嵌入模型。
CrossAttention模块：将文本的语义信息与图像的语义信息进行Attention机制，增强输入文本Prompt对生成图像的控制。
SelfAttention模块：SelfAttention模块的整体结构与CrossAttention模块相同，这是输入全部都是图像信息，不再输入文本信息。
FeedForward模块：Attention机制中的经典模块，由GeGlU+Dropout+Linear组成。
BasicTransformer Block模块：由LayerNorm+SelfAttention+CrossAttention+FeedForward组成，是多重Attention机制的级联，并且每个Attention机制都是一个“残差结构”。通过加深网络和多Attention机制，大幅增强模型的学习能力与图文的匹配能力。
SDXL_Spatial Transformer_X模块：由GroupNorm+Linear+X个BasicTransformer Block+Linear构成，同时ResNet模型的“残差结构”依旧没有缺席。
SDXL_DownBlock模块：由两个ResNetBlock+一个DownSample组成。
SDXL_UpBlock_X模块：由X个ResNetBlock模块组成。
CrossAttnDownBlock_X_K模块：是Stable Diffusion XL Base U-Net中Encoder部分的主要模块，由K个（ResNetBlock模块+SDXL_Spatial Transformer_X模块）+一个DownSample模块组成。
CrossAttnUpBlock_X_K模块：是Stable Diffusion XL Base U-Net中Decoder部分的主要模块，由K个（ResNetBlock模块+SDXL_Spatial Transformer_X模块）+一个UpSample模块组成。
CrossAttnMidBlock模块：是Stable Diffusion XL Base U-Net中Encoder和ecoder连接的部分，由ResNetBlock+SDXL_Spatial Transformer_10+ResNetBlock组成。

可以看到，其中增加的SDXL_Spatial Transformer_X模块（主要包含Self Attention + Cross Attention + FeedForward）数量占新增参数量的主要部分，Rocky在上表中已经用红色框圈出。U-Net的Encoder和Decoder结构也从之前系列的4stage改成3stage（[1,1,1,1] -> [0,2,10]），同时SDXL只使用两次下采样和上采样，而之前的SD系列模型都是三次下采样和上采样。并且比起Stable Diffusion 1.x-2.x，Stable Diffusion XL在第一个stage中不再使用Spatial Transformer Blocks，而在第二和第三个stage中大量增加了Spatial Transformer Blocks（分别是2和10），那么这样设计有什么好处呢？

首先，在第一个stage中不使用SDXL_Spatial Transformer_X模块，可以明显减少显存占用和计算量。然后在第二和第三个stage这两个维度较小的feature map上使用数量较多的SDXL_Spatial Transformer_X模块，能在大幅提升模型整体性能（学习能力和表达能力）的同时，优化了计算成本。整个新的SDXL Base U-Net设计思想也让SDXL的Base出图分辨率提升至1024x1024。在出图参数保持一致的情况下，Stable Diffusion XL生成图片的耗时只比Stable Diffusion多了20%-30%之间，这个拥有2.6B参数量的模型已经足够伟大。

在SDXL U-Net的Encoder结构中，包含了两个CrossAttnDownBlock结构和一个SDXL_DownBlock结构；在Decoder结构中，包含了两个CrossAttnUpBlock结构和一个SDXL_UpBlock结构；与此同时，Encoder和Decoder中间存在Skip Connection，进行信息的传递与融合。

从上面讲到的十四个基本模块中可以看到，BasicTransformer Block模块是整个框架的基石，由SelfAttention，CrossAttention和FeedForward三个组件构成，并且使用了循环残差模式，让SDXL Base U-Net不仅可以设计的更深，同时也具备更强的文本特征和图像体征的学习能力。

接下来，Rocky再给大家讲解CrossAttention模块的一些细节内容，让大家能更好地理解这个关键模块。

Stable Diffusion XL中的Text Condition信息由两个Text Encoder提供（OpenCLIP ViT-bigG和OpenAI CLIP ViT-L），通过Cross Attention组件嵌入，作为K Matrix和V Matrix。与此同时，图片的Latent Feature作为Q Matrix。

但是大家知道Text Condition是三维的，而Latent Feature是四维的，那它们是怎么进行Attention机制的呢？

其实在每次进行Attention机制前，我们需要将Latent Feature从[batch_size,channels,height,width]转换到[batch_size,height*width,channels] ，这样就变成了三维特征，就能够和Text Condition做CrossAttention操作。

在完成CrossAttention操作后，我们再将Latent Feature从[batch_size,height*width,channels]转换到[batch_size,channels,height,width] ，这样就又重新回到原来的维度。

还有一点是Text Condition如何跟latent Feature大小保持一致呢？因为latent embedding不同位置的H和W是不一样的，但是Text Condition是从文本中提取的，其H和W是固定的。这里在CorssAttention模块中有一个非常巧妙的点，那就是在不同特征做Attention操作前，使用Linear层将不同的特征的尺寸大小对齐。

2.4 Text Encoder模型（包含详细图解）

Stable Diffusion XL模型采用的Text Encoder依然是基于CLIP架构的。我们知道，CLIP模型主要包含Text Encoder和Image Encoder两个模块，Stable Diffusion 1.x系列使用的是OpenAI CLIP ViT-L/14（123.65M）中的Text Encoder模型，而Stable Diffusion 2.x系列则使用OpenCLIP ViT-H/14（354.03M）中的Text Encoder模型。

Stable Diffusion XL和Stable Diffusion 1.x-2.x系列一样，只使用Text Encoder模块从文本信息中提取Text Embeddings。

不同的是，Stable Diffusion XL与之前的系列相比使用了两个CLIP Text Encoder，分别是OpenCLIP ViT-bigG（694M）和OpenAI CLIP ViT-L/14（123.65M），从而大大增强了Stable Diffusion XL对文本的提取和理解能力，同时提高了输入文本和生成图片的一致性。

其中OpenCLIP ViT-bigG是一个只由Transformer模块组成的模型，一共有32个CLIPEncoder模块，是一个强力的特征提取模型。其单个CLIPEncoder模块结构如下所示：

# OpenCLIP ViT-bigG中CLIPEncoder模块结构
CLIPEncoderLayer(
    (self_attention): CLIPAttention(
        (k_Matric): Linear(in_features=1280, out_features=1280, bias=True)
        (v_Matric): Linear(in_features=1280, out_features=1280, bias=True)
        (q_Matric): Linear(in_features=1280, out_features=1280, bias=True)
        (out_proj): Linear(in_features=1280, out_features=1280, bias=True)
      )
    (layer_norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
    (mlp): CLIPMLP(
        (activation_fn): GELUActivation()
        (fc1): Linear(in_features=1280, out_features=5120, bias=True)
        (fc2): Linear(in_features=5120, out_features=1280, bias=True)
    )
    (layer_norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
    )

下图是Rocky梳理的SDXL OpenCLIP ViT-bigG的完整结构图，大家可以感受一下其魅力，看着这个完整结构图学习Stable Diffusion XL OpenCLIP ViT-bigG部分，相信大家脑海中的思路也会更加清晰：

Stable Diffusion XL OpenCLIP ViT-bigG Encoder完整结构图

OpenAI CLIP ViT-L/14同样是一个只由Transformer模块组成的模型，一共有12个CLIPEncoder模块，其单个CLIPEncoder模块结构如下所示：

# OpenAI CLIP ViT-L中CLIPEncoder模块结构
CLIPEncoderLayer(
    (self_attention): CLIPAttention(
        (k_Matric): Linear(in_features=768, out_features=768, bias=True)
        (v_Matric): Linear(in_features=768, out_features=768, bias=True)
        (q_Matric): Linear(in_features=768, out_features=768, bias=True)
        (out_proj): Linear(in_features=768, out_features=768, bias=True)
    )
    (layer_norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (mlp): CLIPMLP(
        (activation_fn): QuickGELUActivation()
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
    )
    (layer_norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )

下图是Rocky梳理的SDXL OpenAI CLIP ViT-L/14的完整结构图，大家可以感受一下其魅力，看着这个完整结构图学习Stable Diffusion XL OpenAI CLIP ViT-L/14部分，相信大家脑海中的思路也会更加清晰：

Stable Diffusion XL OpenAI CLIP ViT-L完整结构图

由上面两个结构对比可知，OpenCLIP ViT-bigG的优势在于模型结构更深，特征维度更大，特征提取能力更强，但是其两者的基本CLIPEncoder模块是一样的。

下面Rocky将使用transofmers库演示调用SDXL OpenAI CLIP ViT-L/14 和OpenCLIP ViT-bigG，给大家一个更加直观的SDXL模型的文本编码全过程。

首先是SDXL OpenAI CLIP ViT-L/14的文本编码过程：

from transformers import CLIPTextModel, CLIPTokenizer

# 加载 OpenAI CLIP ViT-L/14 Text Encoder模型和Tokenizer
# SDXL模型权重百度云网盘：关注Rocky的公众号WeThinkIn，后台回复：SDXL模型，即可获得资源链接
text_encoder = CLIPTextModel.from_pretrained("/本地路径/stable-diffusion-xl-base-1.0", subfolder="text_encoder").to("cuda")
text_tokenizer = CLIPTokenizer.from_pretrained("/本地路径/stable-diffusion-xl-base-1.0", subfolder="tokenizer")

# 将输入SDXL模型的prompt进行tokenize，得到对应的token ids特征
prompt = "1girl,beautiful"
text_token_ids = text_tokenizer(
    prompt,
    padding="max_length",
    max_length=text_tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt"
).input_ids

print("text_token_ids' shape:",text_token_ids.shape)
print("text_token_ids:",text_token_ids)

# 将token ids特征输入OpenAI CLIP ViT-L/14 Text Encoder模型中输出77x768的Text Embeddings特征
text_embeddings = text_encoder(text_token_ids.to("cuda"))[0] # 由于Text Encoder模型输出的是一个元组，所以需要[0]对77x768的Text Embeddings特征进行提取
print("text_embeddings' shape:",text_embeddings.shape)
print(text_embeddings)

---------------- 运行结果 ----------------
text_token_ids' shape: torch.Size([1, 77])
text_token_ids: tensor([[49406,   272,  1611,   267,  1215, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407]])
text_embeddings' shape: torch.Size([1, 77, 768])
tensor([[[-0.3885,  0.0230, -0.0521,  ..., -0.4901, -0.3065,  0.0674],
         [-0.8424, -1.1387,  1.2767,  ..., -0.2598,  1.6289, -0.7855],
         [ 0.1751, -0.9847,  0.1881,  ...,  0.0657, -1.4940, -1.2612],
         ...,
         [ 0.2039, -0.7298, -0.3206,  ...,  0.6751, -0.5814, -0.7320],
         [ 0.1921, -0.7345, -0.3039,  ...,  0.6806, -0.5852, -0.7228],
         [ 0.2112, -0.6438, -0.3042,  ...,  0.6628, -0.5576, -0.7583]]],
       device='cuda:0', grad_fn=<NativeLayerNormBackward0>)

接着是SDXL OpenCLIP ViT-bigG的文本编码过程：

from transformers import CLIPTextModel, CLIPTokenizer

# 加载 OpenCLIP ViT-bigG Text Encoder模型和Tokenizer
# SDXL模型权重百度云网盘：关注Rocky的公众号WeThinkIn，后台回复：SDXL模型，即可获得资源链接
text_encoder = CLIPTextModel.from_pretrained("/本地路径/stable-diffusion-xl-base-1.0", subfolder="text_encoder_2").to("cuda")
text_tokenizer = CLIPTokenizer.from_pretrained("/本地路径/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2")

# 将输入SDXL模型的prompt进行tokenize，得到对应的token ids特征
prompt = "1girl,beautiful"
text_token_ids = text_tokenizer(
    prompt,
    padding="max_length",
    max_length=text_tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt"
).input_ids

print("text_token_ids' shape:",text_token_ids.shape)
print("text_token_ids:",text_token_ids)

# 将token ids特征输入OpenCLIP ViT-bigG Text Encoder模型中输出77x1280的Text Embeddings特征
text_embeddings = text_encoder(text_token_ids.to("cuda"))[0] # 由于Text Encoder模型输出的是一个元组，所以需要[0]对77x1280的Text Embeddings特征进行提取
print("text_embeddings' shape:",text_embeddings.shape)
print(text_embeddings)

---------------- 运行结果 ----------------
text_token_ids' shape: torch.Size([1, 77])
text_token_ids: tensor([[49406,   272,  1611,   267,  1215, 49407,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0]])
text_embeddings' shape: torch.Size([1, 77, 1280])
tensor([[[-0.1025, -0.3104,  0.1660,  ..., -0.1596, -0.0680, -0.0180],
         [ 0.7724,  0.3004,  0.5225,  ...,  0.4482,  0.8743, -1.0429],
         [-0.3963,  0.0041, -0.3626,  ...,  0.1841,  0.2224, -1.9317],
         ...,
         [-0.8887, -0.2579,  1.3508,  ..., -0.4421,  0.2193,  1.2736],
         [-0.9659, -0.0447,  1.4424,  ..., -0.4350, -0.1186,  1.2042],
         [-0.5213, -0.0255,  1.8161,  ..., -0.7231, -0.3752,  1.0876]]],
       device='cuda:0', grad_fn=<NativeLayerNormBackward0>)

与传统深度学习中的模型融合类似，Stable Diffusion XL分别提取两个Text Encoder的倒数第二层特征，并进行concat操作作为文本条件（Text Conditioning）。其中OpenCLIP ViT-bigG的特征维度为77x1280，而OpenAI CLIP ViT-L/14的特征维度是77x768，所以输入总的特征维度是77x2048（77是最大的token数，2048是SDXL的context dim），再通过Cross Attention模块将文本信息传入Stable Diffusion XL的训练过程与推理过程中。

OpenCLIP ViT-bigG和OpenAI CLIP ViT-L模型性能

从上图可以看到，OpenCLIP ViT-bigG和OpenAI CLIP ViT-L/14在ImageNet上zero-shot性能分别为80.1%和75.4%，Rocky有点疑惑的是，为什么不用CoCa或者将OpenAI CLIP ViT-L/14替换成penAI CLIP ViT-H呢。

和Stable Diffusion 1.x-2.x一致的是，Stable Diffusion XL输入的最大Token数依旧是77，当输入文本的Token数量超过77后，将通过Clip操作拉回77x2048；如果Token数不足77则会通过padding操作得到77x2048。

与此同时，Stable Diffusion XL还提取了OpenCLIP ViT-bigG的pooled text embedding，将其嵌入到Time Embeddings中（add操作），作为辅助约束条件（强化文本的整体语义信息），但是这种辅助条件的强度是较为微弱的。

和之前的系列一样，SDXL Text Encoder在官方训练时是冻结的，我们在对SDXL模型进行微调训练时，可以同步开启Text Encoder的微调训练，能够使得Text Encoder对生成图片的控制力增强，使其生成内容更加贴近训练集的分布。

2.5 Refiner模型（包含详细图解）

Rocky看到Stable Diffusion XL的Refiner部分时，脑海里马上联想到了DeepFloyd和StabilityAI联合开发的DeepFloyd IF模型。

DeepFloyd IF是一种基于像素的文本到图像三重级联扩散模型，大大提升了扩散模型的图像生成能力。

这次，Stable Diffusion XL终于也开始使用级联策略，在U-Net（Base）之后，级联Refiner模型，进一步提升生成图像的细节特征与整体质量。

通过级联模型提升生成图片的质量，这可以说这是AIGC时代里的模型融合（ensemble）。和传统深度学习时代的多模型融合策略一样，不管是在学术界、工业界还是竞赛界，都是“行业核武”般的存在。

DeepFloyd IF结构图

由于已经有U-Net（Base）模型生成了图像的Latent特征，所以Refiner模型的主要工作是在Latent特征进行小噪声去除和细节质量提升。

U-Net（Base）结构+Refiner结构

Refiner模型和Base模型一样是基于Latent的扩散模型，也采用了Encoder-Decoder结构，和U-Net兼容同一个VAE模型。不过在Text Encoder部分，Refiner模型只使用了OpenCLIP ViT-bigG的Text Encoder，同样提取了倒数第二层特征以及进行了pooled text embedding的嵌入。

下图是Rocky梳理的Stable Diffusion XL Refiner模型的完整结构图，大家可以先感受一下其魅力，在学习Refiner模型时可以与Base模型中的U-Net进行对比，会有更多直观的认识：

Stable Diffusion XL Refiner完整结构图

接下来Rocky给大家分析一下SDXL Refiner模型和SDXL Base模型在结构上的异同：

SDXL Base的Encoder和Decoder结构都采用4个stage，而SDXL Base设计的是3个stage。
SDXL Refiner和SDXL Base一样，在第一个stage中没有使用Attention模块。
在经过第一个卷积后，SDXL Refiner设置初始网络特征维度为384，而SDXL Base 采用的是320。
SDXL Refiner的Attention模块中SDXL_Spatial Transformer结构数量均设置为4。
SDXL Refiner的参数量为2.3B，比起SDXL Base的2.6B参数量略小一些。

SDXL Refiner模型的训练逻辑与SDXL Base一样，不过Refiner模型只在前200个Timesteps上训练（设置的noise level较低）。

在Stable Diffusion XL推理阶段，输入一个prompt，通过VAE和U-Net（Base）模型生成Latent特征，接着给这个Latent特征进行扩散过程加上一定的噪音。在此基础上，再使用Refiner模型进行去噪，以提升图像的整体质量与局部细节。

从下图中可以看到，在使用Refiner模型后，生成图片的背景和人脸部分效果有了一定的提升：

左图表示只使用Base模型，右图表示使用了Base+Refiner模型

可以看到，Refiner模型主要做了图像生成图像（img2img）的工作，其具备很强的迁移兼容能力，可以作为Stable Diffusion、Midjourney、DALL-E、GAN、VAE等生成式模型的级联组件，成为AI绘画领域的一个强力后处理工具，这不管是对学术界、工业界还是竞赛界，都是一个巨大的利好。

Stable Diffusion系列模型效果对比

由上表可以看出，Stable Diffusion XL Base模型的效果已经大幅超过SD 1.5和SD 2.1，当增加Refiner模型之后，完整的Stable Diffusion XL模型达到了更加优秀的图像生成效果。

2.6 SDXL官方训练技巧&细节

Stable Diffusion XL在训练阶段提出了很多优化方法，包括图像尺寸条件化策略，图像裁剪参数条件化策略以及多尺度训练策略。这些优化方法对整个AIGC领域都有很好的参考与借鉴意义，其通用性和迁移性能普惠其他的生成式模型的训练与优化。

【一】图像尺寸条件化

之前在Stable Diffusion的训练过程中，主要分成两个阶段，一个是在256x256的图像尺寸上进行预训练，然后在512x512的图像尺寸上继续训练。

而这两个阶段的训练过程都要对最小图像尺寸进行约束。第一阶段中，会将尺寸小于256x256的图像舍弃；同样的，在第二阶段，会将尺寸小于512x512的图像筛除。这样的约束会导致训练数据中的大量数据被丢弃，从而很可能导致模型性能和泛化性的降低。

下图展示了如果将尺寸小于256x256的图像筛除，整个数据集将减少39%的数据。如果加上尺寸小于512x512的图像，未利用数据占整个数据集的百分比将更大。




针对上述数据集利用率的问题，常规思路可以借助超分模型将尺寸过小的图像放大。但是面对对于图像尺寸过小的场景，目前的超分模型可能会在对图像超分的同时会引入一些噪声伪影，影响模型的训练，导致生成一些模糊的图像。

Stable Diffusion XL为了在解决数据集利用率问题的同时不引入噪声伪影，将U-Net（Base）模型与原始图像分辨率相关联，核心思想是将输入图像的原始高度和宽度作为额外的条件嵌入U-Net模型中，表示为 C_{size} = (height, width) 。height和width都使用傅里叶特征编码进行独立嵌入，然后将特征concat后加在Time Embedding上，将图像尺寸作为条件引入训练过程。这样以来，模型在训练过程中能够学习到图像的原始分辨率信息，从而在推理生成阶段更好地适应不同尺寸的图像生成，而不会产生噪声伪影的问题。

如下图所示，在使用了图像尺寸条件化策略后，Base模型已经对不同图像分辨率有了“自己的判断”。当输入低分辨率条件时，生成的图像较模糊；在不断增大分辨率条件时，生成的图像质量不断提升。

图像尺寸条件化策略让Base模型对图像分辨率有了“自己的判断”

【二】图像裁剪参数条件化

之前的Stable Diffusion系列模型，由于需要输入固定的图像尺寸用作训练，很多数据在预处理阶段会被裁剪。生成式模型中典型的预处理方式是先调整图像尺寸，使得最短边与目标尺寸匹配，然后再沿较长边对图像进行随机裁剪或者中心裁剪。虽然裁剪是一种数据增强方法，但是训练中对图像裁剪导致的图像特征丢失，可能会导致AI绘画模型在图像生成过程中出现不符合训练数据分布的特征。

如下图所示，对一个骑士的图片做了裁剪操作后，丢失了头部和脚部特征，再将裁剪后的数据放入模型中训练，就会影响模型对骑士这个概念的学习和认识。

“骑士”概念特征被破坏的数据

下图中展示了SD 1.4和SD 1.5的经典失败案例，生成图像中的猫出现了头部缺失的问题，龙也出现了体征不完整的情况：

SD1.4和SD1.5的经典失败案例

其实之前NovelAI就发现了这个问题，并提出了基于分桶（Ratio Bucketing）的多尺度训练策略，其主要思想是先将训练数据集按照不同的长宽比（aspect ratio）进行分组（groups）或者分桶（buckets）。在训练过程中，每次在buckets中随机选择一个bucket并从中采样Batch个数据进行训练。将数据集进行分桶可以大量较少裁剪图像的操作，并且能让模型学习多尺度的生成能力；但相对应的，预处理成本大大增加，特别是数据量级较大的情况下。

并且尽管数据分桶成功解决了数据裁剪导致的负面影响，但如果能确保数据裁剪不把负面影响引入生成过程中，裁剪这种数据增强方法依旧能给模型增强泛化性能。所以Stable Diffusion XL使用了一种简单而有效的条件化方法，即图像裁剪参数条件化策略。其主要思想是在加载数据时，将左上角的裁剪坐标通过傅里叶编码后加在Time Embedding上，并嵌入U-Net（Base）模型中，并与原始图像尺寸一起作为额外的条件嵌入U-Net模型，从而在训练过程中让模型学习到对“图像裁剪”的认识。

从下图中可以看到，将不同的 c_{crop} 坐标条件的生成图像进行了对比，当我们设置 c_{crop} = (0,0) 时可以生成主要物体居中并且无特征缺失的图像，而采用其它的坐标条件则会出现有裁剪效应的图像：

SDXL使用不同裁剪坐标获取具有裁剪效应的图像

图像尺寸条件化策略和图像裁剪参数条件化策略都能在SDXL训练过程中使用（在线方式应用），同时也可以很好的迁移到其他AIGC生成式模型的训练中。下图详细给出了两种策略的通用使用流程：

图像尺寸条件化策略和图像裁剪参数条件化策略在SDXL训练时的使用流程

可以看到，SDXL在训练过程中的数据处理流程和之前的系列是一样的，只是需要再将图像原始长宽（width和height）以及图像进行crop操作时的左上角的裁剪坐标top和left作为条件输入。

【三】多尺度训练

Stable Diffusion XL采用了多尺度训练策略，这个是在传统深度学习时代的王牌模型YOLO系列中常用的增强模型鲁棒性与泛化性的策略，终于在AIGC领域应用并常规化了，并且Stable Diffusion XL在多尺度训练的基础上，增加了分桶策略。

SDXL的论文中说训练时采用的是内部数据集作为训练集，Rocky推测大概率是基于LAION数据集为基础构建的。Stable Diffusion XL首先采用图像尺寸条件化和图像裁剪参数条件化这两种策略在256x256和512x512的图像尺寸上分别预训练600000步和200000步（batch size = 2048），总的数据量约等于 （600000 + 200000） x 2048 = 16.384亿。

接着Stable Diffusion XL在1024x1024的图像尺寸上采用多尺度方案来进行微调，并将数据分成不同纵横比的桶（bucket），并且尽可能保持每个桶的像素数接近1024×1024，同时相邻的bucket之间height或者width一般相差64像素左右，Stable Diffusion XL的具体分桶情况如下图所示：

Stable Diffusion XL训练中使用的多尺度分桶训练策略

其中Aspect Ratio = Height / Width，表示高宽比。

在训练过程中，一个Batch从一个桶里的图像采样，并且我们在每个训练步骤（step）中可以在不同的桶之间交替切换。除此之外，Aspect Ratio也会作为条件嵌入到U-Net（Base）模型中，嵌入方式和上面提到的其他条件嵌入方式一致，让模型能够更好地学习到“多尺度特征”。

与此同时，SDXL在多尺度微调阶段依然使用图像裁剪参数条件化策略，进一步增强SDXL对图像裁剪的敏感性。

在完成了多尺度微调后，SDXL就可以进行不同Aspect Ratio的图像生成了，不过官方推荐生成尺寸默认为1024x1024。

【四】使用Offset Noise

在SDXL进行微调时，使用了Offset Noise操作，能够让SDXL生成的图像有更高的色彩自由度（纯黑或者纯白背景的图像）。SD 1.x和SD 2.x一般只能生成中等亮度的图片，即生成平均值相对接近 0.5 的图像（全黑图像为 0，全白图像为 1），之所以会出现这个问题，是因为SD系列模型训练和推理过程的不一致造成的。

SD模型在训练中进行noise scheduler流程并不能将图像完全变成随机高斯噪声，但是推理过程中，SD模型是从一个随机高斯噪声开始生成的，因此就会存在训练与推理的噪声处理过程不一致。

Offset Noise操作是解决这个问题的一种直观并且有效的方法，我们只需要在SD模型的微调训练时，把额外从高斯分布中采样的偏置噪声引入图片添加噪声的过程中，这样就对图像的色彩均值造成了破坏，从而提高了SDXL生成图像的“泛化性能”。

左边未使用Offset Noise，右边使用了Offset Noise

具体的Offset Noise代码如下所示：

def apply_noise_offset(latents, noise, noise_offset, adaptive_noise_scale):
    if noise_offset is None:
        return noise
    if adaptive_noise_scale is not None:
        # latent shape: (batch_size, channels, height, width)
        # abs mean value for each channel
        latent_mean = torch.abs(latents.mean(dim=(2, 3), keepdim=True))

        # multiply adaptive noise scale to the mean value and add it to the noise offset
        noise_offset = noise_offset + adaptive_noise_scale * latent_mean
        noise_offset = torch.clamp(noise_offset, 0.0, None)  # in case of adaptive noise scale is negative

    noise = noise + noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=latents.device)
    return noise

上述代码中的noise_offset默认是采用0.1，SDXL在官方的训练中采用的是0.05。在后面的SDXL训练教程章节中，我们采用的是0.0357，大家可按照实际训练效果调整noise_offset值。

【五】SDXL的条件注入与训练细节

上面我们已经详细讲解了SDXL的四个额外的条件信息注入（pooled text embedding，图像尺寸条件，图像裁剪参数条件和图像多尺寸条件），其中三个图像条件可以像Timestep一样采用傅立叶编码得到Embedding特征，然后再和pooled text embedding特征concat，得到维度为2816的embeddings特征。

接着再将这个embeddings特征通过两个Linear层映射到和Time Embedding一样的维度空间，然后加（add）到Time Embedding上即可作为SDXL U-Net的条件输入，上述流程的具体代码实现如下所示：

import math
from einops import rearrange
import torch

batch_size = 64
# channel dimension of pooled output of text encoder (s)
pooled_dim = 1280
adm_in_channels = 2816
time_embed_dim = 1280

# 生成Timestep Embeddings
def fourier_embedding(timesteps, outdim=256, max_period=10000):
    """
    Classical sinusoidal timestep embedding
    as commonly used in diffusion models
    : param inputs : batch of integer scalars shape [b ,]
    : param outdim : embedding dimension
    : param max_period : max freq added
    : return : batch of embeddings of shape [b, outdim ]
    """
    half = outdim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding

def cat_along_channel_dim(x: torch.Tensor,) -> torch.Tensor:
    if x.ndim == 1:
        x = x[... , None]
    assert x.ndim == 2
    b, d_in = x.shape
    x = rearrange(x, "b din -> (b din)")
    # fourier fn adds additional dimension
    emb = fourier_embedding(x)
    d_f = emb.shape[-1]
    emb = rearrange(emb, "(b din) df -> b (din df)", b=b, din=d_in, df=d_f)
    return emb

# 将SDXL的四个额外条件注入进行concat操作
def concat_embeddings(
    # batch of size and crop conditioning cf. Sec. 3.2
    c_size: torch.Tensor,
    c_crop: torch.Tensor,
    # batch of target size conditioning cf. Sec. 3.3
    c_tgt_size: torch.Tensor ,
    # final output of text encoders after pooling cf. Sec . 3.1
    c_pooled_txt: torch.Tensor,) -> torch.Tensor:
    # fourier feature for size conditioning
    c_size_emb = cat_along_channel_dim(c_size)
    # fourier feature for size conditioning
    c_crop_emb = cat_along_channel_dim(c_crop)
    # fourier feature for size conditioning
    c_tgt_size_emb = cat_along_channel_dim(c_tgt_size)
    return torch.cat([c_pooled_txt, c_size_emb, c_crop_emb, c_tgt_size_emb], dim=1)

# the concatenated output is mapped to the same
# channel dimension than the noise level conditioning
# and added to that conditioning before being fed to the unet
adm_proj = torch.nn.Sequential(
    torch.nn.Linear(adm_in_channels, time_embed_dim),
    torch.nn.SiLU(),
    torch.nn.Linear(time_embed_dim, time_embed_dim)
)

# simulating c_size and c_crop as in Sec. 3.2
c_size = torch.zeros((batch_size, 2)).long()
c_crop = torch.zeros((batch_size, 2)).long ()
# simulating c_tgt_size and pooled text encoder output as in Sec. 3.3
c_tgt_size = torch.zeros((batch_size, 2)).long()
c_pooled = torch.zeros((batch_size, pooled_dim)).long()
 
# get concatenated embedding
c_concat = concat_embeddings(c_size, c_crop, c_tgt_size, c_pooled)
# mapped to the same channel dimension with time_emb
adm_emb = adm_proj(c_concat)
print("c_size:",c_size.shape)
print("c_crop:",c_crop.shape)
print("c_tgt_size:",c_tgt_size.shape)
print("c_pooled:",c_pooled.shape)
print("c_concat:",c_concat.shape)
print("adm_emb:",adm_emb.shape)

---------------- 运行结果 ----------------
c_size: torch.Size([64, 2])
c_crop: torch.Size([64, 2])
c_tgt_size: torch.Size([64, 2])
c_pooled: torch.Size([64, 1280])
c_concat: torch.Size([64, 2816])
adm_emb: torch.Size([64, 1280])

可以看到，上面的代码流程已经清晰的展示了SDXL进行额外条件注入的全部流程。

讲到这里，SDXL在架构上的优化和训练技巧上的优化都已经介绍好了，最后我们在介绍一下SDXL在训练中的配置。和SD 1.x系列一样，SDXL在训练时采用了1000步的DDPM和相同的noise scheduler，同时依旧采用基于预测noise的损失函数，和SD 1.x系列一致：

L_{SDXL}=\mathbb{E}_{\mathbf{x}_{0},\mathbf{\epsilon}\sim \mathcal{N}(\mathbf{0}, \mathbf{I}), t}\Big[ \| \mathbf{\epsilon}- \mathbf{\epsilon}_\theta\big(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\mathbf{\epsilon}, t, \mathbf{c}\big)\|^2\Big]\\

这里的\mathbf{c}为Text Embeddings。

3. 从0到1搭建使用Stable Diffusion XL进行AI绘画（全网最详细讲解）

目前能够加载Stable Diffusion XL模型并进行图像生成的主流AI绘画框架有四种：

diffusers框架
Stable Diffusion WebUI框架
ComfyUI框架
SD.Next框架

为了方便大家使用主流AI绘画框架，Rocky这里也总结汇总了相关的资源，方便大家直接部署使用：

Stable Diffusion WebUI资源包可以关注公众号WeThinkIn，后台回复“WebUI资源”获取。
ComfyUI的500+高质量工作流资源包可以关注公众号WeThinkIn，并回复“ComfyUI”获取。
SD.Next资源包可以关注公众号WeThinkIn，后台回复“SD.Next资源”获取。

接下来，为了让大家能够从0到1搭建使用Stable Diffusion XL这个当前性能优异的AI绘画大模型，Rocky将详细的讲解如何用这四个框架构建Stable Diffusion XL推理流程。那么，跟随着Rocky的脚步，让我们开始吧。

3.1 零基础使用diffusers搭建Stable Diffusion XL推理流程

每次SDXL系列技术在更新迭代时，diffusers库一般都是最先原生支持其功能的，所以在diffusers中能够非常高效的构建Stable Diffusion XL推理流程。但是由于diffusers目前没有现成的可视化界面，Rocky将在Jupyter Notebook中搭建完整的Stable Diffusion XL推理工作流，让大家能够快速的掌握。

首先，我们需要安装diffusers库，并确保diffusers的版本 >= 0.18.0，我们只需要在命令行中输入以下命令进行安装即可：

# 命令中加入：-i https://pypi.tuna.tsinghua.edu.cn/simple some-package 表示使用清华源下载依赖包，速度非常快！
pip install diffusers --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple some-package

显示如下log表示安装成功：
Successfully installed diffusers-0.18.2 huggingface-hub-0.16.4

接着，我们继续安装其他的依赖库：

pip install transformers==4.27.0 accelerate==0.12.0 safetensors==0.2.7 invisible_watermark -i https://pypi.tuna.tsinghua.edu.cn/simple some-package

显示如下log表示安装成功：
Successfully installed transformers-4.27.0 accelerate==0.12.0 safetensors==0.2.7 invisible_watermark-0.2.0

（注意：想要在diffusers中以fp16的精度加载Stable Diffusion XL模型，必须满足transformers库的版本>=4.27.0）

完成了上述依赖库的安装，我们就可以搭建Stable Diffusion XL模型的完整工作流了。

我们先单独使用Stable Diffusion XL中的Base模型来生成图像：

# 加载diffusers和torch依赖库
from diffusers import DiffusionPipeline
import torch

# 加载Stable Diffusion XL Base模型（stable-diffusion-xl-base-1.0或stable-diffusion-xl-base-0.9）
pipe = DiffusionPipeline.from_pretrained("/本地路径/stable-diffusion-xl-base-1.0",torch_dtype=torch.float16, variant="fp16")
# "/本地路径/stable-diffusion-xl-base-1.0"表示我们需要加载的Stable Diffusion XL Base模型路径
# 大家可以关注Rocky的公众号WeThinkIn，后台回复：SDXL模型，即可获得SDXL模型权重资源链接
# "fp16"代表启动fp16精度。比起fp32，fp16可以使模型显存占用减半

# 使用GPU进行Pipeline的推理
pipe.to("cuda")

# 输入提示词
prompt = "Watercolor painting of a desert landscape, with sand dunes, mountains, and a blazing sun, soft and delicate brushstrokes, warm and vibrant colors"

# 输入负向提示词，表示我们不想要生成的特征
negative_prompt = "(EasyNegative),(watermark), (signature), (sketch by bad-artist), (signature), (worst quality), (low quality), (bad anatomy), NSFW, nude, (normal quality)"

# 设置seed，可以固定生成图像中的构图
seed = torch.Generator("cuda").manual_seed(42)

# SDXL Base Pipeline进行推理
image = pipe(prompt, negative_prompt=negative_prompt,generator=seed).images[0]
# Pipeline生成的images包含在一个list中：[<PIL.Image.Image image mode=RGB size=1024x1024>]
#所以需要使用images[0]来获取list中的PIL图像

# 保存生成图像
image.save("SDXL-Base.png")

完成上面的整个代码流程，我们可以生成一张水彩风格的沙漠风景画，如果大家按照Rocky的参数进行操作，应该能确保生成下面的图片：

SDXL Base模型生成的图片

接着，我们将SDXL Base模型和SDXL Refiner模型级联来生成图像：

from diffusers import DiffusionPipeline
import torch

# 下面的五行代码不变
pipe = DiffusionPipeline.from_pretrained("/本地路径/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16")

pipe.to("cuda")

prompt = "Watercolor painting of a desert landscape, with sand dunes, mountains, and a blazing sun, soft and delicate brushstrokes, warm and vibrant colors"

negative_prompt = "(EasyNegative),(watermark), (signature), (sketch by bad-artist), (signature), (worst quality), (low quality), (bad anatomy), NSFW, nude, (normal quality)"

seed = torch.Generator("cuda").manual_seed(42)

# 运行SDXL Base模型的Pipeline，设置输出格式为output_type="latent"
image = pipe(prompt=prompt, negative_prompt=negative_prompt, generator=seed, output_type="latent").images

# 加载Stable Diffusion XL Refiner模型（stable-diffusion-xl-refiner-1.0或stable-diffusion-xl-refiner-0.9）
pipe = DiffusionPipeline.from_pretrained("/本地路径/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16")
# "本地路径/stable-diffusion-xl-refiner-1.0"表示我们需要加载的Stable Diffusion XL Refiner模型，
# 大家可以关注Rocky的公众号WeThinkIn，后台回复：SDXL模型，即可获得SDXL模型权重资源链接

pipe.to("cuda")

# SDXL Refiner Pipeline进行推理
images = pipe(prompt=prompt, negative_prompt=negative_prompt, generator=seed, image=image).images

# 保存生成图像
images[0].save("SDXL-Base-Refiner.png")

完成了上述的代码流程，我们再来看看这次Base模型和Refiner模型级联生成的图片：

SDXL Base模型+Refiner模型级联生成的图片

为了更加直观的对比，我们将刚才生成的两张图片放在一起对比：

我们可以清楚的看到，使用了Refiner模型之后，生成图片的整体质量和细节有比较大的增强改善，构图色彩更加柔和。

当然的，我们也可以单独使用SDXL Refiner模型对图片的质量进行优化提升（img2img任务）：

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained("/本地路径/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16")

pipe = pipe.to("cuda")

image_path = "/本地路径/test.png"

init_image = load_image(image_path).convert("RGB")

prompt = "Watercolor painting of a desert landscape, with sand dunes, mountains, and a blazing sun, soft and delicate brushstrokes, warm and vibrant colors"

negative_prompt = "(EasyNegative),(watermark), (signature), (sketch by bad-artist), (signature), (worst quality), (low quality), (bad anatomy), NSFW, nude, (normal quality)"

seed = torch.Generator("cuda").manual_seed(42)

image = pipe(prompt, negative_prompt=negative_prompt, generator=seed, image=init_image).images[0]

image.save("SDXL-refiner.png")

Rocky这里是使用了未来机甲风格的图片进行测试对比，可以从下图看到，Refiner模型优化图像质量的效果还是非常明显的，图像毛刺明显消除，整体画面更加自然柔和，细节特征也有较好的补充和重建。

SDXL Refiner模型图生图效果

虽然diffusers库是原生支持SDXL模型的，但是在开源社区中流行使用safetensors格式的SDXL模型，所以我们想用diffusers库运行开源社区的很多SDXL模型时，需要首先将其转成diffusers格式。Rocky在这里也总结了一套SDXL模型的格式转换教程，方便大家快速转换格式，使用diffusers库运行模型。主要流程如下所示：

pip install diffusers==0.26.3 transformers==4.38.1 accelerate==0.27.2

git clone https://github.com/huggingface/diffusers.git

cd diffusers/scripts

python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path /本地路径/safetensors格式模型 --dump_path /本地路径/转换后diffusers格式模型的保存路径 --from_safetensors

成功运行上述代码后，我们可以看到一个包含scheduler、vae、unet、text_encoder、tokenizer、text_encoder_2、tokenizer_2文件夹以及model_index.json文件的diffusers格式的SDXL模型。

3.2 零基础使用Stable Diffusion WebUI搭建Stable Diffusion XL推理流程

目前Stable Diffusion WebUI已经全面支持Stable Diffusion XL中的Base模型和Refiner模型。

Stable Diffusion WebUI是AI绘画领域最为流行的框架，其生态极其繁荣，非常多的上下游插件能够与Stable Diffusion WebUI一起完成诸如AI视频生成，AI证件照生成等工作流，可玩性非常强。

接下来，咱们就使用这个流行框架搭建Stable Diffusion XL推理流程吧。

首先，我们需要下载安装Stable Diffusion WebUI框架，我们只需要在命令行输入如下代码即可：

git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git

安装好后，我们可以看到本地的stable-diffusion-webui文件夹。

下面我们需要安装其依赖库，我们进入Stable Diffusion WebUI文件夹，并进行以下操作：

cd stable-diffusion-webui #进入下载好的automatic文件夹中
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple some-package

和SD.Next的配置流程类似，我们还需要配置Stable Diffusion WebUI的repositories插件，我们需要运行下面的代码：

sh webui.sh

#主要依赖包括：BLIP CodeFormer generative-models k-diffusion stable-diffusion-stability-ai taming-transformers

如果发现repositories插件下载速度较慢，出现很多报错，don't worry，大家可以直接使用Rocky已经配置好的资源包，可以快速启动与Stable Diffusion XL兼容的Stable Diffusion WebUI框架。Stable Diffusion WebUI资源包可以关注公众号WeThinkIn，后台回复“WebUI资源”获取。

在完成了依赖库和repositories插件的安装后，我们就可以配置模型了，我们将Stable Diffusion XL模型放到/stable-diffusion-webui/models/Stable-diffusion/路径下。这样以来，等我们开启可视化界面后，就可以选择Stable Diffusion XL模型用于推理生成图片了。

完成上述的步骤后，我们可以启动Stable Diffusion WebUI了！我们到/stable-diffusion-webui/路径下，运行launch.py即可：

python launch.py --listen --port 8888

运行完成后，可以看到命令行中出现的log：

To see the GUI go to: http://0.0.0.0:8888

我们将http://0.0.0.0:8888输入到我们本地的网页中，即可打开如下图所示的Stable Diffusion WebUI可视化界面，愉快的使用Stable Diffusion XL模型进行AI绘画了。

Stable Diffusion WebUI可视化界面

进入Stable Diffusion WebUI可视化界面后，我们可以在红色框中选择SDXL模型，然后在黄色框中输入我们的Prompt和负向提示词，同时在绿色框中设置我们想要生成的图像分辨率（推荐设置成1024x1024），然后我们就可以点击Generate按钮，进行AI绘画了。

等待片刻后，图像就生成好了，并展示在界面的右下角，同时也会保存到/stable-diffusion-webui/outputs/txt2img-images/路径下，大家可以到对应路径下查看。

3.3 零基础使用ComfyUI搭建Stable Diffusion XL推理流程

ComfyUI是一个基于节点式的Stable Diffusion AI绘画工具。和Stable Diffusion WebUI相比，ComfyUI通过将Stable Diffusion模型生成推理的pipeline拆分成独立的节点，实现了更加精准的工作流定制和清晰的可复现性。

同时其完善的模型加载和图片生成机制，让其能够在2080Ti显卡上构建Stable Diffusion XL的工作流，并能生成1024x1024分辨率的图片，如此算力友好，可谓是初学者的福音。

目前ComfyUI已经能够兼容Stable Diffusion XL的Base模型和Refiner模型，下面两张图分别是Rocky使用ComfyUI来加载Stable Diffusion XL Base模型和Stable Diffusion XL Base + Refiner模型并生成图片的完整Pipeline：

ComfyUI加载Stable Diffusion XL Base模型
ComfyUI加载Stable Diffusion XL Base + Refiner模型

大家如果看了感觉复杂，不用担心，Rocky已经为大家保存了这两个工作流，大家只需关注Rocky的公众号WeThinkIn，并回复“ComfyUI”，就能获取这两个工作流以及文生图，图生图，图像Inpainting，ControlNet以及图像超分在内的所有Stable Diffusion经典工作流json文件，大家只需在ComfyUI界面右侧点击Load按钮选择对应的json文件，即可加载对应的工作流，开始愉快的AI绘画之旅。

话说回来，下面Rocky将带着大家一步一步使用ComfyUI搭建Stable Diffusion XL推理流程，从而实现上面两张图的生成过程。

首先，我们需要安装ComfyUI框架，这一步非常简单，在命令行输入如下代码即可：

git clone https://github.com/comfyanonymous/ComfyUI.git

安装好后，我们可以看到本地的ComfyUI文件夹。

ComfyUI框架安装到本地后，我们需要安装其依赖库，我们只需以下操作：

cd ComfyUI #进入下载好的ComfyUI文件夹中
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple some-package

完成这些配置工作后，我们就可以配置模型了，我们将Stable Diffusion XL模型放到ComfyUI/models/checkpoints/路径下。这样以来，等我们开启可视化界面后，就可以选择Stable Diffusion XL模型进行AI绘画了。

接下来，我们就可以启动ComfyUI了！我们到ComfyUI/路径下，运行main.py即可：

python main.py --listen --port 8888

运行完成后，可以看到命令行中出现的log：

To see the GUI go to: http://0.0.0.0:8888

我们将http://0.0.0.0:8888输入到我们本地的网页中，即可打开如上图所示的ComfyUI可视化界面，愉快的使用Stable Diffusion XL模型生成我们想要的图片了。

接下来就是ComfyUI的节点式模块讲解了，首先是只加载Base模型的情况：

Stable Diffusion XL Base模型使用的注释

Rocky已经进行了比较详细的注释，首先大家可以在红框中选择我们的模型（Stable Diffusion XL Base），接着填入Prompt和负向Prompt，并且配置生成推理过程的参数（迭代次数，CFG，Seed等），然后在绿色框中设置好生成图片的分辨率，然后在紫色框中点击Queue Prompt按钮，整个推理过程就开始了。等整个推理过程完成之后，生成的图片会在图中黄色箭头所指的地方进行展示，并且会同步将生成图片保存到本地的ComfyUI/output/路径下。

完成了Stable Diffusion Base模型的推理流程，我们再来看看Base+Refiner模型的推理流程如何搭建：

Stable Diffusion XL Base+Refiner模型使用的注释

和Base模型的构建十分相似，首先大家可以在红框中选择我们的Refiner模型（Stable Diffusion XL Refiner），Refiner模型使用的Prompt和负向Prompt与Base模型一致，并且配置生成推理过程的参数（迭代次数，CFG，Seed等），绿色箭头表示将Base模型输出的Latent特征作为Refiner模型的输入，然后在蓝色框中点击Queue Prompt按钮，整个Refiner精修过程就开始了。等整个推理过程完成之后，生成的图片会在图中紫色箭头所指的地方进行展示，并且会同步将生成图片保存到本地的ComfyUI/output/路径下。

到此为止，Rocky已经详细讲解了如何使用ComfyUI来搭建Stable Diffusion XL模型进行AI绘画，大家可以按照Rocky的步骤进行尝试。

3.4 零基础使用SD.Next搭建Stable Diffusion XL推理流程

SD.Next原本是Stable Diffusion WebUI的一个分支，再经过不断的迭代优化后，最终成为了一个独立版本。

SD.Next与Stable Diffusion WebUI相比，包含了更多的高级功能，也兼容Stable Diffusion, Stable Diffusion XL, Kandinsky, DeepFloyd IF等模型结构，是一个功能十分强大的AI绘画框架。

那么我们马上开始SD.Next的搭建与使用吧。

首先，我们需要安装SD.Next框架，这一步非常简单，在命令行输入如下代码即可：

git clone https://github.com/vladmandic/automatic

安装好后，我们可以看到本地的automatic文件夹。

SD.Next框架安装到本地后，我们需要安装其依赖库，我们只需以下操作：

cd automatic #进入下载好的automatic文件夹中
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple some-package

除了安装依赖库之外，还需要配置SD.Next所需的repositories插件，我们需要运行一下代码：

cd automatic #进入下载好的automatic文件夹中
python installer.py

如果发现extensions插件下载速度较慢，出现很多报错，大家可以直接使用Rocky已经配置好的资源包，可以快速启动SD.Next框架。SD.Next资源包可以关注公众号WeThinkIn，后台回复“SD.Next资源”获取。

在完成了依赖库和repositories插件的安装后，我们就可以配置模型了，我们将Stable Diffusion XL模型放到/automatic/models/Stable-diffusion/路径下。这样以来，等我们开启可视化界面后，就可以选择Stable Diffusion XL模型用于推理生成图片了。

完成上述的步骤后，我们可以启动SD.Next了！我们到/automatic/路径下，运行launch.py即可：

python launch.py --listen --port 8888

运行完成后，可以看到命令行中出现的log：

To see the GUI go to: http://0.0.0.0:8888

我们将http://0.0.0.0:8888输入到我们本地的网页中，即可打开如下图所示的SD.Next可视化界面，愉快的使用Stable Diffusion XL模型进行AI绘画了。

automatic可视化界面

进入SD.Next可视化界面后，我们可以在红色框中选择模型，然后需要修改Settings中的配置，来让SD.Next能够加载Stable Diffusion XL模型。

我们点击上图蓝色框中的Settings，进入Settings配置界面：

automatic框架配置修改【1】
automatic框架配置修改【2】

从上面图示中可以看到，我们需要做的修改是将Settings -> Stable Diffusion -> Stable Diffusion backend设置为diffusers，并在Stable Diffusion refiner栏中选择Refiner模型。

然后我们需要将Settings -> Diffusers Settings-> Select diffuser pipeline when loading from safetensors栏设置为Stable Diffusion XL。

完成了上述的配置修改后，我们就可以使用SD.Next加载Stable Diffusion XL进行AI绘画了！

3.5 SDXL生成图像示例

示例一：未来主义的城市风格

Prompt：Stunning sunset over a futuristic city, with towering skyscrapers and flying vehicles, golden hour lighting and dramatic clouds, high detail, moody atmosphere

Negative Prompt：(EasyNegative),(watermark), (signature), (sketch by bad-artist), (signature), (worst quality), (low quality), (bad anatomy), NSFW, nude, (normal quality)

Stable Diffusion XL Base+Refiner生成结果：

Stable Diffusion XL生成结果：未来主义的城市风格

示例二：天堂海滩风格

Prompt：Serene beach scene with crystal clear water and white sand, tropical palm trees swaying in the breeze, perfect paradise, seascape

Negative Prompt：(EasyNegative),(watermark), (signature), (sketch by bad-artist), (signature), (worst quality), (low quality), (bad anatomy), NSFW, nude, (normal quality)

Stable Diffusion XL Base+Refiner生成结果：

Stable Diffusion XL生成结果：天堂海滩风格

示例三：未来机甲风格

Prompt：Giant robots fighting in a futuristic city, with buildings falling and explosions all around, intense, fast-paced, dramatic, stylized, futuristic

Negative Prompt：(EasyNegative),(watermark), (signature), (sketch by bad-artist), (signature), (worst quality), (low quality), (bad anatomy), NSFW, nude, (normal quality)

Stable Diffusion XL Base+Refiner生成结果：

Stable Diffusion XL生成结果：未来机甲风格

示例四：马斯克风格

Prompt：Elon Musk standing in a workroom, in the style of industrial machinery aesthetics, deutscher werkbund, uniformly staged images, soviet, light indigo and dark bronze, new american color photography, detailed facial features

Negative Prompt：(EasyNegative),(watermark), (signature), (sketch by bad-artist), (signature), (worst quality), (low quality), (bad anatomy), NSFW, nude, (normal quality)

Stable Diffusion XL Base+Refiner生成结果：

Stable Diffusion XL生成结果：马斯克风格
4. 从0到1上手使用Stable Diffusion XL训练自己的AI绘画模型（全网最详细讲解）
4.0 SDXL训练资源分享
SDXL训练脚本：Rocky整理优化过的SDXL完整训练资源SDXL-Train项目，大家只用在SDXL-Train中就可以完成SDXL的模型训练工作，方便大家上手实操。SDXL-Train项目资源包可以通过关注公众号WeThinkIn，后台回复“SDXL-Train”获取。
本文中的SDXL微调训练数据集：二次元人物数据集，大家可以关注公众号WeThinkIn，后台回复“二次元人物数据集”获取。
本文中的SDXL微调训练底模型：WeThinkIn_SDXL_二次元模型，大家可以关注Rocky的公众号WeThinkIn，后台回复“SDXL_二次元模型”获取模型资源链接。
本文中的SDXL LoRA训练数据集：猫女数据集，大家可以关注公众号WeThinkIn，后台回复“猫女数据集”获取。
本文中的SDXL LoRA训练底模型：WeThinkIn_SDXL_真人模型，大家可以关注Rocky的公众号WeThinkIn，后台回复“SDXL_真人模型”获取模型资源链接。
4.1 SDXL训练脉络初识

Stable Diffusion系列模型的训练过程主要分成以下几个步骤，Stable Diffusion XL也不例外：

训练集制作：数据质量评估，标签梳理，数据清洗，数据标注，标签清洗，数据增强等。
训练文件配置：预训练模型选择，训练环境配置，训练步数设置，其他超参数设置等。
模型训练：运行SDXL模型/LoRA模型训练脚本，使用TensorBoard监控模型训练等。
模型测试：将训练好的自训练SDXL模型/LoRA模型用于效果评估与消融实验。

讲完SDXL训练的方法论，Rocky再向大家推荐一些SDXL训练资源：

https://github.com/qaneel/kohya-trainer（本文中主要的训练工程）
https://github.com/Linaqruf/kohya-trainer（此项目中的kohya-trainer-XL.ipynb和kohya-LoRA-trainer-XL.ipynb可以用于制作数据集和配置训练参数）
https://github.com/bmaltais/kohya_ss（此项目可以GUI可视化训练）
Rocky整理优化过的SDXL完整训练资源SDXL-Train项目，大家只用在SDXL-Train中就可以完成SDXL的模型训练工作，方便大家上手实操。SDXL-Train项目资源包可以通过关注公众号WeThinkIn，后台回复“SDXL-Train”获取。

目前我们对SDXL的训练流程与所需资源有了初步的了解，接下来，就让我们跟随着Rocky的脚步，从0到1使用SDXL模型和训练资源一起训练自己的SDXL绘画模型与LoRA绘画模型吧！

4.2 配置训练环境与训练文件

首先，我们需要下载两个训练资源，只需在命令行输入下面的代码即可：

git clone https://github.com/qaneel/kohya-trainer.git

git clone https://github.com/Linaqruf/kohya-trainer.git

qaneel/kohya-trainer项目包含了Stable Diffusion XL的核心训练脚本，而我们需要用Linaqruf/kohya-trainer项目中的kohya-trainer-XL.ipynb和kohya-LoRA-trainer-XL.ipynb文件来生成数据集制作脚本和训练参数配置脚本。

我们打开Linaqruf/kohya-trainer项目可以看到，里面包含了两个SDXL的.ipynb文件：

Linaqruf/kohya-trainer项目

接着我们再打开qaneel/kohya-trainer项目，里面包含的两个python文件就是我们后续的训练主脚本：

qaneel/kohya-trainer项目

正常情况下，我们需要运行Linaqruf/kohya-trainer项目中两个SDXL的.ipynb文件的内容，生成训练数据处理脚本（数据标注，数据预处理，数据Latent特征提取，数据分桶（make buckets）等）和训练参数配置文件。

我们使用数据处理脚本完成训练集的制作，然后再运行qaneel/kohya-trainer项目的训练脚本，同时读取训练参数配置文件，为SDXL模型的训练过程配置超参数。

完成上面一整套流程，SDXL模型的训练流程就算跑通了。但是由于Linaqruf/kohya-trainer项目中的两个.ipynb文件内容较为复杂，整个流程比较繁锁，对新手非常不友好，并且想要完成一整套训练流程，需要我们一直在两个项目之前切换，非常不方便。

所以Rocky这边帮大家对两个项目进行了整合归纳，总结了简单易上手的SDXL模型以及相应LoRA模型的训练流程，制作成SDXL完整训练资源SDXL-Train项目，大家只用在SDXL-Train中就可以完成SDXL的模型训练工作，方便大家上手实操。

SDXL-Train项目资源包可以通过关注公众号WeThinkIn，后台回复“SDXL-Train”获取。

下面是SDXL-Train项目中的主要内容，大家可以看到SDXL的数据处理脚本与训练脚本都已包含在内：

SDXL-Train：Stable Diffusion XL完整训练资源

我们首先进入SDXL-Train项目中，安装SDXL训练所需的依赖库，我们只需在命令行输入以下命令即可：

cd SDXL-Train

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple some-package

# accelerate库的版本需要重新检查一遍，需要安装accelerate==0.16.0版本才能兼容SDXL的训练
pip install accelerate==0.16.0 -i https://pypi.tuna.tsinghua.edu.cn/simple some-package

# Rocky在这里推荐大家安装2.0.1版本的Pytorch，能够兼容SDXL训练的全部流程
pip install torch==2.0.1 -i https://pypi.tuna.tsinghua.edu.cn/simple some-package

在完成上述的依赖库安装后，我们需要确认一下目前的Python、PyTroch、CUDA以及cuDNN的版本是否兼容，我们只需要在命令行输入以下命令即可：

# Python版本推荐3.8或者3.9，两个版本皆可
>>> python
Python 3.9
# 加载PyTroch
>>> import torch
# 查看PyTorch版本
>>> print(torch.__version__)
2.0.1+cu117
# 查看CUDA版本
>>> print(torch.version.cuda)
11.7
# 查看cuDNN版本
>>> print(torch.backends.cudnn.version())
8500
# 查看PyTroch、CUDA以及cuDNN的版本是否兼容，True代表兼容
>>> print(torch.cuda.is_available())
True

如果大家在本地自己验证的时候和Rocky上述的版本一致，说明训练环境已经全部兼容！

安装和验证好所有SDXL训练所需的依赖库后，我们还需要设置一下SDXL的训练环境，我们主要是用accelerate库的能力，accelerate库能让PyTorch的训练和推理变得更加高效简洁。我们只需在命令行输入以下命令，并对每个设置逐一进行填写即可：

# 输入以下命令，开始对每个设置进行填写
accelerate config

# 开始进行训练环境参数的配置
In which compute environment are you running? # 选择This machine，即本机
This machine

# 选择单卡或是多卡训练，如果是多卡，则选择multi-GPU，若是单卡，则选择No distributed training                                                                                                               
Which type of machine are you using?                                                                                        
multi-GPU

# 几台机器用于训练，一般选择1台。注意这里是指几台机器，不是几张GPU卡                                                                                                                  
How many different machines will you use (use more than 1 for multi-node training)? [1]: 1       

# torch dynamo，DeepSpeed，FullyShardedDataParallel，Megatron-LM等环境参数，不需要配置                          
Do you wish to optimize your script with torch dynamo?[yes/NO]: # 输入回车即可                                                           
Do you want to use DeepSpeed? [yes/NO]:  # 输入回车即可                                                                                  
Do you want to use FullyShardedDataParallel? [yes/NO]:    # 输入回车即可                                                                 
Do you want to use Megatron-LM ? [yes/NO]:       # 输入回车即可

# 选择多少张卡投入训练                                                                          
How many GPU(s) should be used for distributed training? [1]:2

# 设置投入训练的GPU卡id，如果是全部的GPU都投入训练，则输入all即可。
What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:all 

# 训练精度，可以选择fp16
Do you wish to use FP16 or BF16 (mixed precision)? 
fp16                             

# 完成配置后，配置文件default_config.yaml会保存在/root/.cache/huggingface/accelerate下                                                                                   
accelerate configuration saved at /root/.cache/huggingface/accelerate/default_config.yaml完成上述的流程后，接下来我们就可以进行SDXL训练数据的制作和训练脚本的配置流程了！

后续进行SDXL与SDXL LoRA模型训练的时候，只需要加载对应的default_config.yaml配置文件即可，具体的调用方法，本文后续的章节会进行详细讲解。

还有一点需要注意的是，我们进行SDXL模型的训练时，SDXL的CLIP Text Encoder会调用CLIP-ViT-bigG-14-laion2B-39B-b160k和clip-vit-large-patch14两个配置文件。一般情况下SDXL模型会从huggingface上将配置文件下载到~/.cache/huggingface/目录中，但是由于网络原因很可能会下载失败，从而导致训练的失败。

所以为了让大家能更方便的训练SDXL模型，Rocky已经将CLIP-ViT-bigG-14-laion2B-39B-b160k和clip-vit-large-patch14这两个配置文件放入SDXL-Train项目的utils_json文件夹中，并且已经为大家配置好依赖路径，大家只要使用SDXL-Train项目便无需做任何修改。如果大家想要修改CLIP-ViT-bigG-14-laion2B-39B-b160k和clip-vit-large-patch14这两个依赖文件夹的调用路径，大家可以找到SDXL-Train/library/sdxl_train_util.py脚本中的第122行，将"utils_json/"部分修改成自己的本地自定义路径比如“/本地路径/utils_json/”即可。

完成上述的流程后，接下来我们就可以进行SDXL训练数据的制作和训练脚本的配置流程了！

4.3 SDXL训练数据集制作

首先，我们需要对数据集进行清洗，和传统深度学习时代一样，数据清洗工作依然占据了AIGC时代模型训练70%-80%左右的时间。

并且这个过程必不可少，因为数据质量决定了机器学习性能的上限，而算法和模型只是在不断逼近这个上限而已。

我们需要筛除分辨率较低、质量较差（比如说768*768分辨率的图片< 100kb）、存在破损以及和任务目标无关的数据，接着再去除数据里面可能包含的水印，干扰文字等污染特征。

同时，我们需要优先保证数据集的质量，在有质量的基础上再去增加数据集的数量与丰富度。

为了满足AI绘画生成图片时的尺寸适应度，我们可以对数据进行多尺度的增强，比如进行1:1，1:2，2:1，1:3，3:4，4:3，9:16，16:9等等尺寸的裁剪与缩放操作。但是切记不能在多尺度增强的时候将图片的主体特征裁剪掉（比如人脸，建筑等）。

完成上述的数据筛选与清洗工作后，我们就可以开始进行数据标注了。

数据标注可以分为自动标注和手动标注。自动标注主要依赖像BLIP（img2caption）和Waifu Diffusion 1.4（img2tag）等能够进行图片生成标签的模型，手动标注则依赖标注人员。

（1）使用BLIP自动标注caption

我们先用BLIP对数据进行自动标注，BLIP输出的是自然语言标签，我们进入到SDXL-Train/finetune/路径下，运行以下代码即可获得自然语言标签（caption标签）：

cd SDXL-Train/finetune/
python make_captions.py "/数据路径" --caption_weights “/本地BLIP模型路径” --batch_size=8 --beam_search --min_length=5 --max_length=75 --debug --caption_extension=".caption" --max_data_loader_n_workers=2

注意：在使用BLIP进行数据标注时需要依赖bert-base-uncased模型，Rocky这边已经帮大家配置好了，大家只要使用SDXL-Train项目便无需做任何修改。同时，如果大家想要修改bert-base-uncased模型的调用路径，可以找到SDXL-Train/finetune/blip/blip.py脚本的第189行，将“../bert-base-uncased”部分修改成自己的本地自定义路径比如“/本地路径/bert-base-uncased”即可。

从上面的代码可以看到，我们第一个传入的参数是训练集的路径。下面Rocky再一一向大家介绍一下其余参数的意义：

--caption_weights：表示加载的本地BLIP模型，如果不传入本地模型路径，则默认从云端下载BLIP模型。

--batch_size：表示每次传入BLIP模型进行前向处理的数据数量。

--beam_search：设置为波束搜索，默认Nucleus采样。

--min_length：设置caption标签的最短长度。

--max_length：设置caption标签的最长长度。

--debug：如果设置，将会在BLIP前向处理过程中，打印所有的图片路径与caption标签内容，以供检查。

--caption_extension：设置caption标签的扩展名，一般为".caption"。

--max_data_loader_n_workers：设置大于等于2，加速数据处理。

讲完了上述的运行代码以及相关参数，下面Rocky再举一个例子， 让大家能够更加直观的感受到BLIP处理数据生成caption标签的过程：

使用BLIP进行数据自动标注的形象例子

上图是单个图像的标注示例，整个数据集的标注流程也是同理的。等整个数据集的标注后，Stable Diffusion XL训练所需的caption标注就完成了。

（2）使用Waifu Diffusion 1.4自动标注tag

接下来我们可以使用Waifu Diffusion 1.4进行自动标注，Waifu Diffusion 1.4输出的是tag关键词，这里需要注意的是，调用Waifu Diffusion 1.4模型需要安装Tensorflow库，并且需要下载特定的版本（2.10.1），不然运行时会报“DNN library is not found“错误。我们只需要在命令行输入以下命令即可：

pip install tensorflow==2.10.1

完成上述的环境配置后，我们依然进入到SDXL-Trian/finetune/路径下，运行以下代码即可获得tag自动标注：

cd SDXL-Train/finetune/
python tag_images_by_wd14_tagger.py "/数据路径" --batch_size=8 --model_dir="/本地路径/wd-v1-4-moat-tagger-v2" --remove_underscore --general_threshold=0.35 --character_threshold=0.35 --caption_extension=".txt" --max_data_loader_n_workers=2 --debug --undesired_tags=""

从上面的代码可以看到，我们第一个传入的参数是训练集的路径。

--batch_size：表示每次传入Waifu Diffusion 1.4模型进行前向处理的数据数量。

--model_dir：表示加载的本地Waifu Diffusion 1.4模型路径。

--remove_underscore：如果开启，会将输出tag关键词中的下划线替换为空格。

--general_threshold：设置常规tag关键词的筛选置信度。

--character_threshold：设置人物特征tag关键词的筛选置信度。

--caption_extension：设置tag关键词标签的扩展名，一般为".txt"。

-max_data_loader_n_workers：设置大于等于2，加速数据处理。

--debug：如果设置，将会在Waifu Diffusion 1.4模型前向处理过程中，打印所有的图片路径与tag关键词标签内容，以供检查。

--undesired_tags：设置不需要输出的tag关键词。

下面Rocky依然用美女图片作为例子， 让大家能够更加直观的感受到Waifu Diffusion 1.4模型处理数据生成tag关键词标签的过程：

使用Waifu Diffusion 1.4模型进行数据自动标注的形象例子

上图是单个图像的标注示例，整个数据集的标注流程也是同理的。等整个数据集的标注后，Stable Diffusion XL训练所需的tag关键词标注就完成了。

上面Rocky是使用了Waifu Diffusion v1.4系列模型中的wd-v1-4-moat-tagger-v2模型，目前Waifu Diffusion v1.4系列模型一共有5个版本，除了刚才介绍到的wd-v1-4-moat-tagger-v2模型，还包括wd-v1-4-swinv2-tagger-v2模型、wd-v1-4-convnext-tagger-v2模型、wd-v1-4-convnextv2-tagger-v2模型以及wd-v1-4-vit-tagger-v2模型。

Rocky也分别对他们的自动标注效果进行了对比，在这里Rocky使用了一张生成的“猫女”图片，分别输入到这五个自动标注模型中，一起来看看不同版本的Waifu Diffusion v1.4模型的效果：

Waifu Diffusion v1.4系列模型不同版本的自动标注效果

从上图可以看到，在将general_threshold和character_threshold同时设置为0.5时，wd-v1-4-moat-tagger-v2模型的标注效果整体上是最好的，内容丰富且最能反应图片中的语义信息。所以在这里，Rocky也推荐大家使用wd-v1-4-moat-tagger-v2模型。

大家也可以在SDXL-Train项目的tag_models文件夹下调用这些模型，进行对比测试，感受不同系列Waifu Diffusion v1.4模型的标注效果。

（3）补充标注特殊tag

完成了caption和tag的自动标注之后，如果我们需要训练一些特殊标注的话，还可以进行手动的补充标注。

SDXL-Trian项目中也提供了对数据进行补充标注的代码，Rocky在这里将其进行提炼总结，方便大家直接使用。

大家可以直接拷贝以下的代码，并按照Rocky在代码中提供的注释进行参数修改，然后运行代码即可对数据集进行补充标注：

import os

# 设置为本地的数据集路径
train_data_dir = "/本地数据集路径"

# 设置要补充的标注类型，包括[".txt", ".caption"]
extension   = ".txt" 

# 设置要补充的特殊标注
custom_tag  = "WeThinkIn"

# 若设置sub_folder = "--all"时，将遍历所有子文件夹中的数据；默认为""。
sub_folder  = "" 

# 若append设为True，则特殊标注添加到标注文件的末尾
append      = False

# 若设置remove_tag为True，则会删除数据集中所有的已存在的特殊标注
remove_tag  = False
recursive   = False

if sub_folder == "":
    image_dir = train_data_dir
elif sub_folder == "--all":
    image_dir = train_data_dir
    recursive = True
elif sub_folder.startswith("/content"):
    image_dir = sub_folder
else:
    image_dir = os.path.join(train_data_dir, sub_folder)
    os.makedirs(image_dir, exist_ok=True)

# 读取标注文件的函数，不需要改动
def read_file(filename):
    with open(filename, "r") as f:
        contents = f.read()
    return contents

# 将特殊标注写入标注文件的函数，不需要改动
def write_file(filename, contents):
    with open(filename, "w") as f:
        f.write(contents)

# 将特殊标注批量添加到标注文件的主函数，不需要改动
def process_tags(filename, custom_tag, append, remove_tag):
    contents = read_file(filename)
    tags = [tag.strip() for tag in contents.split(',')]
    custom_tags = [tag.strip() for tag in custom_tag.split(',')]

    for custom_tag in custom_tags:
        custom_tag = custom_tag.replace("_", " ")
        if remove_tag:
            while custom_tag in tags:
                tags.remove(custom_tag)
        else:
            if custom_tag not in tags:
                if append:
                    tags.append(custom_tag)
                else:
                    tags.insert(0, custom_tag)

    contents = ', '.join(tags)
    write_file(filename, contents)


def process_directory(image_dir, tag, append, remove_tag, recursive):
    for filename in os.listdir(image_dir):
        file_path = os.path.join(image_dir, filename)

        if os.path.isdir(file_path) and recursive:
            process_directory(file_path, tag, append, remove_tag, recursive)
        elif filename.endswith(extension):
            process_tags(file_path, tag, append, remove_tag)

tag = custom_tag

if not any(
    [filename.endswith(extension) for filename in os.listdir(image_dir)]
):
    for filename in os.listdir(image_dir):
        if filename.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
            open(
                os.path.join(image_dir, filename.split(".")[0] + extension),
                "w",
            ).close()

# 但我们设置好要添加的custom_tag后，开始整个代码流程
if custom_tag:
    process_directory(image_dir, tag, append, remove_tag, recursive)

看完了上面的完整代码流程，如果大家觉得代码太复杂，don‘t worry，大家只需要复制上面的全部代码，并将train_data_dir ="/本地数据集路径"和custom_tag ="WeThinkIn"设置成自己数据集的本地路径和想要添加的特殊标注，然后运行代码即可，非常简单实用。

还是以之前的美女图片为例子，当运行完上面的代码后，可以看到txt文件中，最开头的tag为“WeThinkIn”：

手动补充增加特殊tag标签

大家注意，一般我们会将手动补充的特殊tag放在第一位，因为和caption标签不同，tags标签是有顺序的，最开始的tag权重最大，越靠后的tag权重越小。

到这里，Rocky已经详细讲解了在Stable Diffusion XL训练前，如何对数据集进行caption标注，tag标注以及补充一些关键标注的完整步骤与流程，在数据标注完毕后，接下来我们就要进入数据预处理的阶段了。

（4）训练数据预处理

首先，我们需要对刚才生成的后缀为.caption和.txt的标注文件进行整合，存储成一个json格式的文件，方便后续SDXL模型训练时调取训练数据与标注。

我们需要进入SDXL-Train项目的finetune文件夹中，运行merge_all_to_metadata.py脚本即可：

cd SDXL-Train
python ./finetune/merge_all_to_metadata.py "/本地数据路径" "/本地数据路径/meta_clean.json"

如下图所示，我们依旧使用之前的美图女片作为例子，运行完merge_all_to_metadata.py脚本后，我们在数据集路径中得到一个meta_clean.json文件，打开可以看到图片名称对应的tag和caption标注都封装在了文件中，让人一目了然，非常清晰。

meta_clean.json中封装了图片名称与对应的tag和caption标注

在整理好标注文件的基础上，接下来我们需要对数据进行分桶与保存Latent特征，并在meta_clean.json的基础上，将图片的分辨率信息也存储成json格式，并保存一个新的meta_lat.json文件。

我们需要进入SDXL-Train项目的finetune文件夹中，运行prepare_buckets_latents.py脚本即可：

cd SDXL-Train
python ./finetune/prepare_buckets_latents.py "/本地数据路径" "/本地数据路径/meta_clean.json" "/本地数据路径/meta_lat.json" "调用的SDXL模型路径" --batch_size 4 --max_resolution "1024,1024"

运行完脚本，我们即可在数据集路径中获得meta_lat.json文件，其在meta_clean.json基础上封装了图片的分辨率信息，用于SDXL训练时快速进行数据分桶。

meta_lat.json文件在meta_clean.json基础上封装了图片的分辨率信息

同时我们可以看到，美女图片的Latent特征保存为了.npz文件，用于SDXL模型训练时，快速读取数据的Latent特征，加速训练过程。

好的，到目前为止，我们已经完整的进行了SDXL训练所需的数据集制作与预处理流程。总结一下，我们在一张美女图片的基础上，一共获得了以下5个不同的训练配置文件：

meta_clean.json
meta_lat.json
自然语言标注（.caption）
关键词tag标注（.txt）
数据的Latent特征信息（.npz）
SDXL所需的训练配置文件

在完成以上所有数据处理过程后，接下来我们就可以进入SDXL训练的阶段了，我们可以对SDXL进行全参微调（finetune），也可以基于SDXL训练对应的LoRA模型。

4.4 SDXL微调（finetune）训练

微调（finetune）训练是让SDXL全参数重新训练的一种方法，理想的状态是让SDXL模型在原有能力的基础上，再学习到一个或几个细分领域的数据特征与分布，从而能在工业界，学术界以及竞赛界满足不同的应用需求。

Rocky为大家举一个形象的例子，让大家能够能好理解SDXL全参微调的意义。比如我们要训练一个真人写真SDXL模型，应用于写真领域。那么我们首先需要寻找合适的基于SDXL的预训练底模型，比如一个能生成真人图片的SDXL A模型。然后我们用A模型作为预训练底模型，并收集写真行业优质数据作为训练集，有了模型和数据，再加上Rocky为大家撰写的SDXL微调训练全流程攻略，我们就能训练获得一个能生成真人写真的SDXL行业模型，并作为真人写真相关产品的核心大模型。

那么话不多说，下面Rocky将告诉大家从0到1使用SDXL模型进行微调训练的全流程攻略，让我们一起来训练属于自己的SDXL模型吧！

（1）SDXL 微调（finetune）数据集制作

在SDXL全参数微调中，SDXL能够学习到大量的主题，人物，画风或者抽象概念等信息特征，所以我们需要对一个细分领域的数据进行广泛的收集，并进行准确的标注。

Rocky这边收集整理了838张二次元人物数据，包含多样的人物，多样的画风，涵盖了大量的二次元专属信息特征，组成二次元人物数据集，作为本次SDXL微调训练的训练集。

二次元数据集

Rocky一开始收集了5000张数据，经过筛选只剩838张作为最后的模型训练集，数据集质量决定生成效果的上限，所以前期对数据集的清洗工作是非常重要的，Rocky总结了以下的用于SDXL全参微调的数据集筛选要求：

数据尺寸需要在512x512像素以上。
数据的大小最好大于300K。
数据种类尽量丰富，不同主题，不同画风，不同概念都要充分采集。
一个特殊tag对应的图像特征在数据集中需要一致，不然在推理过程触发这个tag时可能会生成多个特征的平均。
每个数据都要符合我们的审美和评判标准！每个数据都要符合我们的审美和评判标准！每个数据都要符合我们的审美和评判标准！

接下来，我们就可以按照本文4.3 Stable Diffusion XL数据集制作章节里的步骤，进行数据的清洗，自动标注，以及添加特殊tag。

Rocky认为对SDXL模型进行微调训练主要有两个目的：增强SDXL模型的图像生成能力与增加SDXL对新prompt的触发能力。

我们应该怎么理解这两个目的呢。我们拿二次元人物数据集为例，我们想要让SDXL模型学习二次元人物的各种特征，包括脸部特征，服装特征，姿势特征，二次元背景特征，以及二次元画风特征等。通过训练不断让SDXL模型“学习”这些数据的内容，从而增强SDXL模型生成二次元人物图片的能力。与此同时，我们通过自动标注与特殊tag，将图片的特征与标注信息进行对应，让SDXL在学习图片数据特征的同时，学习到对应的标注信息，能够在前向推理的过程中，通过二次元的专属标签生成对应的二次元人物图像。

理解了上面的内容，咱们的数据处理部分就告一段落了。为了方便大家使用二次元人物数据集进行后续的SDXL模型微调训练，Rocky这边已经将处理好的二次元人物数据集开源（包含原数据，标注文件，读取数据的json文件等），大家可以关注公众号WeThinkIn，后台回复“二次元人物数据集”获取。

（2）SDXL 微调训练参数配置

本节中，Rocky主要介绍Stable Diffusion XL全参微调（finetune）训练的参数配置和训练脚本。

Rocky已经帮大家整理好了SDXL全参微调训练的全部参数与训练脚本，大家可以在SDXL-Trian项目的train_config文件夹中找到相应的训练参数配置（XL_config文件夹），并且可以在SDXL-Trian项目中运行SDXL_finetune.sh脚本，进行SDXL的全参微调训练。

接下来，Rocky将带着大家从头到尾走通SDXL全参微调训练过程，并讲解训练参数的意义。首先，我们可以看到XL_config文件夹中有两个配置文件config_file.toml和sample_prompt.toml，他们分别存储着SDXL的训练超参数与训练中的验证prompt。

XL_config文件夹中的配置文件config_file.toml和sample_prompt.toml

其中config_file.toml文件中的配置信息包含了sdxl_arguments，model_arguments，dataset_arguments，training_arguments，logging_arguments，sample_prompt_arguments，saving_arguments，optimizer_arguments八个维度的参数信息，下面Rocky为大家依次讲解各个超参数的作用：

[sdxl_arguments]
cache_text_encoder_outputs = true
no_half_vae = true
min_timestep = 0
max_timestep = 1000
shuffle_caption = false

cache_text_encoder_outputs：Stable Diffusion XL训练时需要打开，用于两个Text Encoder输出结果的缓存与融合。注：当cache_text_encoder_outputs设为true时，shuffle_caption将不起作用。

no_half_vae：当此参数为true时，VAE在训练中使用float32精度；当此为false时，VAE在训练中使用fp16精度。

min_timestep：Stable Diffusion XL Base U-Net在训练时的最小时间步长（默认为0）。

max_timestep：Stable Diffusion XL Base U-Net在训练时的最大时间步长（默认为1000）。

shuffle_caption：当设置为true时，对训练标签进行打乱，能一定程度提高模型的泛化性。

[model_arguments]
pretrained_model_name_or_path = "/本地路径/SDXL模型文件"
vae  = "/本地路径/VAE模型文件"

pretrained_model_name_or_path：读取本地Stable Diffusion XL预训练模型用于微调。

vae：读取本地VAE模型，如果不传入本参数，在训练中则会读取Stable Diffusion XL自带的VAE模型。

[dataset_arguments]
debug_dataset = false
in_json = "/本地路径/data_meta_lat.json"
train_data_dir = "/本地路径/训练集"
dataset_repeats = 10
keep_tokens = 0
resolution = "1024,1024"
caption_dropout_rate = 0
caption_tag_dropout_rate = 0
caption_dropout_every_n_epochs = 0
color_aug = false
token_warmup_min = 1
token_warmup_step = 0

debug_dataset：训练时对数据进行debug处理，不让破损数据中断训练进程。

in_json：读取数据集json文件，json文件中包含了数据名称，数据标签，数据分桶等信息。

train_data_dir：读取本地数据集存放路径。

dataset_repeats：整个数据集重复训练的次数。（经验分享：如果数据量级小于一千，可以设置为10；如果数据量级在一千与一万之前，可以设置为5；如果数据量级大于一万，可以设置为2）

keep_tokens：在训练过程中，会将txt中的tag进行随机打乱。如果将keep tokens设置为n，那前n个token的顺序在训练过程中将不会被打乱。

resolution：设置训练时的数据输入分辨率，分别是width和height。

caption_dropout_rate：针对一个数据丢弃全部标签的概率，默认为0。

caption_tag_dropout_rate：针对一个数据丢弃部分标签的概率，默认为0。（类似于传统深度学习的Dropout逻辑）

caption_dropout_every_n_epochs：每训练n个epoch，将数据标签全部丢弃。

color_aug：数据颜色增强，建议不启用，其与caching latents不兼容，若启用会导致训练时间大大增加。

token_warmup_min：在训练一开始学习每个数据的前n个tag（标签用逗号分隔后的前n个tag，比如girl，boy，good）

token_warmup_step：训练中学习标签数达到最大值所需的步数，默认为0，即一开始就能学习全部的标签。

[training_arguments]
output_dir = "/本地路径/模型权重保存地址"
output_name = "sdxl_finetune_WeThinkIn"
save_precision = "fp16"
save_every_n_steps = 1000
train_batch_size = 4
max_token_length = 225
mem_eff_attn = false
xformers = true
max_train_steps = 100000
max_data_loader_n_workers = 8
persistent_data_loader_workers = true
gradient_checkpointing = true
gradient_accumulation_steps = 1
mixed_precision = "fp16"

output_dir：模型保存的路径。

output_name：模型名称。

save_precision：模型保存的精度，一共有[“None”, "float", "fp16", "bf16"]四种选择，默认为“None”，即FP32精度。

save_every_n_steps：每n个steps保存一次模型权重。

train_batch_size：训练Batch-Size，与传统深度学习一致。

max_token_length：设置Text Encoder最大的Token数，有[None, 150, 225]三种选择，默认为“None”，即75。

mem_eff_attn：对CrossAttention模块进行轻量化，能够一定程度上加速模型训练并降低显存占用，开启mem_eff_attn后xformers失效。

xformers：xformers插件可以使SDXL模型在训练时显存减少一半左右。

max_train_steps：训练的总步数。

max_data_loader_n_workers：数据加载的DataLoader worker数量，默认为8。

persistent_data_loader_workers：能够让DataLoader worker持续挂载，减少训练中每个epoch之间的数据读取时间，但是会增加内存消耗。

gradient_checkpointing：设为true时开启梯度检查，通过以更长的计算时间为代价，换取更少的显存占用。相比于原本需要存储所有中间变量以供反向传播使用，使用了checkpoint的部分不存储中间变量而是在反向传播过程中重新计算这些中间变量。模型中的任何部分都可以使用gradient checkpoint。

gradient_accumulation_steps：如果显存不足，我们可以使用梯度累积步数，默认为1。

mixed_precision：训练中是否使用混合精度，一共有["no", "fp16", "bf16"]三种选择，默认为“no”。

[logging_arguments]
log_with = "tensorboard"
logging_dir = "/本地路径/logs"
log_prefix = "sdxl_finetune_WeThinkIn"

log_with：选择训练log保存的格式，可以从["tensorboard", "wandb", "all"]三者中选择，也可以不设置。

logging_dir：设置训练log保存的路径。

log_prefix：增加log文件的文件名前缀，比如sdxl_finetune_WeThinkIn1234567890。

[sample_prompt_arguments]
sample_every_n_steps = 100
sample_sampler = "euler_a"

[saving_arguments]
save_model_as = "safetensors"

sample_every_n_steps：在训练中每n步测试一次模型效果。

sample_sampler：设置训练中测试模型效果时使用的sampler，可以选择["ddim","pndm","lms","euler","euler_a","heun","dpm_2","dpm_2_a","dpmsolver","dpmsolver++","dpmsingle", "k_lms","k_euler","k_euler_a","k_dpm_2","k_dpm_2_a"]，默认是“ddim”。

save_model_as：每次模型权重保存时的格式，可以选择["ckpt", "safetensors", "diffusers", "diffusers_safetensors"]，目前SD WebUI兼容"ckpt"和"safetensors"格式模型。

[optimizer_arguments]
optimizer_type = "AdaFactor"
learning_rate = 1e-7
train_text_encoder = false
max_grad_norm = 0
optimizer_args = [ "scale_parameter=False", "relative_step=False", "warmup_init=False",]
lr_scheduler = "constant_with_warmup"
lr_warmup_steps = 100

optimizer_type：AdamW (default),Lion, SGDNesterov,AdaFactor等。

learning_rate：训练学习率，单卡推荐设置2e-6，多卡推荐设置1e-7。

train_text_encoder：是否在SDXL训练时同步微调Text Encoder。如果设置为true，则在SDXL训练时同时开启Text Encoder模型的微调训练，增强Text Encoder模型对数据集中标签的控制力，能够让生成图片的特征更加趋近于训练数据集分布。

max_grad_norm：最大梯度范数，0表示没有clip。

optimizer_args：设置优化器额外的参数，比如"weight_decay=0.01 betas=0.9,0.999 ..."。

lr_scheduler：设置学习率调度策略，可以设置成linear, cosine, cosine_with_restarts, polynomial, constant (default), constant_with_warmup, adafactor等。

lr_warmup_steps：在启动学习率调度策略前，先固定学习率训练的步数。

到这里，config_file.toml中八个维度的训练超参数就全部讲好了，大家可以根据自己的实际情况这些超参数进行调整。

除了config_file.toml之外，我们配置的文件还有sample_prompt.toml，其主要作用是在训练中阶段性验证模型的性能，里面包含了模型生成验证图片的相关参数：

[prompt]
width = 1024
height = 1024
scale = 7
sample_steps = 28
[[prompt.subset]]
prompt = "1girl, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, green background, hat, hoop earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt"

现在我们已经对SDXL训练的整体参数有了比较充分的了解，下面Rocky再对一些关键参数进行深度的解析，让大家能够更好的理解。

（3）SDXL训练的关键参数详解

【1】pretrained_model_name_or_path对SDXL模型微调训练的影响

pretrained_model_name_or_path参数中我们需要加载本地的SDXL模型作为训练底模型。

在SDXL全参数微调训练中，底模型的选择可以说是最为重要的一环。我们需要挑选一个生成能力分布与训练数据分布近似的SDXL模型作为训练底模型（比如说我们训练二次元人物数据集，可以选择生成二次元图片能力强的SDXL模型）。SDXL在微调训练的过程中，在原有底模型的很多能力与概念上持续扩展优化学习，从而得到底模型与数据集分布的一个综合能力。

【2】xformers加速库对SDXL模型微调训练的影响

当我们将xformers设置为true时，使用xformers加速库能对SDXL训练起到2倍左右的加速，因为其能使得训练显存占用降低2倍，这样我们就能增大我们的Batch Size数。

想要启动xformers加速库，需要先安装xformers库源，这也非常简单，我们只需要在命令行输入如下命令即可：

pip install xformers -i https://pypi.tuna.tsinghua.edu.cn/simple some-package

【3】learning_rate对SDXL模型微调训练的影响

SDXL训练过程对学习率的设置非常敏感，如果我们将学习率设置的过大，很有可能导致SDXL模型训练跑飞，在前向推理时生成非常差的图片；如果我们将学习率设置的过小，可能会导致模型无法跳出极小值点。

Rocky这里总结了相关的SDXL学习率设置经验，分享给大家。如果我们总的Batch Size（单卡Batch Size x GPU数）小于10，可以设置学习率2e-6；如果我们总的Batch Size大于10小于100，可以设置学习率1e-7。

【4】使用save_state和resume对SDXL模型训练的中断重启

在AI绘画领域，很多时候我们需要进行大规模数据的训练优化，数据量级在10万甚至100万以上，这时候整个训练周期需要一周甚至一个月，训练中可能会出现一些通讯/NCCL超时等问题，导致训练中断。

经典NCCL超时问题如下所示：

[E ProcessGroupNCCL.cpp:828] [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=213, OpType=ALLREDUCE, Timeout(ms)=1800000) ran for 1809831 milliseconds before timing out.

这些训练中断问题会导致我们的训练成本大大增加，为了解决这个问题，我们可以在config_file.toml中设置save_state = true，这样我们在训练模型时不单单保存模型权重，还会保存相关的optimizer states等训练状态。

接着，我们在config_file.toml中设置resume = "/本地路径/模型权重保存地址"，重新运行SDXL训练脚本，这时会直接调取训练中断前的模型权重与训练状态，接着继续训练。

（4）SDXL模型训练

完成训练参数配置后，我们就可以运行训练脚本进行SDXL模型的全参微调训练了。

我们本次训练用的底模型选择了WeThinkIn_SDXL_二次元模型，大家可以关注Rocky的公众号WeThinkIn，后台回复“SDXL_二次元模型”获取模型资源链接。

我们打开SDXL_finetune.sh脚本，可以看到以下的代码：

accelerate launch \
  --config_file accelerate_config.yaml \
  --num_cpu_threads_per_process=8 \
  /本地路径/SDXL-Train/sdxl_train.py \
  --sample_prompts="/本地路径/SDXL-Trian/train_config/XL_config/sample_prompt.toml" \
  --config_file="/本地路径/SDXL-Trian/train_config/XL_config/config_file.toml"

我们把训练脚本封装在accelerate库里，这样就能启动我们一开始配置的训练环境了，同时我们将刚才配置好的config_file.toml和sample_prompt.toml参数传入训练脚本中。

接下里，就到了激动人心的时刻，我们只需在命令行输入以下命令，就能开始SDXL的全参微调训练啦：

# 进入SDXL-Trian项目中
cd SDXL-Trian

# 运行训练脚本！
sh SDXL_finetune.sh

训练脚本启动后，会打印出以下的log，方便我们查看整个训练过程的节奏：

running training / 学習開始
  # 表示总的训练数据量，等于训练数据 * dataset_repeats: 1024 * 10 = 10240
  num examples / サンプル数: 10240  
  # 表示每个epoch需要多少step，以8卡为例，需要10240/ (2 * 8) = 640
  num batches per epoch / 1epochのバッチ数: 640
  # 表示总的训练epoch数，等于total optimization steps / num batches per epoch = 64000 / 640 = 100
  num epochs / epoch数: 100
  # 表示每个GPU卡上的Batch Size数，最终的Batch Size还需要在此基础上*GPU卡数，以8卡为例：2 * 8 = 16
  batch size per device / バッチサイズ: 2 
  #表示n个step计算一次梯度，一般设置为1
  gradient accumulation steps / 勾配を合計するステップ数 = 1 
  # 表示总的训练step数
  total optimization steps / 学習ステップ数: 64000

当我们设置1024分辨率+FP16精度+xformers加速时，SDXL模型进行Batch Size = 1的微调训练需要约24.7G的显存，进行Batch Size=14的微调训练需要约32.3G的显存，所以想要微调训练SDXL模型，最好配置一个32G以上的显卡，能让我们更佳从容地进行训练。

到此为止，Rocky已经将SDXL全参微调训练的全流程都做了详细的拆解，等训练完成后，我们就可以获得属于自己的SDXL模型了！

（5）加载自训练SDXL模型进行AI绘画

SDXL模型微调训练完成后，会将模型权重保存在我们之前设置的output_dir路径下。接下来，我们使用Stable Diffusion WebUI作为框架，加载SDXL二次元人物模型进行AI绘画。

在本文3.3节零基础使用Stable Diffusion WebUI搭建Stable Diffusion XL推理流程中，Rocky已经详细讲解了如何搭建Stable Diffusion WebUI框架，未使用过的朋友可以按照这个流程快速搭建起Stable Diffusion WebUI。

要想使用SDXL模型进行AI绘画，首先我们需要将训练好的SDXL二次元人物模型放入Stable Diffusion WebUI的/models/Stable-diffusion文件夹下。

然后我们在Stable Diffusion WebUI中分别选用SDXL二次元人物模型即可：

完成上图中的操作后，我们就可以进行二次元人物图片的生成啦！

下面是使用本教程训练出来的SDXL二次元人物模型生成的图片：

SDXL二次元模型生成图片

到这里，关于SDXL微调训练的全流程攻略就全部展示给大家了，大家如果觉得好，欢迎给Rocky的劳动点个赞，支持一下Rocky，谢谢大家！

如果大家对SDXL全参数微调训练还有想要了解的知识或者不懂的地方，欢迎在评论区留言，Rocky也会持续优化本文内容，能让大家都能快速了解SDXL训练知识，并训练自己的专属SDXL绘画模型！

4.5 基于SDXL训练LoRA模型

目前Stable Diffusion XL全参微调的训练成本是Stable Diffusion之前系列的2-3倍左右，而基于Stable Diffusion XL训练LoRA的成本与之前的系列相比并没有太多增加，故训练LoRA依旧是持续繁荣SDXL生态的高效选择。

如果大家想要了解LoRA模型的核心基础知识，LoRA的优势，热门LoRA模型推荐等内容，可以阅读Rocky之前写的文章：

在本节，Rocky将告诉大家从0到1使用SDXL模型训练对应的LoRA的全流程攻略，让我们一起来训练属于自己的SDXL LoRA模型吧！

（1）SDXL LoRA数据集制作

首先，我们需要确定数据集主题，比如说人物，画风或者某个抽象概念等。本次我们选择用Rocky自己搜集的人物主题数据集——猫女数据集来进行SDXL LoRA模型的训练。

猫女数据集

确定好数据集主题后，我们需要保证数据集的质量，Rocky总结了以下的SDXL LoRA训练数据集筛选要求：

当我们训练人物主题时，一般需要10-20张高质量数据；当我们训练画风主题时，需要100-200张高质量数据；当我们训练抽象概念时，则至少需要200张以上的数据。
不管是人物主题，画风主题还是抽象概念，一定要保证数据集中数据的多样性（比如说猫女姿态，角度，全身半身的多样性）。
每个数据都要符合我们的审美和评判标准！每个数据都要符合我们的审美和评判标准！每个数据都要符合我们的审美和评判标准！

所以Rocky这次挑选的猫女数据集一共有22张图片，包含了猫女的不同姿态数据，并且每张图也符合Rocky的审美哈哈。

接下来，我们就可以按照本文4.3 Stable Diffusion XL数据集制作章节里的步骤，进行数据的清洗，自动标注，以及添加特殊tag——即触发词。在这里，我们要在标注文件的开头添加“catwomen”作为猫女的触发词。

除了对数据进行标注，我们还需要对数据的标注进行清洗，删除一些概念与触发词重合的标签。为什么我们要进行数据标注的清洗呢？因为如果不对标注进行清洗，会导致训练时的tag污染。

我们拿猫女数据集为例，我们想要让SDXL LoRA模型学习猫女的主要特征，包括脸部特征，服装特征（最具猫女特点的黑色皮衣和黑色眼罩）等，我们想让“catwomen”学习到这些特征。但是自动标注会给数据打上一些描述脸部特征和服装特征的tag，导致猫女的主要特征被这些tag分走，从而导致tag污染。这样就会导致很多精细化的特征丢失在自动标注的tag中，使得SDXL LoRA在生成猫女图片时缺失黑色皮衣或者黑色眼罩等。

所以我们需要删除自动标注的脸部，服装等tag，从而使得保留下来的触发词等标签是SDXL LoRA模型着重需要学习的。

一张一张手动删除标签费时费力，Rocky这里推荐大家使用Stable Diffusion WebUI的一个数据标注处理插件：stable-diffusion-webui-dataset-tag-editor，可以对标签进行批量处理，非常方便。

完成上述步骤，咱们的数据处理部分就告一段落了。为了方便大家使用猫女数据集进行后续的LoRA训练，Rocky这边已经将处理好的猫女数据集开源（包含原数据，标注文件，读取数据的json文件等），大家可以关注公众号WeThinkIn，后台回复“猫女数据集”获取。

（2）SDXL LoRA训练参数配置

大家可以在SDXL-Trian项目中train_config/XL_LoRA_config路径下找到SDXL LoRA的训练参数配置文件config_file.toml和sample_prompt.toml，他们分别存储着SDXL_LoRA的训练超参数与训练中的验证prompt信息。

其中config_file.toml文件中的配置文件包含了sdxl_arguments，model_arguments，dataset_arguments，training_arguments，logging_arguments，sample_prompt_arguments，saving_arguments，optimizer_arguments以及additional_network_arguments九个个维度的参数信息。

训练SDXL_LoRA的参数配置与SDXL全参微调的训练配置有相同的部分（上述的前八个维度），也有LoRA的特定参数需要配置（additional_network_arguments）。

下面我们首先看看这些共同的维度中，有哪些需要注意的事项吧：

[sdxl_arguments] # 与SDXL全参微调训练一致
cache_text_encoder_outputs = true
no_half_vae = true
min_timestep = 0
max_timestep = 1000
shuffle_caption = false

[model_arguments] # 与SDXL全参微调训练一致
pretrained_model_name_or_path = "/本地路径/SDXL模型文件"
vae  = "/本地路径/VAE模型文件"  # 如果只使用模型自带的VAE，不读取额外的VAE模型，则需要将本行直接删除

[dataset_arguments] # 与SDXL全参微调训练不一致
#LoRA训练过程中取消了caption_dropout_rate = 0，caption_tag_dropout_rate = 0，
#caption_dropout_every_n_epochs = 0这三个参数，因为本身LoRA的模型容量较小，不需要再进行类标签Dropout的操作了。
debug_dataset = false
in_json = "/本地路径/data_meta_lat.json"
train_data_dir = "/本地路径/训练集"
dataset_repeats = 1
keep_tokens = 0
resolution = "1024,1024"
color_aug = false
token_warmup_min = 1
token_warmup_step = 0

[training_arguments] # 与SDXL全参微调训练不一致
# SDXL_LoRA增加了sdpa参数，当其设置为true时，训练中启动scaled dot-product attention优化，这时候就不需要再开启xformers了
output_dir = "/本地路径/模型权重保存地址"
output_name = "sdxl_lora_WeThinkIn"
save_precision = "fp16"
save_every_n_epochs = 1
train_batch_size = 4
max_token_length = 225
mem_eff_attn = false
sdpa = true
xformers = false
max_train_epochs = 100 #max_train_epochs设置后，会覆盖掉max_train_steps，即两者同时存在时，以max_train_epochs为准
max_data_loader_n_workers = 8
persistent_data_loader_workers = true
gradient_checkpointing = true
gradient_accumulation_steps = 1
mixed_precision = "fp16"

[logging_arguments] # 与SDXL全参微调训练一致
log_with = "tensorboard"
logging_dir = "/本地路径/logs"
log_prefix = "sdxl_lora_WeThinkIn"

[sample_prompt_arguments] # 与SDXL全参微调训练一致
sample_every_n_epochs = 1
sample_sampler = "euler_a"

[saving_arguments] # 与SDXL全参微调训练一致
save_model_as = "safetensors"

[optimizer_arguments] # 与SDXL全参微调训练不一致
optimizer_type = "AdaFactor"
learning_rate = 1e-5 # 训练SDXL_LoRA时，学习率可以调的大一些，一般比SDXL全参微调的学习率大10倍左右，比如learning_rate = 1e-5
max_grad_norm = 0
optimizer_args = [ "scale_parameter=False", "relative_step=False", "warmup_init=False",]
lr_scheduler = "constant_with_warmup"
lr_warmup_steps = 100

除了上面的参数，训练SDXL_LoRA时还需要设置一些专属参数，这些参数非常关键，下面Rocky将给大家一一讲解：

[additional_network_arguments]
no_metadata = false
network_module = "networks.lora"
network_dim = 32
network_alpha = 16
network_args = [ "conv_dim=32", "conv_alpha=16",]
network_train_unet_only = true

no_metadata：保存模型权重时不附带Metadata数据，建议关闭，能够减少保存下来的LoRA大小。

network_module：选择训练的LoRA模型结构，可以从["networks.lora", "networks.dylora", "lycoris.kohya"]中选择，最常用的LoRA结构默认选择"networks.lora"。

network_dim：设置LoRA的RANK，设置的数值越大表示表现力越强，但同时需要更多的显存和时间来训练。

network_alpha：设置缩放权重，用于防止下溢并稳定训练的alpha值。

network_args：设置卷积的Rank与缩放权重。

下面表格中Rocky给出一些默认配置，大家可以作为参考：

network_category	network_dim	network_alpha	conv_dim	conv_alpha
LoRA	32	1	-	-
LoCon	16	8	8	1
LoHa	8	4	4	1

如果我们想要训练LoRA，我们需要设置network_module = "networks.lora"，同时设置network_dim和network_alpha，和上面的配置一致。

如果我们想要训练LoCon，我们需要设置network_module = "lycoris.kohya"和algo="locon"，同时设置network_dim和network_alpha：

network_module = "lycoris.kohya"
algo = "locon"
network_dim = 32
network_alpha = 16
network_args = [ "conv_dim=32", "conv_alpha=16",]

如果我们想要训练LoHa，我们需要设置network_module = "lycoris.kohya"和algo="loha"，同时设置network_dim和network_alpha：

network_module = "lycoris.kohya"
algo = "loha"
network_dim = 32
network_alpha = 16
network_args = [ "conv_dim=32", "conv_alpha=16",]

network_train_unet_only：如果设置为true，那么只训练U-Net部分。

（3）SDXL LoRA关键参数详解

【1】train_batch_size对SDXL LoRA模型训练的影响

和传统深度学习一样，train_batch_size即为训练时的batch size，表示一次性送入SDXL LoRA模型进行训练的图片数量。

一般来说，较大的batch size往往每个epoch训练时间更短，但是显存占用会更大，并且收敛得慢（需要更多epoch数）。较小的batch size每个epoch训练时间长，但是显存占用会更小，并且收敛得快（需要更少epoch数）。

但是有研究表明这个结论会在batch size大于8000的时候才会体现，所以在实际的训练时，如果GPU数不大于8卡的话，还是需要尽可能占满GPU显存为宜，比如64-96之间（理论上batch size = 2^n时计算效率较高），训练一般都能取得不错效果。

上面的结论在训练SDXL大模型时是非常适用的，不过我们在训练SDXL LoRA模型时，一般来说数据量级是比较小的（10-300为主），所以在这种情况下，我们可以设置batch size为2-6即可。

【2】pretrained_model_name_or_path对SDXL LoRA模型训练的影响

pretrained_model_name_or_path参数中我们需要加载本地的SDXL模型作为训练底模型。

底模型的选择至关重要，SDXL LoRA的很多底层能力与基础概念的学习都来自于底模型的能力。并且底模型的优秀能力需要与我们训练的主题，比如说人物，画风或者某个抽象概念相适配。如果我们要训练二次元LoRA，则需要选择二次元底模型，如果我们要训练三次元LoRA，则需要选择三次元底模型，以此类推。

【3】network_dim对SDXL LoRA模型训练的影响

network_dim即特征维度，越高表示模型的参数量越大，设置高维度有助于LoRA学习到更多细节特征，但模型收敛速度变慢，同时也更容易过拟合，需要的训练时间更长。所以network_dim的设置需要根据任务主题去调整。

一般来说，在SDXL的1024*1024分辨率训练基础上，可以设置network_dimension = 128，此时SDXL LoRA大小约为686MB。

【4】network_alpha对SDXL LoRA模型训练的影响

network_alpha是一个缩放因子，用于缩放模型的训练权重 W ， W = W_{in} \times alpha / dim 。network_alpha设置的越高，LoRA模型能够学习更多的细节信息，同时学习速率也越快，推荐将其设置为network_dimension的一半。

（4）SDXL LoRA模型训练

完成训练参数配置后，我们就可以运行训练脚本进行SDXL_LoRA模型的训练了。

我们本次训练用的底模型选择了WeThinkIn_SDXL_真人模型，大家可以关注Rocky的公众号WeThinkIn，后台回复“SDXL_真人模型”获取模型资源链接。

我们打开SDXL_fintune_LoRA.sh脚本，可以看到以下的代码：

accelerate launch \
  --config_file accelerate_config.yaml \
  --num_cpu_threads_per_process=8 \
  /本地路径/SDXL-Train/sdxl_train_network.py \
  --sample_prompts="/本地路径/SDXL-Train/train_config/XL_LoRA_config/sample_prompt.toml" \
  --config_file="/本地路径/SDXL-Train/train_config/XL_LoRA_config/config_file.toml"

我们把训练脚本封装在accelerate库里，这样就能启动我们一开始配置的训练环境了，同时我们将刚才配置好的config_file.toml和sample_prompt.toml参数传入训练脚本中。

接下里，就到了激动人心的时刻，我们只需在命令行输入以下命令，就能开始SDXL_LoRA训练啦：

# 进入SDXL-Trian项目中
cd SDXL-Trian

# 运行训练脚本！
sh SDXL_fintune_LoRA.sh

当我们基于SDXL训练SDXL LoRA模型时，我们设置分辨率为1024+FP16精度+xformers加速时，进行Batch Size = 1的微调训练需要约13.3G的显存，进行Batch Size=8的微调训练需要约18.4G的显存，所以想要微调训练SDXL LoRA模型，最好配置一个16G以上的显卡，能让我们更佳从容地进行训练。

（5）加载SDXL LoRA模型进行AI绘画

SDXL LoRA模型训练完成后，会将模型权重保存在我们之前设置的output_dir路径下。接下来，我们使用Stable Diffusion WebUI作为框架，加载SDXL LoRA模型进行AI绘画。

在本文3.3节零基础使用Stable Diffusion WebUI搭建Stable Diffusion XL推理流程中，Rocky已经详细讲解了如何搭建Stable Diffusion WebUI框架，未使用过的朋友可以按照这个流程快速搭建起Stable Diffusion WebUI。

要想使用SDXL LoRA进行AI绘画，首先我们需要将SDXL底模型和SDXL LoRA模型分别放入Stable Diffusion WebUI的/models/Stable-diffusion文件夹和/models/Lora文件夹下。

然后我们在Stable Diffusion WebUI中分别选用底模型与LoRA即可：

Stable Diffusion WebUI中使用SDXL LoRA流程

完成上图中的操作后，我们就可以进行猫女图片的生成啦！

【1】训练时的底模型+猫女LoRA

首先我们使用训练时的底模型作为测试底模型，应选用训练好的猫女LoRA，并将LoRA的权重设为1，看看我们生成的图片效果如何：

自训练SDXL LoRA生成猫女图片

我们可以看到，生成的猫女图片的完成度还是非常好的，不管是整体质量还是细节都能展现出猫女该有的气质与魅力。并且在本次训练中猫女的手部特征也得到了较好的学习，优化了一直困扰AI绘画的手部问题。

【2】设置LoRA的不同权重

接下来，我们设置LoRA的权重分别为[0.2, 0.4, 0.6, 0.8, 1]，进行对比测试，看看不同SDXL LoRA权重下的图片生成效果如何：

SDXL LoRA的权重分别为[0.2, 0.4, 0.6, 0.8, 1]时的图片生成效果

从上图的对比中可以看出，当SDXL LoRA设置的权重越高时，训练集中的特征越能载生成图片中展现，比如说猫女的人物特征，猫女的服装特征以及生成的猫女有无面罩等。但LoRA的权重也不是也高越好，当设置权重为0.6-0.8之间，生成的图片会有更多的泛化性。

【3】切换不同的底模型

完成了在单个底模型上的SDXL LoRA不同权重的效果测试，接下来我们切换不同的底模型，看看会生成的猫女图片会有什么变化吧。

首先，我们将底模型切换成SDXL Base模型，使用猫女LoRA并设置权重为1：

SDXL Base模型 + 猫女LoRA生成图片

从上面的图中可以看出，使用SDXL Base模型作为底模型后，生成的猫女图片整体质感已经发生改变，背景也有了更多光影感。

我们再使用SDXL的二次元模型作为底模型，同样使用猫女LoRA并设置权重为1：

SDXL 二次元模型+ 猫女LoRA生成图片

可以看到，换用二次元模型作为底模型后，生成的猫女图片整体质感开始卡通化。但是由于训练数据集中全是三次元图片，所以二次元底模型+三次元LoRA生成的图片并没有完全的二次元化。

【4】使用不同提示词改变图片风格

最后，我们再尝试通过有添加提示词prompt，来改变生成的猫女图片的风格。

首先，我们在提示词prompt中加入赛博朋克风格“Cyberpunk style”，这是生成的猫女图片中就会加入赛博朋克元素了：

赛博朋克风格的猫女

到这里，关于SDXL LoRA的全流程攻略就全部展示给大家了，大家如果觉得好，欢迎给Rocky的劳动点个赞，支持一下Rocky，谢谢大家！

如果大家对SDXL LoRA还有想要了解的知识或者不懂的地方，欢迎在评论区留言，Rocky也会持续优化本文内容，能让大家都能快速了解SDXL LoRA知识，并训练自己的专属LoRA模型！

4.6 SDXL训练结果测试评估

之前的章节讲述了SDXL模型微调和SDXL LoRA模型训练后的效果测试评估流程，那么在本小节，Rocky向大家介绍一下AI绘画模型测试评估的一些通用流程与技巧。

在进行AI绘画时，我们需要输入正向提示词（positive prompts）和负向提示词（negative prompts）。

正向提示词一般需要输入我们想要生成的图片内容，包括我们训练好的特殊tag等。

不过在正向提示词的开头，一般都需要加上提高生成图片整体质量的修饰词，Rocky这里推荐一套“万金油”修饰词，方便大家使用：

(masterpiece,best quality,ultra_detailed,highres,absurdres:1.2)

负向提示词一般需要输入我们不想生成的内容，在这里Rocky再分享一套基于SDXL的“万金油”负向提示词，方便大家使用：

(worst quality, low quality, ugly:1.4), poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, blurry, (bad art, bad anatomy:1.4), blurred, text, watermark

当然的，我们也可以使用ChatGPT辅助生成提示词，在此基础上我们再加入训练好的特殊tag并对提示词进行修改润色。

在我们进行模型测试的时候，如果存在生成图片质量不好，生成图片样式单一或者生成图片崩坏的情况，就需要优化数据或者参数配置，重新训练了。

4.7 SDXL训练经验分享（持续更新！）

在本节中，Rocky将向大家持续分享关于SDXL和SDXL LoRA等模型的训练经验，大家有自己的经验，也欢迎在评论区补充，我们一起让AIGC和AI绘画领域更加繁荣！

1. 数据质量标准：图片的文件容量（占用的内存）越大越好，图像的分辨率越大越好，同时数据内容的是符合我们训练的领域的专业审美与标准的。

2. 数据质量最为重要：切勿在没有保证数据质量的前提下，盲目扩充数据数量，这样只会导致低质量数据污染整个数据集，从而导致模型训练后的图片生成效果不理想。

3. 数据集中类别均衡：确保数据集中每种风格、主题、概念等各种维度的类别有大致相同的数量。使用均衡的数据集训练出来的SDXL模型，具备多功能性，可以作为未来训练的基础SDXL模型。

4. 细分类别精细化训练：对整个数据集，我们进行统一的训练，整体上效果是可以保证的。但是可能会存在一些难类别/难样本/难细节等，在这些场景中模型的训练效果不好。这时可以针对这些bad case，补充庞大且高质量的素材进行优化训练（每个素材强化训练80-100步）。

5. 数据标签质量：每张图片都需要经过仔细的数据标注，能够较好的表达呈现图片中的内容，才算是一个好的标签。高质量的数据标签能够增强模型用最少的提示词生成高质量图片的能力和生成图片内容的准确性。

6. 数据增强手段：数据多尺寸处理、素材去水印、图像特殊部分（手部、服装等）mask处理等。

Rocky将持续把Stable Diffusion XL训练的经验与思考分享出来，大家先点赞收藏敬请期待！！！

5. 生成式模型的性能测评

到目前为止，AIGC领域的测评过程整体上还是比较主观。Stable Diffusion XL在性能测评时使用了FID（Fréchet inception distance），CLIP score以及人类视觉系统（HVS）评价这三个指标作为文生图的标价指标，其中人类视觉系统依旧是占据主导地位。

下面，跟着Rocky一起来看看各个评价指标的意义与特点，以及在不同的实际场景中，该如何对生成式模型做出有效的评估。

5.1 FID（Fréchet inception distance）

FID（Fréchet inception distance）表示生成图像与真实图像之间的相似性，即图像的真实度。FID表示的是生成图像的特征向量与真实图像的特征向量之间的距离，该距离越近（值越小），表明生成模型的效果越好，即图像的清晰度高，且多样性丰富。

FID是通过Inception模型进行计算的。主要流程是将生成图像和真实图像输入到Inception模型中，并提取倒数第二层的2048维向量进行输出，最后计算两者特征向量之间的距离。

由于Stable Diffusion XL模型是文生图模型，并不存在原本的真实图像，所以一般选用COCO验证集上的zero-shot FID-30K（选择30K的样本）与生成图像进行求FID操作，并将其中最小的FID用于不同模型之间的性能对比。

但是有研究指出，FID对深入的文本理解，独特艺术风格之间的精细区分，以及明显的视觉美感等AIGC时代特有的特征不能很好的评估。同时也有研究表明，zero-shot FID-30K与视觉美学呈负相关。

5.2 CLIP score

CLIP score可以用于评估文生图/图生图中生成图像与输入Prompt文本以及生成图像与输入原图像的匹配度。以文生图为例，我们使用CLIP模型将Prompt文本和生成图像分别转换为特征向量，然后计算它们之间的余弦相似度。当CLIP Score较高时，表示生成图像与输入Prompt文本之间的匹配度较高；当CLIP Score较低时，表示生成图像与输入Prompt文本之间的匹配度较低。




Stable Diffusion XL比起之前的系列：FID指标上升，CLIP score指标提升

从上表可以看出，SDXL在CLIP score上得分最高，由于使用了两个CLIP Text Encoder进行约束，SDXL对文本内容的控制力确实强了不少。但是SDXL在FID指标上也同步上升了，并且比之前的系列都要高，刚才在介绍FID指标的内容中，Rocky已经阐述过很多研究都表明FID在AIGC时代不能很好的作为美学评价指标，甚至与视觉美学呈负相关，在SDXL的论文中更加证实了这一点。所以在此基础上，最后又增加了人类视觉系统作为评价指标（同样的参数让SD系列的不同模型生成图像，并让人工评价出最好的图像），在人类评估者眼中，明显更加喜欢SDXL生成的图片，结果如下所示：

在人类评估者眼中，Stable Diffusion XL以48.44%的胜率超过之前的系列
5.3 Aesthetics Scorer（美学评分）

除了上述提到的三种评价指标，我们还可以用Aesthetics Scorer（美学评分）对Stable DIffusion系列模型生成的图片进行评分。Aesthetics Scorer背后的评价标准数据集是基于LAION-Aesthetics，也就是Stable Diffusion系列训练时有用到的数据集，所以Aesthetics Scorer有一定的可靠性。

Aesthetics Scorer越高，表示图像美学质量越好
5.4 与Midjourney系列进行对比

完成和Stable Diffusion之前系列的对比之后，SDXL还与Midjourney v5.1和Midjourney v5.2进行了对比。

这里主要使用谷歌提出的文生图测试prompts：PartiPrompts（P2）作为测试基准。PartiPrompts（P2）包含了1600多个英文prompts，覆盖了不同的类别与复杂度，可以说是一个高价值基准。

PartiPrompts（P2）包含的prompts内容范围

在具体比较时，从PartiPrompts（P2）的每个类别中选择五个随机prompts，并由Midjourney v5.1和SDXL分别生成四张1024×1024分辨率的图像。然后将这些图像提交给AWS GroundTruth工作组，工作组根据图像与prompts的匹配度进行投票（Votes）。投票结果如下图所示，总体而言，SDXL的生成效果略高于Midjourney v5.1（54.9%：45.1%）：

Stable Diffusion XL VS Midjourney v5.1

除了Midjourney v5.1之外，SDXL还与DeepFloyd IF、DALLE-2、Bing Image Creator和Midjourney v5.2进行了定性比较，这些都是AIGC领域的高价值与强生态模型，对比结果已经在下图展示，大家可以对比了解一下：

SDXL与DeepFloyd IF、DALLE-2、Bing Image Creator以及Midjourney v5.2的定性比较结果

总的来说，不管是在传统深度学习时代还是在AIGC时代，生成式模型的生成效果一直存在难以量化和难以非常客观评价的问题，需要人类视觉系统去进行兜底与约束。

那么，在不同的实际场景中，我们该如何对SDXL等生成式模型做出务实的评价呢？

5.5 不同实际场景的务实评估

其实在不同的实际场景中，最务实有效的AIGC模型评估手段是判断模型是否具备价值。

从投资视角看，价值是有产生类ChatGPT效应或者类妙鸭相机流量的潜在势能，比如在团队内部或者让潜在投资人产生了“哇塞时刻”。

从CEO角度看，价值是在ToB领域能解决客户的问题；在ToC领域能获得用户的好评并存在成为爆款的潜在可能性。

从CTO角度看，价值是能够在严谨的验证中发表论文；能够在顶级算法竞赛中斩获冠军；能够作为产品解决方案的一部分，让产品“焕然一新”。

由于模型本身是“黑盒”，无法量化分析模型结构本身，而评估模型生成的内容质量的指标又离不开人类视觉系统的支持。那就把模型融入到产品中，融入到算法解决方案中，在一次次的实际商业“厮杀”中，来反馈验证模型的性能。

6. SDXL Turbo模型核心基础内容完整讲解

2023年11月29日，StabilityAI官方发布了最新的快速文生图模型SDXL Turbo，目前代码、模型和技术报告已经全部开源。

SDXL Turbo模型生成图片效果（分辨率512x512）
6.1 SDXL Turbo整体架构初识

SDXL Turbo模型是在SDXL 1.0模型的基础上设计了全新的蒸馏训练方案（Adversarial Diffusion Distillation，ADD），经过蒸馏训练得到的。SDXL Turbo模型只需要1-4步就能够生成高质量图像，这接近实时的性能，无异让AI绘画领域的发展更具爆炸性，同时也为未来AI视频的爆发奠定坚实的基础。

SDXL Turbo模型本质上依旧是SDXL模型，其网络架构与SDXL一致，可以理解为一种经过蒸馏训练后的SDXL模型，优化的主要是生成图像时的采样步数。

不过SDXL Turbo模型并不包含Refiner部分，只包含U-Net（Base）、VAE和CLIP Text Encoder三个模块。在FP16精度下SDXL Turbo模型大小6.94G（FP32：13.88G），其中U-Net（Base）大小5.14G，VAE模型大小167M以及两个CLIP Text Encoder：一大一小分别是1.39G和246M。

6.2 SDXL Turbo核心原理详解

既然我们已经知道SDXL Turbo模型结构本质上是和SDXL一致，那么其接近实时的图片生成性能主要还是得益于最新的Adversarial Diffusion Distillation（ADD）蒸馏方案。

模型蒸馏技术在传统深度学习时代就应用广泛，只是传统深度学习的落地场景只局限于ToB，任务范围不大且目标定义明确，大家往往人工设计轻量型的目标检测、分割、分类小模型来满足实际应用需求，所以当时模型蒸馏技术显得有些尴尬。

但是到了AIGC时代，大模型成为“AI舞台“上最耀眼的明星，让模型蒸馏技术重新繁荣，应用于各个大模型的性能实时化中，Rocky相信模型蒸馏技术将在AIGC时代成为一个非常关键的AI技术工具。

接下来，就让我们一起解析ADD蒸馏方案的核心知识吧。首先ADD蒸馏方案的整体架构如下图所示：

SDXL Turbo的Adversarial Diffusion Distillation（ADD）蒸馏方案

ADD蒸馏方案的核心流程包括：将预训练好的SDXL 1.0 Base模型作为学生模型（预训练好的网络能显著提高对抗性损失（adversarial loss）的训练效果），它接收经过forward diffusion process后的噪声图片，并输出去噪后的图片，然后用这个去噪后的图片与原图输入判别器中计算adversarial loss以及与教师模型（一个冻结权重的强力Diffusion Model）输出的去噪图片计算distillation loss。ADD蒸馏算法中主要通过优化这两个loss来训练得到SDXL Turbo模型：

adversarial loss：借鉴了GAN的思想，设计了Hinge loss（支持向量机SVM中常用的损失函数）作为SDXL Turbo模型的adversarial loss，通过一个Discriminator来辨别学生模型（SDXL 1.0 Base模型）生成的图像和真实的图像，以确保即使在一个或两个采样步数的低步数状态下也能有高图像保真度，同时避免了其他蒸馏方法中常见的失真或模糊问题。
distillation loss：经典的蒸馏损失函数，让一个强力Diffusion Model作为教师模型并冻结参数，让学生模型（SDXL 1.0 Base模型）的输出和教师模型的输出尽量一致，具体计算方式使用的是机器学习中经典的L2损失。

最后，ADD蒸馏训练中总的损失函数就是adversarial loss和distillation loss的加权和，如下图所示，其中权重 \lambda =2.5：

6.3 SDXL Turbo效果测试

因为SDXL Turbo网络结构与SDXL一致，所以大家可以直接在Stable Diffusion WebUI上使用SDXL Turbo模型，我们只需按照本文3.3章中的教程使用Stable Diffusion WebUI即可。

同时ComfyUI也支持SDXL Turbo的使用：ComfyUI SDXL Turbo Examples，然后我们按照本文3.1章的教程使用ComfyUI工作流即可运行SDXL Turbo。

ComfyUI运行SDXL Turbo模型

当然的，diffusers库最早原生支持SDXL Turbo的使用运行，可以进行文生图和图生图的任务，相关代码和操作流程如下所示：

# 加载diffusers和torch依赖库
from diffusers import AutoPipelineForText2Image
import torch

# 构建SDXL Turbo模型的Pipeline，加载SDXL Turbo模型
pipe = AutoPipelineForText2Image.from_pretrained("/本地路径/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
# "/本地路径/sdxl-turbo"表示我们需要加载的SDXL Turbo模型，
# 大家可以关注Rocky的公众号WeThinkIn，后台回复：SDXL模型，即可获得资源链接，里面包含SDXL Turbo模型权重文件
# "fp16"代表启动fp16精度。比起fp32，fp16可以使模型显存占用减半。

# 使用GPU进行Pipeline的推理
pipe.to("cuda")

# 输入提示词
prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

# Pipeline进行推理
image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
# Pipeline生成的images包含在一个list中：[<PIL.Image.Image image mode=RGB size=1024x1024>]
#所以需要使用images[0]来获取list中的PIL图像

运行上面的整个代码流程，我们就能生成一张小浣熊的图片了。这里要注意的是，SDXL Turbo模型在diffusers库中进行文生图操作时不需要使用guidance_scale和negative_prompt参数，所以我们设置guidance_scale=0.0。

diffusers库中使用SDXL-Turbo模型进行文生图

接下来，Rocky再带大家完成SDXL Turbo模型在diffusers中图生图的整个流程：

from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image

pipe = AutoPipelineForImage2Image.from_pretrained("/本地路径/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

init_image = load_image("/本地路径/用于图生图的原始图片").resize((913, 512))

prompt = "Miniature model, axis shifting, reality, clarity, details, panoramic view, suburban mountain range, game suburban mountain range, master work, ultra-high quality, bird's-eye view, best picture quality, 8K, higher quality, high details, ultra-high resolution, masterpiece, full of tension, realistic scene, top-level texture, top-level light and shadow, golden ratio point composition, full of creativity, color, future city, technology, smart city, aerial three-dimensional transportation, pedestrian and vehicle separation, green building, macaron color, gorgeous, bright"

image = pipe(prompt, image=init_image, num_inference_steps=2, strength=0.5, guidance_scale=0.0).images[0]

运行上面的整个代码流程，我们就能生成一张新的城郊山脉的图片。需要注意的是，当在diffusers中使用SDXL Turbo模型进行图生图操作时，需要确保num_inference_steps * strength大于或等于1。因为前向推理的步数等于int(num_inference_steps * strength)步。比如上面的例子中，我们就使用SDXL-Turbo模型前向推理了0.5 * 2.0 = 1 步。

diffusers库中使用SDXL-Turbo模型进行图生图

Stability AI官方发布的技术报告中表示SDXL Turbo和SDXL相比，在推理速度上有大幅的提升。在A100上，SDXL Turbo以207ms的速度生成一张512x512的图像（prompt encoding + a single denoising step + decoding, fp16），其中U-Net部分耗时占用了67ms。

Rocky也测试了一下SDXL Turbo的图像生成效率，确实非常快，在V100上，4 steps生成512x512尺寸的图像基本可以做到实时响应（1.02秒，平均1 step仅需250ms）。

SDXL Turbo与其他模型耗时对比

在我们输入完最后一个prompt后，新生成的图像就能马上显示，推理速度确实超过了Midjourney、DALL·E 3以及之前的Stable Difusion系列模型，可谓是“天下武功，无坚不破，唯快不破”的典范。SDXL Turbo在生成速度快的同时，生成的图像质量也非常高，可以比较精准地还原prompt的描述。

为了测试SDXL Turbo的性能，StabilityAI使用相同的文本提示，将SDXL Turbo与SDXL、LCM-XL等不同版本的文生图模型进行了比较。测试结果显示，在图像质量和Prompt对齐方面，SDXL Turbo只用1个step，就击败了LCM-XL用4个steps生成的图像，并且达到了SDXL 1.0 Base通过50个steps生成的图像效果。

SDXL Turbo 1个step 生成图像效果

接着当我们将采样步数提高到4时，SDXL Turbo在图像质量和Prompt对齐方面都已经略微超过SDXL 1.0 Base模型：

SDXL Turbo 4个step 生成图像效果

论文里表示目前SDXL Turbo只能生成512x512像素的图片，Rocky推测当前开源的SDXL Turbo只在单一尺寸上进行了蒸馏训练，后续估计会有更多优化版本发布。Rocky也在512x512像素下测试了不同steps（1-8 steps）时SDXL Turbo的图片生成效果，对比结果如下所示：

SDXL Turbo不同steps效果测试

可以看到当steps为1和4时，效果都非常好，并且4steps比1step效果更好，这是可以理解的。不过当steps大于4之后，生成的图像明显开始出现过拟合现象。总的来说，如果是急速出图的场景，可以选择1 step；如果想要生成更高质量图像，推荐选择4 steps。

同时Rocky测试了一下SDXL Turbo在不同尺寸（768x768，1024x1024，512x768，768x1024，768x512，1024x768共6中尺寸）下的图像生成质量，可以看到除了1024x1024存在一定的图片特征不完善的情况，其余也具备一定的效果，但是整体上确实不如512x512的效果好。

SDXL Turbo不同尺寸效果测试

SDXL Turbo的一个直接应用，就是与游戏相结合，获得2fps的风格迁移后的游戏画面：

使用SDXL Turbo对游戏画面进行2fps的风格迁移

SDXL Turbo的另外一个直接应用是成为SDXL的“化身”，代替SDXL去快速验证各种AI绘画工作流的有效性，助力AI绘画领域的持续繁荣与高效发展。

SDXL Turbo发布后，未来AI绘画和AI视频领域有了更多的想象空间。一定程度上再次整合加速了AIGC领域的各种工作流应用，未来的潜力非常大。不过由于SDXL Turbo模型需要通过蒸馏训练获得，并且其中包含了GAN的对抗损失训练，在开源社区中像训练自定义的SDXL模型一样训练出特定SDXL Turbo模型，并且能保证出图的质量，目前来看是有一定难度的。

7. Playground v2.5核心基础内容完整讲解

2024年2月28号，Playground发布了最新的文生图模型Playground v2.5，其是Playground v2.0模型的升级版本。Playground v2.5在美学质量，颜色和对比度，多尺度生成以及以人为中心的细节处理等方面有比较大的提升，使得Playground v2.5显著优于主流的开源模型（如SDXL、Playground v2和PixArt-⍺）和闭源模型（如Midjourney v5.2和DALL-E 3）。

Playground v2.5模型生成图片示例

下面，Rocky就和大家一起研究探讨Playground v2.5的优化方法。

7.1 Playground v2.5模型整体架构讲解

Playground v2.5的模型架构与SDXL的模型架构完全一致，在此基础上Playground v2.5设计了针对性的优化训练方法（增强颜色和对比度、改善多种长宽比的图像生成效果，以及改善以人为中心的细节）来显著提升生成图片的质量。

总的来说，把Playground v2.5看作是SDXL的调优版本也不为过，这些优化训练的方法具备很强的兼容性，能在AI绘画领域的其他模型上进行迁移应用，非常有价值！

（1）增强颜色和对比度

基于Latent Diffusion架构的AI绘画模型通常难以生成色彩鲜艳、对比度高的图像，这是SD 1.x系列以来就存在的问题。尽管SDXL在生成图片的美学质量上相比之前的版本有了显著的改进，但它的色彩和对比度依然较为柔和，有概率无法生成纯色图像或者无法将生成主体放置在纯色背景上。

这个问题源于扩散模型在扩散过程设置的noise schedule：Stable Diffusion的信噪比太高，即使在离散噪声水平达到最大时也是如此。Offset Noise和Zero Terminal SNR等工作能够优化这个问题，因此SDXL在训练的最后阶段采用了Offset Noise，有一定的优化效果。

Playground v2.5模型则采取了更为直接的方法，从零开始使用EDM框架训练，而不是在SDXL的基础上进行微调训练。EDM框架为Playground v2.5带来了显著的性能优势，EDM的noise schedule在最后的Timestep中展示出接近零的信噪比，我们可以不再使用offset noise的情况下增强AI绘画模型出图的颜色和对比度，同时能让模型在训练中更快的收敛。

下图是Playground v2.5模型和SDXL模型的生成图片效果对比，可以看到Playground v2.5模型生成的图片对比度更好：

Playground v2.5模型和SDXL模型的生成图片效果对比

下图是Playground v2.5模型生成图片的颜色与对比度质量：

Playground v2.5模型生成图片的颜色与对比度质量

（2）改善多种长宽比的图像生成效果

在AI绘画领域中，生成多种长宽比（1:1、1:2、2:1、3:4、4:3、9:16、16:9等）尺寸的高质量图像是AI绘画模型能够在学术界、工业界、竞赛界以及用户侧受欢迎的重要能力。

目前很多主流的AI绘画模型在训练时一般只采用一种分辨率（比如通过随机或中心裁剪获得1:1尺寸）进行训练，虽然这些AI绘画模型中包含了卷积结构，卷积的平移不变性能一定程度上适应在推理时生成任何分辨率的图像，但是如果训练时只使用了一种尺寸，那么AI绘画模型大概率会过拟合在一个尺寸上，在其他不同尺寸上的出图效果会有问题（泛化性差、图像结构错误、内容缺失等）。

为了解决这个问题，SDXL尝试了NovelAI提出的一种宽高比分桶策略，确实是有一定的效果。但是如果数据集的分辨率分布本身已经很不均衡了，这时SDXL仍然会学习到数据集里的长宽尺寸的不均衡偏见。

Playground v2.5在SDXL的分桶策略基础上，设计了更加细分的数据Bucket管道，来确保一个更平衡的桶采样策略。新的分桶策略避免了AI绘画模型在训练时对尺寸的灾难性遗忘，并帮助AI绘画模型不会偏向某个特定的尺寸。

Playground v2.5模型生成不同尺寸图片的效果
Playground v2.5模型和SDXL模型在不同尺寸上的效果对比

（3）改善以人为中心的细节

改善以人为中心的细节，即使AI绘画模型的输出结果与人类偏好对齐。在AIGC时代，不管是AI绘画大模型还是AI对话大模型，都存在模型产生“幻觉”的情况。这在AI绘画领域的具体表现包括写实人物的人体特征崩坏（如手、脸、躯干等结构错误和特征缺失），这使得一张构图和风格都非常高质量的图片也可能存在“恐怖谷”效应。

为了缓解这个问题，Playground v2.5设计了一种类似于SFT（SFT是在LLMs中常用的策略，用于使模型与人类偏好对齐并减少错误）的对齐策略，用于AI绘画模型。新的对齐策略使Playground v2.5在至少四个重要的以人为中心的类别上超越了SDXL的效果：

面部细节、清晰度和生动度
眼睛的形状和凝视状态
头发的纹理质地
整体的光照、颜色、饱和度和景深
Playground v2.5生成效果示例
7.2 Playground v2.5模型效果测试

Playground官方在AIGC时代引入了互联网产品思维，直接将Playground v2.5模型上线到产品中让用户进行测试评估（用户评估，User Evaluations）。Playground官方认为这是收集用户反馈、测评AI绘画模型的最佳环境与方式，并且是最严格的测试，能判断AI绘画模型是否真正为用户提供了有价值和收到用户喜爱，从而能够反哺整个AI绘画生态与社区。

用户评估测试模型效果与生成图片质量

我们首先来看一下Playground官方的模型测试评估效果。

首先将Playground v2.5与世界级的主流开源AI绘画模型SDXL（提高了4.8倍）、PixArt-α（提高了2.4倍）和Playground v2（提高了1.6倍），以及世界级的闭源模型Midjourney v5.2（提高了1.2倍）和DALL·E 3（提高了1.5倍）进行了对比测试，Playground v2.5的美学质量显著优于当前最先进的这些AI绘画模型。

接着，在9:16、2:3、3:4、1:1、4:3、3:2、16:9等多分辨率尺寸生成质量方面，Playground v2.5也远远超过了SDXL模型。

人像写实图像是AI绘画领域的生成主力军，能占到AI绘画生成总量的80%左右。所以针对人像生成这块，官方将Playground v2.5与SDXL、RealStock v2两个模型进行测试对比，RealStock v2模型在SDXL的基础上针对人像场景进行了微调训练。从下图中可以看出，Playground v2.5在人像生成方面的美学质量远远超过了这两个模型。

最后，在MJHQ-30K评测上，Playground v2.5在总体FID和所有类别FID指标（所有FID指标都是在1024x1024的分辨率下计算的）上都超过了Playground v2和SDXL，特别是在人物和时尚类别上。MJHQ-30K评测结果与用户评估的结果一致，这也表明人类偏好与MJHQ-30K基准的FID得分之间存在着相关性。

官方对Playground V2.5的全面测试评估

目前Playground v2.5模型可以在diffusers中直接使用，可以进行文生图和图生图的任务，生成1024x1024分辨率及以上的高质量图片。相关代码和操作流程如下所示：

# 注意需要安装diffusers >= 0.27.0
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "/本地路径/playground-v2.5-1024px-aesthetic",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

# # Optional: Use DPM++ 2M Karras scheduler for crisper fine details
# from diffusers import EDMDPMSolverMultistepScheduler
# pipe.scheduler = EDMDPMSolverMultistepScheduler()

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt=prompt, num_inference_steps=50, guidance_scale=3).images[0]

我们在使用Playground v2.5模型时，推荐使用EDMDPMSolverMultistepScheduler调度器，能够生成更加清晰的图像，这时推荐设置guidance_scale = 3.0。如果我们使用EDMEulerScheduler调度器，推荐设置guidance_scale = 5.0。

Playground官方认为SDXL系列模型架构还有很大的优化空间，主要方向包括：更好的文本-图像对齐效果、增强模型的泛化能力、增强Latent特征空间、更精确的图像编辑等。

8. AI绘画领域的未来发展
8.1 AI绘画的“数据工厂”

看完本文的从0到1进行Stable Diffusion XL模型训练章节内容，大家可能都有这样一种感觉，整个数据处理流程占据了整个SDXL模型开发流程的60-70%甚至更多的时间，并且在数据量级不断增大的情况下，数据产生的内容护城河开始显现，模型的“数据飞轮”效应会给产品带来极大的势能。

可以看到，数据侧是如此重要，AIGC时代也需要像传统深度学习时代一样，设立“数据工厂”为AIGC模型的优化迭代提供强有力的数据支持。

在传统深度学习时代，数据工厂能够将图片中的目标类别进行精准标注；而到了AIGC时代，数据工厂的标注内容就发生了变化，需要对图片进行理解，并且通过文字与关键词完整描述出图片中的内容。

除此之外，数据工厂的数据收集、数据整理、数据供给等功能都是跨周期的需求，在传统深度学习时代已经证明，数据工厂能大大增强算法团队的工作效率，为整个AI算法解决方案和AI产品的开发、交付、迭代提供有力支持。

Rocky相信，未来AIGC时代的数据工厂会是一个非常大的机会点，作为AIGC领域的上游产业，不管是大厂自建数据工厂，还是专门的AIGC数据工厂公司，都会像在传统深度学习时代那样，成为AIGC产业链中不可或缺的关键一环。

8.2 AI绘画的“工作流”产品

2022年是AIGC元年，各路大模型争先恐后的发布，“百模大战”一触即发。

但是，市场真的需要这么多大模型吗？用户真的需要这么多大模型吗？

答案显而易见是否定的。并且大模型并不是万能的，有着明显的能力边界，单个大模型不足以形成足够的技术护城河与产品普惠。

历史不总是重复，但是会押韵。从AIGC之前的深度学习时代，也有非常多的公司迷信当时的人脸检测、人脸识别、目标检测、图像分割、图像分类算法。但时间告诉我们，深度学习时代的红利与价值，最后只赋能了互联网行业、安防行业、智慧城市行业以及智慧工业等强B端行业。

那么，AIGC时代和深度学习时代的区别是什么呢？

也很简单，AIGC时代有类似移动互联网时代那样的ToC可能性。因为不管是开源社区，还是AIGC时代的互联网大厂和AI公司，亦或者是绘画领域专业的个人，都开始尝试大模型+辅助工具的“工作流”产品，并且让这些产品触达每个用户。像妙鸭相机等APP已经显现移动互联网时代独有的快速商业落地与流量闭环能力。

Rocky在这里再举一些热门AI绘画“工作流”产品例子：

Stable Diffusion + LoRA + ControlNet的“三巨头”组合。
Stable Diffusion + 其他生成式模型（ChatPT，GAN等）的组合。
Stable Diffusion + 深度学习模型（分类，分割，检测等）的组合。
Stable Diffusion + 视频模型的组合。
Stable Diffusion + 数字人模型的组合。
......

这些“工作流”式的AIGC产品，不管是对ToB还是ToC，都有很强的普惠性与优化迭代可能性。同时可以为工业界、学术界以及竞赛界的AI绘画未来发展带来很多的势能与“灵感”，Rocky相信这会成为AI绘画行业未来持续繁荣的核心关键。

8.3 AI绘画模型未来需要突破的瓶颈

SDXL虽然是先进的开源AI绘画模型，但是其同样存在一些需要突破的瓶颈。

SDXL有时仍难以处理包含详细空间安排和详细描述的复杂提示词。同时手部生成依旧存在一定的问题，会出现多手指、少手指、鸡爪指、断指以及手部特征错乱等情况。除此之外，两个概念的互相渗透、互相混淆，或者是概念溢出等问题还是存在。

下图中展示了这些问题，左上角的图片展示了手部生成失败的例子；左下角和右下角的图片展示了概念渗透，互相混淆的例子；右上角的图片展示了对复杂描述与详细空间安排的提示词生成失败的例子：

SDXL生成的一些失败案例与瓶颈

客观来说，目前AI绘画领域的模型一定程度上都存在上述的瓶颈，既然有瓶颈，那么就有突破的方向与动力。

SDXL论文中也给出了future work，向我们阐述了未来图像生成式模型的研究重点（科研界风向标）：

Single stage（单阶段模型）：目前，SDX是二阶段的级联模型，使得显存占用和运算耗时都增大了。所以未来的一个价值点是想办法提出和SDXL有相似性能或者更强性能的单阶段模型。
Text synthesis（文本合成）：持续优化Text Encoder模型，提升模型对文本的理解能力。使用更加细粒度的文本编码器或者使用额外的约束增加文本与图像的一致性，或许是AI绘画领域的X因素。
Architecture（架构）：SDXL论文中简单地尝试了基于纯Transformer的架构，比如UViT和DiT，目前暂没有太大的进展。但是SDXL论文中认为通过精心的超参数研究，最终将能够扩展到以Transformer结构为主导的更高效的模型架构中。
Distillation（蒸馏）：通过指导性蒸馏、知识性蒸馏和逐步蒸馏等技术，来降低SDXL推理所需的计算量，并减少推理耗时。
New Model（新模型）：未来基于连续时间的EDM框架是非常有势能的候选扩散模型框架，因为它允许更大的采样灵活性，并且不需要噪声调度修正。
8.4 构建AI绘画产品的开发流程

就如同深度学习时代一样，构建AIGC时代算法产品与算法解决方案的开发流程是每个AIGC公司的必修课。

与传统深度学习时代作类比的话，我们可以发现AIGC时代中的算法产品开发流程有很多与之相似的地方。

下面是Rocky在AIGC时代积累的算法产品开发流程方法论，大家可以参考：

产品需求定义（与传统深度学习类似）
数据收集、筛选、标注（与传统深度学习不同，需要细分领域的专家对数据进行评估）
模型选择（与传统深度学习无脑选择的YOLO，ResNet，U-Net不同，AIGC时代对模型的选择也许要根据细分领域进行评估）
模型训练（与传统深度学习类似）
模型测试评估（与传统深度学习不同，需要细分领域的专家通过Prompt工程挖掘评估模型能力）
前处理与后处理（与传统深度学习类似）
工程化部署（与传统深度学习类似，在传统深度学习只能ToB的基础上，多了ToC可能性，既能端侧部署，也能上线部署）
8.5 AI绘画的多模态发展

在2023年OpenAI的开发者大会上，GPTs这个重量级产品正式发布，让人感到惊艳的同时，其生态快速繁荣。

如果说2023年年初是“百模大战”的话，那么2023年的年末就是“千GPTs大战”。

如此强的ToC普惠是传统深度学习时代未曾出现的，就连移动互联网时代在如此的势能面前都稍逊一筹，然而AIGC时代才刚刚开始。

所以Rocky认为AIGC时代的AI产品与应用会以AI绘画+AI视频+AI对话+AI语音+AI大模型+数字人等多模态的形式呈现，AI绘画会成为AI应用的关键一环，发挥重要作用。多模态的AI产品形态，会极大增强AI产品的ToC/ToB的普惠势能，也是AIGC时代发展的必经之路。

8.6 AI绘画的轻量化与端侧部署

在传统深度学习时代，AI模型的轻量化和端侧部署为ToB的可能性打下了坚实的基础，在端侧快速高效的使用目标检测、图像分割、图像分类、目标跟踪等算法，能够为智慧城市、智慧交通、智慧工业能领域创造非常大的价值。

历史不会重复，但会押韵。在AIGC时代中，AI绘画模型的轻量化和端侧部署依旧在ToC和ToB领域有着巨大的势能与市场，在可预见的未来，如果每个人都能方便快速的在端侧设备中使用AI绘画模型生成内容，这将大大推动各行各业的变革与重构。

同时，实时高效的生成AIGC内容，将是未来的元宇宙时代的“核心基建”。

更多思考与感悟，Rocky会持续补充，大家敬请期待！码字确实不易，希望大家能一键三连，多多点赞！

9. 推荐阅读

Rocky会持续分享AIGC的干货文章、实用教程、商业应用/变现案例以及对AIGC行业的深度思考与分析，欢迎大家多多点赞、喜欢、收藏和转发，给Rocky的义务劳动多一些动力吧，谢谢各位！

9.1 深入浅出完整解析Stable Diffusion（SD）核心基础知识

Rocky也对Stable Diffusion 1.x-2.x系列模型的核心基础知识做了全面系统的梳理与解析：

9.2 深入浅出完整解析Stable Diffusion中U-Net的前世今生与核心知识

Rocky对Stable Diffusion中最为关键的U-Net结构进行了深入浅出的全面解析，包括其在传统深度学习中的价值和在AIGC中的价值：

9.3 深入浅出完整解析LoRA(Low-Rank Adaptation)模型核心基础知识

对于AIGC时代中的“ResNet”——LoRA模型，Rocky也进行了深入浅出的全面讲解：

9.4 深入浅出完整解析ControlNet核心基础知识

AI绘画作为AIGC时代的一个核心方向，开源社区已经形成以Stable Difffusion为核心，ConrtolNet和LoRA作为首要AI绘画辅助工具的变化万千的AI绘画工作流。

ControlNet正是让AI绘画社区无比繁荣的关键一环，它让AI绘画生成过程更加的可控，更有助于广泛地将AI绘画应用到各行各业中：

9.5 深入浅出完整解析主流AI绘画框架核心基础知识

AI绘画框架正是AI绘画“工作流”的运行载体，目前主流的AI绘画框架有Stable Diffusion WebUI、ComfyUI以及Fooocus等。在传统深度学习时代，PyTorch、TensorFlow以及Caffe是传统深度学习模型的基础运行框架，到了AIGC时代，Rocky相信Stable Diffusion WebUI就是AI绘画领域的“PyTorch”、ComfyUI就是AI绘画领域的“TensorFlow”、Fooocus就是AI绘画领域的“Caffe”：

9.6 手把手教你成为AIGC算法工程师，斩获AIGC算法offer！

在AIGC时代中，如何快速转身，入局AIGC产业？如何成为AIGC算法工程师？如何在学校中系统性学习AIGC知识，斩获心仪的AIGC算法offer？

Don‘t worry，Rocky为大家总结整理了全面的AIGC算法工程师成长秘籍，为大家答疑解惑，希望能给大家带来帮助：

9.7 AIGC产业的深度思考与分析

2023年3月21日，微软创始人比尔·盖茨在其博客文章《The Age of AI has begun》中表示，自从1980年首次看到图形用户界面（graphical user interface）以来，以OpenAI为代表的科技公司发布的AIGC模型是他所见过的最具革命性的技术进步。

Rocky也认为，AIGC及其生态，会成为AI行业重大变革的主导力量。AIGC会带来一个全新的红利期，未来随着AIGC的全面落地和深度商用，会深刻改变我们的工作、生活、学习以及交流方式，各行各业都将被重新定义，过程会非常有趣。

那么，在此基础上，我们该如何更好的审视AIGC的未来？我们该如何更好地拥抱AIGC引领的革新？Rocky准备从技术、产品、商业模式、长期主义等维度持续分享一些个人的核心思考与观点，希望能帮助各位读者对AIGC有一个全面的了解：

9.8 算法工程师的独孤九剑秘籍

为了方便大家实习、校招以及社招的面试准备，同时帮助大家提升扩展技术基本面，Rocky将符合大厂和AI独角兽价值的算法高频面试知识点撰写总结成《三年面试五年模拟之独孤九剑秘籍》，并制作成pdf版本，大家可在公众号WeThinkIn后台【精华干货】菜单或者回复关键词“三年面试五年模拟”进行取用：

Rocky一直在运营技术交流群（WeThinkIn-技术交流群），这个群的初心主要聚焦于AI行业话题的讨论与研究，包括但不限于算法、开发、竞赛、科研以及工作求职等。群里有很多AI行业的大牛，欢迎大家入群一起交流探讨～（请备注来意，添加小助手微信Jarvis8866，邀请大家进群～）
