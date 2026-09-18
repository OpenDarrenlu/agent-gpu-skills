# 深入浅出完整解析Stable Diffusion中U-Net的前世今生与核心知识

**作者**: Rocky Ding​北京科技大学 工学硕士

**原文链接**: https://zhuanlan.zhihu.com/p/642354007

---

​
目录
收起
1. 传统深度学习时代的U-Net
1.1 U-Net的“AI江湖”印象
1.2 U-Net的核心结构与细节
1.3 是什么让U-Net通向AIGC
2. Stable Diffusion中的U-Net
2.1 U-Net在Stable Diffusion中扮演的角色
2.2 Stable Diffusion中U-Net的完整核心结构
2.3 U-Net在AIGC时代中的核心结构与细节
2.4 GroupNorm
3. U-Net在Stable Diffusion中的训练和推理
3.1 U-Net在Stable Diffusion中的训练过程
3.2 U-Net在Stable Diffusion中的推理过程
4. 推荐阅读
4.1 深入浅出完整解析Stable Diffusion XL核心基础知识
4.2 深入浅出完整解析Stable Diffusion核心基础知识
4.3 深入浅出完整解析ControlNet核心基础知识
4.4 深入浅出完整解析LoRA核心基础知识
4.5 手把手教你如何成为AIGC算法工程师，斩获AIGC算法offer！
4.6 AIGC产业深度思考与分析
4.7 算法工程师的独孤九剑秘籍

本文的专栏：算法兵器谱
我的公众号：WeThinkIn
更多高价值内容欢迎关注我的知乎，公众号，专栏～

码字确实不易，希望大家能多多点赞！

大家好，我是Rocky。

2022年，Stable Diffusion横空出世，成为AI行业从传统深度学习时代过渡至AIGC时代的标志模型，并为工业界和投资界注入了新的活力，让AI再次性感。

在Stable Diffusion系列的第一篇文章中，Rocky已经详细讲解了Stable Diffusion的核心基础知识：

同时Rocky也深入浅出完整解析了Stable Diffusion XL模型的核心基础知识：

本文作为Stable Diffusion系列的第三篇文章，Rocky将深入浅出的讲解Stable Diffusion中U-Net的核心知识，包括U-Net在传统深度学习中的核心价值与在AICG中的核心价值，让我们来看看U-Net是如何在两个时代中同时从容，并大放异彩的。同时，Rocky也希望我们能借助Stable Diffusion系列文章更好的入门Stable Diffusion及其背后的AIGC领域。

话不多说，在Rocky毫无保留的分享下，让我们开始学习吧！

So，enjoy：


1. 传统深度学习时代的U-Net
1.1 U-Net的“AI江湖”印象

在2015年，传统深度学习时代的早期，U-Net: Convolutional Networks for Biomedical Image Segmentation（U-Net）正式发表，图像分割领域迎来了它的“ResNet”。

U-Net起初在生物医学图像这个细分领域取得了最佳的工业界效果，由于其简洁，高效，稳定的特性，随机被广泛的应用于图像分割的各个方向，比如智慧交通，智慧城市，工业检测等

可以说在传统深度学习时代，不管是实际业务、AI竞赛还是科研，U-Net都成为了当仁不让的图像分割通用Baseline。但是让人没想到的是，在8年后的AIGC时代到来后，U-Net顺应了时代的潮流，依旧爆发出了鲜活的生命力与价值。

1.2 U-Net的核心结构与细节

(1) Encoder-Decoder结构

U-Net最经典的特征是其Encoder-Decoder的结构，这样的结构简洁且高效，并且具备对称的“艺术”美感，也让U-Net具备了极强的生命力与适应性。

传统深度学习时代的U-Net

其中左半部分的Encoder模块负责进行特征的提取与学习，Encoder模块可以由ResNet、VGG、EfficientNet等一流特征提取模型担任，所以Encoder模块具备较强的工程潜力与科研势能。与此同时Encoder模块可以增加对扰动噪声的鲁棒性，减少过拟合的风险，降低运算量以及增加感受野的大小等作用。

而右半部分的Decoder模块则负责将feature map恢复到原始分辨率，并使skip-connection这个关键一招融合了浅层的位置信息与深层的语义信息。与此同时，Decoder模块和Encoder模块一样可以由ResNet、VGG、EfficientNet等一流模型担任，从而使得U-Net的变体非常繁荣，增加了工程“魔改”的可玩性。

(2) U-Net结构细节挖掘

讲完Encoder-Decoder结构的整体框架，Rocky再向大家介绍一下Encoder-Decoder结构中的一些能够成为通用范式和经典Tricks的细节操作。

从上图的Encoder-Decoder结构中可以看到，U-Net是一个全卷积神经网络，网络最后一层使用了浅蓝色箭头，表示1*1卷积，其完全取代了全连接层，使得模型的输入尺寸不再受限制，极大增强了U-Net在各种应用场景的兼容性。

上图中的蓝色和白色框表示feature map，深蓝色箭头表示 3x3 卷积，padding=0 ，stride=1其用于特征提取。由于padding=0，所以每次经过卷积运算，feature map将有一定程度的下采样。深红色箭头表示max pooling，stride=2，用于降低维度。将卷积和max pooling两者结合，能够对feature map进行特征提取的同时从容进行下采样。

max pooling操作

上图中的绿色箭头表示Upsample操作，对feature map进行上采样从而恢复维度。

Upsampling常用的方式有两种：转置卷积和插值。两者的详细内容可以阅读Rocky之前的文章：【三年面试五年模拟】算法工程师的独孤九剑秘籍（前十二式汇总篇）V1版。而在U-Net中，使用了bilinear双线性插值。

在Encoder和Decoder两个模块之间，使用skip-connection作为桥梁，用于特征融合，将浅层的位置信息与深层的语义信息进行concat操作。图中用灰色箭头表示skip-connection，其中“copy”就是concat操作，而“crop”则通过裁剪使得两个特征图尺寸一致。

1.3 是什么让U-Net通向AIGC

讲完U-Net在传统深度学习时代的核心知识点与价值，接下来Rocky再阐述一下为何在AIGC时代，U-Net成为了Stable Diffusion这个划时代模型的关键结构。

在投资界有一句话，叫“在上个时代适应的越好的人，很有可能是下一个时代最大的失败者”。这个逻辑套用在技术上再合适不过了，有太多技术产生，也有太多技术消亡，而学习技术并从事技术行业的工人们就会背负更多的沉没成本与风险。

但是U-Net不这么认为，其同时成为了AIGC与传统深度学习这两个时代的弄潮儿，在AIGC时代，U-Net有了新的内涵和面貌，并且“文艺复兴”。

那么，是什么让U-Net能够通向AIGC，跨过周期呢？

主要有以下四个特质：

U-Net中Encoder模块的压缩特质。作为Encoder模块最初的应用，输入的图像经过下采样，抽取出比原图小得多的高维特征，相当于进行了压缩操作。这和Stable diffusion的latent逻辑不谋而合，随即在AIGC时代“文艺复兴”。
U-Net中Decoder模块的去噪特质，作为Decoder模块最初的应用，在AIGC时代“文艺复兴”。
U-Net整体结构上的简洁、稳定和高效，使得其在Stable Diffusion中能够从容的迭代去噪声，能够撑起Stable Diffusion的整个图像生成逻辑。
Encoder-Decoder结构的强兼容性，让U-Net不管是在分割领域，还是在生成领域，都能和Transformer等新生代模型的从容融合。

U-Net发表8年后的AIGC时代里，正是这些特质让U-Net顺应了时代的潮流，依旧爆发出了鲜活的生命力与价值。

2. Stable Diffusion中的U-Net
2.1 U-Net在Stable Diffusion中扮演的角色
Stable Diffusion结构图，U-Net在最核心的位置

Stable Diffusion中的U-Net包含约860M的参数，在float32的精度下，约占3.4G的存储空间。

在上图中可以看到，U-Net是Stable Diffusion中的核心模块。U-Net主要在“扩散”循环中对高斯噪声矩阵进行迭代降噪，并且每次预测的噪声都由文本和timesteps进行引导，将预测的噪声在随机高斯噪声矩阵上去除，最终将随机高斯噪声矩阵转换成图片的隐特征。

在U-Net执行“扩散”循环的过程中，Content Embedding始终保持不变，而Time Embedding每次都会发生变化。每次U-Net预测的噪声都在Latent特征中减去，并且将迭代后的Latent作为U-Net的新输入。

总的来说，如果说Stable Diffusion是“优化噪声的艺术”，那么U-Net将是这个“艺术”的核心主导者。

2.2 Stable Diffusion中U-Net的完整核心结构

在讲解Stable Diffusion中U-Net的各个核心模块之前，我们先看看U-Net在Stable Diffusion中的完整结构：

Stable Diffusion U-Net完整结构图
2.3 U-Net在AIGC时代中的核心结构与细节

Stable Diffusion中的U-Net，在Encoder-Decoder结构的基础上，增加了Time Embedding模块，Spatial Transformer(Cross Attention)模块和self-attention模块。

(1) Time Embedding模块

首先，什么是Time Embedding呢？

Time Embedding（时间嵌入）是一种在时间序列数据中用于表示时间信息的技术。时间序列数据是指按照时间顺序排列的数据，例如股票价格、天气数据、传感器数据等。时间嵌入的目的是将时间作为一个特征进行编码，以便在深度学习模型中更好地学习时间相关性特征。

Time Embedding的基本思想是将时间信息映射到一个连续的向量空间，使得时间之间的关系可以被模型学习和利用。

Time Embedding的使用可以帮助深度学习模型更好地理解时间相关性，从而提高模型的性能。比如在Stable Diffusion中，将Time Embedding引入U-Net中，帮助其在扩散过程中从容预测噪声。

Stable Diffusion需要迭代多次对噪音进行逐步预测，使用Time Embedding就可以将time编码到网络中，从而在每一次迭代中让U-Net更加合适的噪声预测。

讲完Time Embedding的核心基础知识，我们再解析一下Stable Diffusion中U-Net的Time Embeddings模块是如何构造的：

可以看到，Time Embeddings模块 + Encoder模块中原本的卷积层，组成了一个Residual Block结构。它包含两个卷积层，一个Time Embedding和一个skip Connection。而这里的全连接层将Time Embedding变换为和Latent Feature一样的维度。最后通过两者的加和完成time的编码。

(2) Spatial Transformer(Cross Attention)模块

在Stable Diffusion中，使用了Spatial Transformer来表示类Cross Attention模块。

按照惯例，我们先理解一下什么是Cross Attention？

Cross Attention是一种多头注意力机制，它可以在两个不同的输入序列之间建立关联，并且可以将其中一个输入序列的信息传递给另一个输入序列。

在计算机视觉中，Cross Attention可以用于将图像与文本之间的关联建立。例如，在图像字幕生成任务中，Cross Attention可以将图像中的区域与生成的文字之间建立关联，以便生成更准确的描述。

Stable Diffusion中使用Cross Attention模块控制文本信息和图像信息的融合交互，通俗来说，控制U-Net把噪声矩阵的某一块与文本里的特定信息相对应。

讲完Cross Attention的核心基础知识，我们再解析一下Stable Diffusion中U-Net的Cross Attention模块是如何构造的：

Stable Diffusion中的CrossAttention结构

可以看到，Latent Feature和Context Embedding作为输入，将两者进行Cross Attenetion操作，将图像信息和文本信息进行了融合，整体上是一个经典的Transformer流程。

2.4 GroupNorm

Rocky在这里再讲一个Stable Diffusion中U-Net的细节Trick，那就是U-Net中全部采用GroupNorm进行归一化。

GroupNorm有如下的优点：

独立于Batch：GroupNorm不依赖于Batch大小，这使得它在处理小Batch数据或者Batch大小变化较大的情况时仍能保持稳定性。这在生成任务中尤其重要，因为它允许使用更小的Batch而不会牺牲性能。
提升训练稳定性：在生成对抗网络（GANs）等生成任务中，模型训练可能非常不稳定。GroupNorm可以帮助增强模型的训练稳定性，从而产生更高质量的生成结果。
减少内存消耗：由于GroupNorm允许使用更小的Batch而不影响性能，因此可以减少训练期间的内存消耗，这对于资源限制较大的环境特别重要。

关于GroupNorm的详细知识，大家可以阅读Rocky之前的文章：

3. U-Net在Stable Diffusion中的训练和推理
3.1 U-Net在Stable Diffusion中的训练过程

在Stable Diffusion中，U-Net在不断的训练过程中主要学会了一件事，那就是去噪！去噪！还是tmd去噪！

想要让U-Net能够高效去噪，并获得图像的隐特征，我们就要让U-Net知道什么是噪声数据。

于是我们在训练的预处理过程中，向训练集有策略地加入噪声。

这个加噪策略主要包括设定不同级别的噪声，比如说0-100共101个强度的噪声，在每个Batch中，随机加入1-n个101强度序列中的噪声，生成噪声图片。

数据加噪策略

加噪+噪声强度+加噪次数+原数据集，构成了Stable Diffusion中U-Net训练数据的基石。

有了数据预处理的大逻辑，在训练过程中，U-Net需要在已知噪声强度的条件下，不断学习提升从噪声图片中计算出噪声的能力。

需要注意的是，Stable Diffusion中的U-Net并不直接输出无噪声的原数据，而是去预测原数据上所加过的噪声。

Stable Diffusion中U-Net的训练过程

如上图所示，Stable Diffusion中U-Net的训练一共分四步：

从训练集中选取一张加噪过的图片和噪声强度，比如上图的加噪街道图和噪声强度3。
将数据输入U-Nnet，并且预测噪声矩阵。
将预测的噪声矩阵和实际噪声矩阵（Label）进行误差的计算。
通过反向传播更新U-Net的参数。
3.2 U-Net在Stable Diffusion中的推理过程

在推理阶段中，我们将U-Net预测的噪声不断在噪声图片中减去就能恢复出图片的隐特征了。

当我们完成了U-Net在Stable Diffusion中的训练，如果我们再将噪声强度和噪声图输入U-Net，那么U-Net就能较准确地预测出有加在原素材上的噪声：

Stable Diffusion中U-Net预测噪声

有了U-Net对噪声的强预测能力，在Stable Diffusion的推理过程中，我们就可以使用U-Net循环预测噪声，并在噪声图上逐步减去这些被预测出来的噪声，从而得到一个我们想要的高质量的图像隐特征，去噪流程如下图所示：

Stable Diffusion推理过程
4. 推荐阅读

Rocky会持续分享AIGC的干货技术教程，经典模型讲解，实用的工具应用以及对AIGC行业的深度思考，欢迎大家多多点赞，喜欢，收藏，给Rocky的义务劳动多一些动力吧，谢谢各位！

4.1 深入浅出完整解析Stable Diffusion XL核心基础知识

在此之前，Rocky也对Stable Diffusion XL的核心基础知识作了比较系统的梳理与总结：

4.2 深入浅出完整解析Stable Diffusion核心基础知识

当然的，Rocky也对Stable Diffusion的核心基础知识作了比较系统的梳理与总结：

4.3 深入浅出完整解析ControlNet核心基础知识

AI绘画作为AIGC时代的图像内容核心方向，开源社区已经形成以Stable Difffusion为核心，ConrtolNet和LoRA作为首要AI绘画辅助工具的变化万千的AI绘画工作流。

ControlNet正是让AI绘画社区无比繁荣的关键一环，它让AI绘画生成过程更加的可控，有助于更广泛地将AI绘画应用到各行各业中。

同时，由于Stable Diffusion + LoRA + ControlNet三巨头的强强联合，形成了一个变化多样的AI绘画“大框架”。

4.4 深入浅出完整解析LoRA核心基础知识

对于AIGC时代中的“ResNet”——LoRA，Rocky也进行了讲解，大家可以按照Rocky的步骤方便的进行LoRA模型的训练，繁荣整个AIGC生态：

4.5 手把手教你如何成为AIGC算法工程师，斩获AIGC算法offer！

在AIGC时代中，如何快速转身，入局AIGC产业？成为AIGC算法工程师？如何在学校中学习AIGC系统性知识，斩获心仪的AIGC算法offer？

Don‘t worry，Rocky为大家总结整理了全维度的AIGC算法工程师成长秘籍，为大家答疑解惑，希望能给大家带来帮助：

4.6 AIGC产业深度思考与分析

2023年3月21日，微软创始人比尔·盖茨在其博客文章《The Age of AI has begun》中表示，自从1980年首次看到图形用户界面（graphical user interface）以来，以OpenAI为代表的科技公司发布的AIGC模型是他所见过的最具革命性的技术进步。

Rocky也认为，AIGC及其生态链，会成为AI行业重大变革的主导力量。AIGC会带来一个全新的红利期，未来随着AIGC的全面落地和深度商用，会深刻改变我们的工作，生活，学习以及交流方式，许多行业都将被重新定义，过程会非常有趣。

2023年的“疯狂三月”，世界上主要科技公司与研究机构们争先恐后发布关于AIGC的最新进展，让人目不暇接，吃瓜群众们纷纷惊呼不已。那么，在狂欢过后，我们该如何更好的审视AIGC的未来？我们该如何更好地拥抱AIGC引领的革新？接下来Rocky准备从技术，产品，长期主义等维度分享一些个人的核心思考与观点，希望能帮助各位读者对AIGC有一个全面的了解。

4.7 算法工程师的独孤九剑秘籍

为了便于大家实习，校招以及社招的面试准备与技术基本面的扩展提升，Rocky将符合大厂和潜力独角兽价值的算法高频面试知识点撰写总结成《三年面试五年模拟之独孤九剑秘籍》，并制作成pdf版本，大家可在公众号WeThinkIn后台【精华干货】菜单或者回复关键词“三年面试五年模拟”进行取用。

Rocky一直在运营技术交流群（WeThinkIn-技术交流群），这个群的初心主要聚焦于AI行业话题的讨论与研究，包括但不限于算法，开发，竞赛，科研以及工作求职等。群里有很多AI行业的大牛，欢迎大家入群一起交流探讨～（请添加小助手微信Jarvis8866，邀请大家进群～）
