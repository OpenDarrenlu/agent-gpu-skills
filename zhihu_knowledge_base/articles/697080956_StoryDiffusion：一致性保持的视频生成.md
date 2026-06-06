# StoryDiffusion：一致性保持的视频生成

**作者**: meton​北京大学 计算机硕士

**原文链接**: https://zhuanlan.zhihu.com/p/697080956

---

论文阅读： storydiffusion http://arxiv.org/abs/2405.01434

无论是图片生成还是视频生成，内容一致性保持都是一个巨大的挑战。例如要根据文本描述生成漫画，那么我们总是希望漫画分镜图之间的主体特征、背景、主题属性可以保持一致。从漫画生成进一步延伸，如果希望将生成的漫画进一步插帧生成关于某段故事的视频，我们也希望视频帧之间有很高的一致性。
论文中作者提出一种基于diffusion model的两阶段视频生成方法，可以生成高度一致性保持的视频结果。


第一阶段 是提出一种免训练的热拔插attention模块CAB，可以直接将开源文生图基座模型（例如stable diffusion xl）的attention模块直接替换，然后来生成一批一致性保持度高的图片。以漫画生成场景为例：
● 选择一个开源文生图基座模型，例如SDXL，将网络结构中的attention替换成CAB模块，这个过程不需要重新训练微调。
● 对于一段较长的漫画故事文本描述，分拆成一批（几句）用于不同漫画分镜生成的prompt描述，对于这批prompt描述我们希望SDXL可以生成对应的一批主体一致性高度保持的漫画图。
● SDXL的模型输入是一个batch的prompt描述，对应的batch内的噪声图输入通过编码器变成batch的image latent（维度NxC）后，在这个batch中，某个image_latent作为Q输入attention模块时，这个image_latent会和batch内随机采样的其他image_latent进行concat后转化为attention操作的K和V（维度（S+N)xC ），然后再进行attention操作。这个过程的意义在于对于batch内某一个图片的生成中，也会考虑当前batch内其他图片的信息，这样可以更好的保持主体和属性的一致性，这个过程attention的权重没有改变，也没有重新训练。
● 通过CAB模块，可以生成一批主体特性/属性高度一致的漫画分镜图




第二阶段 对于前面生成的一批漫画分镜图，想进一步生成一致性高度保持的视频，作者在video diffusion model的基础上去集成一个semantic space motion predictor模块来实现，具体的操作是：
● 对于漫画的某两帧，利用image encoder转换到latent空间后（维度2xNxC），在latent空间上进行插帧变成FxNxC的latent特征，之后通过一个参数可训的semantic space motion 模块（其实就是一堆tansformer bock），之后后变成一批具有一致性语义的latent 特征（维度FxNxC）。
● 之后这些latent特征可以作为一种condition信息 拼接 prompt文本嵌入信息然后通过cross attention注入到常规的video diffusion训练中。

训练之后的视频生成模型，可以将前一阶段的每两个漫画帧作为输入，生成一段短视频。之后将多段短视频拼接起来后可以得到更长的视频，作者展示的是10s左右的漫画视频。
