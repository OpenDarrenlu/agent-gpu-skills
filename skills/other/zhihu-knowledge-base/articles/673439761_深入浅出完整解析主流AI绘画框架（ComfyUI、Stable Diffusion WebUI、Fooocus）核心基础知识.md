# 深入浅出完整解析主流AI绘画框架（ComfyUI、Stable Diffusion WebUI、Fooocus）核心基础知识

**作者**: Rocky Ding​北京科技大学 工学硕士

**原文链接**: https://zhuanlan.zhihu.com/p/673439761

---

​
目录
收起
1. 从0到1保姆级Stable Diffusion WbUI使用教程（全网最详细）
1.1 从0到1在端脑云平台快速上手使用Stable Diffusion WebUI
1.2 从0到1在本地安装部署Stable Diffusion WebUI进行AI绘画（全网最详细讲解）
1.3 Stable Diffusion WebUI文生图全解析
1.4 Stable Diffusion WebUI图生图全解析
1.5 Stable Diffusion WebUI Extras页面使用解析
1.6 Stable Diffusion WebUI PNG Info、Checkpoint Merge页面使用解析
2. 从0到1深入浅出全面解析Stable Diffusion WebUI热门插件
2.1 ADetailer插件功能
2.2 深入浅出完整解析EasyPhoto的AI写真功能
2.3 深入浅出完整解析Facechain的AI写真功能
2.4 ControlNet插件
2.5 Segment Anything插件
2.6 AnimateDiff插件
2.7 Tiled Diffusion/VAE插件
2.8 SuperMerge插件
3. 从0到1保姆级ComfyUI使用教程（全网最详细）
3.1 从0到1在端脑云平台快速上手使用ComfyUI
3.2 零基础ComfyUI框架原理拆解（全网最详细讲解）
3.3 ComfyUI特性
3.4 ComfyUI文生图全解析
3.5 ComfyUI图生图全解析
3.6 ComfyUI中LoRA、ControlNet、GAN使用全解析
4. 从0到1保姆级Fooocus使用教程
5. 从0到1保姆级Stable Diffusion WebUI-forge使用教程
6. 推荐阅读
6.1 深入浅出完整解析Stable Diffusion 3（SD 3）和FLUX.1系列核心基础知识
6.2 深入浅出完整解析Stable Diffusion XL（SDXL）核心基础知识
6.3 深入浅出完整解析Stable Diffusion（SD）核心基础知识
6.4 深入浅出完整解析Stable Diffusion中U-Net的前世今生与核心知识
6.5 深入浅出完整解析LoRA（Low-Rank Adaptation）模型核心基础知识
6.6 深入浅出完整解析ControlNet核心基础知识
6.7 深入浅出完整解析Sora等AI视频大模型核心基础知识
6.8 深入浅出完整解析AIGC时代Transformer核心基础知识
6.9 手把手教你成为AIGC算法工程师，斩获AIGC算法offer！
6.10 AIGC产业的深度思考与分析
6.11 算法工程师的独孤九剑秘籍
6.12 深入浅出完整解析AIGC时代中GAN系列模型的前世今生与核心知识
本文的专栏：Rocky Ding的AI算法兵器谱
我的公众号：WeThinkIn
更多AI行业干货内容欢迎关注我的知乎，公众号，专栏～

码字不易，希望大家能多多点赞，给Rocky更多坚持写下去的动力，谢谢大家！

大家好，我是Rocky。

2022年是AIGC时代的元年，AI绘画领域全面爆发。Stable Diffusion系列模型、LoRA系列模型、ControlNet系列模型、GAN系列模型以及各种AI绘画辅助插件组成的变化万千的“工作流”让AI绘画在ToB和ToC方向都有了不可限量的可能性，同时开源生态持续繁荣发展，让AI绘画彻底破圈！

那么这些变化万千的AI绘画“工作流”在哪些载体上运行呢？或者说海量的AI绘画内容是从哪里源源不断地生产出来呢？这里就要提到我们本文的核心内容——AI绘画框架了。

AI绘画框架正是AI绘画“工作流”的运行载体，目前主流的AI绘画框架有Stable Diffusion WebUI、ComfyUI以及Fooocus。在传统深度学习时代，PyTorch、TensorFlow以及Caffe是传统深度学习模型的运行基础框架，到了AIGC时代，Rocky相信Stable Diffusion WebUI就是AI绘画领域的“PyTorch”、ComfyUI就是AI绘画领域的“TensorFlow”、Fooocus就是AI绘画领域的“Caffe”。

因此在本文中，Rocky主要对上述三个AI绘画框架的全维度各个方面都做一个深入浅出的分析总结（从0到1对Stable Diffusion WebUI/ComfyUI/Fooocus进行安装搭建，从0到1使用Stable Diffusion WebUI/ComfyUI/Fooocus进行AI绘画的保姆级教程，深入浅出介绍Stable Diffusion WebUI/ComfyUI/Fooocus的各模块功能介绍，深入浅出介绍Stable Diffusion WebUI/ComfyUI/Fooocus的高阶用法，Stable Diffusion WebUI/ComfyUI/Fooocus的最新资源汇总，相关配套工具使用等），和大家一些探讨学习，让我们在AIGC时代更好地融入和从容。

相信很多刚入门AI绘画的小伙伴都对如何配置AI绘画框架一头雾水，一时半会儿也找不到快速入门的AI绘画应用平台，同时苦于没有算力支持AI绘画模型的使用，这些因素让很多热爱AI绘画的小伙伴们望而止步。

现在我们不用担心这些了，因为端脑云平台出现了，端脑云是由端脑科技开发的 AIGC 工具平台，会不断集合市面上主流的AI绘画框架，通过分布式算力技术整合优化了分散的计算资源，为大家提供强大的算力支持。

总而言之，端脑云平台能够帮助我们快速入门使用AI绘画领域热门的AI绘画框架，无需购买设备或本地部署应用，同时相比阿里云等中心化的云算力平台，端脑云的综合成本会降低90%以上。（方便使用！花费成本低！快速上手！适合所有AI绘画领域的学习者和使用者！）

目前端脑云平台支持的AI绘画工具如下所示，基本上包含了当今的最强AI绘画生产力工具：

Stable Diffusion WbUI保姆级应用：一键部署AI绘画环境，全面支持huggingface和C站。
Stable Diffusion WbUI高级版应用：一键部署AI绘画环境，支持命令行交互，支持百度网盘上传文件。
ComfyUI保姆级应用：基于节点流的Stable Diffusion，具备精准的工作流制定和完善的可复现性。
Fooocus保姆级应用：类似Stable Diffusion，一款简单易用的AI绘画工具。

端脑云平台资源分享：

端脑云福利传送门：Cephalon Cloud
持续更新：主流AIGC应用平台持续更新，目前已上线的有Stable Diffusion、Jupyter、Web SSH、SSH等。
即开即用：云端部署，无需下载，开箱即用，体验便捷，对新手非常友好。
社区镜像：定制专属社区镜像，一键使用，无需重复部署。
性价比之最：主打全网最高性价比的算力服务（4090显卡包月价格低至0.59/小时）。
社区共建：连接更多AIGC行业资源与人脉，共建端脑AI社区。

接下来，就跟着Rocky的脚步，我们一起来从0到1全方位使用端脑云平台进行AI绘画的创作吧！

1. 从0到1保姆级Stable Diffusion WbUI使用教程（全网最详细）

Stable Diffusion WebUI是一个基于gradio架构的AI绘画框架，不仅支持Stable Diffusion的最基础的文生图、图生图以及图像inpainting功能，还支持Stable Diffusion的很多拓展功能，很多与Stable Diffusion相关的拓展应用都可以用插件的方式安装在Stable Diffusion WebUI上，非常方便实用。

Stable Diffusion WebUI界面

得益于新手友好的操作流程与丰富的插件生态，Stable Diffusion WebUI已经成为AI绘画领域最火爆的AI绘画框架。目前端脑云平台已经集成了Stable Diffusion WebUI，我们能够方便的在端脑云平台上使用Stable Diffusion Web UI进行AI绘画。

接下来，就让我们跟随着Rocky的脚步，一起使用端脑云平台中的Stable Diffusion WebUI进行AI绘画！

1.1 从0到1在端脑云平台快速上手使用Stable Diffusion WebUI

我们首先打开端脑云网站：

可以看到端脑云平台中有一个官方应用栏目，我们可以在里面找到Stable Diffusion WbUI框架、Stable Diffusion WbUI高级版框架、ComfyUI框架、Fooocus框架等热门AI绘画应用框架。

我们点击Stable Diffusion WbUI框架，先选择应用类型，我们选择“Stable Diffusion WbUI”。然后我们选择使用方式，我们选择“应用模式”。接着我们再选择GPU性能，我们根据自己的需求，选择适合的GPU即可。最后我们点击“立即使用”，就能开启一个新的应用了！

如下图所示，我们进入端脑云界面后，我们点击左侧的“我的应用”，可以看到正在运行的Stable Diffusion WbUI框架，我们点击右侧的“进入应用”，即可开始Stable Diffusion WbUI框架的使用：

第一次打开Stable Diffusion WbUI框架时，会弹出一个登陆页面，我们只需要将端脑云界面上的登录账号和登录密码输入即可登录：

登陆后，我们就能在端脑云上看到Stable Diffusion WbUI框架的可视化界面：

我们可以看到，端脑云上已经集成了大量的Stable Diffusion WbUI的插件与依赖。主要的热门插件包括SadTalker、Temporal-Kit、Deforum、OpenPose编辑器、Inpatinting Anything、超级模型融合、WD1.4标签器、ADetailer、AnimateDiff等，可谓是非常丰富与齐全，基本上能满足我们的所有AI绘画与AI视频的使用需求。

大家可以通过Cephalon Cloud注册来方便地使用Stable Diffusion WbUIAI绘画框架。

接下来Rocky输入如下的Prompt：

1girl,bangs,black_hair,blunt_bangs,bracelet,branch,building,city,city_lights,cityscape,couch,fishnets,indoors,ivy,jewelry,leaf,looking_at_viewer,night,night_sky,on_couch,palm_tree,plant,potted_plant,short_hair,sitting,skyline,skyscraper,solo,star_(sky),starry_sky,tanabata,tanzaku,tree,(Snakeskin mesh tights),(Fishnet shirt),single_thighhigh。

可以看到，下图中生成了一张我们想要的图片：

更多AI绘画玩法，大家可以在端脑云上逐步尝试学习！

1.2 从0到1在本地安装部署Stable Diffusion WebUI进行AI绘画（全网最详细讲解）

【1】本地安装Stable Diffusion WebUI

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

【2】Stable Diffusion WebUI操作页面显示更多参数设置

当我们刚安装好Stable Diffusion WebUI，并打开操作界面时，我们会发现有一些重要的参数配置是没有显示的，比如说：VAE设置、Clip skip设置、Eta noise seed delta设置等。

我们可以通过点击Stable Diffusion WebUI操作界面的Settings按钮，进入Settings界面。然后再点击User interface按钮，可以看到Quicksettings list栏目，我们在栏目中分别搜索点击sd_vae、CLIP_stop_at_last_layers、eta_noise_seed_delta，选中这三个参数后，我们点击界面上方的Apply settings按钮，再点击Reload UI按钮，即可重启保存修改配置。完整流程如下图所示：

Stable Diffusion WebUI操作界面增加参数设置栏的流程

等重启完成后，我们可以看到Stable Diffusion WebUI操作界面上方多出了SD VAE栏目、Clip skip栏目以及Eta noise seed delta栏目，我们可以方便地调节这些关键参数。

1.3 Stable Diffusion WebUI文生图全解析

Stable Diffusion WebUI文生图的操作界面主要分为：模型区域、功能区域、参数区域、出图区域四个部分。

Stable Diffusion WebUI文生图的操作界面
Prompt（正向提示词）：希望SD模型生成的内容。
Negative Prompt（反向提示词）：不希望SD模型生成的内容。
Sampling method（采样方法）：推荐选择 Euler a 或 DPM++ 系列，采样速度较快。
Sampling steps（迭代步数）：设置的数值越大，生成的图像质量越好，同时生成时间也越长，一般设置为20-50区间就能生成质量较高的图片。
Hires. fix（超分辨率精绘）：当我们开启Hires. fix后，在图像生成过程中会將SD系列模型原始生成的图当作“草图”，先用超分算法（Upscale）把原始生成图放大到我們指定的倍数（Upscale by），再將超分后的“草图”用图生图（img2img）的方式重新生成一次，最后获得优化生成后的图片。我们可以调整的参数还有Hires steps（精绘步数）和Denoising strength（噪声强度），Hires steps设置为20-30即可，Denoising strength设置为0.4-0.7即可。使用Hires. fix可以大幅提高生成图片中人物的臉部、眼睛、头发细节质量，同时图片整体的精細度也有肉眼可见的提升。
Refiner：SDXL的Refiner，在使用SDXL模型时可以启动用于提升生成图片的质量。
Width/Height（生成图片的宽高）：设置的越大越消耗显存，生成时间也越长。使用SD模型时可以设置512x512、512x768等尺寸；使用SDXL模型时可以设置768x1024、1024x1024以及更大的尺寸。
CFG Scale（classifier-free guidance scale，提示词相关性）：CFG Scale设置的越大越与提示词相关，设置的越小则越不相关，一般建议4-12区间即可。过低的CFG会让出图饱和度偏低，过高的CFG则会出现粗矿的线条或过度锐化的图像，甚至于画面出现严重的崩坏。
Batch count：表示一共生成图片的批次数，是串行处理的，如果Batch count设置为2，Batch size设置为4，那么我们分两次并行生成4张图片，一共生成8张图片（2x4）。
Batch size：表示一个Batch中生成图片的数量，是并行处理的，Batch size设置的越大，占用的显存越大。
Seed（种子数）：设置为-1 表示每次生成图片时使用随机种子，相同的种子数可以保持生成图像的一致性，如果觉得生成图片的构图不错，但对风格不满意，可以将种子数固定，再调整参数进行优化生成。
Script：包含了一些组合操作，比如Prompt matrix、Prompts from file or textbox、X/Y/Z plot等。

在这里Rocky也向大家详细讲解一下Stable Diffusion WebUI文生图功能的底层Pipeline代码：

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm


model_id = "runwayml/stable-diffusion-v1-5"
# 1. 加载autoencoder
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
# 2. 加载tokenizer和text encoder 
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
# 3. 加载扩散模型UNet
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
# 4. 定义noise scheduler
noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False, # don't clip sample, the x0 in stable diffusion not in range [-1, 1]
    set_alpha_to_one=False,
)

# 将模型复制到GPU上
device = "cuda"
vae.to(device, dtype=torch.float16)
text_encoder.to(device, dtype=torch.float16)
unet = unet.to(device, dtype=torch.float16)

# 定义参数
prompt = [
    "A dragon fruit wearing karate belt in the snow",
    "A small cactus wearing a straw hat and neon sunglasses in the Sahara desert",
    "A photo of a raccoon wearing an astronaut helmet, looking out of the window at night",
    "A cute otter in a rainbow whirlpool holding shells, watercolor"
]
height = 512
width = 512
num_inference_steps = 50
guidance_scale = 7.5
negative_prompt = ""
batch_size = len(prompt)
# 随机种子
generator = torch.Generator(device).manual_seed(2023)


with torch.no_grad():
 # 获取text_embeddings
 text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
 # 获取unconditional text embeddings
 max_length = text_input.input_ids.shape[-1]
 uncond_input = tokenizer(
     [negative_prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
 )
      uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
 # 拼接为batch，方便并行计算
 text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

 # 生成latents的初始噪音
 latents = torch.randn(
     (batch_size, unet.in_channels, height // 8, width // 8),
     generator=generator, device=device
 )
 latents = latents.to(device, dtype=torch.float16)

 # 设置采样步数
 noise_scheduler.set_timesteps(num_inference_steps, device=device)

 # scale the initial noise by the standard deviation required by the scheduler
 latents = latents * noise_scheduler.init_noise_sigma # for DDIM, init_noise_sigma = 1.0

 timesteps_tensor = noise_scheduler.timesteps

 # Do denoise steps
 for t in tqdm(timesteps_tensor):
     # 这里latens扩展2份，是为了同时计算unconditional prediction
     latent_model_input = torch.cat([latents] * 2)
     latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t) # for DDIM, do nothing

     # 使用UNet预测噪音
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

     # 执行CFG
     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

     # 计算上一步的noisy latents：x_t -> x_t-1
     latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
    
 # 注意要对latents进行scale
 latents = 1 / 0.18215 * latents
 # 使用vae解码得到图像
    image = vae.decode(latents).sample

【1】Prompt语法深入浅出讲解

在Stable Diffusion WebUI中，正向提示词（Prompt）是引导SD系列模型生成图像内容的关键信息。我们可以在提示词中使用“()”可以增加SD模型对括号内提示词的注意力，即增加提示词权重；而使用“[]”则会减少提示词的权重。

下面是一些具体的使用例子：

a (word) - 提示词的权重增加1.1倍 
a ((word)) - 提示词的权重增加1.21倍（= 1.1 * 1.1） 
a [word] - 提示词的权重减少1.1倍
a (word:1.5) - 提示词的权重增加1.5倍
a (word:0.25) - 提示词的权重减少4倍（= 1 / 0.25） 
a \(word\) - 使用转义字符取消()的改变提示词权重效果，让（）成为普通字符
不同提示词权重下的SD模型图片生成效果

那么Stable Diffusion WebUI是如何让这些权重注意力起作用呢？

简单来说，在输入提示词到Stable Diffusion Text Encoder之前，WebUI通过上面讲到的权重规则将提示词分离成Text（文本）和Weight（权重）两个独立的部分。接着将Text部分输入Tokenizer模块转换成tokens，然后再输入Text Encoder模型编码成Text Embeddings。完成了上面的操作，我们再将权重和对应的Text Embeddings相乘，再乘上一个正则化系数，将对应的Embeddings向量进行缩放（scale），代表着这个提示词的增强和减弱。完整的提示词权重规则代码在stable-diffusion-webui/modules/prompt_parser.py脚本中，完整的提示词权重生效代码在stable-diffusion-webui/modules/sd_hijack_clip.py脚本中，大家可以按需查阅。

下面有详细的流程注解，方便大家直观的理解WebUI的提示词权重生效逻辑：

# 输入提示词
Prompt = (woman)

# 通过prompt_parser.py脚本解析成Text（文本）和Weight（权重）两个独立的部分
parsed: [['woman', 1.1]]

# 将Text（文本）输入Tokenizer模块转换成tokens
tokenized: [[2308]]

# 将tokens输入Text Encoder模型编码成Text Embeddings
z_original = tensor([[ 0.5410,  1.2411, -0.4761,  ..., -0.3535, -1.1046, -1.7921]])
original_mean = z_original.mean()

# 将Text Embeddings与对应的Weight（权重）相乘
z_new = tensor([[ 0.5951,  1.3652, -0.5237,  ..., -0.3888, -1.2150, -1.9714]])
new_mean = z_new.mean()

# 计算正则化系数
r = (original_mean / new_mean)

# 最终的Text Embeddings
z = z_new * r
z = tensor([[ 0.5943,  1.3634, -0.5230,  ..., -0.3883, -1.2134, -1.9688]])

除了正向提示词，负向提示词（Negative Prompt）同样重要，负向提示词的主要作用是指导SD系列模型避免生成包含某些特定元素或特征的图像。这是通过在图像生成过程中显式地告诉模型哪些内容是不希望出现的来实现的。通过这种方式，Negative Prompts增加了对生成结果的控制，帮助我们更精确地定制想要的输出图像。

下面Rocky就带着大家详细了解一下负向提示词（Negative Prompt）是如何在SD系列模型中起作用的：

【2】Negative Prompt深入浅出讲解

我们知道SD系列模型采用了CFG技术来提升生成图像的质量。当我们使用CFG技术后，SD系列模型的去噪过程不仅仅依赖条件扩散模型，也依赖无条件扩散模型。具体的计算公式如下所示：

\begin{aligned}\text{pred_noise} &= w \cdot \text{cond_pred_noise} + (1-w) \cdot \text{uncond_pred_noise} \\ &= w\mathbf{\epsilon}_\theta\big(\mathbf{x}_t, t, \mathbf{c}\big)+(1-w)\mathbf{\epsilon}_\theta\big(\mathbf{x}_t, t\big) \end{aligned}\\ =\mathbf{\epsilon}_\theta\big(\mathbf{x}_t, t\big)+w(\mathbf{\epsilon}_\theta\big(\mathbf{x}_t, t, \mathbf{c}\big)-\mathbf{\epsilon}_\theta\big(\mathbf{x}_t, t\big)) \\

这里的w为guidance scale，当w越大时，condition起的作用越大，即生成的图像和输入文本更加一致。我们说的Negative Prompt就是无条件扩散模型的Text输入，当我们不输入Negative Prompt时会将Text置为空字符串来实现无条件扩散模型。

在模型推理过程中，我们可以使用不为空的Negative Prompt来避免模型生成的图像包含我们不想要的内容，因为从上述公式可以看到这里的无条件扩散模型就是我们想远离的特征分布。

当我们使用Negative Prompt时，会将无条件扩散模型中原本是空字符的unconditional_conditioning用Negative Prompt替代。

下面的代码展示了我们只使用Prompt和同时使用Prompt与Negative Promtpt的区别：

# 只使用Prompt
prompts = ["a castle in a forest"]
batch_size = 1

c = model.get_learned_conditioning(prompts)
uc = model.get_learned_conditioning(batch_size * [""])

samples_ddim, _ = sampler.sample(conditioning=c, unconditional_conditioning=uc, [...])

# 同时使用Prompt和Negative Prompt
prompts = ["a castle in a forest"]
negative_prompts = ["grainy, fog"]

c = model.get_learned_conditioning(prompts)
uc = model.get_learned_conditioning(negative_prompts)

samples_ddim, _ = sampler.sample(conditioning=c, unconditional_conditioning=uc, [...])


# SD系列模型在不使用Negative Prompt和使用Negative Prompt时，生成过程的区别
if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
    e_t = self.model.apply_model(x, t, c)
else:
    x_in = torch.cat([x] * 2)
    t_in = torch.cat([t] * 2)
    c_in = torch.cat([unconditional_conditioning, c])
    e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
    e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

当我们同时使用Prompt和Negative Prompt时，SD系列模型在生成图像的过程中会靠近Prompt中描述的概念与内容，而远离Negative Prompt中描述的概念与内容。

基本的Negative Prompt主要有：worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting

生成二次元图片常用的Negative Prompt主要有：bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn, realistic photo, huge eyes, worst face, 2girl, long fingers, disconnected limbs

生成真实场景图片常用的Negative Prompt主要有：bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn, cartoon, cg, 3d, unreal, animate, amputation, disconnected limbs

下面的代码形象展示了不使用Negative Prompt和使用Negative Prompt之间的逻辑区别：

# 不使用Negative Prompt
prompts = ["a castle in a forest"]
batch_size = 1

c = model.get_learned_conditioning(prompts)
uc = model.get_learned_conditioning(batch_size * [""])

samples_ddim, _ = sampler.sample(conditioning=c, unconditional_conditioning=uc, [...])

# 使用Negative Prompt

prompts = ["a castle in a forest"]
negative_prompts = ["grainy, fog"]

c = model.get_learned_conditioning(prompts)
uc = model.get_learned_conditioning(negative_prompts)

samples_ddim, _ = sampler.sample(conditioning=c, unconditional_conditioning=uc, [...])

【3】Hires. fix功能深入浅出解析

Hires. fix（超分辨率精绘）功能是Stable Diffusion WebUI中img2img部分的一个非常重要的功能，其能够大幅度改善生成图片质量、增加细节并提升图片的分辨率。

下图是使用Hires. fix功能前后的效果对比：

使用Hires. fix功能前后的生成图像效果对比

可以看到，图片生成质量有了明显的提升，那么Hires. fix功能的核心流程是什么样的呢？

Rocky帮大家总结归纳了一下，主要可以分成下面几个步骤：

使用SD模型进行文生图（txt2img）过程，生成初始图片。
使用超分模型（GAN系列超分模型和生成式超分模型）对初始图片进行超分。
使用SD模型进行图生图（img2img）过程，生成精绘的图片。
输出最终的生成图片。

下面是Hires. fix功能的完整图解，大家能够直观的了解Hires. fix功能的全流程：

Stable Diffusion WebUI Hires. fix功能的全流程

接下来我们讲解一下Hires.fix功能包含的参数，由于WebUI中默认有一些参数没有展示，所以我们通过下面的步骤，就能展示Hires.fix功能的全部参数，有助于我们更加灵活地调节配置：

完成上述的四个步骤后，我们就能在WebUI界面看到完整的Hires. fix功能参数了：

WebUI Hires. fix功能完整参数

从上图可以看到，Hires. fix功能的完整参数包括：

Upscaler：超分模型选择，推荐选择R-ESRGAN 4x+、R-ESRGAN 4x+ Anime6B、4x-UltraSharp等。
Hires steps：图生图过程采样步数，如果设置Hires steps = 0，则执行与第一阶段文生图相同的采样步数（Hires steps = Sampling steps），推荐设置10-15效果最佳。
Denoising strength：图生图过程加入的噪声强度，推荐设置0.4-0.5之间。
Upscale by：使用超分模型对图像超分的倍数，最大可以设置为4倍。
Resize width to和Resize height to：除了设置Upscale by，我们还可以设置明确的图像超分超分尺寸，Resize width to和Resize height to分别表示图像超分后的宽和高。
Hires checkpoint：表示在图生图阶段使用哪个SD系列模型作为底模型，如果设置Use same checkpoint，则选择与第一阶段相同的SD系列模型作为底模型。
Hires sampling method：表示在图生图阶段使用哪个采样算法，如果设置Use same sampler，则选择与第一阶段相同的采样算法。
Prompt for hires fix pass：表示在图生图阶段使用的提示词，如果设置为空，则执行与第一阶段文生图相同的提示词。
Negative for hires fix pass：表示在图生图阶段使用的负向提示词，如果设置为空，则执行与第一阶段文生图相同的负向提示词。
Sampling steps设置为20时，Hires. fix功能不同Hires steps的效果对比
Hires. fix功能不同Denoising strength的效果对比

有了Hires. fix功能，我们就可以用一个小分辨率（512x512）为基准，生成1024x1024乃至2048x2048分辨率的高质量图像，同时不会出现生成图像内容结构混乱的情况。

【四】采样方法介绍

SD系列模型的文生图和图生图功能都需要用到采样方法，下面Rocky就带着大家一起了解一下目前的主流采样方法：

1. Euler：使用欧拉方法进行采样，它是一种简单的数值积分方法，适用于简单的扩散模型，但可能不够准确。

2. LMS：代表最小均方（Least Mean Square）方法，它是一种迭代算法，通过根据观测误差来调整模型参数，以逐步提高采样准确性。

3. Heun：使用Heun方法进行采样，也称为改进的欧拉方法，它在欧拉方法的基础上进行了改进，提供了更准确的数值积分结果。

4. DPM2：代表动态粒子扩散模型（Dynamic Particle Model 2），是一种基于粒子的采样方法，通过在扩散模型中移动粒子来进行采样。

5. DPM++2S a：DPM++2S a是DPM++2S方法的改进版本，它可能在采样效果上有所改善。

6. DPM++2M：DPM++2M是DPM++2S方法的另一个改进版本，它可能在采样效果上有所改善。

7. DPM++SDE：DPM++SDE代表DPM++稳定扩散估计器，采用稳定性差分方程（Stabilized Differential Equation）方法进行采样。

8. DPM++2M SDE：DPM++2M SDE是DPM++2M方法与稳定性差分方程（Stabilized Differential Equation）方法的结合，可能提供更准确的采样结果。

9. DPM fast：DPM fast是DPM方法的一种快速版本，它在时间效率上进行了优化，可能会比其他方法更快，但可能损失一些采样准确性。

10. DPM adaptive：DPM adaptive是DPM方法的一种自适应版本，它根据模型的情况自动调整采样策略，以提供更准确的采样结果。

11. LMS Karras：LMS Karras是LMS方法的Karras改进版本，可能在采样效果上有所改善。

12. DPM2 Karras：DPM2 Karras是DPM2方法的Karras改进版本，可能在采样效果上有所改善。

13. DPM2 a Karras：DPM2 a Karras是DPM2 a方法的Karras改进版本，可能在采样效果上有所改善。

14. DPM++2S a Karras：DPM++2S a Karras是DPM++2S a方法的Karras改进版本，可能在采样效果上有所改善。

15. DPM++2M Karras：DPM++2M Karras是DPM++2M方法的Karras改进版本，可能在采样效果上有所改善。

16. DPM++SDE Karras：DPM++SDE Karras是DPM++SDE方法的Karras改进版本，可能在采样效果上有所改善。

17. DPM++2M SDE Karras：DPM++2M SDE Karras是DPM++2M SDE方法的Karras改进版本，可能在采样效果上有所改善。

18. DDIM：代表可微分扩散不变量采样方法（Differentiable Diffusion Invariant Moment），是一种基于扩散不变量的采样方法，可以提供更精确的采样结果。

19. PLMS：代表部分最小均方（Partial Least Mean Square）方法，它是一种迭代算法，类似于LMS方法，但在计算上更高效。

20. UniPC：代表统一粒子扩散模型（Unified Particle Model），是一种基于粒子的采样方法，适用于多种扩散模型。一种可以在5~10步实现高质量图像的方法。

祖先采样器：名称中带有a标识的采样器都是祖先采样器。这一类采样器在每个采样步中都会向图像添加噪声，导致结果具有随机性。部分没有带a的采样器也属于祖先采样器，如Eular a，DPM2 a，DPM++2S a，DPM++2S a KARRAS，DPM++ SDE，DPM++SDE KARRAS。

Karras Noise Schedule：带有Karras字样的采样器，最大的特色是使用了Karras论文中的噪声计划表，主要表现是去噪的程度在开头会比较高，在接近尾声时会变小，有助于提升图像质量。

1.4 Stable Diffusion WebUI图生图全解析

在Stable Diffusion WebUI中，图生图（img2img）功能可以生成与原图一样构图的图像，或者指定一部分内容进行重绘编辑（inpainting）。

Stable Diffusion WebUI图生图的操作界面同样也分为：模型区域、功能区域、参数区域、出图区域四个部分。

上述界面中Prompt、Negative Prompt、Sampling method、Sampling steps、Batch count、Batch size、Width、Height、CFG Scale、Seed等参数的含义是与文生图一致的。下面Rocky向大家详细讲解WebUI图生图功能中一些特殊的参数，希望能给大家带来帮助：

Resize mode：一共有四种图像的缩放模式。Just resize表示只调整图片大小，如果输入与输出长宽比例不同，图片会被拉伸。Crop and resize表示裁剪与调整大小，如果输入与输出长宽比例不同，会以图片中心向四周，将比例外的部分进行裁剪。Resize and fill表示调整大小与填充，如果输入与输出分辨率不同，会以图片中心向四周，将比例内多余的部分进行填充。
Denoising strength：进行图生图功能的重绘幅度，设置的值越大越自由发挥，越小则越和原图接近。
Resize to：表示在进行图生图时，将输出图像尺寸固定到特定的宽高上。
Resize by：表示在进行图生图时，将输出图像尺寸固定为输入图像的特定倍数上。
Mask blur：表示在进行Inpainting操作时，对Mask掩膜边缘应用的模糊程度。它通过在掩膜的边缘区域添加一定程度的模糊，使得掩膜的过渡区域更加平滑，从而使修补区域与原始图像之间的衔接更加自然。
Mask mode：表示设置的Mask蒙版模式，Inpaint masked表示只重绘涂色部分，Inpaint not masked表示重绘涂色之外的部分。
Masked Content：表示在进行Inpainting操作时，对Mask掩膜区域在修补之前填充的初始内容进行设定。这个初始内容不会直接出现在最终的输出结果中，但会影响修补过程中的计算，从而影响最终生成的图像。简而言之，Masked Content决定了在开始修补掩膜区域之前，该区域内部被视为是什么内容。其中四个选项分别为：Fill用于删除对象，生成与背景一致的内容；Original用于细微修改，保留原始特征；Latent Noise用于生成全新内容，增加创造性；Latent Nothing用于在潜在空间中提供空白输入，依赖模型推断。
Inpaint area：表示进行Inpainting的重绘区域。Whole Picture（整张图片）表示对整个图像的尺寸进行处理，即使只修改了部分区域。Only Masked（仅掩膜区域）表示仅对掩膜区域进行处理，其余部分尺寸保持不变。
Only masked padding,pixels：可以指定的一个整数值（以像素为单位）。它用于在Mask掩膜区域的周围添加一定数量的像素，从而扩大SD模型在处理掩膜区域时所考虑的上下文范围。

Stable Diffusion WebUI的图生图的底层Pipeline代码如下所示：

import PIL
import numpy as np
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm


model_id = "/本地路径/stable-diffusion-v1-5"
# 1. 加载autoencoder
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
# 2. 加载tokenizer和text encoder 
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
# 3. 加载扩散模型UNet
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
# 4. 定义noise scheduler
noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False, # don't clip sample, the x0 in stable diffusion not in range [-1, 1]
    set_alpha_to_one=False,
)

# 将模型复制到GPU上
device = "cuda"
vae.to(device, dtype=torch.float16)
text_encoder.to(device, dtype=torch.float16)
unet = unet.to(device, dtype=torch.float16)

# 预处理init_image
def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

# 参数设置
prompt = ["A fantasy landscape, trending on artstation"]
num_inference_steps = 50
guidance_scale = 7.5
strength = 0.8
batch_size = 1
negative_prompt = ""
generator = torch.Generator(device).manual_seed(2023)

init_image = PIL.Image.open("init_image.png").convert("RGB")

with torch.no_grad():
 # 获取prompt的text_embeddings
 text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
 # 获取unconditional text embeddings
 max_length = text_input.input_ids.shape[-1]
 uncond_input = tokenizer(
     [negative_prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
 )
      uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
 # 拼接batch
 text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

 # 设置采样步数
 noise_scheduler.set_timesteps(num_inference_steps, device=device)
 # 根据strength计算timesteps
 init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
 t_start = max(num_inference_steps - init_timestep, 0)
 timesteps = noise_scheduler.timesteps[t_start:]


 # 预处理init_image
 init_input = preprocess(init_image)
    init_latents = vae.encode(init_input.to(device, dtype=torch.float16)).latent_dist.sample(generator)
    init_latents = 0.18215 * init_latents

 # 给init_latents加噪音
 noise = torch.randn(init_latents.shape, generator=generator, device=device, dtype=init_latents.dtype)
 init_latents = noise_scheduler.add_noise(init_latents, noise, timesteps[:1])
 latents = init_latents # 作为初始latents


 # Do denoise steps
 for t in tqdm(timesteps):
     # 这里latens扩展2份，是为了同时计算unconditional prediction
     latent_model_input = torch.cat([latents] * 2)
     latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t) # for DDIM, do nothing

     # 预测噪音
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

     # CFG
     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

     # 计算上一步的noisy latents：x_t -> x_t-1
     latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
    
 # 注意要对latents进行scale
 latents = 1 / 0.18215 * latents
    # 解码
    image = vae.decode(latents).sample

我们接下来再对图生图的一些关键参数进行详细的讲解。

【Masked content参数详解】

下面是Masked content参数设置不同的效果区别：

【Resize mode参数详解】

下面是Resize mode参数设置不同的效果区别：

【Soft inpainting参数详解】

Soft Inpainting（软重绘）是一种在 Inpainting 过程中，采用软过渡技术的重绘方法。与传统的硬边界修补不同，Soft Inpainting 在处理掩膜区域时，使用了软化的边界和权重，使得修补区域与周围环境更好地融合，避免了突兀的边缘或明显的拼接痕迹。

在WebUI界面中，Soft Inpainting 作为 Inpainting 设置中的一个选项，我们可以根据需要启用或禁用该功能。

Soft Inpainting的核心作用：

1. 改善边缘过渡：在进行Inpainting时，如果掩膜区域的边缘过于锐利，模型可能会在修补区域与原始图像之间产生明显的分界线。这会导致生成的图像看起来不自然，修补痕迹明显。Soft Inpainting通过软化掩膜边缘，使模型在处理修补区域时，能够逐渐过渡到原始图像，从而减少边缘伪影，提升视觉效果。

2. 提升修补质量：软化的边缘使得模型在生成内容时，可以更好地考虑周围像素的信息，生成的内容与原始图像更加一致，细节更加丰富。

Soft Inpainting核心原理：

1. 掩膜权重的软化：传统 Inpainting使用二值化的掩膜，掩膜区域的像素值为 1，非掩膜区域为 0。这种硬边界可能导致过渡不自然。与此同时，Soft Inpainting对掩膜进行软化处理，使得掩膜的边缘区域具有介于 0 和 1 之间的权重值。掩膜值越接近 1，表示该区域越需要被修补；越接近 0，表示需要保留原始内容。

2. 模型的处理方式：在修补过程中，模型根据软化后的掩膜权重，对生成内容和原始图像进行加权融合。由于掩膜边缘的权重是连续变化的，生成的内容在边缘区域会与原始图像平滑过渡，避免了硬边界导致的突兀效果。

\text{输出像素} = (\text{掩膜权重}) \times \text{生成内容} + (1 - \text{掩膜权重}) \times \text{原始像素} \\

3. 掩膜模糊与 Soft Inpainting 的关系：掩膜模糊（Mask Blur）通过对掩膜进行模糊处理，使边缘过渡更加平滑。Soft Inpainting在掩膜模糊的基础上，进一步利用软化的掩膜权重，在生成过程中进行权重融合。二者配合使用，可以显著提升修补效果。

【图生图界面中img2img、Sketch、inpaint、inpaint Sketch这四者的功能区别】

一、img2img（图像到图像）：img2img（图像到图像）功能允许用户输入一张初始图像，模型根据该图像和提示词（Prompt），生成一张新的图像。生成的图像在内容、风格或细节上可能与原图像有所不同，具体取决于提示词和设置的参数。

二、Sketch（草图）：Sketch功能允许用户在空白画布上绘制草图，模型根据用户绘制的轮廓和提示词，生成完整的图像。草图作为图像的基本结构，引导模型生成符合预期的内容。

三、Inpaint（图像修补）：Inpaint功能允许用户在已有的图像上指定需要修改的区域（通过掩膜），模型在该区域内根据提示词生成新的内容，而不影响图像的其他部分。

四、Inpaint Sketch（修补草图）：Inpaint Sketch功能结合了Inpaint和Sketch的特点。用户在指定的掩膜区域内绘制草图，模型根据草图和提示词，在该区域生成新的内容。这样，用户可以更精确地控制修补区域的结构和细节。

四者的功能区别总结如下所示：

功能	输入内容	处理范围	主要作用
img2img	原始图像 + 提示词	整张图像	基于原始图像生成新的图像，改变风格、内容等
Sketch	用户绘制的草图 + 提示词	根据草图确定的范围	根据草图生成完整的图像，从零开始创作
Inpaint	原始图像 + 掩膜 + 提示词	掩膜区域	修改或替换图像的特定区域
Inpaint Sketch	原始图像 + 掩膜 + 掩膜内的草图 + 提示词	掩膜区域	在掩膜区域内根据草图精确生成新的内容

Stable Diffusion WebUI的图生图页面中还有Inpaint选项，我们可以使用Inpaint功能对输入的图像进行重绘。

下面是Inpaint这个工作流的主要流程代码：

import PIL
import numpy as np
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm

def preprocess_mask(mask):
    mask = mask.convert("L")
    w, h = mask.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    mask = mask.resize((w // 8, h // 8), resample=PIL.Image.NEAREST)
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    mask = mask[None].transpose(0, 1, 2, 3)  # what does this step do?
    mask = 1 - mask  # repaint white, keep black
    mask = torch.from_numpy(mask)
    return mask

def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

model_id = "/本地路径/stable-diffusion-v1-5"
# 1. 加载autoencoder
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
# 2. 加载tokenizer和text encoder 
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
# 3. 加载扩散模型UNet
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
# 4. 定义noise scheduler
noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False, # don't clip sample, the x0 in stable diffusion not in range [-1, 1]
    set_alpha_to_one=False,
)

# 将模型复制到GPU上
device = "cuda"
vae.to(device, dtype=torch.float16)
text_encoder.to(device, dtype=torch.float16)
unet = unet.to(device, dtype=torch.float16)

prompt = "a mecha robot sitting on a bench"
strength = 0.75
guidance_scale = 7.5
batch_size = 1
num_inference_steps = 50
negative_prompt = ""
generator = torch.Generator(device).manual_seed(0)

with torch.no_grad():
    # 获取prompt的text_embeddings
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    # 获取unconditional text embeddings
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [negative_prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    # 拼接batch
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # 设置采样步数
    noise_scheduler.set_timesteps(num_inference_steps, device=device)
    # 根据strength计算timesteps
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = noise_scheduler.timesteps[t_start:]


    # 预处理init_image
    init_input = preprocess(input_image)
    init_latents = vae.encode(init_input.to(device, dtype=torch.float16)).latent_dist.sample(generator)
    init_latents = 0.18215 * init_latents
    init_latents = torch.cat([init_latents] * batch_size, dim=0)
    init_latents_orig = init_latents
    # 处理mask
    mask_image = preprocess_mask(input_mask)
    mask_image = mask_image.to(device=device, dtype=init_latents.dtype)
    mask = torch.cat([mask_image] * batch_size)
    
    # 给init_latents加噪音
    noise = torch.randn(init_latents.shape, generator=generator, device=device, dtype=init_latents.dtype)
    init_latents = noise_scheduler.add_noise(init_latents, noise, timesteps[:1])
    latents = init_latents # 作为初始latents


    # Do denoise steps
    for t in tqdm(timesteps):
        # 这里latens扩展2份，是为了同时计算unconditional prediction
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t) # for DDIM, do nothing

        # 预测噪音
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # 计算上一步的noisy latents：x_t -> x_t-1
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
        
        # 将unmask区域替换原始图像的nosiy latents
        init_latents_proper = noise_scheduler.add_noise(init_latents_orig, noise, torch.tensor([t]))
        latents = (init_latents_proper * mask) + (latents * (1 - mask))

    # 注意要对latents进行scale
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample

在Stable DIffusion WebUI的图生图页面中，还包含了两个标签反推功能，分别是Interrogate CLIP和InterrogateDeepBooru。其中CLIP模型根据图像内容，生成自然语言标签；而DeepBooru则根据图像内容生成tag标签。两者的具体效果如下图所示：

CLIP反推生成效果与DeepBooru反推生成效果对比

想要使用CLIP模型进行标签反推，我们需要进行如下配置：

将model_base_caption_capfilt_large.pth模型放入“/本地路径/stable-diffusion-webui/models/BLIP/”路径下。
使用CLIP反推时可能会出现报错：报“downloading default CLIP interrogate categories: FileExistsError”。原因是我们还缺少interrogate相关配置文件，下载后解压并将interrogate文件夹放到“/本地路径/stable-diffusion-webui/“路径下，同时删除interrogate_tmp文件夹。
将bert-base-uncased模型放入“/本地路径/stable-diffusion-webui/”路径下。

完成上面的三个模型与依赖文件的配置，即可运行CLIP模型！

想要使用DeepBooru模型进行标签反推，我们需要进行如下配置：

将model-resnet_custom_v3.pt模型放入“/本地流经/stable-diffusion-webui/models/torch_deepdanbooru/”路径下。

完成上面的配置，即可运行DeepBooru模型！

当我们看到一些AI绘画生成的具备艺术价值的图片，可以方便地使用Interrogate CLIP和InterrogateDeepBooru快速反推获得提示词。

1.5 Stable Diffusion WebUI Extras页面使用解析

Stable Diffusion WebUI Extras页面如下所示：

Stable Diffusion WebUI Extras页面

Stable Diffusion WebUI Extras页面中的功能主要包括：图像超分、人脸修复、图像裁剪、图像变换等AIGC时代的AI绘画前处理与后处理，其中的核心功能是图像超分。

为什么WebUI中要将图像超分和人脸修复等功能单独在一个页面进行配置呢？本质上是因为AI绘画领域通过以AI绘画大模型为核心构建的完整工作流，让AI绘画在各个细分领域生根发芽，而图像超分和人脸修复等功能正是AI绘画工作流中不可或缺的部分。

在Stable Diffusion WebUI框架中，一种集成了13种图像超分算法，包括：Lanczos、Nearest、ESRGAN_4x、LDSR、R-ESRGAN 2x+、R-ESRGAN 4x+、R-ESRGAN 4x+ Anime6B、R-ESRGAN AnimeVideo、R-ESRGAN General 4xV3、R-ESRGAN General WDN 4xV3、ScuNET、ScuNET PSNR以及SwinIR_4x。

Lanczos和Nearest属于传统超分算法，核心是Nearest插值和Lanczos插值。这两种超分算法仅使用图像的像素值执行数学运算来放大图像尺寸并填充新像素。当图像本身被损坏或扭曲，这些算法就无法准确地填充缺失的有效信息。

与此同时，ESRGAN_4x、LDSRR-ESRGAN 2x+、R-ESRGAN 4x+、R-ESRGAN 4x+ Anime6B、R-ESRGAN AnimeVideo、R-ESRGAN General 4xV3、R-ESRGAN General WDN 4xV3等超分算法都是基于GAN模型架构的，在对图像进行超分的同时能够较好的填充细节，下图是传统超分算法与R-ESRGAN超分模型的效果对比：

R-EsRGAN模型与Lanczos算法的超分效果比对

除了基于GAN模型架构的超分算法，也有基于扩散模型架构的超分算法，比如LDSR（Latent Diffusion Super Resolution），虽然它的效果很好，但是超分耗时较久，所以在实时场景中一般不建议使用。

SRGAN模型倾向于保留精细的细节，并产生清晰锐利的图像。

Real-ESRGAN模型与ESRGAN模型相比，倾向于产生更平滑的图像，同时在真实图像上的效果有比较多的提升。

那么哪种图像超分算法效果最好呢？

当我们使用真实图像进行超分处理时，推荐使用R-ESRGAN 4x+超分算法。
当我们使用二次元图像进行超分处理时，推荐使用R-ESRGAN 4x+ Anime6B超分算法。

在Stable Diffusion WebUI Extras页面中，我们除了能够进行图像超分处理，我们还能够对图像进行人脸修复操作。

我们可以选择GFPGAN和CodeFormer两个算法进行人脸修复（Face restoration）操作，我们可以使用一个或者两个组合的方式，具体效果如下：

GFPGAN和CodeFormer算法人脸修复效果对比

想要使用GFPGAN算法进行人脸修复，我们需要进行如下配置：

将GFPGANv1.4.pth、detection_Resnet50_Final.pth和parsing_parsenet.pth模型放入“/本地路径/stable-diffusion-webui/models/GFPGAN/”路径下。

想要使用CodeFormer算法进行人脸修复，我们需要进行如下配置：

将codeformer-v0.1.0.pth模型放入“/本地路径/stable-diffusion-webui/models/Codeformer/”路径下。
将detection_Resnet50_Final.pth和parsing_parsenet.pth模型放入“/本地路径/stable-diffusion-webui/repositories/CodeFormer/weights/facelib/”路径下。
1.6 Stable Diffusion WebUI PNG Info、Checkpoint Merge页面使用解析
2. 从0到1深入浅出全面解析Stable Diffusion WebUI热门插件

再本章节中，Rocky将带着大家详细了解Stable Diffusion WebUI生态中的热门插件，这些插件都给我们带来了巨大的使用价值，提高了AI绘画工作流的效率，为AI绘画生态的繁荣作出了很大的贡献。

在网络良好的情况下，我们在SD WebUI中跳转到Extentions，然后选择install from URL。输入对应插件的路径，点击下方的install即可安装对应的插件。

在安装过程中，会自动安装依赖包，我们需要耐心等待一下。安装完需要重启WebUI。

除了上述这种安装形式，我们还可以使用项目源码安装，直接进入到Stable Diffusion WebUI的extensions文件夹，在命令行输入git clone即可。下载完成后，我们需要重新启动WebUI，便会检查需要的环境库并且安装依赖。

2.1 ADetailer插件功能

ADetailer插件能够自动修复Stable Diffusion生成图像的脸部和手部的崩坏。

ADetailer能够自动识别生成图像中的人脸，并对脸部进行mask覆盖，从而能够施展Inpainting功能对图像中的人脸部分进行重绘，从而优化脸部的质量与细节。

ADetailer插件本质上是一个节省时间、简化图像修复增强过程的算法解决方案，将原来的人工手动mask填充+手动Inpainting过程完全自动化了。

在整个过程中，就涉及到传统深度学习时代的YOLO系列检测模型用于检测图像中的人脸、人体、手部等特征。目前涉及到的模型以及功能如下所示：

Face_xxxx：检测并重新绘制人脸
Hand_xxxx：检测并重新绘制手部
Person_xxxx：检测并重新绘制整个人
Mediapipe_face_xxxxx：检测和重绘人脸
使用ADetailer对脸部进行修复
使用ADetailer对手部进行修复
使用ADetailer对全身进行修复

与此同时，在ADetailer进行自动化重绘时，我们可以输入特定的提示词和负向提示词，更加精准的控制Inpainting过程。

值得一提的是，ADetailer还可以与ControlNet组合使用，让整个生成过程更加可控。

2.2 深入浅出完整解析EasyPhoto的AI写真功能

AI写真是AIGC时代中AI绘画领域的一个重要商业变现与可持续发展方向。妙鸭相机作为AIGC时代的一款收费产品，成功向大家展示了如何凭借少量的人脸图片进行AI写真生成，能够迅速提供真、像、美的AI个人写真，在极短的时间内便拥有了众多的付费客户。

目前，作为开源版的妙鸭相机——EasyPhoto插件已经能够很好的适配Stable Diffusion WebUI，EasyPhoto允许用户上传几张同一个人的照片，快速的训练Lora模型，再结合用户上传的模板图片，快速生成真、像、美的AI写真图片。

EasyPhoto效果展示

下面Rocky将带着大家一起学习EasyPhoto的AI写真工作流，由于其在Stable Diffusion WebUI中已经开发出相应的插件，我们能够方便地使用。

EasyPhoto项目地址： EasyPhoto：Your Smart AI Photo Generator

EasyPhoto工作流主要分训练和推理两个阶段。

EasyPhoto训练阶段

我们可以将EasyPhoto的训练阶段整体上分成数据预处理和LoRA模型训练两个部分，完整流程图如下所示：

EasyPhoto训练过程完整流程图

在数据预处理部分，又可以分为人像得分排序、Top-k个人像选取、显著性分割和图像修复四个步骤。

首先我们需要对用户上传的人像数据进行打分与排序，整个流程需要结合人脸特征向量、图像质量评分和人脸偏移角度技术。

我们可以通过人脸特征向量选出最像自己的图片，因为每个人在不同时期的样貌是有些许差异的，所以我们需要选出输入图片中最像自己的图片用于后续的LoRA模型训练。

人脸特征向量计算需要先进行人脸检测；然后进行人脸对齐，使其成为一张标准的人脸（人脸“正则化”）；最后通过人脸识别提取出这个人脸的特征向量。 经过人脸特征向量的提取后，我们可以得到一个特征向量来表示人脸，如下所示：

# 这是一个的128维的人脸特征向量
feature_vector = [
    0.12, -0.23, 0.33, 0.45, -0.56, 0.67, -0.78, 0.89, 0.91, -1.02, 
    1.13, -1.24, 1.35, -1.46, 0.57, -0.68, 0.79, -0.80, 0.81, -0.92,
    0.93, -1.04, 1.15, -1.26, 1.37, -1.48, 1.59, -1.60, 0.61, -0.72,
    0.73, -1.84, 1.95, -1.06, 0.17, -0.28, 0.39, -0.40, 0.41, -0.52,
    0.53, -1.64, 1.75, -1.86, 0.97, -0.08, 0.19, -0.30, 0.31, -0.42,
    0.43, -1.54, 1.65, -1.76, 0.87, -0.98, 0.09, -0.10, 0.11, -0.22,
    0.23, -1.34, 1.45, -1.56, 0.67, -0.78, 0.89, -0.90, 0.91, -0.02,
    0.13, -1.24, 1.35, -1.46, 0.57, -0.68, 0.79, -0.80, 0.81, -0.92,
    0.93, -1.04, 1.15, -1.26, 1.37, -1.48, 1.59, -1.60, 0.61, -0.72,
    0.73, -1.84, 1.95, -1.06, 0.17, -0.28, 0.39, -0.40, 0.41, -0.52,
    0.53, -1.64, 1.75, -1.86, 0.97, -0.08, 0.19, -0.30, 0.31, -0.42,
    0.43, -1.54, 1.65, -1.76, 0.87, -0.98, 0.09, -0.10, 0.11, -0.22
]

整个人脸特征向量计算的过程可以用下图表示：

人脸特征向量计算过程示意图

通过对人脸特征向量之间的比对，我们就可以判断人脸之间的相似程度了。

在计算出人脸特征向量后，我们需要进行人脸偏移角度的计算。人脸偏移角度的计算方法有很多，EasyPhoto中通过双眼的旋转角度计算人脸的偏移角度。我们以水平线为基准，双眼的旋转角度就是眼睛连线相对于水平线的倾斜角，如下图所示：

人脸偏移角度

如果这个倾斜角为0，则代表双眼完全正视，一般情况下，如果人像存在侧拍、侧身、歪头等情况，倾斜角是不会为0的，因此我们可以通过计算倾斜角度来选出最正的人像，用于接下来的图像质量评分计算和在预测阶段中作为参考人像进行使用，进行人脸融合。

完成人脸特征向量计算和人脸偏移角度计算后，我们就可以进行人像的排序。

我们先对人脸偏移角度进行归一化：

人脸偏移角度为0的时候，得分为1
人脸偏移角度为90的时候，得分为0

我们再根据人脸特征向量，计算用户上传图像自身与自身的相似程度，首先计算人像数据的平均特征，然后计算每一张图片与平均特征的相似程度，相似程度也用一个0-1之间的得分来表示。

我们接着将人脸特征相似度得分与人脸偏移得分进行相乘，选出得分最高的，作为参考人像。同时使用人脸特征相似度得分与图像质量评分进行相乘，选出得分最高的Top-K个人像进行训练。

完成上面的工作后，我们还需要进行人像图片进行显著性分割，我们在训练阶段想要LoRA模型学习到的是人像特征而不是其它的特征，所以训练数据中人相特征越显著越好。

所以我们对Top-k个人像数据使用图像分割模型进行显著性分割，将背景进行了去除，再通过人脸检测框裁剪出人脸周围的区域。

接下来我们就进入图像预处理阶段的最后一个步骤——图像修复。

由于用户的输入图片不一定是高质量图片，图片可能存在模糊、噪声、不清晰等问题，所以我们需要使用一些修复算法与超分算法将低质量图像进行修复后再用于LoRA模型的训练。不然LoRA模型会学到模糊、噪声、不清晰等不好的特征，从而影响图像生成效果。

我们用之前的图像质量评分判断选出质量最低的一些图像，使用GPEN人像修复增强算法进行图像修复和超分，同时使用ABPN人像美肤算法提升人像的皮肤质感，从而提升了图片综合质量。

完成了上面的数据预处理流程，下面我们可以进行人脸LoRA模型的训练了。

EasyPhoto分别在Stable Diffusion的Text Encoder和U-Net部分的自注意力机制中添加LoRA进行训练。

下面是一些LoRA模型训练的关键参数，我们可以按需进行调整：

参数名	含义
resolution	训练时输入图片的分辨率，默认值为512
validation & save steps	验证图片与保存中间权重的steps数，默认值为100，代表每100步验证一次图片并保存LoRA权重
max train steps	最大训练步数，默认值为800
max steps per photo	每张图片的最大训练次数，默认为200
train batch size	训练的批次大小，默认值为1
gradient accumulation steps	是否进行梯度累计，默认值为4，若train batch size设置为4，每个Step相当于输入四张图片进行训练
dataloader num workers	数据加载的works数量，windows下无法使用，Linux正常设置
learning rate	训练LoRA模型的学习率，默认为1e-4
rank LoRA	权重的特征长度，默认为128
network alpha	LoRA训练的正则化参数，一般为rank的二分之一，默认为64

在EasyPhoto中，最终LoRA模型的训练步数等于：

Final training step = Min(photo_num * max_steps_per_photos, max_train_steps)

以上面的默认参数为例，如果图片数量小于4，则训练步数为200x图片数量；如果大于等于4，则训练步数为800。

在LoRA模型的训练过程中，我们还需要通过LoRA模型融合来获取最佳的LoRA模型。

直接取最后保存的LoRA模型并不能保证就是最符合人像特征的。因此，EasyPhoto在训练中加入了LoRA模型融合的机制。

EasyPhoto会在每100个Step处添加一次模型验证并且保存LoRA权重，模型验证时使用一些模板图像进行图生图功能，生成人像正脸照。以默认参数为例，我们一共进行800步的训练，那么我们可以得到8个LoRA模型和8组验证结果，我们提取验证结果与训练图片的人脸特征向量，进行人脸相似度的计算。

再将能够生成人脸相似度较高的几个LoRA模型进行融合。由于每组验证结果包含多张图片（默认为4张），我们会根据每组验证结果中图片被选中的比例作为这个LoRA模型在进行模型融合时的权重。

到这里，EasyPhoto的训练阶段就完整的介绍好了，Rocky再给大家进行总结梳理一下，方便大家快速学习：

上传5到20张人像数据作为训练集，包括不同的角度和光照，同时最好是半身照片且不要佩戴眼镜。
采用人脸特征相似度和图像质量对所有图片进行评分，筛选最佳图像和Top-K图像。
采用人脸检测和显著性分割，对Top-K图像进行人脸检测和抠图，并去除背景。
采用图像修复模型和美肤模型优化部分低质量人脸数据，提升训练数据的整体质量。
对处理后的训练图片进行标注，并设置超参数用于人脸LoRA模型的训练。
LoRA模型的训练过程中采用基于人脸特征相似度的验证步骤，间隔一定的step保存模型权重，并根据相似度来融合LoRA模型。

EasyPhoto推理阶段

EasyPhoto在推理阶段主要采用StableDiffusion模型 + 人像LoRA模型 + ControlNet模型进行AI写真的生成，整体上是包含初步重建、边缘完善和后处理三个阶段。

EasyPhoto推理过程完整流程图

首先我们进入初步重建阶段，这个阶段又可以分为人脸融合、人脸裁剪与仿射变换、Stable Diffusion重建与颜色转移三个部分。

我们先将在训练阶段筛选出的最佳人像图像作为目标脸型，通过人脸融合算法与模版图中的人脸进行融合，生成一张与目标脸型相似，且具有模版人脸外貌特征的新图像，为后续的AI写真提供了一个较好的基础图像。

人脸融合算法（cv_unet_face_fusion_torch）使用多尺度属性编码器提取模板图属性特征，使用预训练人脸识别模型提取用户图的ID特征，再通过引入可行变特征融合结构， 将ID特征嵌入属性特征空间的同时，以光流场的形式实现面部的自适应变化，最终融合结果真实、高保真和一定程度内对目标用户脸型的自适应感知。

完成了人脸融合后，我们需要进行人脸裁剪与仿射变换。

我们找出在训练阶段验证时生成的和用户相似度最高的正脸图片，在此基础上我们裁剪这个正脸照片并且进行仿射变换，利用五个人脸关键点，将其贴到模板图像上，得到一个Replaced Image，将在后续使用Stable Diffusion重建时提供openpose信息。

有了融合人脸图像和仿射变换的人脸图像后，我们就可以进行Stable Diffusion重建和颜色转移了。

在这一步中我们主要使用训练阶段获得的LoRA模型进行人脸重建生成，但只使用LoRA模型是不够的，很容易生成不和谐的图像，我们需要施加一些ControlNet控制。

在这里EasyPhoto使用了三个ControlNet控制和一个Mask：

使用人脸融合图像的Canny控制（防止人像崩坏）
使用人脸融合图像的颜色控制（使生成的颜色且符合模板）
使用Replaced Image的openpose+Face pose控制（使得眼睛与轮廓更像本人）
使用训练获得的LoRA模型
使用Mask对人像区域进行重建生成

重建生成完成后的图像可能存在一些颜色的偏移，EasyPhoto会使用一个color_transfer方法，保证重建后的图片与原图的颜色协调。

完成上面的操作后，我们就可以进入边缘完善这个阶段了。

边缘完善阶段又可以分为两步，分别是人脸融合和Stable Diffusion重建。

与初步重建阶段类似，在边缘完善阶段我们再做一次人脸融合来提升人脸的相似度。

在完成初步重建后，生成的人像图片整体效果已经不错了，但在边缘上可能依旧存在不和谐的问题。因此，EasyPhoto进行了第二次Stable Diffusion重建进行图像的边缘完善。同样需要在使用LoRA模型的基础上施加一些Controlnet控制，EasyPhoto主要使用了两个ControlNet控制和一个Mask：

使用人脸融合图像的tile控制（防止颜色过于失真）
使用人脸融合图像的canny的控制（防止人像崩坏）
使用训练获得的LoRA模型
通过Mask对人像周围区域进行重建（不是人像区域）

最后，我们需要进行后处理操作，一共包含了人像美肤和超分辨率重建两个步骤，能够让生成的AI写真更美更清晰。

到这里，EasyPhoto的推理阶段就完整的介绍好了，Rocky再给大家进行总结梳理一下，方便大家快速学习：

对输入的模板图进行人脸检测(crop & warp)并结合数字分身进行模板图的人脸替换。
挑选用户输入的最佳ID Photo和模板照片进行人脸融合。
使用融合后的图片作为基底图片，使用替换后的人脸作为ControlNet条件，加上数字分身对应的LoRA模型，进行img2img的局部重绘生成，同时进行图像颜色上的协调。
再次进行人脸融合，同时使用Stable Diffusion + LoRA+ControlNet进行图像边缘完善。
采用人脸美肤模型和超分辨率重建模型生成优美的高清AI写真图。
2.3 深入浅出完整解析Facechain的AI写真功能

除了EasyPhoto插件外，FaceChain项目也能完成AI写真功能，并且FaceChain具备多张图片精细化生成AI写真和一张图片快速生成AI写真（10秒左右）两种适应不同场景需求的功能。

FaceChain是阿里达摩院发布的一个功能上对标“秒鸭相机”的开源项目，顾名思义，FaceChain就是对人脸（face）做一连串（chain）的操作处理。

那么这些处理包括哪些呢，又是怎么串联起来的呢？接下来Rocky就分别从精细化生成AI写真和一张图片快速生成AI写真两个功能向大家详细介绍。

我们首先来介绍FaceChain精细化生成AI写真的功能，其完整工作流程如下所示：

FaceChain精细化生成AI写真工作流

训练阶段

输入：用户上传的清晰人脸图像。
输出：人脸LoRA模型。

首先，我们分别使用基于朝向判断的图像旋转模型、基于人脸检测和关键点模型的人脸精细化旋转方法处理用户上传图像，得到包含正向人脸的图像。

接着，我们使用人体解析模型和人像美肤模型，以获得高质量的人脸训练图像。

然后，我们使用人脸属性模型和文本标注模型，结合标签后处理方法，产生训练图像的精细化标签。

最后，我们使用上述图像和标签数据基于Stable Diffusion模型训练人脸LoRA模型。

推断阶段

输入：训练阶段用户上传的图像和预设的Prompt提示词。
输出：个人写真图像。

首先，我们将人脸LoRA模型和风格LoRA模型的权重融合到Stable Diffusion模型中。

接着，我们使用Stable Diffusion模型的文生图功能，基于预设的输入提示词初步生成个人写真图像。

然后，我们使用人脸融合模型进一步改善写真图像的人脸细节，其中用于融合的模板人脸通过人脸质量评估模型在训练图像中择优挑选。

最后，我们使用人脸识别模型计算写真图像与模板人脸的相似度，以此对写真图像进行排序，并输出排名靠前的n个人写真图像作为最终输出结果。

讲完了FaceChain精细化生成AI写真的功能，接下来Rocky再和大家一起分析FaceChain一张图片快速生成AI写真的功能。

最新的FaceChain FACT版本中，用户仅需要提供一张照片即可10秒钟获得独属于自己的个人写真（支持多种风格）。FaceChain可实现兼具可控性与ID保持能力的无限风格写真与固定模板写真功能，同时对ControlNet和LoRA具有优秀的兼容能力。

下面是FaceChain FACT版本的完整工作流：




2.4 ControlNet插件

ControLnet的相关核心知识与实战应用，大家可以直接阅读Rocky在持续更新的《深入浅出完整解析ControlNet核心基础知识》文章：

码字确实不易，希望大家能多多点赞！！！

2.5 Segment Anything插件

在WebUI中，Segment-Anything插件将自然语言、目标检测与图像分割的能力结合，使得我们可以通过简单的文字描述完成复杂的目标检测和图像分割任务。它在提高图像处理的自动化和精度方面有显著优势，特别是在大规模数据标注、精细分割和复杂场景的多对象处理上，非常适合需要高效处理和标注大量图像数据的场景。

Segment Anything插件主要结合了两种强大的AI算法技术——Grounding DINO和Segment Anything Model (SAM)，从而实现复杂图像中的对象检测和分割任务。我们通过自然语言或者标签提示，精准地识别和分割图像中的目标区域，广泛应用于图像理解、图像编辑、自动化标注等场景。

Grounding DINO是一种强大的目标检测模型，它的独特之处在于能够根据自然语言提示进行目标检测。相比传统的目标检测模型（如 Faster R-CNN 或 YOLO），Grounding DINO 允许用户通过输入具体的描述，来定位目标对象。比如输入 "a cat on the sofa"，模型就能在图像中找到沙发上的猫。

Segment Anything Model (SAM)是一种通用的分割模型，能够处理不同类型的图像分割任务，甚至在无监督情况下也能分割出非常复杂的物体。SAM 可以从像素级别精确分割出图像中的区域，因此非常适合用于精细化处理图像。

Segment Anything插件的功能极大地增强了图像分割的灵活性和易用性：

自动化标注：由于Grounding DINO和SAM的自动化能力，我们不再需要手动标记数据。只需提供简单的描述性标签，插件就能够自动生成高质量的分割结果并保存为标注数据。这对于减少标注时间和提高数据集的质量具有重要意义。
图像理解与分析：在图像理解任务中，该插件可以用于自动提取图像中的目标对象，并且可以通过自然语言输入高效地完成复杂的目标检测和分割任务。
高效图像编辑：特别适合图像编辑或标注任务，我们可以快速提取出图像中的特定对象进行后续处理，如替换背景、修改颜色等。
支持细粒度分割：Segment Anything插件不仅可以处理较大区域的分割任务，也能够胜任更精细的对象分割任务。例如，在图像中要分割一只猫的具体部分如“猫的耳朵”，通过语言描述（如“the cat's ear”），插件可以首先通过 Grounding DINO 定位猫，然后通过 SAM 模型对耳朵进行精细分割。这对于需要高精度的图像处理任务（如医疗影像分析、自动驾驶场景理解等）尤为重要。
多对象分割：Segment Anything插件还支持多对象分割。如果图像中有多个目标对象，我们可以通过提供多个提示词来分别识别和分割不同的对象。比如在一张包含猫和狗的图片中，我们可以分别输入“cat”和“dog”，模型会依次分割出图像中的猫和狗，并对它们进行标注。这种多对象分割能力非常适合应用于复杂场景，如无人驾驶中对不同物体（如行人、车辆、障碍物）的分割识别。
2.6 AnimateDiff插件

AnimateDiff是一个文生视频的算法，我们可以输入一段文本提示词，从而生成大约几秒钟的短视频。同时，将AnimateDiff与ControlNet进行组合使用，可以进行视频生成视频和视频编辑功能。

AnimateDiff最大的特点是其加入了运动模型模块（Motion Modeling Module）。该模块的作用是学习和捕捉运动的先验知识，使生成的动画在帧与帧之间具有连贯性和合理性。它可以理解为一种对运动模式和规律的学习器，通过对大量视频数据的学习，掌握不同物体的运动方式和变化趋势，从而在生成动画时能够根据文本提示和已有的图像信息，合理地预测和生成下一帧的图像内容，实现动画片段中的运动平滑度和内容一致性。

上图中展示的是AnimateDiff模型在训练阶段和推理阶段的整体流程，旨在将静态图像生成模型（Text-to-Image，T2I）扩展为文本驱动的视频生成模型。下面我们一起详细了解训练和推理过程。

【AnimateDiff训练阶段】

1. 输入视频片段：在训练阶段，模型使用一系列视频片段作为输入。这些视频片段包含多个连续帧，为模型提供了学习运动模式的丰富数据。

2. 编码过程：视频片段被输入到编码器（图中的ε）中，转换为潜在的运动特征表示 z_{0:N} ，这一步提取了视频的运动信息和帧间的动态关系。

3. 添加噪声：在训练中，会给这些潜在特征添加一定的随机噪声（符合正态分布 N(0, I)），这是扩散模型的一部分，用于训练模型在多样化的输入情况下保持一致。

4. 基础 T2I 模型和运动建模模块：

- 基础 T2I 模型：图中的灰色部分表示已经预训练好的 T2I 模型的权重，这部分在训练过程中被“冻结”，即不再更新参数。

- 运动模块：图中的蓝色部分表示新加入的、专门用于运动建模的模块。在训练阶段，这个模块从头开始初始化，并通过与视频片段数据的训练来学习合理的运动先验。

5. 损失计算：通过扩散过程，模型输出带有噪声的图像，与目标噪声图像进行比较，计算损失。这一损失用于优化运动建模模块，使其能够从视频数据中学习到运动信息。运动建模模块的最终训练目标是：

\mathcal{L}=\mathbb{E}_{\mathcal{E}(x_0^{1:N}),y,\epsilon\sim\mathcal{N}(0,I),t}\left[\|\epsilon-\epsilon_\theta(z_t^{1:N},t,\tau_\theta(y))\|_2^2\right]

【AnimateDiff推理阶段】

1. 插入运动建模模块：在推理阶段，训练好的运动建模模块被插入到已经预训练好的 T2I 模型中。这样，T2I 模型就具备了生成连续帧的能力，而不仅仅是生成单张静态图像。

2. 生成过程：

- 初始噪声生成：从标准正态分布N(0, I)中采样初始噪声，这与生成单张图像的过程类似。

- 迭代去噪：通过插入了运动建模模块的模型，开始迭代去噪过程。这一步的目标是逐渐将噪声转换为具有连续运动特征的图像序列。

- 解码生成：经过一系列去噪和生成步骤，最终得到一个符合输入文本描述的动画图像序列。

3. 视频输出：最终生成的结果是一个动画序列，即一段短视频，展示了根据输入文本描述生成的动画内容。

2.7 Tiled Diffusion/VAE插件

Stable Diffusion WebUI的Tiled Diffusion/VAE插件项目地址：multidiffusion-upscaler-for-automatic1111

ComfyUI的Tiled Diffusion/VAE插件项目地址：ComfyUI-TiledDiffusion

Tiled Diffusion/VAE插件能够让我们在显存不足的情况下，进行图像的超分辨率重建、2k-8k的高分辨率图像生成以及图像子区域的编辑生成。

接下来我们就逐一介绍这个插件各个功能的详细作用。

首先我们讲解一下Tiled VAE的核心功能，Tiled VAE的核心目标是通过分块处理大图像，解决显存不足的问题，同时保持生成图像中的子分块图像的边缘无缝连接。使用到的核心技术是通过估计GroupNorm的参数来实现无缝生成。下面是Tiled VAE的具体流程：

1. 将大图像分割成多个小块（tiles）。

2. 在编码器（encoder）和解码器（decoder）中，每个小块会分别进行填充（padding），通常为11/32像素。

3. 将原始的VAE前向传播过程被分解为一个任务队列和一个任务处理器。任务处理器开始处理每个小块。当需要GroupNorm时，任务处理器会暂停，存储当前的GroupNorm的均值和方差，并将数据发送到RAM（Random Access Memory，随机存取存储器），然后转向处理下一个小块。

4. 在所有GroupNorm的均值和方差被汇总后，任务处理器会对每个小块应用GroupNorm并继续处理。同时使用Zigzag执行顺序（如从左到右，再从右到左）处理这些小块，可以减少数据在 RAM 和 VRAM 之间的传输次数。这种顺序避免了频繁切换数据块，从而提高了处理效率。

在我们安装好Tiled Diffusion/VAE插件后，我们可以在WebUI界面看到如下的Tiled VAE参数配置框：

我们点击启动Tiled VAE，其余参数可以保持不变。如果大家的显存实在太小，上述参数配置的情况下仍出现CUDA显存溢出的情况，可以再次调小“编码器分块大小”和“解码器分块大小”这两个参数。同时，如果我们设置的tile size尺寸太小，可能会导致生成图片的整体颜色变得灰暗且不清晰，这是可以开启“快速编码器颜色修复”功能进行修复。

讲完了Tiled VAE的核心功能，接下来，我们再来解读Tiled Diffusion的核心功能。

Tiled Diffusion能够在低显存（6GB及一下）情况下，对图像进行超分辨率重建、生成2k-8k分辨率的高质量高清大图以及图像子区域的编辑生成。

我们要知道的是，Tiled Diffusion的底层思想来源于MultiDiffusion。一般来说，我们进行图像的Tiled处理时，会将输入的噪声图像划分成不同的区块，针对每一个区块分别进行特定的图像特征生成。但是这样处理会引入一个问题，那就是不同的区块之间会出现明显的边缘不连贯问题。针对这个问题，MultiDiffusion中提出将不同区块的不连续部分进行融合，然后再进行全局去噪过程，这样能保证最终生成图像的特征一致性与边缘平滑。

下面是Tiled Diffusion的完整流程：

1. 将Latent特征进行分块处理。

2. Stable Diffusion预测每个分块图像的噪声，并进行去噪操作。

3. 将去噪后的分块图像加在一起，同时每个像素的值会被除以它

4. 将所有的分块图像拼接在一起，同时重叠的像素会取平均值或者使用融合后的噪声对整个图像进行一步去噪。

5. 重复上述1-4步骤，直到所有采样时间步完成。

接下来，我们详细讲解MultiDiffusion核心原理，作为进一步的深入浅出理解。

MultiDiffusion是一个统一的生成框架，无需任何训练或微调。它通过一个最小二乘优化任务，将同一个Stable Diffusion模型在多个不同的图像空间和条件空间中生成的crop区域约束在一起，虽然最初各个crop区域生成的内容可能会不一致，不过不用担心，MultiDiffusion会通过全局去噪，不断融合各个crop区域，从而生成不同场景的高质量和连贯性的完整图像。同时MultiDiffusion方法在很大程度上依赖于Stable Diffusion的生成先验，最终生成结果的质量取决于Stable Diffusion提供的扩散路径。

我们先来看一下Stable Diffusion模型的图像生成过程，它是MultiDiffusion方法的基础：

\Phi: \mathcal{I} \times \mathcal{Y} \rightarrow \mathcal{I}\\

它的图像空间为\mathcal{I} =\mathbb{R}^{H \times W \times C} ，条件空间为 \mathcal{Y} ，其中去噪采样过程如下：

I_{T}, I_{T-1}, \ldots, I_{0} \quad \text { s.t. } \quad I_{t-1}=\Phi\left(I_{t} \mid y\right) \\

这个过程展示了Stable Diffusion逐渐将高斯噪声 I_T 去噪转化成干净图像 I_0 。

接下来我们再看一下MultiDiffusion通用生成框架的图像生成过程：

\Psi: \mathcal{J} \times \mathcal{Z} \rightarrow \mathcal{J}\\

它的图像空间为 \mathcal{J}=\mathbb{R}^{H^{\prime} \times W^{\prime} \times C} ，条件空间为 \mathcal{Z} 。从初始噪声开始，按图像切片序列生成一系列图像：

J_{T}, J_{T-1}, \ldots, J_{0} \quad \text { s.t. } \quad J_{t-1}=\Psi\left(J_{t} \mid z\right)\\下面我们开始建立MultiDiffusion和Stable Diffusion之间的关联。我们定义一组目标图像空间和参考图像空间之间的区域映射 F_{i}: \mathcal{J} \rightarrow \mathcal{I} ，以及一组相应的条件空间之间的映射 \lambda_{i}: \mathcal{Z} \rightarrow \mathcal{Y} ，其中 i \in[n]=\{1, \ldots, n\} 。其中映射关系可表示为：

I_{t}^{i}=F_{i}\left(J_{t}\right),\quad y_{i}=\lambda_{i}(z)， i \in[n]=\{1, \ldots, n\}\\

我们的目标是使每个MultiDiffusion采样 J_t ， 对于所有i \in\{1, \ldots, n\} ，尽可能使得 F_i(J_t) 接近常规Stable Diffusion的采样 \Phi\left(I_{t}^{i} \mid y_{i}\right)，我们可以采用下面的优化过程达到目标：

\Psi\left(J_{t} \mid z\right)=\underset{J \in \mathcal{J}}{\arg \min } \mathcal{L}_{\mathrm{FTD}}\left(J \mid J_{t}, z\right)\\

\mathcal{L}_{\mathrm{FTD}}\left(J \mid J_{t}, z\right)=\sum_{i=1}^{n}\left\|W_{i} \otimes\left[F_{i}(J)-\Phi\left(I_{t}^{i} \mid y_{i}\right)\right]\right\|^{2} \\

其中， W_{i} \in \mathbb{R}_{+}^{H \times W} 是per-pixel权重， \otimes 是Hadamard乘积。

如下图所示，MultiDiffusion生成过程是从噪声图像 J_T 开始，在每个采样步骤中，通过一个优化任务，使得目标图像 J_t 的每个crop F_i(J_t) 将尽可能接近参考图像去噪采样 \Phi\left(F_{i}\left(J_{t}\right)\right) 。虽然各个采样\Phi\left(F_{i}\left(J_{t}\right)\right) 可能会将图像拉向不同的方向，但MultiDiffusion会将这些不一致的方向融合到全局去噪\Psi(J_t) 中，从而产生高质量的连贯图像。

由于F_{i}是直接从目标图像 J_{t} 中进行crop的过程。在这种情况下 \mathcal{L}_{\mathrm{FTD}}\left(J \mid J_{t}, z\right) 的优化问题可以通过二次最小二乘 (LS)分析法解决， J 的最优解的每个像素是Reference Model所有扩散采样的加权平均：

\Psi\left(J_{t} \mid z\right)=\sum_{i=1}^{n} \frac{F_{i}^{-1}\left(W_{i}\right)}{\sum_{j=1}^{n} F_{j}^{-1}\left(W_{j}\right)} \otimes F_{i}^{-1}\left(\Phi\left(I_{t}^{i} \mid y_{i}\right)\right)\\

下面是MultiDiffusion的核心代码，其提取图像各区块，并进行去噪采样的过程就生动的展现出来了：

with torch.autocast('cuda'):
    for i, t in enumerate(self.scheduler.timesteps):
        count.zero_()
        value.zero_()

        for h_start, h_end, w_start, w_end in views:
            # TODO we can support batches, and pass multiple views at once to the unet
            latent_view = latent[:, :, h_start:h_end, w_start:w_end]

            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latent_view] * 2)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds)['sample']

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the denoising step with the reference model
            latents_view_denoised = self.scheduler.step(noise_pred, t, latent_view)['prev_sample']
            value[:, :, h_start:h_end, w_start:w_end] += latents_view_denoised
            count[:, :, h_start:h_end, w_start:w_end] += 1

        # take the MultiDiffusion step
        latent = torch.where(count > 0, value / count, value)


接下来，我们再看看如何以MultiDiffusion为核心，进行高分辨率图像的生成和图像子区域编辑（Region-based）生成。

首先我们使用MultiDiffusion进行高分辨率图像的生成。对于高分辨率图像生成任务，此时MultiDiffusion对应的空间尺寸大于Stable Diffusion输出域的空间尺寸(H^{\prime} \geq H, W^{\prime} \geq W)。 F_{i}(J) 是高分辨率图像 J 中一个尺寸为 H \times W 的crop区域（属于Stable Diffusion空间域），通过预定义滑动窗口的方式获取覆盖高分辨率图像 J 的 n 个这样的crop区域，设置权重 W_i=1 ，文本提示 y_{i}=\lambda_{i}(z)，MultiDiffusion采样过程 \Psi 定义为：

\Psi\left(J_{t}, z\right)=\underset{J \in \mathcal{J}}{\arg \min } \sum_{i=1}^{n}\left\|F_{i}(J)-\Phi\left(F_{i}(J), y_{i}\right)\right\|^{2}\\

上述公式同样是一个最小二乘问题，我们可以根据分析法计算它的最优解。

总的来说，MultiDiffusion融合了Stable Diffusion提供的多条扩散采样路径。如下图所示，H\times4W的高分辨率图像是我们想要生成的，下图(a) 展示了在四种非重叠crop区域上独立使用Stable Diffusion时的生成结果。正如我们之前分析的那样，生成的crop区域之间没有一致性，相当Stable Diffusion生成四个随机样本进行拼凑。下图(b)展示了超参数相同的情况下，MultiDiffusion生成过程可以融合这些最初不相关的扩散采样路径，最终生成连贯的高分辨率图像。

MultiDiffusion生成高分辨率图像的效果

最后我们再学习一下图像子区域编辑（Region-based）生成功能。

在基于Region-based的图像子区域编辑生成任务中，我们给定一组区域掩码 \left\{M_{i}\right\}_{i=1}^{n} \subset\{0,1\}^{H \times W} 和一组对应的文本提示词 \left\{y_{i}\right\}_{i=1}^{n} \subset \mathcal{Y}^{n} ，我们的目标是生成图像 I \in \mathcal{I} ，在每个区域中包含文本提示词所对应的内容，也就是说图像 I \otimes M_{i} 应该体现文本提示词 y_i 。我们设置区域映射 F_{i}(I)=I ，权重 W_{i}=M_{i} ，那么MultiDiffusion采样过程 \Psi 定义为：

\Psi\left(J_{t}, z\right)=\underset{J \in \mathcal{I}}{\arg \min } \sum_{i=1}^{n}\left\|M_{i} \otimes\left[J-\Phi\left(J_{t} \mid y_{i}\right)\right]\right\|\\

这也是一个最小二乘问题，我们可以根据分析法计算它的最优解。

在MultiDiffusion的每一步中， J_t 的每个像素都是由包含它的所有区域 M_i 的加权融合而来。这样我们仅需给出Bounding Box约束，MultiDiffusion就可以生成符合文本提示词描述的各种高质量图像。

同时MultiDiffusion可以进一步支持从紧密的mask获得高保真图像。考虑到结构布局是在扩散过程的早期就确定的，因此我们应该尽量使得 \Phi\left(J_{t} \mid y_{i}\right) 在生成过程的早期就关注区域 M_i 以匹配所需的布局，然后再考虑图像全局连贯性。基于此，区域映射 F_i 可以定义为：

F_{i}\left(J_{t}, t\right)=\left\{\begin{array}{ll} J_{t}, & \text { if } t \leq T_{\text {init }} \\ M_{i} \otimes J_{t}+\left(1-M_{i}\right) \otimes S_{t}, & \text { if } T_{i n t}<t \leq T \end{array}\right.\\

其中 S_t 作为背景，是具有预设颜色的随机图像。 T_{int} 设置为生成过程的 20%（即 T_{int} = 800 ）。

讲完详细的原理，我们在WebUI中对图像子区域进行编辑生成的实践。我们首先需要创建一个文生图画布，然后接着在这个画布中启动我们的编辑区域，我们可以选择多个编辑区域，每个编辑区域我们都可以拖动鼠标进行移动和调整区域大小。

一般来说，我们先设定一个全局的编辑区域，用于生成我们想要的背景内容。然后接着我们再创建几个局部编辑区域，用于生成我们想要的前景内容（任务、建筑、产品等）。具体如下图所示：

Tiled Diffusion还可以帮助Dtable Diffusion模型在有限的显存中生成超大分辨率（2k~8k）的图像，并且无需任何后处理即可生成无缝输出。但是与之相对应的，生成速度显著慢于常规的生成方法。

在参数选择上，我们可以直接参考下图所示的参数配置：

总的来说，Tiled Diffusion功能本质上是对图像的局部Tile区块和整体图像区块的重绘过程。

Tiled Diffusion还可以对图像进行超分辨率重建和超分辨率重绘。这是我们需要选择一个超分模型，先将图像进行超分辨率重建，然后再按照上述讲到的Tiled Diffusion处理流程，对超分后的图像进行切片重绘和融合。

2.8 SuperMerge插件

在WebUI中，SuperMerger插件是用于AI绘画模型合并和优化的一个功能强大且灵活的工具。它可以对多个不同的深度学习模型进行权重合并，并根据指定的参数生成一个新的合成模型。其核心作用是帮助我们优化AI绘画模型性能，或利用不同模型的优势生成更加多样化的模型。

SuperMerger可以将两个或多个模型的权重按照指定的比例合并。例如，我们可以通过 50% 的权重来自模型 A，另外 50% 的权重来自模型 B，生成一个新的模型。这个过程允许用户结合多个模型的优势，生成一个在特定任务上表现更好的模型。

SuperMerger可以通过模型合并生成一个新的模型，并将其保存为一个独立的模型文件。这意味着用户不需要从头开始训练新模型，而是可以通过现有模型的组合和调整生成新的变种模型。这个功能对AI绘画研究和应用开发非常重要，因为它大大降低了训练成本，尤其是当训练一个大规模模型需要大量资源时，SuperMerger 提供了一种快速优化的方案。

SuperMerger 插件为 WebUI 提供了一个强大的模型合并和优化工具，允许我们结合不同AI绘画模型的优点，以低成本的方式生成高效、灵活的新AI绘画模型。它通过提供直观的操作界面，简化了复杂的模型合并任务，使研究者和开发者能够更快速地迭代和探索新模型，极大提高了工作效率。

在具体功能上，我们可以使用SuperMerger融合出全新的SD系列模型和LoRA系列模型：

SD模型 + SD模型 -> 新SD模型
SD模型 + LoRA模型 -> 新SD模型
LoRA模型 + LoRA模型 -> 新LoRA模型
SD模型 - SD模型 -> 新SD模型

那么在我们进行AI绘画大模型融合时，该如何设置模型权重配比呢？Don't worry，Rocky也持续整理归纳了一些AI绘画大模型融合的权重配比参数，希望能给大家带来帮助：

融合LoRA模型时，LoRA的权重设置经验：

1. 影响除脸之外的部分（保持不变脸情况下改变其他部分）
（强）1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1
（弱）1,1,1,1,1,1,0.2,1,0.2,0,0,0.8,1,1,1,1,1

2. 影响脸部（脸型、发型、眼型、瞳色等）
（强）1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0
（弱）1,0,0,0,0,0,0,0,0.8,1,1,0.2,0,0,0,0,0
（微弱）1,0,0,0,0,0,0,0,0.2,0.6,0.8,0.2,0,0,0,0,0

3. 影响手部
1,0,1,1,0.2,0,0,0,0,0,0,0,0,0,0,0,0

4. 影响服装（搭配tag使用）
1,1,1,1,1,0,0.2,0,0.8,1,1,0.2,0,0,0,0,0

5. 影响动作（搭配tag使用）
1,0,0,0,0,0,0.2,1,1,1,0,0,0,0,0,0,0

6. 影响上色风格（搭配tag使用）
1,0,0,0,0,0,0,0,0,0,0,0.8,1,1,1,1,1

7. 角色（去风格化）
1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0

8. 背景（去风格化）
1,1,1,1,1,1,0.2,1,0.2,0,0,0.8,1,1,1,0,0

9. 减弱过拟合（等同于OUTALL）
1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1

Rocky也会持续更新沉淀，码字不易，希望大家能够多多点赞！！！

3. 从0到1保姆级ComfyUI使用教程（全网最详细）

ComfyUI框架作为目前AI绘画领域的主流节点式框架，端脑云平台也完全集成进来了，大家可以高效地构建属于自己的独家AI绘画工作流！

与Stable Diffusion WebUI一样，ComfyUI也是基于gradio架构的AI绘画框架。ComfyUI最大的特点是有一个基于节点流程的Stable Diffusion应用操作界面，可以构建丰富多样的AI绘画工作流，实现更加精准的AI绘画功能定制，同时让AI绘画功能的流程更加解耦和可复现。

ComfyUI框架经典工作流界面

AI绘画工作流中的每一个模块都有特定的的功能，我们可以通过调整模块的排列组合达到不同的AI绘画功能，但节点式的工作流也提高了一部分使用门槛。

与此同时，ComfyUI对内部的AI绘画流程进行了优化，生成图片的耗时比Stable Diffusion WebUI减少了10%-25%左右（不同GPU的提升效果不同）。在生成超大图片（分辨率1024x1024以上）时也不会导致显存溢出，但可能会因为图像切块运算导致生成的图像出现割裂和不自然的现象。

3.1 从0到1在端脑云平台快速上手使用ComfyUI

我们只需要按照之前章节中讲过的创建新应用的流程，创建一个ComfyUI的应用即可使用ComfyUI绘画框架！

如上图所示，我们打开端脑云平台界面，点击左侧的“我的应用”，就可以看到正在运行中的ComfyUI应用，我们点击右侧的“进入应用”，即可使用ComfyUI框架进行AI绘画了！

同时，我们点击“文件管理”按钮，就能进入后台管理ComfyUI框架的SD系列模型，LoRA模型，ControlNet模型等核心组件，方便的搭建自己的AI绘画模型武器库。大家可以点击如下链接进行注册使用，挖掘脑云平台上ComfyUI框架的更多玩法：Cephalon Cloud

从上图可以看到，ComfyUI框架中每个模块都是解耦的，Load Checkpoint模块用于加载SD系列模型；CLIP Text Encode模块用于输入正向提示词和负向提示词；Empty Latent Image用于生成初始的Latent特征；Ksampler模块用于控制扩散过程；VAE Decode模块用于将Latent Feature重建成像素级的图像。

我们接着输入以下提示词：Landscape painting, a thatched wooden house with bamboo fences in front of it covered in blooming roses (close-up), flowers and plants on the ground, stone path, sunset glow, red clouds all over the sky, dreamy, fantastical, cinematic lighting, beautiful, aesthetic, wlop, digital painting, trending on ArtStation, highly detailed, epic composition, official media, 8k uhd, distant

并且点击“Queue Prompt”，开始生成图片：

上面的工作流是经典的文生图工作流，那么图生图工作流、ControlNet工作流、Inpainting工作流、图像超分工作流等更多玩法，还需要大家自己亲自使用构建。

3.2 零基础ComfyUI框架原理拆解（全网最详细讲解）

从之前的章节我们已经知道ComfyUI 是AIGC时代的“基于有向无环图（DAG）的可视化推理管线”AI绘画框架。ComfyUI把“模型加载、算子节点、图调度执行、API/服务层、扩展节点生态”等模块清晰分层，核心在于把Stable Diffusion、FLUX.1、Wan2.1等AIGC大模型推理流程拆成节点，每个节点都有自己独立的功能，每个模块通过输入和输出的线连在一起变成一个完整的工作流，最后组合成图后由执行引擎调度运行。

正是因为ComfyUI简单易懂和极具扩展性的设计思想，目前在AI绘画/AIGC图像生成领域中得到了广泛的应用，延展出繁荣的生态。

接下来Rocky通过对ComfyUI中代表性的处理流程进行深入浅出的分析，希望能够帮助读者理解ComfyUI中的各种核心实现，以及与外部节点算法库集成的各种细节，从而可以解决应用ComfyUI过程中出现的各种问题，或者基于ComfyUI进行二次开发工作。

【一】运行入口与核心服务进程代码解析

核心文件名称	作用讲解
main.py	常用启动入口（解析 CLI 配置、初始化模型路径、前端资源、启动服务器）
server.py	WebSocket/HTTP 服务层与会话管理，处理前端通信、任务提交/进度回传、图的执行触发
protocol.py	与前端/客户端通信协议的小型约束
app/	app_settings.py、frontend_management.py：前端静态资源、设置管理。
model_manager.py、user_manager.py：运行期模型/用户管理。
custom_node_manager.py：自定义节点扫描/装载。
logger.py：日志集成。
api_server/	若开启独立 API 模式，这里提供 REST 路由与服务工具（routes/、services/、utils/file_operations.py）
execution.py	计算图与调度执行：把前端/协议层提交的“图”编译为可执行计划，协调调度、错误恢复、资源管理，与 comfy/ 模型层打通。
comfy_execution/	图执行引擎：
graph.py、graph_utils.py：图结构、拓扑排序、依赖与节点 IO 管理。
caching.py：节点级缓存/中间结果复用策略（哈希输入/参数以避免重复计算）。
progress.py：进度、步骤回调、可中断机制。
validation.py：图与节点输入校验。
utils.py：执行期工具。
node.py	DAG内置节点的实现。内置标准节点的集中定义（图像加载/保存、提示词、采样、ControlNet、LoRA、VAE、调度器等）。节点声明输入/输出类型、执行逻辑，供图引擎装配。
comfy_extras/	扩展节点与功能（更广的 IO、图像处理、后处理、实验性算子等）
custom_nodes/	用户侧自定义节点存放目录（样例 example_node.py.example）。启动时由 app/custom_node_manager.py 自动发现并注册。
comfy_api_nodes/	基于 Comfy API 的“节点化封装”，把三方/外部推理服务（如 OpenAI、Stability、Runway、Luma、Pika 等）包装为可组合节点：
nodes_*.py：各厂商/平台的节点集合。
apis/、util/、mapper_utils.py：统一鉴权、参数映射、错误处理与工具。
允许在同一图中混合本地推理与云端推理。
comfy/	（推理与模型底层）
文生图/图生图核心：sd.py（除SD采样算法外其他模块包含VAE、CLIP等算法模块的前后处理逻辑）、model_base.py（在开源SD代码的基础上进一步封装了diffusion的一些关键模型对象）、model_management.py（实现模型在不同设备上的加载、卸载与显存管理等功能）、model_patcher.py（对模型进行权重修改，支持LORA动态加载等功能）、model_sampling.py、sample.py、samplers.py（核心采样算法前后处理相关逻辑）、extra_samplers/。
文本/图像编码器：text_encoders/、image_encoders/、audio_encoders/、clip_*.py、sd1_clip.py、sdxl_clip.py。
控制/适配：controlnet.py（controlnet等相关控制型算法模型的处理逻辑）、t2i_adapter/、lora.py（lora模型加载的相关逻辑）、lora_convert.py、weight_adapter/。
Diffusers 适配：diffusers_load.py、diffusers_convert.py。
其它底层：ops.py（算子）、utils.py（通用工具）、options.py（可选项）、latent_formats.py、rmsnorm.py 等。
大量具体模型实现在 ldm/ 子目录（Stable Diffusion/UNet/VAE 等结构与组件）。
models/	模型权重的默认目录结构（如 checkpoints/、vae/、loras/ 等）。
folder_paths.py	统一管理与发现模型/资源的查找路径、覆盖机制（结合 extra_model_paths.yaml.example）。
middleware/	cache_middleware.py：请求/响应或任务级缓存挂钩，可与执行缓存协作，提高多用户/多请求复用率。
工程与配置	pyproject.toml、requirements.txt：依赖与打包配置。
comfy_config/：配置解析（config_parser.py）与类型（types.py）。
middleware/、utils/、node_helpers.py、latent_preview.py、cuda_malloc.py：辅助模块与性能工具。
script_examples/：Python 客户端示例（HTTP/WebSocket 调用、提交图、接收图像）。
output/：默认输出目录。

接下来Rocky再将上面表格中的核心文件代码已流程图的形式展现，让大家能够更加直观的学习感受ComfyUI的魅力：

ComfyUI模型管理机制：

ComfyUI自带的model loader大部分会先把模型加载到内存中，然后在运行时按需加载到GPU（KSampler节点运行前）；

Controlnet模型在ControlNetLoader节点会加载到内存中，然后在ControlnetApply节点基于strength生成一个新的模型副本，最终在KSampler节点统一加载到GPU中；

CheckpointLoaderSimple节点在一些条件下底模会直接加载到显存中，后续若底模引用发生变化（比如加了LoRA相关Patch），则会在KSampler节点卸载原底模，加载新底模；

核心模型加载到GPU上都是通过comfy/model_management.py::load_models_gpu()API来实现，最终加载到GPU上的模型都通过comfy/model_management.py::current_loaded_models 数组来缓存。

核心模型从GPU卸载到Memory，会调用comfy/model_management.py::cleanup_models()API来实现；卸载时机是每次进行workflow的调度前，会检查comfy/model_management.py::current_loaded_models 数组中每个模型的引用计数，如果无其他对象引用该模型，则会进行模型卸载。

ComfyUI每次调度workflow过程中每个节点的执行结果会被缓存，下次请求执行时会通过节点参数以及前置依赖节点的参数是否变化或者节点的ISCHANGED()方法来判断是否清理缓存和重新执行当前节点。节点的缓存清理和模型的引用计数相关联 。

到这里，我们已经知道ComfyUI是基于节点式的AI绘画框架了，那么我们该如何本质的理解其工作原理与框架思想呢？

我们先拆解一下Stable Diffusion模型的完整工作流程，具体如下所示：

Stable Diffusion模型的完整工作流程

可以看到，Stable Diffusion模型的完整推理流程可以解耦成Condition注入、扩散模型去噪、VAE编码与解码等几个模块，ComfyUI正是基于这些工作模块，对每个模块单独进行封装，制作出了一个个AI绘画节点（node），我们可以方便的对这些节点进行连接与搭建，从而形成各式各样的AI绘画工作流。

接下来，我们再对每个节点进行详细的讲解。

首先是获取输入Prompts的Tokens。由于计算机不理解原始的输入文字，因此首要任务是使用分词器将每个单词转换为token符号。

使用分词器将输入Prompts转换成Token符号
print(tokenizer('dog in batman costume'))
{'input_ids': [49406, 1929, 530, 7223, 7235, 49407], 'attention_mask': [1, 1, 1, 1, 1, 1]}

如果我们打开在Stable Diffusion官方提供的tokenizer文件夹，就可以找到的文件vocab.json，里面包含了几乎所有输入文字的对应Token符号字典。

"<|startoftext|>": 49406,

"dog</w>": 1929,

"in</w>": 530,

"batman</w>": 7223,

"costume</w>": 7235,

"<|endoftext|>": 49407,

我们完成输入文本到Tokens的转化后，接下来需要进行Token到Text Embeddings特征的编码。

Token到Text Embeddings特征的编码

在我们获得Text Embeddings特征后，还需要通过Attention机制将其转换成Transformer特征，注入到Stable Diffusion的扩散模型中。

Text Embeddings特征通过Attention机制转换成Transformer特征

有了上面的这些文本特征信息，再加上图像的Latent特征，我们就可以使用Stable Diffusion的扩散模型部分进行去噪生成过程了。

Stable Diffusion的扩散模型部分的去噪生成过程

注意，Stable Diffusion 1.x、2.x以及XL版本的扩散模型部分是基于U-Net架构的；如果是Stable Diffusion 3或者FLUX.1版本，其扩散模型部分是基于Transformer架构的。

同时我们需要配置调度器的TimeSteps数值：

scheduler.set_timesteps(inference_steps)
print(scheduler.timesteps)

tensor([999.0000, 964.5517, 930.1035, 895.6552, 861.2069, 826.7586, 792.3104,
        757.8621, 723.4138, 688.9655, 654.5172, 620.0690, 585.6207, 551.1724,
        516.7241, 482.2758, 447.8276, 413.3793, 378.9310, 344.4828, 310.0345,
        275.5862, 241.1379, 206.6897, 172.2414, 137.7931, 103.3448,  68.8966,
         34.4483,   0.0000]) 

上述的TimeSteps有30个步骤，同时间隔相同的距离（34.4483个单位）。

最后，我们需要VAE的编码器将输入图像转换为潜在空间中的张量，并且使用VAE的解码器将潜在空间中的张量转换为图像。

VAE的编码器和解码器的编码与解码过程

未完待续，希望大家能持续关注！！！

3.3 ComfyUI特性
简洁的节点/图形/流程图界面，用于快速实验和创建复杂的AI绘画工作流，无需编写任何代码。
完全支持 SD系列和FLUX.1系列等主流AI绘画大模型。
异步队列系统。
只重新执行同一工作流中在两次执行之间发生变化的部分。
如果没有GPU也能使用，自动切换至CPU。
支持加载ckpt、safetensors、pt和diffusers等不同格式的AI绘画模型。
支持LoRA系列模型、ControlNet系列模型、GAN系列模型、Embeddings系列模型、传统深度学习模型等AI行业的所有主流模型。
生成的PNG文件中包含了完整的工作流信息，能够方便在ComfyUI上再次加载。
可以使用Json形式保存/加载工作流。
支持AI绘画模型的融合实验和模型保存（Model Merging）。
与Stable Diffusion WebUI相比，启动速度极快。
3.4 ComfyUI文生图全解析
3.5 ComfyUI图生图全解析
3.6 ComfyUI中LoRA、ControlNet、GAN使用全解析
4. 从0到1保姆级Fooocus使用教程

端脑云平台也集成了AI 绘画工具 Fooocus，Fooocus 是对 stable diffusion 和Midjourney设计的重新思考：一方面保留了 SD 的开源属性，可以部署到本地免费使用；另一方面在操作界面上向 midjourney 学习，省去了 WebUI 中复杂的参数调节，让用户可以专注于提示和图像。可以和 Stable diffusion WebUI 一样部署到本地免费使用，同时具备 midjourney 一样便捷的操作界面。

我们只需要按照上面的创建新应用的流程，创建一个Fooocus的应用即可使用Fooocus绘画框架！

如上图所示我们在端脑云平台左侧点击“我的应用”后，就能看到正在运行中的Fooocus应用，点击进入应用之后，可以看到Fooocus绘画框架的操作界面非常简洁。整个界面中只有生成图像展示窗口、正向提示词和生成按钮 3 项。同时勾选“Advanced”会弹出高级设置的窗口，可以调整画面宽高比、风格、图像数量、种子值、反向提示词、模型、lora 权重比值、图像锐利程度等。勾选“Input Image”可以上传图片进行超分，作为底图，进行重绘或者AI扩图。

我们在Prompt框中输入如下Prompt：1girl, solo, black_hair, hair_ornament, jewelry, closed_mouth, upper_body, flower, earrings, hair_flower, mole, lips, grey_eyes, eyelashes, makeup, facial_mark, lipstick, red_flower, mole_under_mouth, red_lips。

然后点击Generate按钮，即可生成图片：

当我们点击“Input Image”后，就会出现如下的信息栏：

在Upscale or Variation栏中，我们可以上传图片，进行图像超分或者变换任务。在Image Prompt栏中，我们上传图片用作图像提示词，给我们的图像生成过程附加参考图。而Inpaint or Outpaint栏中，我们可以对上传的图片进行重绘和扩图。

接下来我们再看看“Advanced”按钮的功能，我们点击“Advanced”按钮，可以看到右侧出现了高级设置的窗口：

在高级设置的窗口的Setting栏中，我们可以设置生成图片的Performance，图片尺寸，图片数量，负向提示词等配置。

在Style栏中，端脑云平台中已经集成了100多个图像的生成风格，包括写实、胶片、电影质感、动漫、水彩、黏土、3D、等距、像素、霓虹、赛博朋克、波普、纸艺等各个方面，我们可以方便的选择我们想要的风格，生成相应的图像。

在Model栏中，我们可以选择SD系列模型，端脑云平台中已经给出了非常多的高质量SD模型供我们选择，也有很多LoRA模型可以进行搭配。

在最后一个Advanced栏中，我们可以调整图像生成过程中的Guidance Scale值和Image Sharpness值。

到这里，端脑云平台中的Fooocus绘画框架的基本使用流程和参数已经讲好了，大家可以尝试使用一下，有非常多的玩法等待大家挖掘！大家可以点击如下链接进行注册使用：Cephalon Cloud

5. 从0到1保姆级Stable Diffusion WebUI-forge使用教程
6. 推荐阅读

Rocky会持续分享AIGC的干货文章、实用教程、商业应用/变现案例以及对AIGC行业的深度思考与分析，欢迎大家多多点赞、喜欢、收藏和转发，给Rocky的义务劳动多一些动力吧，谢谢各位！

6.1 深入浅出完整解析Stable Diffusion 3（SD 3）和FLUX.1系列核心基础知识

Rocky也对Stable Diffusion 3和FLUX.1的核心基础知识作了全面系统的梳理与解析：

6.2 深入浅出完整解析Stable Diffusion XL（SDXL）核心基础知识

Rocky也对Stable Diffusion XL的核心基础知识作了全面系统的梳理与解析：

6.3 深入浅出完整解析Stable Diffusion（SD）核心基础知识

Rocky也对Stable Diffusion 1.x-2.x系列模型的核心基础知识做了全面系统的梳理与解析：

6.4 深入浅出完整解析Stable Diffusion中U-Net的前世今生与核心知识

Rocky对Stable Diffusion中最为关键的U-Net结构进行了深入浅出的全面解析，包括其在传统深度学习中的价值和在AIGC中的价值：

6.5 深入浅出完整解析LoRA（Low-Rank Adaptation）模型核心基础知识

对于AIGC时代中的“ResNet”——LoRA模型，Rocky也进行了深入浅出的全面讲解：

6.6 深入浅出完整解析ControlNet核心基础知识

AI绘画作为AIGC时代的一个核心方向，开源社区已经形成以Stable Difffusion为核心，ConrtolNet和LoRA作为首要AI绘画辅助工具的变化万千的AI绘画工作流。

ControlNet正是让AI绘画社区无比繁荣的关键一环，它让AI绘画生成过程更加的可控，更有助于广泛地将AI绘画应用到各行各业中：

6.7 深入浅出完整解析Sora等AI视频大模型核心基础知识

AI绘画和AI视频是两个互相促进、相互交融的领域，2024年无疑是AI视频领域的爆发之年，Rocky也对AI视频领域核心的Sora等大模型进行了全面系统的梳理与解析：

6.8 深入浅出完整解析AIGC时代Transformer核心基础知识

在AIGC时代中，Transformer为AI行业带来了深刻的变革。Transformer架构正在一步一步重构所有的AI技术方向，成为AI技术架构大一统与多模态整合的关键核心基座，大有一统“AI江湖”之势。Rocky也对Transformer模型进行持续的深入浅出梳理与解析：

6.9 手把手教你成为AIGC算法工程师，斩获AIGC算法offer！

在AIGC时代中，如何快速转身，入局AIGC产业？如何成为AIGC算法工程师？如何在学校中系统性学习AIGC知识，斩获心仪的AIGC算法offer？

Don‘t worry，Rocky为大家总结整理了全面的AIGC算法工程师成长秘籍，为大家答疑解惑，希望能给大家带来帮助：

6.10 AIGC产业的深度思考与分析

2023年3月21日，微软创始人比尔·盖茨在其博客文章《The Age of AI has begun》中表示，自从1980年首次看到图形用户界面（graphical user interface）以来，以OpenAI为代表的科技公司发布的AIGC模型是他所见过的最具革命性的技术进步。

Rocky也认为，AIGC及其生态，会成为AI行业重大变革的主导力量。AIGC会带来一个全新的红利期，未来随着AIGC的全面落地和深度商用，会深刻改变我们的工作、生活、学习以及交流方式，各行各业都将被重新定义，过程会非常有趣。

那么，在此基础上，我们该如何更好的审视AIGC的未来？我们该如何更好地拥抱AIGC引领的革新？Rocky准备从技术、产品、商业模式、长期主义等维度持续分享一些个人的核心思考与观点，希望能帮助各位读者对AIGC有一个全面的了解：

6.11 算法工程师的独孤九剑秘籍

为了方便大家实习、校招以及社招的面试准备，同时帮助大家提升扩展技术基本面，Rocky将符合大厂和AI独角兽价值的算法高频面试知识点撰写总结成《三年面试五年模拟之独孤九剑秘籍》，并制作成pdf版本，大家可在公众号WeThinkIn后台【精华干货】菜单或者回复关键词“三年面试五年模拟”进行取用：

6.12 深入浅出完整解析AIGC时代中GAN系列模型的前世今生与核心知识

GAN网络作为传统深度学习时代的最热门生成式Al模型，在AIGC时代继续繁荣，作为Stable Diffusion系列模型的“得力助手”，广泛活跃于Al绘画的产品与工作流中：

Rocky一直在运营技术交流群（WeThinkIn-技术交流群），这个群的初心主要聚焦于AI行业话题的讨论与研究，包括但不限于算法、开发、竞赛、科研以及工作求职等。群里有很多AI行业的大牛，欢迎大家入群一起交流探讨～（请备注来意，添加小助手微信Jarvis8866，邀请大家进群～）
