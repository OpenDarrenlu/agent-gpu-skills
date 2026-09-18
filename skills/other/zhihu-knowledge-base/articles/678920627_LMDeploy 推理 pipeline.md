# LMDeploy 推理 pipeline

**作者**: lvhan程序媛

**原文链接**: https://zhuanlan.zhihu.com/p/678920627

---

LMDeploy 从 0.2.0 版本，正式发布 pipeline 接口。它是 LMDeploy 对底层推理引擎的高度抽象和封装。我们希望它能像 transformer 里的 pipeline 一样，给上层的应用带来便捷。
目前，pipeline 还只是针对的 LLM 模型。后续，我们会扩展到多模态模型。
pipeline 初始版难免有很多不足，还请大家体谅和包涵。我们会悉听大家的意见和建议，不断改进和完善。
接下来，本文将从基本用法开始，向大家展示 pipeline 的使用方法和技巧。在此之前，请首先安装 lmdeploy，版本不低于 0.2.0

pip install lmdeploy
基本用法
from lmdeploy import pipeline
pipe = pipeline('internlm/internlm2-chat-7b')
response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
print(response)

在这个例子中，pipeline默认申请50%显存，用来存储推理过程中产生的k/v。对于 7B 模型来说，如果显存小于 40G，会出现 OOM。
当遇到 OOM 时，请大家按照下面的方法调节k/v cache分配比例：

from lmdeploy import pipeline, TurbomindEngineConfig

# 调低 k/v cache内存占用比例为 20%
backend_config = TurbomindEngineConfig(cache_max_entry_count=0.2)

pipe = pipeline('internlm/internlm2-chat-7b', 
                backend_config=backend_config)
response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
print(response)

当然，更好的做法是，从空闲显存中按照一定的比例为k/v cache开辟空间。我们会在后续的版本中加以完善，让大家有更好的体验。
在推理时，如果需要生成参数，比如sampling参数、期望生成的token个数等，可以像下面的例子，增加GenerationConfig参数:

from lmdeploy import pipeline, GenerationConfig

gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=1024)
pipe = pipeline('internlm/internlm2-chat-7b')
response = pipe(['Hi, pls intro yourself', 'Shanghai is'],
                gen_config=gen_config)
print(response)

关于 LMDeploy GenerationConfig，我们计划与 transformers 库的 GenerationConfig 对齐。

更多用法

在接下来的章节中，我们通过具体的示例，展示如何使用 pipeline 实现张量并行、长文本推理、量化模型推理等功能。

张量并行

初始化 pipeline 时，设置引擎参数，把 tp 置为需要用的 GPU 个数。tp 的值务必为2的整数次幂。

from lmdeploy import pipeline, TurbomindEngineConfig

backend_config = TurbomindEngineConfig(tp=2)

gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=1024)
pipe = pipeline('internlm/internlm2-chat-20b',
                backend_config=backend_config)
response = pipe(['Hi, pls intro yourself', 'Shanghai is'],
                gen_config=gen_config)
print(response)
长文本推理

LMDeploy 实现了外推方法之一 Dynamic NTK 方式，算法原理可以参考 Scaling Laws of RoPE-based Extrapolation。LMDeploy 的实现方式与 transformer 的 LlamaDynamicNTKScalingRotaryEmbedding 是对齐的。
下面展示的是使用 LMDeploy 把 InternLM2 模型的上下文外推到 210K。

from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

backend_config = TurbomindEngineConfig(rope_scaling_factor=2.0,
                                       session_len=210000,
                                       tp=1)
pipe = pipeline('internlm/internlm2-chat-7b', backend_config=backend_config)
# prompt 可以替换为长文本的输入
prompt = '你好'

gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=1024)
response = pipe(prompt, gen_config=gen_config)
print(response)
# Response(text='你好！很高兴能为你提供帮助。有什么我可以为您做的吗？', generate_token_len=15, finish_reason='stop')

在这个例子中，文本长度由引擎配置中的session_len指定。它表示模型最大的上下文窗口大小，为输入、输出token之和，推荐不小于 210K。
Dynamic NTK 外推方法的系数通过rope_scaling_factor设置。我们可以近似的认为、该值越高，模型在长文表现更好，但有损失短文表现的风险（ 32K 以下的文本不受影响，一般在过大时会导致 32K~64K 之间的语言质量下降）。由于 InternLM2 各版本的训练数据、流程都有差别，InternLM2 的 20B 模型的长文本训练数据量是7B 模型的三分之一左右，所以对应的经验最佳 rope_scaling_factor 设置也有区别，下表是几个推荐在长文本环境使用的版本及其推荐参数。

推荐参数	InternLM2-7B	InternLM2-Chat-7B-SFT	InternLM2-Chat-20B-SFT	InternLM2-Chat-7B	InternLM2-Chat-20B
rope_scaling_factor	5.0	2.0	3.0	2.0	3.0

InternLM2 团队采用 OpenCompass的“大海捞针”来评测模型的长文本性能，评测方法和结果请参考这篇文章

量化模型推理

这个章节，我们略过模型量化过程，把重点集中在已量化好 4bit 模型的推理上。对于量化过程感兴趣的朋友们，请移步到 LMDeploy 文档的量化章节
LMDeploy 支持 huggingface hub 上通过 AWQ 算法量化的 4bit 模型推理。比如 lmdeploy 空间、internlm 空间和 TheBloke 空间下的模型。比如：

from lmdeploy import pipeline, TurbomindEngineConfig

# 参数 model_format为awq，表示模型是 awq 量化的4bit模型。
backend_config = TurbomindEngineConfig(model_format='awq', tp=1)

# 推理 internlm2 的量化模型
pipe = pipeline("internlm/internlm2-chat-7b-4bits",
                backend_config=backend_config)
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)

# 推理 thebloke 空间下的 llama2-13b 量化模型
from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
pipe = pipeline("TheBloke/LLaMA2-13B-Tiefighter-AWQ",
                backend_config=TurbomindEngineConfig(model_format='awq'),
                chat_template_config=ChatTemplateConfig(model_name='llama2')
                )
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)
```
未完待续
pipeline中多种样式的 prompt
如何在pipeline中使用 LMDeploy 另一推理引擎 PyTorchEngine
参考文档
https://zhuanlan.zhihu.com/p/678784248
