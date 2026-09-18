# 用 tensor-parallel 让大模型多卡并发推理 🚀

**作者**: Uranus​清华大学 计算机系博士在读

**原文链接**: https://zhuanlan.zhihu.com/p/619528375

---

​
目录
收起
参考

2024/02/26 Update：tensor parallel 在主流的推理框架已经很好的支持了，vLLM 和 lightllm 都是很好的选择。现在 tensor-parallel 这个项目的意义主要在做一些实验上，真实场景下不再适用。

上一篇文章中我用 Alpaca-LoRA 的代码结合 Luotuo作者翻译的 Alpaca 数据集 fine-tune 了一个自己的中文 chatbot。fine-tune 后发现两个问题：

单卡推理实在有点慢
对于比较复杂的输入，单卡的 varm 有时也会显得不堪重负

而今天偶然了解到了 tensor-parallel，这个库可以帮助我们很轻松地把模型训练与推理的 workload 平均分布到多块 GPU。一方面推理的速度上来了，另一方面 vram 的负载平衡也让复杂的 prompt 能被轻松处理。

话不多说，先上 demo！

首先 import 相关的 libs：

# torch version 2.0.0
import torch
# tensor-parallel version 1.0.22
from tensor_parallel import TensorParallelPreTrainedModel
# transformer version 4.28.0.dev0
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

加载 LLaMA-7B 并转化为 TensorParallelPreTrainedModel：

model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf", torch_dtype=torch.float16)
model = TensorParallelPreTrainedModel(model, ["cuda:0", "cuda:1"])

此时通过 nvitop，我们可以看到两张显卡的 vram 占用非常平均：

使用 tensor-parallel 加载模型的效果

加载 tokenizer 并进行推理：

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

tokens = tokenizer("Hi, how are you?", return_tensors="pt")
tokenizer.decode(model.generate(tokens["input_ids"].cuda(0), attention_mask=tokens["attention_mask"].cuda(0))[0])

输出：

 'Hi, how are you? I'm a 20 year old girl from the Netherlands'

另一个 example，让模型的输出长一点：

tokens = tokenizer("Once upon a time, there was a lonely computer ", return_tensors="pt")
tokenizer.decode(model.generate(tokens["input_ids"].cuda(0), attention_mask=tokens["attention_mask"].cuda(0), max_length=256)[0])

输出：

'Once upon a time, there was a lonely computer. It was a very old computer, and it had been sitting in a box for a long time. It was very sad, because it had no friends.\nOne day, a little girl came to the computer. She was very nice, and she said, “Hello, computer. I’m going to be your friend.”\nThe computer was very happy. It said, “Thank you, little girl. I’m very happy to have you as my friend.”\nThe little girl said, “I’m going to call you ‘Computer.’”\n“That’s a good name,” said Computer.\nThe little girl said, “I’m going to teach you how to play games.”\n“That’s a good idea,” said Computer.\nThe little girl said, “I’m going to teach you how to do math.”\nThe little girl said, “I’m going to teach you how to write stories.”\nThe little girl said, “I’m going to teach you how to draw pictures.”\nThe little girl said, “I’m going to teach you how to play music.”\nThe little girl said, “I’m'

因为是未经 fine-tune 的 LLaMA-7B，输出的效果一般。不过这不是重点，重点是我们的推理逻辑平均分布到了两块 GPU 上。tensor-parallel works！

我还把这个 demo 贴到了 tensor-parallel 相关的 issue 里，感兴趣的朋友可以去看看。

有了 tensor-parallel，我们还可以让 GPT4ALL，Alpaca-LoRA 等等模型并发地运行在我们的多块 GPU 上！等我后续有时间跑跑看，感兴趣的朋友可以保持关注 :)

参考

https://github.com/BlackSamorez/tensor_parallel

LLaMa 7B

tensor_parallel int8 LLM
