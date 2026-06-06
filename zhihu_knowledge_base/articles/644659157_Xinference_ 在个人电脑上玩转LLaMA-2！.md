# Xinference: 在个人电脑上玩转LLaMA-2！

**作者**: Uranus​清华大学 计算机系博士在读

**原文链接**: https://zhuanlan.zhihu.com/p/644659157

---

Fine-tune LLaMA-2 的文章来啦：Uranus：如此简单！LLaMA-2 finetune 实战！

我为所有想在个人电脑上尝试 LLaMA-2 的小伙伴准备了一个教程，可以在个人电脑上体验 LLaMA-2（甚至不需要显卡）。未来微调 LLaMA-2（7B，13B，和 70B）的教程也会陆续放出。对微调感兴趣的小伙伴可以先关注我的账号！

这次 LLaMA-2 的发布让我对开源大模型更加充满了信心！ Meta 再次证明了自己是 “真 OpenAI” ！YYDS！

LLaMA-2

LLaMA-2 共发布了 7B，13B，34B（暂时还没放出）和 70B，将免费提供用于研究和商业用途。预训练过程的使用的数据相比于第一代，增长了40%，上下文长度也增加了一倍，并且采用了分组注意力机制来提升性能。其微调版本 LLaMA-2-chat，总共收集了超过 100w 条人工标注用于 RLHF。根据热心网友测算，70B 参数的模型的训练成本将达到 260w 美元。更多评价可以参考我在这个问题下的回答。

使用 Xinference 快速体验 LLaMA-2

Xinference 是一个模型推理框架，支持包括 LLM，multimodal model 等多种模型。Xinference 可以让你在个人电脑（Win，macOS，Linux）上一键体验最前沿的开源模型，提供命令行与 web UI 方便用户快速体验模型。

Xinference 可以帮助你快速体验 LLaMA-2 在内的开源 LLM！甚至不需要显卡！让我们先来看看效果 :)

一首关于 LLaMA 和 GPT 的诗

除了 LLaMA-2 以外，Xinference 还支持以下模型，并提供了多种 quantization 规格，以满足不同需求，这个列表目前还在快速扩充中！

baichuan
chatglm
chatglm2
wizardlm-v1.0
wizardlm-v1.1
vicuna-v1.3
orca

本地体验 Xinference 异常简单。首先通过 PyPI 安装，我们强烈推荐使用一个新的虚拟环境来避免可能的依赖冲突：

pip install "xinference[all]"


之后，启动 Xinference 即可：

xinference
基于 Xinference 开发应用

Xinference 不仅可以让你通过命令行和聊天界面体验这些模型，而且可以让你在 AI 应用开发时，方便地对开源LLM 进行私有化部署，它将带来以下好处：

安全：私有化部署下，数据完全不外流，因此数据泄露的风险大大降低。
定制化：可以基于开源基础模型，使用自己的数据集进行微调，定制化模型。
成本更低：相比于 OpenAI 的 LLM 服务，私有化的 LLM 可以在定制化的基础上，以更小的模型达到相似的效果。这可以大大降低硬件需求，并提高推理效率。

Xinference 提供了与 OpenAI 兼容的 RESTful API，并提供了相应的 client 方便用户以编程的方式使用 Xinference：

from xinference.client import Client

client = Client("http://localhost:9997")

model_uid = client.launch_model(
    model_name="llama-2-chat",
    model_size_in_billions=13,
    quantization="q4_1"
)

model = client.get_model(model_uid)

chat_history = []
prompt = "What is the largest animal?"
model.chat(
    prompt,
    chat_history,
    generate_config={"max_tokens": 1024}
)

我们目前正在积极推动 Xinference 成为 LangChain 与 LlamaIndex 的内置 LLM 之一，届时使用 Xinference 开发 AI 应用将更加方便。

未来微调 LLaMA-2（7B，13B，和 70B）的教程也会陆续放出，对微调感兴趣的小伙伴也可以先关注我的账号 :)
