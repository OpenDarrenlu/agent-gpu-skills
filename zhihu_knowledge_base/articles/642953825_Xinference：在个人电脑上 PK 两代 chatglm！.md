# Xinference：在个人电脑上 PK 两代 chatglm！

**作者**: Uranus​清华大学 计算机系博士在读

**原文链接**: https://zhuanlan.zhihu.com/p/642953825

---

​
目录
收起
Xinference 是什么
体验 Xinference
大规模部署 Xinference
基于 Xinference 快速打造 AI 应用
总结

chatglm 一经发布便成为了国内 LLM 圈顶流。而前一阵子 chatglm2 发布带来了性能，上下文长度，以及推理速度的全面提升。

然而想体验 chatglm2 却没那么容易，尽管是 6b 的 “小” 模型，你也至少需要一块 80 系 16GB（以 4080 为例，价格在9k左右）起步的显卡才能顺利运行。

加载 chatglm2 的显存开销

那如果想要对比两代 chatglm 呢？听起来很酷，但光是 chatglm2 便需要将近 13GB 显存，两代加在一起的话，90 系（以 4090 为例，价格在 13k 左右）拥有 24 GB 显存的卡也表示扛不住啊！

但这一切 Xinference 可以帮你做到！甚至不需要显卡！

让我们先来看看效果 :)

chatglm vs chatglm2
Xinference 是什么

照例先放 GitHub 链接：

Xinference 是一个分布式的模型推理框架，用来支持包括 LLM，multimodal model 等多种模型。

对于个人用户，Xinference 可以让你在个人电脑（Win，macOS，Linux）上一键体验最前沿的开源模型，甚至可以同时与两个 LLM 对话来比较模型的好坏！
对于企业用户，Xinference 能够帮助你在计算集群上轻松地部署并管理模型，享受私有化部署带来的安全，定制化，以及低成本。

下面是目前 Xinference 支持的模型列表，而这个列表目前还在快速扩充！

Name	Type	Language	Format	Size (in billions)	Quantization
baichuan	Foundation Model	en, zh	ggmlv3	7	‘q2_K’, ‘q3_K_L’, … , ‘q6_K’, ‘q8_0’
chatglm	SFT Model	en, zh	ggmlv3	6	‘q4_0’, ‘q4_1’, ‘q5_0’, ‘q5_1’, ‘q8_0’
chatglm2	SFT Model	en, zh	ggmlv3	6	‘q4_0’, ‘q4_1’, ‘q5_0’, ‘q5_1’, ‘q8_0’
wizardlm-v1.0	SFT Model	en	ggmlv3	7, 13, 33	‘q2_K’, ‘q3_K_L’, … , ‘q6_K’, ‘q8_0’
vicuna-v1.3	SFT Model	en	ggmlv3	7, 13	‘q2_K’, ‘q3_K_L’, … , ‘q6_K’, ‘q8_0’
orca	SFT Model	en	ggmlv3	3, 7, 13	‘q4_0’, ‘q4_1’, ‘q5_0’, ‘q5_1’, ‘q8_0’
体验 Xinference

本地体验 Xinference 异常简单。首先通过 PyPI 安装，我们强烈推荐使用一个新的虚拟环境来避免可能的依赖冲突：

pip install "xinference[all]"

之后，启动 Xinference 即可：

xinference

Xinference 启动后，会打印服务的 endpoint，并启动浏览器打开 web UI。此时你只需要：

选择你想要体验的模型，模型大小以及量化规格
点击 create，如果模型没有缓存在本地，此时会开始下载
第一次使用时下载模型
开始和模型对话吧！
00:41
大规模部署 Xinference

Xinference 可以部署在一台个人电脑，也可以轻松 scale out 到一个计算集群！

说到这里，不得不提私有化部署 LLM 能够带来的好处：

安全：私有化部署下，数据完全不外流，因此数据泄露的风险大大降低。
定制化：可以基于开源基础模型，使用自己的数据集进行微调，定制化模型。
低成本：相比于 OpenAI 的 LLM 服务，私有化的 LLM 可以在定制化的基础上，以更小的模型达到相似的效果。这可以大大降低硬件需求，并提高推理效率。

在私有化部署的场景下，Xinference 的内置资源调度可以帮助用户提高集群的吞吐量，并降低推理延迟。

此外，还可以基于 Xinference 提供的 API 进行动态的模型加载与释放，进一步提高集群利用率。

为了方便企业用户管理模型，Xinference 提供了与 OpenAI 兼容的 RESTful API，并提供了相应的 client 方便用户以编程的方式使用 Xinference：

from xinference.client import Client

client = Client("http://localhost:9997")
model_uid = client.launch_model(model_name="chatglm2")
model = client.get_model(model_uid)

chat_history = []
prompt = "What is the largest animal?"
model.chat(
    prompt,
    chat_history,
    generate_config={"max_tokens": 1024}
)

返回值：

{
  "id": "chatcmpl-8d76b65a-bad0-42ef-912d-4a0533d90d61",
  "model": "56f69622-1e73-11ee-a3bd-9af9f16816c6",
  "object": "chat.completion",
  "created": 1688919187,
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The largest animal that has been scientifically measured is the blue whale, which has a maximum length of around 23 meters (75 feet) for adult animals and can weigh up to 150,000 pounds (68,000 kg). However, it is important to note that this is just an estimate and that the largest animal known to science may be larger still. Some scientists believe that the largest animals may not have a clear \"size\" in the same way that humans do, as their size can vary depending on the environment and the stage of their life."
      },
      "finish_reason": "None"
    }
  ],
  "usage": {
    "prompt_tokens": -1,
    "completion_tokens": -1,
    "total_tokens": -1
  }
}
基于 Xinference 快速打造 AI 应用

Xinference 可以非常容易地与 LangChain，LlamaIndex 等流行的库进行集成。我们计划近期让 Xinference 成为 LangChain 的内置 LLM，届时使用 Xinference 开发 AI 应用将更加方便。

总结

对于个人用户 Xinference 可以让你在个人电脑上体验最前沿的开源模型。而对于企业用户，Xinference 能够帮助你在计算集群上轻松地部署并管理模型，享受私有化部署带来的安全，定制化，以及低成本。

执行：

$ pip install "xinference[all]"

即刻尝试使用 Xinference 吧！
