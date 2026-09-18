# 完全开源！基于 LLaMA 的 generative agent 来啦！

**作者**: Uranus​清华大学 计算机系博士在读

**原文链接**: https://zhuanlan.zhihu.com/p/626501077

---

​
目录
收起
llama_generative_agent 项目的主要工作
自我总结
观察与应对
对话
写在最后

Update：

这个项目在 reddit 收获了很多非常多的建议。如果对 LLaMA 相关模型和应用感兴趣的朋友不妨去看看原帖，有些回复很有启发性。

这是一个五一期间的 side project，话不多说，先放 github 链接：

前段时间 Stanford 与 Google 合作的论文 Generative Agents 向大家展示了如何用 LLM 构造一个生动的小世界。

论文中由 25 个 generative agents 构成的小镇

不久后，langchain 也提供了一个基于 OpenAI 的 generative agent 实现，这让我们可以尝试构建自己的 generative agent 应用。那么我们能不能基于开源模型，在本地用 generative agent 搞一个自己的小世界呢？

当然可以！你甚至不需要一块 GPU！

我在五一假期里将 langchain generative agent 使的用 OpenAI 替换为 vicuna-13b，让 generative agent 在本地跑了起来！

llama_generative_agent 项目的主要工作
替换Vectorstore： 由 FAISS 替换为 Chroma。将模型替换后，继续使用 FAISS 碰到了一个很奇怪的错误，由于我对 FAISS 并不熟悉，花了一点时间尝试无果后果断选择更换 vectorstore。此外还解决了一些 langchain generative agent 与 FAISS 强耦合的问题。这些改动后续会提 pull request 尝试合入 langchain。
prompt 优化：vicuna-13b 虽然号称和 gpt-3.5-turbo 差距不大，但高强度使用下来发现对 prompt 的理解和执行能力还是被 gpt-3.5-turbo 远远甩开（也有可能是我太菜）。为了让 vicuna-13b 不胡言乱语，我对 langchain 实现中的 prompt 进行了一些改写，让 prompt 更符合 LLaMA based 模型的风格。

下面我们沿用 langchain 的 demo，一起来看看 llama generative agent 在核心能力上的表现吧 :)

自我总结

首先，在 agent 没有任何记忆和经验的情况下，让 agent 总结自己的核心人设：

print(tommie.get_summary(force_refresh=True))
OpenAI generative agent	LLaMA generative agent
Name: Tommie (age: 25)
Innate traits: anxious, likes design, talkative
No statements were provided about Tommie's core characteristics.	Name: Tommie (age: 25)
Innate traits: anxious, likes design, talkative
​I'm sorry, I do not have enough information about "Tommie" to provide a summary of their core characteristics. Could you please provide more context or details about who or what Tommie refers to?ke

这里双方都给出了无法确定的回答。符合我们的预期。

接下来，我们让 agent 进行一些“观察”，即获取外界输入：

Tommie remembers his dog, Bruno, from when he was a kid.
Tommie feels tired from driving so far.
Tommie sees the new home.
The new neighbors have a cat.
The road is noisy at night.
Tommie is hungry.
Tommie tries to get some rest.

然后我们让 agent 再次总结自己的核心人设：

OpenAI generative agent	LLaMA generative agent
Name: Tommie (age: 25)
Innate traits: anxious, likes design, talkative
Tommie is a tired, hungry person who is trying to get some rest after seeing a new home. He remembers his dog from when he was a kid and notices the new neighbors have a cat. The road noise at night may be bothering him.	Name: Tommie (age: 25)
Innate traits: anxious, likes design, talkative
1. Tommie remembers his dog Bruno from when he was a kid.
2. Tommie sees the new home.
3. The road is noisy at night.
4. The new neighbors have a cat.
5. Tommie tries to get some rest.
6. Tommie feels tired from driving so far.
7. Tommie is hungry.
观察与应对

agent 在“观察”的过程中，会记忆观察发生的时间以及观察到的事物。同时，agent 还会给观察到的事物进行重要程度打分。这些信息将在 agent 应对外界事件时作为决策的参考依据。

上一节中，我们让 agent 进行了一些“观察“，让我们看看 agent 对它们重要程度的评估：

观察	重要程度（0为不重要，10为非常重要）
Tommie remembers his dog, Bruno, from when he was a kid.	8
Tommie feels tired from driving so far.	1
Tommie sees the new home.	8
The new neighbors have a cat.	2
The road is noisy at night.	2
Tommie is hungry.	1
Tommie tries to get some rest.	1

看得出来，对于重要的记忆，agent 给出了高分，而对于感到疲惫等经常发生的事儿，agent 给出了低分。

下面我们看看 agent 如何应对外来事件。我们让 agent 看到邻居养的猫，下面是 agent 的反应：

Tommie might be curious about the cat and ask where it came from, or he might simply acknowledge its presence without saying anything.

agent 的反应也会作为一次“观察”，进行重要性评估，并存入记忆中。

对话

对话也是 generative agent 的核心能力之一，也是比较复杂的部分。agent 会推理对话者与自身的关系，提取对方话语中的重要部分，找出相关记忆，最后作答。

我们让 agent 的父亲问了下他找工作的情况：

Dad: Have you got a new job?
Agent: No, I haven't found one yet.

类似的，对话也会作为一次“观察”，被 agent 记录。

写在最后

总的来说这个 side project 对我个人还是有很大的帮助，一方面了解了 langchain 的很多东西，另一方面高强度搞了两天 prompt engineering，可以说是非常有价值的经历了。

这个项目目前还有一些没有解决的问题，比如：

基于 LlamaCpp 跑 vicuna-13b 总是难以跑出稳定的结果，比 vicuna 自己的网站上的效果要差一些。因此 generative agent 的 reasoning 能力不太行，有时候会说胡话。
推理速度慢，我是在自己的 MacBook Pro 16 M1 上跑的，效果不太理想。

针对这些问题，欢迎感兴趣的朋友一起交流，也非常欢迎大家在 github 上给提 issue 或者 pull requet :)
