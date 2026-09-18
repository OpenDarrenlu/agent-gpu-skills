# Reasoning LLM（二）：过程监督与结果监督

**作者**: 紫气东来​上海交通大学 工学硕士

**原文链接**: https://zhuanlan.zhihu.com/p/17569409591

---

​
目录
收起
一、监督的辩论与奖励的设计
1.1 概念说明
1.2 奖励模型的作用
二、奖励模型的训练
2.1 ORM 的训练
2.2 PRM 的训练
三、过程奖励的再思考
参考资料

在上一期中介绍过，MCTS 在 simulation 过程中的关键步骤就是评估，在这个过程中既设计过程评估，又涉及结果评估。

在学生时代，我们常常会听到一个经典的辩论议题，即“结果更重要，还是过程更重要”，这是一个深刻的人生哲学问题，对 LLM 亦如是。本文将围绕 LLM 的过程监督(process supervision, PS)和结果监督(outcome supervision, OS)问题展开以下讨论：

为什么要引入过程监督，其逻辑和依据是什么 ？
过程监督与结果监督如何提高 LLM 的表现 ？
如何实现过程监督与结果监督，有何方法和路径 ？
在一个具体场景中，如何组合使用以提高效果 ？
一、监督的辩论与奖励的设计
1.1 概念说明

对 LLM 的输出进行监督既是评估的手段，又是改进优化的依据。对于 LLM 当前的监督方式，主要有两种：

结果监督

通常使用结果奖励模型(Outcome Reward Model, ORM)对 LLM 的输出进行评估，这是最直接最基础的方法，其通常对比较直接明确的问题比较有效，比较类似于 system 1。但是对于一些需要复杂推理的问题，则会存在以下弊端：

模型中间的思考过程是未知的，解释性和可靠性不高
无法真正实现对齐人类，即没有“思考”过程

过程监督

通常使用过程奖励模型(Process Reward Model, PRM) 对于具有较长或多步推理链的推理任务的每个步骤分配奖励来监督模型输出。它会对推理过程中的每一个环节进行评估，这就使得它能够更精确地指出解决方案中可能出现错误的具体位置，比较类似于 system 2。还是以数学问题解答为例，PRM 会检查每一步的运算是否正确、逻辑是否连贯等。对于 LLM 的过程监督还可以分为两种粒度：

token 粒度：即计算token分数的均值，或者 last token 的分数，经典的 RLHF 即采用这种方法，最后计算 GAE
step 粒度：即按照步骤计算分数，这种方式更加灵活，也更接近人类的思考方式

对于不同的 RM，有一个标准的榜单 reward-bench，截止笔者的写稿时间，榜单上 TOP10 的模型为：

1.2 奖励模型的作用

无论是结果监督还是过程监督，其根本目标都是使 LLM 更好地对其人类，以提高其在实际场景中的表现。那么如果使用奖励模型来实现上述目标，其具体路径和方法是什么呢？主要有以下3种形式：

1.2.1 训练数据增强

即通过奖励模型评分结合拒绝抽样来采用数据选择过程， 其本质是通过这种方式产生更高质量的数据。这种方法当前已被广泛用在 SFT 数据构造上，而早期采用这种方法的典型案例即 Llama 2 的对齐过程，其 V1~V3 版本的模型都是 reject sampling 的方式得到的。

1.2.2 强化学习训练

这种方式是最广泛最经典的奖励模型的用法，即在强化学习训练并提供有效的奖励信号，进一步提升模型性能。如果说 reject sampling 是一种静态的监督增强（即离线产生数据之后不会随模型训练而变化），那么在 RLHF 中，可以认为是一种动态的监督的方式。最经典的案例就是 RLHF，其中包含 4 个模型：Policy Model, Value Model, Reward Model, Reference Model，相关细节在笔者之前的文章中已多次讨论，在此不予赘述。

由于 RLHF 训练成本的高昂，后续的研究对此做了一些精简，比较经典的就是 DeepSeek 的 GRPO，该方法抛弃了 Value Model 而仅保留 Reward Model 作为监督的依据，同样可以采用两种监督方式，概括而言：

结果监督：对于每个输入产生多个输出，使用 Reward Model 打分并标准化，然后计算每个输出的分数与均值的差异，再进行策略优化；
过程监督：与上述方法不同的是，过程监督对结果的每一步进行打分，最后优势函数的计算即所有步骤优势的和，再进行策略优化。

通过这种方式，充分利用 Reward Model 对 LLM 的训练过程进行监督。

1.2.3 推理性能增强

除了作用于训练过程，奖励模型还可以作用于推理过程，通过结合 sampling，Best-of-N，MCTS 等策略，选择奖励模型判断得分最高的回答，通过花费多步和更多的推理时间产生更好的结果。

那么我们不妨在此列举部分 RM 的相关工作及其总结：

work	RM 作用	paper	code
DeepSeek-R1
(2025.1)	强化学习训练
训练数据增强	https://github.com/deepseek-ai/DeepSeek-R1	
Kimi-K1.5
(2025.1)	强化学习训练	https://github.com/MoonshotAI/Kimi-k1.5	
PRIME
(2025.1)	强化学习训练	https://huggingface.co/blog/ganqu/prime	https://github.com/PRIME-RL/PRIME
rStar-Math
(2025.1)	推理增强(MCTS)	https://arxiv.org/pdf/2501.04519	
Meta-CoT
(2025.1)	优化训练
推理增强	https://arxiv.org/pdf/2501.04682	
Macro-o1
(2024.11)	推理增强(MCTS)	https://arxiv.org/pdf/2411.14405	https://github.com/AIDC-AI/Marco-o1
OpenR
(2024.10)	数据生成
优化训练
推理增强	https://arxiv.org/pdf/2410.09671	https://github.com/openreasoner/openr




二、奖励模型的训练
2.1 ORM 的训练

ORM 的训练是最常见也最典型的，但是其数据构造和训练过程通常是经验性，本篇力求在此基础上更多讨论其基本形式、理论依据及注意事项。

2.1.1 基本形式

奖励模型建模的基本形式有2种：Bradley-Terry model 与 Preference Model

Bradley-Terry reward model

通常的 ORM 的训练数据是以 pairwise 的形式出现的，而构建 pairwise 数据则是通过“竞技场”式的排序过程来获得的。那么这种构造方式的合理性如何，其理论依据是什么，本节将通过 Bradley-Terry 模型来进行一些探究。

Bradley-Terry 模型是对象之间成对比较结果的概率模型。给定从某个总体中抽取的一对项目 i 和 j ，估计成对比较 i>j 结果为真的概率，即 P(i>j)=\frac{u(i)}{u(i)+u(j)}=\frac{\exp (r(i))}{\exp (r(i))+\exp (r(j))}=\operatorname{softmax}(r(i), r(j)) \\ 其中 u(i) 表示分配给个体 i 的正实值分数， r(i) 表示该值的对数值。

在上述过程中， Bradley-Terry 模型把相对的排序变成了绝对的分数，这正是reward model 的目的。然而在实际中，LLM 竞技场可以通过大量比较来为每一个模型赋予一个合理的分数，但是在 reward model 构造时却很难为每一个 prompt 进行充分的比较而赋予 response 合理的分数，即这种情况是稀疏的。

ORM 数据集的形式通常为\mathcal{D}=\{x_i, y^w_i,y^l_i\}_{i=1}^N\\其中x_i代表prompt，y^w_i, y^l_i分别代表被人类偏好和不被偏好的response。这里在使用 Bradley-Terry Model 时实际上包含了 2 条假设：

成对偏好可以用分数奖励代替；
基于分数奖励训练的奖励模型可以从收集的数据推广到策略采样的分布外数据

基于此，奖励函数的整体优化目标可以表述为：

\begin{aligned} \mathcal{L}\left(r, \mathcal{D}\right) & =-\mathbb{E}_{x, y_w, y_l \in \mathcal{D}}\left[\log \left(P\left(y_w>y_l \mid x\right)\right)\right] \\ & =-\mathbb{E}_{x, y_w, y_l \in \mathcal{D}}\left[\log \left(\frac{e^{r\left(x, y_w\right)}}{e^{r\left(x, y_w\right)}+e^{r\left(x, y_l\right)}}\right)\right] \\ & =-\mathbb{E}_{x, y_w, y_l \in \mathcal{D}}\left[\log \left(\frac{1}{1+e^{-\left(r\left(x, y_w\right)-r\left(x, y_l\right)\right)}}\right)\right] \\ & =-\mathbb{E}_{x, y_w, y_l \in \mathcal{D}}\left[\log \left(\sigma\left(r\left(x, y_w\right)-r\left(x, y_l\right)\right)\right)\right] . \end{aligned}\\
