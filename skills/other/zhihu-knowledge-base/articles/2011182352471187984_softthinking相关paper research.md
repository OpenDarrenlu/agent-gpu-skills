# softthinking相关paper research

**作者**: 东川路第一伊蕾娜​Open and open again

**原文链接**: https://zhuanlan.zhihu.com/p/2011182352471187984

---

​
目录
收起
1.Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space
1.论文信息：
2.核心方法推导： 我们从计算概率出发：
2.LLMS ARE SINGLE-THREADED REASONERS: DEMYSTIFYING THE WORKING MECHANISM OF SOFT THINKIN
1.论文信息：
2.对soft-thinking范式的改进：
3.实验结果
3.SofT-GRPO: Surpassing Discrete-Token LLM Reinforcement Learning via Gumbel-Reparameterized Soft-Thinking Policy Optimization
1.论文信息：
2.方法介绍： Soft-GRPO是 off policy的方法： 首先，对 Q 对问题都采样一组 G 个sof-thinking CoTs ，从而得到：
3.实验：
4.Multiplex Thinking: Reasoning via Token-wise Branch-and-Merge
1.论文信息：
2.方法介绍：
总结：
1.Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space
1.论文信息：

论文地址：https://arxiv.org/pdf/2505.15778

代码地址：https://github.com/eric-ai-lab/Soft-Thinking

2.核心方法推导： 我们从计算概率出发：

𝑝
𝑡
=
𝜋
𝜃
(
⋅
|
[
𝑄
,
(
𝑠
1
,
𝑠
2
,
.
.
.
,
𝑠
𝑡
−
1
)
]
)

其中，Q 表示问题，
𝑠
𝑡
表示第 t 个位置的 token，
𝜋
𝜃
表示模型策略，
𝑝
𝑡
表示输入问题和前 t - 1个token时，第 t 个位置在全词表上的概率分布。

随后，softthinking引入了soft-token，定义为：

𝑠
𝑡
=
∑
𝑖
=
1
𝑉
𝑝
𝑖
𝑒
𝑖

这里 V 代表词表，
𝑒
𝑖
代表词表第 i 个位置的 embedding，所以第 t 个位置的soft-token被定义为：根据全词表的概率分布对每个词的embedding加权求和。

停止条件：因为soft-token 没有对应的词表单词，所以需要保留原本的多项式采样一次，当采样到的token是的时候停止。

cold stop：soft-thinking还存在问题，LLM遇到OOD的 soft tokens，可能会出现无限重复的复读问题。所以作者设置了连续熵低于阙值就早停的方案。

2.LLMS ARE SINGLE-THREADED REASONERS: DEMYSTIFYING THE WORKING MECHANISM OF SOFT THINKIN
1.论文信息：

论文地址：2508.03440v4

2.对soft-thinking范式的改进：

2.1 论文复现：

论文首先从对 soft-thinking 的复现出发：

模型：Deepseek-R1-Distill-Qwen-32B, QwQ-32B, Skywork-OD1-32B 三个中尺寸模型。

Benchmark：包括数学（AIME24,AIME25,MATH500,AMC23,GPQA）和代码（HumanEval,MBPP,LiveCodeBench）两个领域的benchmark。

结论：soft-token效果在不同模型上和大多数数据集 没有sampling好。这就引出了一个问题：为什么模型没有像 soft-token 提出那样，模型能同时推理很多路径，获得更好的性能？

2.2 对soft-token的分析（“Greedy Pitfall”现象）： 2.2.1 对最后一层输出的分析： 作者首先用 JS 散度分析了 soft-token P_{st}， top-1大的tokenP_1, 和第 2 大概率大tokenP_2作为输入时的输出。这里 JS 散度是 KD 散度的改进形式，其计算方式为： 给定两个分布 P(x),Q(x), 先计算其平均分布：

M(x)= \frac{1}{2}(P(x)+Q(x))
进而计算 JS 散度：

JS(P||Q)=\frac{1}{2}KL(P||M)+\frac{1}{2}KL(Q||M)

如果采用ln 作为对数的底数，JS取值范围为[0,ln2] （[0,0.693]），越靠近0表示P和Q分布越相似，越靠近1表示P和Q分布越不相似。

模型：QwQ-32B

数据：aime-24，aime-25 大概分析了 $10^6$个 token ，从而保证统计规律。

结论：soft-token 和 top-1 token 作为输入的输出的 JS 的分布集中在 0 附近，说明soft-token 作为输入的效果接近等价于 top1 token。而 soft-token 和第 2 大概率大token做为输入的输出 集中在0.6以上，非常接近最大值ln2，说明 soft-token 作为作为输入的效果和第 2 大概率大token关系不大。

soft-token 并没有让模型张出“三头六臂”并行处理很多路径，模型会按照接近greedy的方式处理并输出。

问题：为什么处理输入会和greedy相似呢？在层的传递中发生了什么呢？

2.2.2 对中间层输出的分析：

为了构造一个平衡的 Soft Token，作者手动将：第一个 token 的概率设为 0.6 token1，第二个 token 的概率设为 0.4 token2。

隐含假设：理想情况下soft-token 作为输入的输出top2K 应该同时包括token1 和 token2 各自作为输入的输出的topK token。

然后把soft-token ，token1，token2分布作为输入，然后采用 soft-token 作为输入输出的 top-10 token 和 token1，token2 作为输入的输出的top-5 token。恒量每一层输出的交集大小。

每一层输出这里采用了Logit Lens技术：

理想的transformer是第一层->第二层->第三层->....->最后一层。正常情况最后一层会用hidden state $h_N$，乘以输出矩阵$W$得到 logits，再经过softmax处理得到输出的概率。Logit Lens技术则是用每一层的hidden state 都分别乘以输出矩阵$W$得到 logits，再经过softmax处理得到每一层的概率。

结论：前几层模型确实同时考虑了两个路径，但经过后面层的处理，逐渐只考虑了1st token作为输入。

LLM是一个“剪枝器”，它会逐渐把推理路径精剪，输出竞争力更大的路径。

2.3 对soft-token的改进：

作者为了打破上述“Greedy Pitfall”现象，尝试在解码方法中加入随机性，需要满足：

有效性（Validness）：它仍然必须是词表 V 上的一个合法概率分布。
随机性（Randomness）：它应当是无偏的（unbiased），并且能够反映原始 $s_t$​ 中的预测信息。
Soft 性（Softness）: 它必须保持为soft 分布形式，而不能退化为 one-hot 向量（即不能变成只对应单个 token 的硬选择）。

本文尝试了两种trick， DIRICHLET SAMPLING 和 GUMBEL-SOFTMAX TRICK：

2.3.1 DIRICHLET SAMPLING：

在概率单纯形 $\Delta^{n-1}$ 上，最常见的分布或许就是狄利克雷分布。

我们先理解一下狄利克雷分布：

概率单纯形 ($\Delta^{n-1}$)：你可以把它想象成一个所有维度相加等于 1 的空间，这里就相当于模型预测的概率 p 。

狄利克雷分布：我们用一个例子解释：[0.7,0.2,0.1]三个组成一个概率单纯形。狄利克雷分布是用来生成这个概率的概率，它不会问下一次是A还是B，而是会问加入噪声后 A,B,C 的概率是多少？而我们观察相同噪声下，不同$\alpha$的输入是多少：

α=[0.7,0.2,0.1] -> [0.99,0.005,0.005] (施加扰动后很极端)

α=[70,20,10] -> [0.68,0.22,0.10]（施加扰动后变化范围合理）

所以本文没有使用$Dir(p)$，而是加了一个缩放参数 $\gamma$，从$Dir(\gamma p)$采样。

2.3.2 GUMBEL-SOFTMAX TRICK 模型输出一个概率分布\pi, 有 i = 1,2,3…,n 类。算法独立采样 g_i 代表第 i 类施加的noise。其中：

log \pi_i =log (\frac{e^{z_i}}{\sum_je^{z_j}})=z_i-ln(\sum_je^{z_j})

而对于每个\pi_i,ln(\sum_je^{z_j})是一个公共项，无论是给 \ln \pi_i 加噪声，还是直接给神经网络输出的 z_i 加噪声，最后的 argmax 结果是完全一样的。 所以得到了:

y_i=\frac{exp((log \pi_i+g_i)/τ)}{\sum_{k=1}^nexp((log \pi_k+g_k)/τ}

其中 τ 代表了温度，计算得到了加入噪声后的概率。

作用：Gumbel-Softmax 分布在离散的一热编码（one-hot-encoded）类别分布与连续的类别密度之间起到了插值（平滑过渡）的作用。原本分布可能是[0.5,0.4,0.1]，加上噪声后可能变成[0.4,0.45,0.15] 可以注意到了second token。

3.实验结果

实施细节：Dirichlet Sampling：设置$\gamma$范围为[1,10],一次增长1.0，4.0最好；

Gumbel-Softmax trick：温度选择 τ 范围为[0.3,0.9] ,一次增长0.1，0.5最好。

结果：Gumbel 方法在所有三个模型主流的数学和代码榜单上都超过了sampling方法和 Dirichlet sampling 。

3.SofT-GRPO: Surpassing Discrete-Token LLM Reinforcement Learning via Gumbel-Reparameterized Soft-Thinking Policy Optimization
1.论文信息：

论文地址：arXiv:2511.06411

2.方法介绍： Soft-GRPO是 off policy的方法： 首先，对 Q 对问题都采样一组 G 个sof-thinking CoTs ，从而得到：

(p_1,...p_{V}) = \pi_{old}(\cdot|[Q,(s_1,...,s_{t-1})])

其中，p_i 代表第 i 个位置的概率，而s_t代表模型生成第 t 个位置的token。

g_i'=log p_i + ϵ_i

这里ϵ_i \sim G(0,1)表示加入的噪声，而g_i'表示加入噪声后的 logits。

y_i'=\frac{exp(g_i'/τ_g)}{\sum_{i=1}^{V}exp(g_i'/τ_g)}

这里y_i'表示第 i 个位置加入噪声后的logits归一化后的概率，而 τ 代表温度。 因为RLVR需要利用概率计算梯度，而soft-thinking怎么算概率呢？

本文采用了一个有意思的观点：不直接算 Soft Token 的概率，而是算生成它的那组“噪声”的概率，把采样的 Gumbel 噪声 \epsilon看作是 RL 的 Action。通过取对数和求和计算：“刚才生成的这组特定的 Gumbel 噪声 \epsilon 出现的总概率密度是多少？”

p(g'|[Q,(s_1,...,s_{t-1}),\theta_{old}])=p(\epsilon) = \prod_{i=1}^{|T|} e^{-\epsilon_i - e^{-\epsilon_i}} 这里通过连乘，计算了生成第 t 个位置 噪声的概率，也是第 t 个位置 soft-token的概率。 代入g_i' -log p_i= ϵ_i,我们可以得到： p(g' | [Q, (s_1, \dots, s_{t-1})], \theta) = \exp \left( \sum_{i=1}^{|T|} -(g'_i - \log p_i) - \exp(-(g'_i - \log p_i)) \right)

其核心贡献是采用计算噪声概率的巧妙思路计算出了soft-token的概率。

3.实验：

参数&细节设置：top-p 为0.95，top-k为5，Gumbel-Softmax 的 temperature为0.1，LLM生成答案的 temperature为1。采用SGLang推理，verl-0.4.x训练。

数据集：训练采用DeepScaler，推理采用AIME2024,AIME202, AMC23,MATH-500,GSM8K 5个数学数据集和 GPQA Diamond，HumanEval，MBPP 3个代码数据集个

baseline：正常的cot模型（discrete-token CoT），。soft-thinkin模式（Gumbel-Softmax），g正常的cot模+GRPO，型soft-thinkin+GRPO，soft-thinkin+soft GRPO

结果： 1.Pass@1小幅度提升：优于传统 GRPO（离散 token RL），平均 Pass@1小幅度提升，并且传统 GRPO + soft-thinking提升 不稳定（有时甚至下降），而SofT-GRPO稳定提升，说明SofT-GRPO 是专门为 soft-thinking 设计的 RL。 2.Pass@K 提升也没那么大：Pass@16和Pass@32都提升/基本持平，说明soft-GRPO可以尽可能强化多个 token。 实验结果来看还是不够有说服力。

4.Multiplex Thinking: Reasoning via Token-wise Branch-and-Merge
1.论文信息：

论文地址:Multiplex Thinking: Reasoning via Token-wise Branch-and-Merge

2.方法介绍：

假设给定从数据集 D 中采样的问题 q , LLM $\pi_{\theta}$采样了一条长为 L 的轨迹 $c = (c_1,c_2,...,c_L)$和答案 y 。而 Multiplex Thinking 会采样 K 条路径，比如在第 i 个位置会采样得到$k_{i,1}, k_{i,2},...,k_{i,K}$然后聚合得。

s_i=\frac{1}{K}\sum_{j=1}^Kz_{i,j}

其中，z_{i,j}是 k_{i,j}对应的 one-hot 向量。所以 s_i 表示采样得到的选择每个token的概率。 当 K = 1 ，对应我们平时采用的cot。而当 K → ∞ 完整代表了 LM head的分布。 假设embedding矩阵为E \in R^{V \times d}, 然后 w_i 模型输出的概率（均匀平均等价上面 p_i），然后我们可以计算得到这里的soft-token:

（a）均匀平均：如果w_i为 1，那么multiplex token 是：采样 token embedding 的平均值（之前的soft token）

（b）LM-head 重加权：

w_i[v] = K \cdot \frac{\mathbb{1}[s_i[v]>0] \cdot \pi_\theta(v|e(q), c_{<i})}{\sum_{u=1}^V \mathbb{1}[s_i[u]>0] \cdot \pi_\theta(u|e(q), c_{<i})}

即仅对采样集合中出现的令牌进行重加权，并根据模型的语言模型头（LM-head）概率对它们进行缩放。优势：减少随机性偏差：如果运气好抽到比较多，可以把它“拉回”到模型真正的置信度水平。 选择K次采样 等价于：当分布非常尖锐且熵较低时，采样结果会坍缩为同一个令牌，退回到标准的离散行为；相反，高熵分布则会产生多样化的混合结果，从而在单个连续向量中编码探索过程。由于独立性假设，生成特定多重令牌的概率是可以因子化分解的。因此，整个推理轨迹 c 的对数概率等于其所有组成部分的离散采样样本的对数概率之和：

\log \pi(c|e(q)) = \sum_{i=1}^{|c|} \sum_{j=1}^{K} \log \pi_\theta(k_{i,j} | e(q), c_{<i})

3.实验：

部署： 训练细节：通过GRPO训练，采用 batch size = 128，训练 300个steps，最大输出为4096，一次生成8条，温度为1，top-p为1；

推理细节：top-p 设置为0.95，64次平均报告了pass@1，也在top-p报告了pass@k ∈ {1, 2, 4, … , 1024}

模型：DeepSeek-R1-Distill-Qwen-1.5B 和 DeepSeek-R1-Distill-Qwen-7B 数据集：DeepScaleR-Preview-Dataset训练，采用: AIME 2024 , AIME 2025 , AMC 2023, MATH-500, Minerva Math, and OlympiadBench 评估。 baselines: Discrete CoT , Stochastic Soft Thinking,Discrete RL 它没有和soft-GRPO比较，也没有测试code

3.1模型边界评估：评估pass@k

结果：在pass@k 上，multiplex thinking效果提升很大;

3.2模型pass@1 评估：

结果：1.5B的模型在5个数据集上有提升，而7B模型在所有数据集都有提升;

3.3 采样次数k对模型的影响：

根据不同 k 下，计算pass@1 。




Multiplex Thinking-I 为无需训练版本，对照从而判断题升来自Multiplex Thinking 还是 RL
结果：Multiplex Thinking-I 性能和 Stochastic Soft Thinking 持平，RL后效果更佳。


总结：


最早的 soft thinking 被证明效果相似于 greedy sampling，随后的 Stochastic Soft Thinking 和 Multiplex Thinking-I 改进版本明显有所提升。
而 Stochastic Soft Thinking 分支发展了 soft-GRPO ，用计算噪声概率的方式 算出了soft-token的概率，从而解决了该方法RL的最大挑战；
Multiplex Thinking-I 分支则更佳适合RL一点，其RL效果也很强，尤其是pass@K 上的提升。

