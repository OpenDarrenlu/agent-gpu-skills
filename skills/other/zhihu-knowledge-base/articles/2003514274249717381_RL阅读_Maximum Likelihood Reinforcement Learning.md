# RL阅读:Maximum Likelihood Reinforcement Learning

**作者**: 东川路第一伊蕾娜​Open and open again

**原文链接**: https://zhuanlan.zhihu.com/p/2003514274249717381

---

这篇论文最近很火，希望能还原一下读的心路历程。

abstract：

RL（比如GRPO）会隐式地诱导出一个关于正确轨迹的似然函数。然而，作者观察到，强化学习并未最大化这一似然，而仅仅优化了其低阶近似。因此他们提出了最大似然强化学习。

Introduction：最大似然估计Maximum likelihood (ML) and 强化学习reinforcement learning (RL)是两个成功的优化范式。如果input x 输出正确答案的概率为 
𝑝
𝜃
(
𝑥
)
 ,那么 
∇
𝜃
𝐽
𝑅
𝐿
=
𝐸
𝑥
[
∇
𝜃
𝑝
𝜃
(
𝑥
)
]
 ,而 
∇
𝜃
𝐽
𝑀
𝐿
=
𝐸
𝑥
[
∇
𝜃
𝑙
𝑜
𝑔
𝑝
𝜃
(
𝑥
)
]
=
𝐸
𝑥
[
1
𝑝
𝜃
(
𝑥
)
∇
𝜃
𝑝
𝜃
(
𝑥
)
]
 。

最大似然估计：在不可微问题中，这一目标难以被直接优化。

然后我们看方法，我很好奇它是怎么实现的🤔

接下来开始推导公式：

，
𝑋
，
𝑌
 分别是输入，输出空间。 
𝑥
∼
𝜌
 是每个任务的分布。对于每个输入 
𝑥
 ，我们假设 
𝑦
∗
(
𝑥
)
 是评判结果。那么模型参数为 
𝜃
 ，预测正确概率为 
𝑝
𝜃
(
𝑦
|
𝑥
)
 ，其中 
𝑝
𝜃
(
|
𝑥
)
∈
𝑌
 是模型（输入为x的）条件输出。因为y一般不是直接输出的。比如RLVR 我们会先生成一个轨迹，所以引入 
𝑧
∈
𝑍
 。最终输出 
𝑦
=
𝑓
(
𝑧
)
 (比如f可以是\boxed{}从里面正则提取出来)

pass比例： 
𝑝
𝜃
𝑝
𝑎
𝑠
𝑠
(
𝑥
)
=
𝑝
𝜃
(
𝑦
∗
|
𝑥
)
=
𝐸
𝑦
∼
𝑝
𝜃
(
|
𝑥
)
[
𝐼
{
𝑦
=
𝑦
∗
(
𝑥
)
}
]
 (单个)

对于k个： 
𝑝
𝑎
𝑠
𝑠
@
𝑘
=
𝑃
(
∃
𝑖
∈
[
𝑘
]
,
𝑠
.
𝑡
.
𝑦
𝑖
=
𝑦
∗
(
𝑥
)
)

接下来我们考虑ML 和 RL 用上面符号表达：

ML： 
𝐽
𝑀
𝐿
(
θ
)
:=
𝐸
𝑥
∼
ρ
[
𝑙
𝑜
𝑔
𝑝
θ
(
𝑦
∗
(
𝑥
)
|
𝑥
)
]
𝑤
𝑖
𝑡
ℎ
𝑝
θ
(
𝑦
∗
(
𝑥
)
|
𝑥
)
=
𝐸
𝑧
∼
𝑚
θ
(
·
|
𝑥
)
[
𝐼
{
𝑓
(
𝑧
)
=
𝑦
∗
(
𝑥
)
}
]
 RL:

J_{RL}(θ) := E_{x∼ρ} E_z∼m_θ(·|x) [r(x, z)] = E_{x∼ρ} [p^{pass}_θ (x)]

而如果ML 公式用pass 比例的方式表达，那就是 J_{ML}(x)=logp = log(1-(1-p))=-\sum_{k=1}^{∞}\frac{(1-p)^k}{k}=-\sum_{k=1}^{∞}\frac{fail@k(x)}{k}=-\sum_{k=1}^{∞}\frac{fail@k(x)}{k}=-\sum_{k=1}^{∞}\frac{1-pass@k(x)}{k}

那么 ∇_{\theta}J_{ML}(x)=\sum_{i=1}^{∞}\frac{1}{k}∇_{\theta}pass@k(x) ,而 ∇_{\theta}J_{RL}(x)=∇_{\theta}pass@1(x)

对比可以看出，RL只优化pass@1，而ML 优化pass@k.

那么可以定义 J^{T}_{MAXRL}(x)=-\sum_{k=1}^{T}\frac{(1-p)^k}{k}=\sum_{k=1}^T\frac{1}{k}pass@k(x)

T=1时，是强化学习。T→ ∞ 时，是最大似然。

进而，我们开始推导求解器：

K&gt;=1，否则为0（不是除以0）

\frac{1}{K}\sum_{i=1}^Nr_iS_i 恰好为ML 公式的期望。

然后我们考虑有效性也就是方差：而这里估计器本质还是policy gradient ，方差其实还是挺大的（采样预算有限前提下）。而引入baseline降低方差的前提是baseline b不依赖当前 action。但这里的K 和每个rollout有关（N采样次数无关，而正确次数K和rollout有关系）

而这里引入了一个baseline： \frac{1}{N}\sum_{i=1}^{N}∇_{\theta}logm_{\theta}(z_i|x=\frac{1}{N}\sum_{i=1}^{N}S_i ,它期望为0。

我证明是这样的 ∫m_{\theta}(z|x)dz=1 , ∇_{\theta}∫m_{\theta}(z|x)dz=∇_{\theta}1=0 ,那么 ∫m_{\theta}(z|x)∇_{\theta}logm_{\theta}(z|x)dz=0 ,提取log项从而得到。

此时目标函数为 \frac{1}{K}\sum_{i=1}^Nr_iS_i-\frac{1}{N}\sum_{i=1}^NS_i=\sum_{i=1}^N(\frac{r_i}{K}-\frac{1}{N})S_i （K=0时）

然后我们看算法流程：

这里修改了advantage的计算方式，其他基本不变。

而： ∇_θJ = E_{x∼ρ}[w(p_θ(x)) ∇_θp_θ(x)] ，对于RL，GRPO，MaxRL，ML，这个w分别是：

GRPO这里，思考z-score归一化下面除以std，那个就是这里的权重。

GRPO虽然能够很好近似pass rate比较小的部分，但如果pass rate比较高，就和最大似然拉开差距，因此，当非常容易的输入存在时，GRPO 会对这些样本分配更高的权重。

实验部分，包括了三种场景：（1）近似无限数据的设置（2）一个固定的训练数据集（3）对推理模型进行数学问题求解任务上的训练与评测。

baseline：（1）leave-one-out baseline（RLOO） 的 REINFORCE（2）GRPO

（ML是无限采样，这里无法实现）

我比较关注（3），就重点看了一下。

Qwen3-1.7B-Base and Qwen3-4B-Base训练POLARIS-53K，在AIME 2025, BeyondAIME , MATH-500 ，Minerva 评测。

可以看出MaxRL 在pass@k上 全方位超越了GRPO（GRPO相对base基本没有提升）。

训练动态的pass@1基本差距不大，但MaxRL后期更有优势。

从数学推导到模拟，再到实验验证，这个方法在pass@k上超越了GRPO，但pass@1 提升不大，也是很不错的工作了。
