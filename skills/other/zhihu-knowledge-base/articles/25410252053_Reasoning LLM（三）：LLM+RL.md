# Reasoning LLM（三）：LLM+RL

**作者**: 紫气东来​上海交通大学 工学硕士

**原文链接**: https://zhuanlan.zhihu.com/p/25410252053

---

​
目录
收起
一、从 RL 角度再思考 Post-training
1.1 RLHF 中的 RL 基础概念
1.2 基于策略梯度的更新方法
1.3 重要性采样与梯度裁剪
二、主流算法的思路及其比较
2.1 PPO 与 VC-PPO
2.2 RLOO
2.3 GRPO 与 Dr,GRPO
2.4 REINFORCE++ 与 ReMax
2.5 DAPO 与 VAPO
2.6 GSPO
三、从理论角度再思考 LLM + RL
3.1 DPO：有监督对齐
3.2 统一视角理解从 SFT 到 RL
参考资料

自 RLHF 提出以来，LLM+RL 的研究一直是热点，从最开始的对齐人类偏好到后来的训练复杂推理模型，RL 已俨然成为一个基础步骤和必要环节。笔者之前的文章中也多次讨论过相关内容，但主要侧重于算法原理和实现过程，但其中的很多根本性问题未做过多讨论，因此本篇将从经典算法到最新的算法展开，主要讨论以下问题：

RLHF 的本质是什么，及其与 RL 和有监督学习（SL）的关系
当前的各种优化方法的优化点是什么，其立足角度和收益如何
使用 RL 训练逻辑推理模型，有哪些经验及注意事项
一、从 RL 角度再思考 Post-training

RLHF 是一种使用强化学习方法将 LLM 对齐人类的技术，并逐步延伸到复杂推理模型的训练中，在发展过程中也出现多种算法，包括 PPO，DPO，GRPO，RLOO，REINFORCE 及其衍生版本。在本节中，我们将试图从最基础的 PPO 算法出发，结合 RL 理论，试图从一个统一的视角来理解不同算法的本质、取舍及效果。

首先我们不妨来回顾一下 RLHF (PPO) 的工作流程，这是我们接下来讨论的框架和基础：

首先使用经典的有监督方法训练得到一个 SFT 模型，以作为 RL 训练的初始模型( 
𝜋
𝜃
 )以及计算 KL 散度的参考模型( 
𝜋
𝑟
𝑒
𝑓
 )；
使用人工标注的偏好数据集训练奖励模型 
𝑟
𝜙
(
𝑥
,
𝑦
)
 , 最场景的方式是 Bradley-Terry 模型，即通过有监督学习将人类有偏好的结果赋予更高的分数，更详细内容可回看前文；
进行 PPO 强化学习训练，该过程可分为2步：
采样阶段：
对每个提示 x ，用 \pi_\theta 生成回答 y ，并计算奖励 r(x, y) = r_\phi(x, y) - \beta D_{KL}({\pi_\theta(y|x)},{\pi_{{ref}}(y|x)}) （包含KL惩罚项避免偏离SFT）。
计算优势估计 \hat{A} （常见的是GAE）和回报值，并将以上结果存入 experience buffer。
优化阶段：
策略更新：最大化PPO-Clip目标函数： \mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E} \left[ \min \left( \frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)} \hat{A}, \text{clip} \left( \frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}, 1-\epsilon, 1+\epsilon \right) \hat{A} \right) \right]
同时加入熵正则项鼓励探索。
价值函数更新：最小化价值函数损失 \begin{equation} \mathcal{L}_{\text {critic }}(\phi)=\hat{\mathbb{E}}_t\left[\left\|V_\phi\left(s_t\right)-\widehat{R}_t\right\|^2\right] \end{equation} 拟合回报值, 其中 \widehat{R}_t 表示状态 s_t 的实际回报值，可以估计为 \begin{equation} \hat{R}_t=\sum_{k=0}^{\infty} \gamma^k r_{t+k} \end{equation} 。
1.1 RLHF 中的 RL 基础概念

RL 中有一些必要的概念与构成，如环境、状态、动作、奖励等，下面我们将 RLHF 中的各环节对应起来。

状态(State): 初始状态即由用户输入 prompt x ，后续随着生成过程逐渐增加，如上图
动作(Action)：
token level: 该方式基于一种认识：每一个 token 是一个 action，动作空间由模型词汇表中的所有token组成，即大小为 vocab_size。这种方式事实上与自然语言的特点并不相容，即人类通常不会逐字去评估一段话，而是看完一整句或整段后进行评估。
response level：即将一个完整的 response 作为 action，通过多次生成来互相比较并计算优势值。
奖励(Reward): 策略 \pi_\theta 根据当前上下文生成 token，逐步构建的回答 y=(y_1,y_2,...,y_T) , 在评估时，通常以最后一个 token EOS 的分数代替（这是最常见的做法，上图显示就是这种）。
序列决策：经典 RL 处理的都是序列决策问题，即一次探索由多个交互（动作）构成。而在 RLHF 中，如果我们将 response 看做 action，那么 RLHF 实际上就是一种单步的、即时奖励的 RL 。
环境(Environment): 在经典 online RL 中，环境是客观的、能够提供即时反馈，从而帮助模型进行更新；而在 RLHF 中环境实际上由策略模型和奖励模型本身构成，而奖励模型还是预先训练好的、静态的，从这个角度来说，RLHF 更类似于 offline RL 或监督学习，而不是 online RL。
策略(Policy): 在 RLHF 中，策略即 LLM 本身
概念	经典 RL	RLHF
状态空间	物理状态（如像素、坐标）	文本上下文（token序列）
动作空间	离散/连续控制（如按键）	生成下一个 token（词汇表分布）
奖励	环境反馈（如得分）	奖励模型或规则
可观测性	通常完全观测	完全观测（文本历史已知）
1.2 基于策略梯度的更新方法

RLHF 中的算法都是基于策略梯度的 RL 算法，即通过深度学习模型为策略建模，输入某个状态，然后输出动作的概率分布，目标是寻找一个最优策略并最大化该策略在环境中的期望回报。

依据训练数据产生的方式，主流的 RLHF 方法可以大致分为两大类：

On-Policy: 在训练过程中，模型主动生成自己的数据样本，进而根据模型当前状态持续探索和更新，这种方式的瓶颈主要在数据生成过程（耗费计算资源和时间），典型方法的如 PPO（此处姑且这么认为，后文讨论）
Off-Policy: 训练依赖于预先收集的数据（或由另一个策略生成的数据），无需实时生成，这种方式样本效率高，典型方法如 DPO

接下来以 PPO 算法为例来回顾 RLHF 中策略更新的核心过程，一个典型的 PPO 算法包括以下组成：

Actor: 即策略模型，用以生成输出，并根据反馈进行更新
Critic: 扮演教练的角色，为每个生成的输出提供即时反馈，并随着模型能力的提升与 Actor 模型同步更新
Reward Model: 作为裁判，分配最终得分或偏好评估，预先由偏好数据训练得到，在PPO训练过程中保持不变
Reference Model: 作为策略模型的参考，防止演员模型偏离原始预训练分布太远

接下来简要回顾策略优化的过程，强化学习中的目标是优化策略以最大化期望回报 \begin{equation} \pi^*=\arg \max _\pi J(\pi) \end{equation} \\ 策略的回报定义在所有可能的轨迹上 \begin{equation} J\left(\pi_\theta\right)=\int_\tau P(\tau \mid \pi) R(\tau)=\mathbb{E}_{\tau \sim \pi}[R(\tau)] \end{equation}\\ 轨迹 \begin{equation} \tau=\left(s_0, a_0, s_1, a_1, \ldots\right) \end{equation} 的概率可以表示为 \begin{equation} P(\tau \mid \pi)=\rho_0\left(s_0\right) \prod_{t=0}^{T-1} P\left(s_{t+1} \mid s_t, a_t\right) \pi\left(a_t \mid s_t\right) \end{equation}\\ 其中轨迹总回报被定义为 \begin{equation} R(\tau)=\sum_{t=0}^{\infty} \gamma^t r_t \end{equation} 。

在深度学习中，我们通常通过最小化损失函数并使用随机梯度下降来更新参数。然而，由于我们的目标是最大化回报，我们使用随机梯度上升来更新策略 \begin{equation} \theta_{k+1}=\theta_k+\alpha \nabla_\theta J\left(\pi_\theta\right) \end{equation}\\ 关于 \nabla_\theta J\left(\pi_\theta\right) 的推导可以参考笔者之前的文章 深度强化学习（五）：策略梯度的方法 - 知乎 在此直接给出结果 \begin{aligned} \nabla_\theta J\left(\pi_\theta\right) &=\mathrm{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta\left(a_t \mid s_t\right) \cdot \Psi_t\right] \\ &\approx \frac{1}{\mathcal{D}} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^T \nabla_\theta \log \pi_\theta\left(a_t \mid s_t\right) \Psi_t \end{aligned} \\ 其中 \Psi_t 可以有以下多种形式：

（1） \sum_{t=0}^{T} \gamma^t r_t ：轨迹的总回报

（2） \sum_{t=t^\prime}^{T} \gamma^{t-t^\prime} r_t : 动作 a_{t^\prime} 之后的回报，无偏但方差较大

（3）\sum_{t=t^\prime}^{T} \gamma^{t-t^\prime} r_t - b(s_{t^\prime}) : 增加偏置项的改进版本，以减小方差

（4）Q^{\pi_\theta}\left(s_t, a_t\right) ：动作价值函数

（5）A^{\pi_\theta}\left(s_t, a_t\right) : 优势函数，把状态价值函数 V 作为基线，从 Q 函数减去这个 V 函数则得到了 A 函数，RLHF 的主要采用形式，且PPO, GRPO, RLOO等不同算法的差异也主要体现在这里

（6）r_t+\gamma V^{\pi_\theta}\left(s_{t+1}\right)-V^{\pi_\theta}\left(s_t\right)：时序差分残差，利用 Q=r+\gamma V 公式即可得到

以上几种方式更加深入的讨论和区别，及其在不同算法上的使用，将在第二节进行深入讨论。

1.3 重要性采样与梯度裁剪

在策略梯度中，除了 \Psi_t 外，另一个关键的部分即 \frac{1}{\mathcal{D}} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^T \nabla_\theta \log \pi_\theta\left(a_t \mid s_t\right) 采样，如果按照这种方式进行更新，则在每次更新参数后都需要重新采样轨迹，而不能进行多步迭代后，因此样本效率比较低，那么该如何提高样本效率呢？ 答案是重要性采样 (Importance Sampling) 。

重要性采样将已知分布的样本转换为未知分布的样本，如下所示 p(x) 为未知分布，而 q(x) 为已知分布： E_{x \sim p(x)}[f(x)] =\int p(x) f(x) d x =\int \frac{q(x)}{q(x)} p(x) f(x) d x =\int q(x) \frac{p(x)}{q(x)} f(x) d x =E_{x \sim q(x)}\left[\frac{p(x)}{q(x)} f(x)\right]\\ 这样就将 off-policy 的形式转化成了 on-policy 的形式，则策略梯度可改写为 \begin{aligned} \nabla_\theta J\left(\pi_\theta\right) &= \frac{1}{\mathcal{D}} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^T \nabla_\theta \log \pi_\theta\left(a_t \mid s_t\right) \Psi_t \\ &= \frac{1}{\mathcal{D}} \sum_{\tau \in \mathcal{D}} (\sum_{t=0}^T \nabla_\theta\log \frac{ \pi_\theta\left(a_t \mid s_t \right) }{ \pi_{\theta_{old}} \left(a_t \mid s_t \right)}\Psi_t) \end{aligned} \\ 这样就从 \pi_\theta 采样变成了从 \pi_{\theta_{old}} 采样。

这么描述可能还不够直观，我们不妨通过当前的主流框架来具体说明其实现过程，以 OpenRLHF 为例，其中与 batch size 相关的有4个参数：

rollout_batch_size

即在 rollout 阶段利用当前的策略模型 \pi_{\theta_{old}}与环境（如用户对话或文本生成任务）交互，生成多条轨迹（即模型输出的文本序列或对话回合），一次 rollout 产生的样本总数为 rollout_batch_size * n_samples_per_prompt

micro_rollout_batch_size

在样本产生后，还需要经过 critic 得到 value，经过 RM 得到 reward，经过 Reference Model 得到 log probs，而由于资源所限，这些模型每次只能处理部分样本，这就是 micro_rollout_batch_size， 以上结果完成后即存入 experience 中供后续训练。

train_batch_size

一批样本被分割成多个子批次，每个子批次的批次大小为 train_batch_size，用于 PPO 更新（即小步更新）。train_batch_size 是所有GPU 的全局数量。

micro_train_batch_size

类似于梯度累积，一个前向传递的 micro_train_batch_size，以速度换取 GPU 内存。该值表示每个 GPU 的本地 batch size 值。

现不考虑工程问题（micro_rollout_batch_size 与 micro_train_batch_size），以 PPO (n_samples_per_prompt=1) 为例来说明，当rollout_batch_size=1024, train_batch_size=128
时，即在 rollout 阶段，每次从 prompt 数据集中选择 1024 个输入每个产生1条输出，共1024个输出的experience，在训练阶段，每次从 experience 中选择 128 个样本进行策略更新，因此完成更新共需要 8 步。尽管在这个过程里模型更新了 8 次，但样本都是由起始阶段的 \pi_{\theta_{old}}产生的，从这个角度来说，RLHF 也并非真正意义上的 on-policy 算法，而是用 off-policy + 重要性采样实现模拟 on-policy 的效果。

现在我们尝试基于以上过程推导出损失函数。假设当前策略为 \pi_\theta ，参数为 \theta 。我们考虑如何借助当前的找到一个更优的参数 \theta^\prime ，使得 J(\pi_{\theta^\prime}) \geq J(\pi_{\theta}) 。具体来说，由于初始状态 s_0 的分布和策略无关，因此上述策略 \pi_\theta 下的优化目标 J(\pi_{\theta}) 可以写成在新策略 \pi_{\theta^\prime} 的期望形式： \begin{aligned} J(\pi_\theta) & =\mathbb{E}_{s_0}\left[V^{\pi_\theta}\left(s_0\right)\right] \\ & =\mathbb{E}_{\pi_{\theta^{\prime}}}\left[\sum_{t=0}^{\infty} \gamma^t V^{\pi_\theta}\left(s_t\right)-\sum_{t=1}^{\infty} \gamma^t V^{\pi_\theta}\left(s_t\right)\right] \\ & =-\mathbb{E}_{\pi_{\theta^{\prime}}}\left[\sum_{t=0}^{\infty} \gamma^t\left(\gamma V^{\pi_\theta}\left(s_{t+1}\right)-V^{\pi_\theta}\left(s_t\right)\right)\right] \end{aligned}\\则新旧策略的目标函数之间的差距为： \begin{aligned} J\left(\theta^{\prime}\right)-J(\theta) & =\mathbb{E}_{s_0}\left[V^{\pi_{\theta^{\prime}}}\left(s_0\right)\right]-\mathbb{E}_{s_0}\left[V^{\pi_\theta}\left(s_0\right)\right] \\ &= \mathbb{E}_{\pi_{\theta^{\prime}}}\left[\sum_{t=0}^{\infty} \gamma^t r\left(s_t, a_t\right)\right]+\mathbb{E}_{\pi_{\theta^{\prime}}}\left[\sum_{t=0}^{\infty} \gamma^t\left(\gamma V^{\pi_\theta}\left(s_{t+1}\right)-V^{\pi_\theta}\left(s_t\right)\right)\right] \\ &= \mathbb{E}_{\pi_{\theta^{\prime}}}\left[\sum_{t=0}^{\infty} \gamma^t\left[r\left(s_t, a_t\right)+\gamma V^{\pi_\theta}\left(s_{t+1}\right)-V^{\pi_\theta}\left(s_t\right)\right]\right] \\ & =\mathbb{E}_{\pi_{\theta^{\prime}}}\left[\sum_{t=0}^{\infty} \gamma^t A^{\pi_\theta}\left(s_t, a_t\right)\right] \\ & =\sum_{t=0}^{\infty} \gamma^t \mathbb{E}_{s_t \sim P_t^{\pi_{\theta^{\prime}}}} \mathbb{E}_{a_t \sim \pi_{\theta^{\prime}}\left(\cdot \mid s_t\right)}\left[A^{\pi_\theta}\left(s_t, a_t\right)\right] \\ \end{aligned}\\ 只要我们能找到一个新策略，使得 \sum_{t=0}^{\infty} \gamma^t \mathbb{E}_{s_t \sim P_t^{\pi_{\theta^{\prime}}}} \mathbb{E}_{a_t \sim \pi_{\theta^{\prime}}\left(\cdot \mid s_t\right)}\left[A^{\pi_\theta}\left(s_t, a_t\right)\right] \geq 0 ，就能保证策略性能单调递增，即实现策略更新。由于上式是通过 \pi_{\theta^\prime} 进行采样的，同样可以采用重要性采样，这样就可以得到替代的优化目标： L_\theta\left(\theta^{\prime}\right)=J(\theta)+\mathbb{E}_{s \sim P_t^{\pi_{\theta^{\prime}}}} \mathbb{E}_{a \sim \pi_\theta(\cdot \mid s)}\left[\frac{\pi_{\theta^{\prime}}(a \mid s)}{\pi_\theta(a \mid s)} A^{\pi_\theta}(s, a)\right]\\ 在 PPO 中，可以进一步简化优化目标，即 \begin{aligned} & \operatorname{max}_\theta ~~~~ \hat{\mathbb{E}}_t\left[\frac{\pi_\theta\left(a_t \mid s_t\right)}{\left.\pi_{\theta_{\text {old }}}\left(a_t \mid s_t\right) \right.}\hat{A}_t\right] \\ & \text { s.t }~~~~ \hat{\mathbb{E}}_t\left[\operatorname{KL}\left(\pi_{\theta_{\text {old }}}\left(\cdot \mid s_t\right), \pi_\theta\left(\cdot \mid s_t\right)\right)\right] \leq \delta \end{aligned}\\ 根据拉格朗日乘子法将 KL 散度以惩罚项的形式放到目标函数中，使之变成一个无约束的优化问题 \mathcal{L}_{\mathrm{ppo}-\text { penalty }}(\theta)=\hat{\mathbb{E}}_t\left[\frac{\pi_\theta\left(a_t \mid s_t\right)}{\pi_{\theta_{\text {old }}}\left(a_t \mid s_t\right)} \hat{A}_t\right]-\beta \mathrm{KL}\left(\pi_{\theta_{\text {old }}}\left(\cdot \mid s_t\right), \pi_\theta\left(\cdot \mid s_t\right)\right)\\ PPO 的另一种截断的形式更加直接，它在目标函数中进行限制，以保证新的参数和旧的参数的差距不会太大 (这也是最常见的形式)，即： \mathcal{L}_{\mathrm{ppo}-\mathrm{clip}}(\theta)=\hat{\mathbb{E}}_t\left[\min \left(\frac{\pi_\theta\left(a_t \mid s_t\right)}{\pi_{\theta_{\text {old }}}\left(a_t \mid s_t\right)} \hat{A}_t, \operatorname{clip}\left(\frac{\pi_\theta\left(a_t \mid s_t\right)}{\pi_{\theta_{\text {old }}}\left(a_t \mid s_t\right)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_t\right)\right]\\

二、主流算法的思路及其比较

在第一节中，我们简要回顾了 RL 在 Post-traing 中的核心过程，但是在 1.2 小节中留下了 \Psi_t 未做深入讨论，事实上这也是当前 REINFORCE, ReMax, RLOO, PPO, GRPO, Dr.GRPO 的差异所在。在进行比较之前，我们需要确定几个概念，弄清了这几个关键概念的计算过程，也就理解了不同算法的差别。

Return (回报)：从时间步数 t 开始的回报 G_t ​为轨迹的总（折现）奖励 G_t=r_t+\gamma r_{t+1}+\gamma^2 r_{t+2}+\cdots=\sum_{k=0}^{\infty} \gamma^k r_{t+k}\\ 由此还可以得到其递推关系 G_t=\gamma G_{t+1}+r_{t+1} 。

状态值函数定义为在策略 \pi 下的当前状态期望回报 V_\pi(s)=\mathbb{E}_\pi\left[G_t \mid s_t=s\right]

动作值函数被定义为在策略 \pi 下的当前状态下动作的期望回报 Q_\pi(s, a)=\mathbb{E}_\pi\left[G_t \mid s_t=s, a_t=a\right]

以上二者的关系为，在使用策略 \pi 中，状态 s 的价值等于在该状态下基于策略 \pi 采取所有动作的概率与相应的价值相乘再求和的结果: V_\pi(s)=\sum_{a \in A} \pi(a \mid s) Q_\pi(s, a)\\使用策略 \pi 时，状态 s 下采取动作 a 的价值等于即时奖励加上经过衰减后的所有可能的下一个状态的状态转移概率与相应的价值的乘积： Q_\pi(s, a)=r(s, a)+\gamma \sum_{s^{\prime} \in S} P\left(s^{\prime} \mid s, a\right) V_\pi\left(s^{\prime}\right)\\Advantage(优势) 表示相对于基线的差异，通常把状态价值函数 V 作为基线，从 Q 函数减去这个 V 函数则得到了 A 函数，即 A(s_t,a_t)=Q_\pi(s_t,a_t)-V_\pi(s_t)=r_t+\gamma V_\pi\left(s_{t+1}\right)-V_\pi\left(s_t\right)\\

2.1 PPO 与 VC-PPO

优势函数A很难精确计算，对于如何估计优势函数，目前比较常用的一种方法为广义优势估计（Generalized Advantage Estimation，GAE），接下来我们简单介绍一下 GAE 的做法。

首先，用 \delta_t=r_t+\gamma V\left(s_{t+1}\right)-V\left(s_t\right) 表示时序差分误差，其中 V 是一个已经学习的状态价值函数。于是，根据多步时序差分的思想，有： \begin{array}{ll} A_t^{(1)}=\delta_t & =-V\left(s_t\right)+r_t+\gamma V\left(s_{t+1}\right) \\ A_t^{(2)}=\delta_t+\gamma \delta_{t+1} & =-V\left(s_t\right)+r_t+\gamma r_{t+1}+\gamma^2 V\left(s_{t+2}\right) \\ A_t^{(3)}=\delta_t+\gamma \delta_{t+1}+\gamma^2 \delta_{t+2} & =-V\left(s_t\right)+r_t+\gamma r_{t+1}+\gamma^2 r_{t+2}+\gamma^3 V\left(s_{t+3}\right) \\ \quad \vdots & \vdots \\ A_t^{(k)}=\sum_{l=0}^{k-1} \gamma^l \delta_{t+l} & =-V\left(s_t\right)+r_t+\gamma r_{t+1}+\ldots+\gamma^{k-1} r_{t+k-1}+\gamma^k V\left(s_{t+k}\right) \end{array}\\ 然后，GAE 将这些不同步数的优势估计进行指数加权平均： \begin{aligned} A_t^{G A E} & =(1-\lambda)\left(A_t^{(1)}+\lambda A_t^{(2)}+\lambda^2 A_t^{(3)}+\cdots\right) \\ & =(1-\lambda)\left(\delta_t+\lambda\left(\delta_t+\gamma \delta_{t+1}\right)+\lambda^2\left(\delta_t+\gamma \delta_{t+1}+\gamma^2 \delta_{t+2}\right)+\cdots\right) \\ & =(1-\lambda)\left(\delta\left(1+\lambda+\lambda^2+\cdots\right)+\gamma \delta_{t+1}\left(\lambda+\lambda^2+\lambda^3+\cdots\right)+\gamma^2 \delta_{t+2}\left(\lambda^2+\lambda^3+\lambda^4+\cdots\right)+\cdots\right) \\ & =(1-\lambda)\left(\delta_t \frac{1}{1-\lambda}+\gamma \delta_{t+1} \frac{\lambda}{1-\lambda}+\gamma^2 \delta_{t+2} \frac{\lambda^2}{1-\lambda}+\cdots\right) \\ & =\sum_{l=0}^{\infty}(\gamma \lambda)^l \delta_{t+l} \end{aligned}\\ 其中， \lambda \in [0,1] 是在 GAE 中额外引入的一个超参数,

当 \lambda =0 时， A_t^{GAE}=r_t+\gamma V\left(s_{t+1}\right)-V\left(s_t\right) ，也即是仅仅只看一步差分得到的优势；
当 \lambda =1 时， A_t^{G A E}=\sum_{l=0}^{\infty} \gamma^l \delta_{t+l}=\sum_{l=0}^{\infty} \gamma^l r_{t+l}-V\left(s_t\right) ，则是看每一步差分得到优势的完全平均值。

这样就得到了完整版本的 PPO 损失函数的公式，即 J_{PPO}(\theta)=\min \left(\frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text {old }}(a \mid s)}} A^{GAE}, \operatorname{clip}\left(\frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text {old }}}(a \mid s)}, 1-\epsilon, 1+\epsilon\right) A^{GAE} \right) \\下面通过一个例子来说明主要元素的计算过程, GAE 的代码实现如下：

def get_gae_advantages_and_returns(
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Input:
        - values: Token level value, shape: (bs, response_len)
        - rewards: Token level reward, shape: (bs, response_len)
        - action_mask: [EOS] mask. The token after [EOS] have mask zero. shape: (bs, response_len)

        Output:
        - advantages: shape: (bs, response_len)
        - returns: Tensor of shape (bs, response_len)
        """
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

可以看到，PPO 是 token-level 的动作，而其奖励确实 response-level（只在 <EOS> 位置有奖励值），这种方式实际上会导致奖励非常稀疏。

主流框架的实现可参考：

trl/ppo_trainer
OpenRLHF/get_advantages_and_returns
verl/compute_gae_advantage_return

标准的 PPO 算法直接来源于经典 RL ，对语言序列的特点考虑的有限，特别是在长文本上，对此 VC-PPO 做了一些特异性修正，下面理解一下其核心点：长序列中的值模型偏差及其消除

在 PPO 中，VM 是由 RM 初始化的，而由于 reward 是 response-level 的，即取的是 <EOS> 位置的分数，而分配给靠前 token 的分数较低；而 value 是 token-level 的，在给定策略下估计所有在之前的 token 的预期累积奖励。在早期训练阶段，由于 GAE 的反向计算，每个时间步 t 都会存在正偏差，并沿着轨迹累积。因此在 VC-PPO中，value model 在 policy model 基础上经过预训练以减少初始化的偏差。

VC-PPO 另一个关键点是 Decoupled-GAE，在优化 value model 时，由于基于规则的 reward 较准确，使用\lambda_{critic} = 1，而计算 policy loss 中的advantage时，使用\lambda_{policy} = 0.95来降低方差。

另外为了适应长 CoT，VC-PPO 提出了一种长度自适应更新策略，确保 TD 误差在短序列和长序列上分布更加均匀，即 \sum_{t=0}^{\infty} \lambda_{\text {policy }}^t \approx \frac{1}{1-\lambda_{\text {policy }}}=\alpha l \\ \lambda_{policy} = 1- \frac{1}{\alpha l}\\ 这种方式可以更加灵活有效地处理不同长度的序列。

2.2 RLOO

RLOO核心实现细节在于，它使用批次中其他样本的平均奖励来计算基线，而不是平均所有奖励。具体来说，在每个时间步 t ，我们从 s_t 采样 K 个输出样本，因此对于一个 prompt 来说，其基线可以定义如下： b\left(c, a_k\right)=\frac{1}{K-1} \sum_{i=1, i \neq k}^K r\left(s, a_i\right)\\ 在此情况下，其优势函数为： A\left(s, a_k\right)=r\left(s, a_k\right)-b\left(s, a_k\right)=\frac{K}{K-1}\left(r\left(s, a_k\right)-\frac{1}{K} \sum_{i=1}^K r\left(s, a_i\right)\right) \\下面通过一个例子来说明主要元素的计算过程

可以看到 RLOO 与 PPO 的计算过程明显不同，主要体现在：

PPO 的计算过程是 token-level 的（即将 token 看做 action），而 RLOO 的计算过程是 response-level 的（即将 response 看做 action）
RLOO 没有 PPO 中的 value model(critic) ，而需要对每个 prompt 产生多个输出，即显存占用变少而产生样本变多；
PPO 使用 GAE 的方式计算 A 函数，而 RLOO 通过与基线的差值来计算 A 函数

当然，RLOO 也可以通过 token-level 的方式进行计算，更多实现可参考：

trl/rloo_trainer (token-level & response-level)
OpenRLHF (token-level)
verl/compute_rloo_outcome_advantage (response-level)
2.3 GRPO 与 Dr,GRPO

DeepSeek提出的 GRPO 基本上结合了 PPO 与 RLOO 的多采样技巧, 其通过简化值估计并将相同的价值分配给每个 token（即在提示的完成中，每个 token 都分配相同的价值，而不是标准值函数中的折扣奖励）来实现这一点，然后通过蒙特卡洛估计优势或基线。

具体来说，对于给定的prompt s 的一组多个相应 \left\{a_1, a_2, \ldots, a_G\right\} 其损失函数可写成 J(\theta)=\frac{1}{G} \sum_{i=1}^G\left[\min \left(\frac{\pi_\theta\left(a_i \mid s\right)}{\pi_{\theta_{d d}}\left(a_i \mid s\right)} A_i, \operatorname{clip}\left(\frac{\pi_\theta\left(a_i \mid s\right)}{\pi_{\theta_{d d}}\left(a_i \mid s\right)}, 1-\varepsilon, 1+\varepsilon\right) A_i\right)\right]\\ 同样可将其拓展为 token-level 的损失函数 J(\theta)=\frac{1}{G} \sum_{i=1}^G \frac{1}{\left|a_i\right|} \sum_{t=1}^{\left|a_i\right|}\left[\operatorname {min} \left(\frac{\pi_\theta\left(a_{i, t} \mid s_{i, t}\right)}{\pi_{\theta_{\text {old }}\left(a_{i, t} \mid s_{i, t}\right)}} A_{i, t} , \operatorname{clip}\left(\frac{\pi_\theta\left(a_{i, t} \mid s_{i, t}\right)}{\pi_{\theta_{d d}}\left(a_{i, t} \mid s_{i, t}\right)}, 1-\varepsilon, 1+\varepsilon\right) A_{i, t} \right) \right] \\ 其中第 i 个response 的优势计算方式如下： A_i=\frac{r_i-\operatorname{mean}\left(r_1, r_2, \cdots, r_G\right)}{\operatorname{std}\left(r_1, r_2, \cdots, r_G\right)}\\ 直观地说，GRPO 的更新是在批次内比较单个问题与多个答案，这是一种非常简单的方式来计算优势，即特定动作相对于给定状态的平均动作有多好。相对于 PPO，GRPO 通常使用每个提示的样本数量要高得多。代码实现如下：

    def compute_grpo_advantages(rewards, num_samples, adv_type):
        """Function that computes GRPO advantages and returns from rewards.
        Input:
        - values: Token level value, shape: (bs*num_samples, response_len)
        - num_samples: samples generated by LLM for one prompt
        - adv_type: ['grpo','dr_grpo'] for GRPO or Dr.GRPO

        Output:
        - advantages: shape: (bs, num_samples)

        """
        rewards = rewards.sum(-1).view(-1, num_samples) # (bs, num_samples)

        # Compute monte carlo trajectory-level advantage
        values = rewards.mean(dim=1) #(bs,)
        values = values.repeat_interleave(num_samples, dim=0).view(-1, num_samples) #(bs, num_samples)

        advantages = rewards - values # (bs, num_samples)
        if adv_type == "grpo":
            # Additionally normalize by std.
            std_grouped_rewards = rewards.std(dim=1)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(num_samples, dim=0).view(-1, num_samples)
            advantages = advantages / (std_grouped_rewards + 1e-8)
        return advantages

GRPO 的优势计算存在偏差权衡。通过标准差进行归一化会奖励那些在批次中答案正确性变化较小的提问。对于几乎全部正确或全部错误的答案，标准差会较低，优势会较高。

Dr. GRPO 提出在这种情况下移除标准差项，但这会以降低那些全部错误但有少数正确答案的提问的权重为代价，这可能会被视为有价值的信号。

Dr. GRPO 移除了标准差归一化项。这解决了对低奖励方差问题（即几乎所有答案都是正确或错误的）的偏差，在常数缩放因子下等同于 RLOO 估计 \tilde{A}_i=r_i-\operatorname{mean}\left(r_1, r_2, \cdots, r_G\right)=r_i-\frac{1}{G} \sum_{j=1}^{G} r_j \\ 回顾一下 RLOO 的优势估计 A_i^{\mathrm{RLOO}}=r_i-\frac{1}{G-1} \sum_{j=1, i \neq j}^G r_j\\ 则有以下缩放等价性 \begin{aligned} \frac{G}{G-1} \bar{A}_i & =\frac{G}{G-1}\left(r_i-\frac{1}{G} \sum_{j=1}^G r_j\right) \\ & =\frac{G}{G-1} r_i-\frac{1}{G-1} \sum_{j=1}^G r_j \\ & =\frac{G}{G-1} r_i-\frac{1}{G-1} \sum_{j=1, j \neq i}^G r_j-\frac{1}{G-1} r_i \\ & =r_i\left(\frac{G}{G-1}-\frac{1}{G-1}\right)-\frac{1}{G-1} \sum_{j=1, j \neq i}^G r_j \\ & =r_i-\frac{1}{G-1} \sum_{j=1, j \neq i}^G r_j \\ & =A_i^{\mathrm{RLOO}} \end{aligned}\\ 关于 GRPO 还有一个细节是关于 KL 散度的，即在通常的 GRPO 实现中，KL 并不直接作用为 reward 的惩罚项，而是通常直接作用到损失函数中 (DAPO 与 Dr. GRPO 证明KL并非必须项)，即 J(\theta)=\frac{1}{G} \sum_{i=1}^G \sum_{t=1}^{\left|a_i\right|}\left[\operatorname {min} \left(\frac{\pi_\theta\left(a_{i, t} \mid s_{i, t}\right)}{\pi_{\theta_{\text {old }}\left(a_{i, t} \mid s_{i, t}\right)}} A_{i, t} , \operatorname{clip}\left(\frac{\pi_\theta\left(a_{i, t} \mid s_{i, t}\right)}{\pi_{\theta_{d d}}\left(a_{i, t} \mid s_{i, t}\right)}, 1-\varepsilon, 1+\varepsilon\right) A_{i, t} \right)-\beta D_{K L}\left(\pi_\theta\left(\cdot \mid s_{i, t}\right) \| \pi_{r e f}\left(\cdot \mid s_{i, t}\right)\right) \right] \\ 下面通用通过一个例子来说明其主要元素的计算过程

主流框架的实现可参考：

trl/grpo_trainer
OpenRLHF/grpo & dr_grpo
verl/compute_grpo_outcome_advantage
Dr_GRPO/compute_advantages
2.4 REINFORCE++ 与 ReMax

前边介绍的方法都一定程度使用了一些假设条件，同时由于训练成本高昂，很难同时训练多种方法进行比较选择，那么使用是否有一种 baseline 作为参照呢？REINFORCE 就是这样一种大道至简的方法。

事实上，REINFORCE++ 的 A 函数计算方式与 Dr.GRPO 完全一致（也不需要 value 模型），即 A_i={r_i-\operatorname{mean}\left(r_1, r_2, \cdots, r_G\right)}\\ 与Dr.GRPO 不同的是其优势值会拓展到 token 级别，因此 reward 的计算方法（使用 KL 惩罚项）及重要性采样与clip 又与 PPO 保持一致，因此这是一种非常理想的基线方法。

# REINFORCE++ 算法的 advantage 的计算过程
scores[i] = scores[i] - id2mean[index[i]]
scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
scores = verl_F.masked_whiten(scores, response_mask) * response_mask

在 PPO 中，计算 GAE 需要状态值函数，这在LLM训练后难以准确获得，并且存储时占用显存，训练时计算量大。ReMax 通过在每个时间步 t 即时采样贪婪轨迹的回报来消除对 V_{\pi_\theta}(s_t) 的需求。ReMax使用贪婪生成的回答（greedy response）的奖励作为基准值（baseline value）来构建梯度估计器，具体方式如下：

2.5 DAPO 与 VAPO

DAPO 与 VAPO 都是近期字节 seed 团队提出的 SOTA 的 RL 方法，针对现有方法及特定场景提出了很多有价值的技术点，下边我们来一看究竟。

DAPO 对 GRPO 进行了 4 项修改，以更好地适应需要长序列和增加新、未充分利用的标记概率的推理语言模型，其修改后的损失函数为 \begin{aligned} & \mathcal{J}_{\mathrm{DAPO}}(\theta)= {\frac{1}{\sum_{i=1}^G\left|o_i\right|} \sum_{i=1}^G \sum_{t=1}^{\left|o_i\right|} \min \left(R_{i, t}(\theta) \hat{A}_{i, t}, \operatorname{clip}\left(R_{i, t}(\theta), 1-\varepsilon_{\text {low }}, 1+\varepsilon_{\text {high }}\right) \hat{A}_{i, t}\right) } \\ & ~~~~~~~~~~~\text { s.t. } \quad 0<\mid\left\{o_i \mid \text { is_equivalent }\left(a, o_i\right)\right\} \mid<G, \end{aligned}\\ 其中 R_{i, t}(\theta)=\frac{\pi_\theta\left(o_{i, t} \mid q, o_{i,<t}\right)}{\pi_{\theta_{\text {old }}}\left(o_{i, t} \mid q, o_{i,<t}\right)}, \quad \hat{A}_{i, t}=\frac{r_i-\operatorname{mean}\left(r_1,r_2,\dots ,r_G\right)}{\operatorname{std}\left(r_1,r_2,\dots,r_G\right)} \\ 接下来分别简要介绍其 4 处主要的改进点：

（1）提高探索：使用两个不同的 clip 超参数 \epsilon_{low} 和 \epsilon_{high}

在原始的 PPO 算法中，使用 clip 是为了让策略在有限范围内探索，以避免波动过大。在 LLM 的训练过程中，这种做法可能会导致熵坍缩现象：随着训练的进行，策略的熵迅速下降，某些组的采样响应几乎相同，这就影响模型能力的进一步提高。

RL 的关键即探索和利用的平衡，从 token-level 的视角来看，如果取 \epsilon = 0.2 ，有两个动作 \pi_{\theta_{\mathrm{old}}}\left(o_1 \mid q\right)=0.01, \pi_{\theta_{\mathrm{old}}}\left(o_2 \mid q\right)=0.9 此时如果更新概率 \pi_{\theta}\left(o_1 \mid q\right)=0.012, \pi_{\theta}\left(o_2 \mid q\right)=1.08 即会被截断，此时低概率的 token 增加的范围非常有限，因此将 \epsilon 的上下限解耦，增大 \epsilon_{high} 可以有效提高LLM的探索能力（特别是低概率 token 上）。

另一方面，为什么不扩大 \epsilon_{low} 呢？如果 \epsilon_{low} 也增大，即允许 \pi_{\theta}\left(o_i \mid q\right) 更大程度接近 0，这样会导致采样空间的压缩。

（2）动态采样，有效学习

在 GRPO 中，如果一个 prompt 过于简单或者过于困难，导致一组的输出全对或者全错，那么这组的优势为零，进而导致梯度为零，策略无法更新，这样的话样本效率比较低。过滤掉这类样本将有效提高模型训练效率，具体做法即损失函数的约束条件。

（3）样本区分：token-level 损失

在 GRPO 中，损失是样本级的，即首先在每个样本内按 token 平均损失，然后跨样本汇总损失，这样每个样本在最终损失计算中被赋予相同的权重。这样对于长输出，可能会带来以下影响：

高质量长输出：可能会导致模型无法有效学习其中的推理模式；
低质量长输出，如乱码重复，也无法进行有效惩罚，导致熵和长度不健康增加；

因此DAPO增加长度的平均项，使无论奖励出现在哪个长度的响应中，该模式都将被同等程度地促进或抑制。

（4）减少噪声：超长过滤

在样本生成过程中，通常会设置一个最大长度，超过该长度的样本会被截断，被截断的样本通常没有最终的结果，因此也无法做出准确评估，这就会造成训练过程的噪声。DAPO 过滤掉了这类样本，显著提高了训练稳定性及性能。

在前文中，我们已经介绍了主要的 LLM+RL 训练方法，包括：大本大宗的 PPO，炙手可热的 GRPO，作为基线的 REINFORCE，修正主义的 DAPO。那么是否有一种方法能够集以上方法之大成呢？我想当前 VAPO 算是一个，这么说的原因体现在：

从 PPO 继承了损失函数的基本形式，特别是对于值函数的利用参考了 VC-PPO
从 GRPO 中继承了 Group-Sampling
从 DAPO 中继承了 Clip-Higher

此外，VAPO 还有一个重要贡献是：稀疏奖励信号及其处理

在 RLHF 中，奖励信号本来就是稀疏的（只在 EOS 位置有奖励值），而基于验证器的奖励模型通常提供二元反馈（0 或 1），这种反馈比 RM 输出的连续值更加稀疏。在复杂推理场景中，CoT 显著延长输出长度，不仅增加计算时间，还降低获得非零奖励的频率。在策略优化中，具有正确答案的采样响应可能极为稀缺且宝贵。这就会遇到 RL 最常见的探索-利用困境：

一方面，模型必须保持相对较高的不确定性。这使它能采样各种不同的响应，增加针对给定提示生成正确答案的可能性。
另一方面，算法需要有效地利用通过艰苦探索获得的正确采样响应——以增强学习效率。

为了解决这一问题，VAPO 除了继承了 DAPO 中的Clip-Higher 和 Group-Sampling，还提出了一个正例 LM 损失。在复杂推理任务的强化学习背景下，一些任务表现出显著的低准确率，大多数训练样本都给出错误答案。传统的策略优化策略在抑制错误样本生成概率时，在强化学习训练过程中效率低下，因为试错机制会带来巨大的计算成本。鉴于这一挑战，当策略模型采样到正确答案时，最大化正确答案的效用至关重要。为了应对这一挑战，通过在强化学习训练过程中对正确结果采样引入额外的负对数似然（NLL）损失 \mathcal{L}_{\mathrm{NLL}}(\theta)=-\frac{1}{\sum_{o_i \in \mathcal{J}}\left|o_i\right|} \sum_{o_i \in \mathcal{J}} \sum_{t=1}^{\left|o_i\right|} \log \pi_\theta\left(a_t \mid s_t\right)\\ 其中 \mathcal{J} 表示正确答案的集合。最终的负对数似然损失通过权重系数 \mu 与策略梯度损失相结合，共同作为更新策略模型的指标：

\mathcal{L}(\theta)=\mathcal{L}_{\mathrm{PPO}}(\theta)+\mu * \mathcal{L}_{\mathrm{NLL}}(\theta)\\ 通过这种方法大大提高了强化学习训练过程中正样本的利用效率。

2.6 GSPO

从上文可知，GRPO 的 reward 和 advantage 都是 response-level 的，但是其重要性采样确实 token-level 的。在数学上，重要性采样理论要求我们对从一个分布中采出的多个样本求平均，才能准确修正分布的偏差。而 GRPO 在每个时间步，只基于一个采样出的token y_t 来计算权重，这个权重充满了随机噪声，失去了修正分布的意义。这种噪声会随着回答的变长而不断累积，从而引发灾难性的模型崩溃。

GSPO 定义了 response-level 的重要性比率，并在 sequence 层面执行裁剪、奖励和优化。设 $$x$$ 为prompt， $$\pi_{\theta_\text{old}}$$ 为用于采样回复的策略， \{y_i\}_{i=1}^G 为采样得到的回复组， \widehat{A}_{i} 为各个回复的组内相对优势， \pi_\theta 为需优化的当前策略。GSPO 采用以下优化目标： \begin{aligned} \mathcal{J}_{\mathrm{GSPO}}(\theta) & =\mathbb{E}_{x \sim \mathcal{D},\left\{y_i\right\}_{i=1}^G \sim \pi_{\theta_{\text {old }}}(\cdot \mid x)}\left[\frac{1}{G} \sum_{i=1}^G \min \left(s_i(\theta) \widehat{A}_i, \operatorname{clip}\left(s_i(\theta), 1-\varepsilon, 1+\varepsilon\right) \widehat{A}_i\right)\right] \\ s_i(\theta) & =\left(\frac{\pi_\theta\left(y_i \mid x\right)}{\pi_{\theta_{\text {old }}}\left(y_i \mid x\right)}\right)^{\frac{1}{\left|y_i\right|}}=\exp \left(\frac{1}{\left|y_i\right|} \sum_{t=1}^{\left|y_i\right|} \log \frac{\pi_\theta\left(y_{i, t} \mid x, y_{i,<t}\right)}{\pi_{\theta_{\text {old }}}\left(y_{i, t} \mid x, y_{i,<t}\right)}\right) . \end{aligned}\\

s_i(\theta) 即为 GSPO 基于序列似然定义的重要性比率，并使用了长度归一化（几何平均值）以降低方差并统一 s_i(\theta) 的数值范围。

这种做法带来几个效果：

在相同计算资源下，GSPO 在训练奖励和下游任务性能上，都稳定且持续地优于 GRPO
裁剪悖论：GSPO 所裁剪的 token 比例比 GRPO 要高上两个数量级，但却具有更高的训练效率。这进一步表明 GRPO 采用的 token 级别的优化目标是有噪和低效的，而 GSPO 的序列级别的优化目标则提供了更可靠、有效的学习信号。
MoE 友好：当采用 GRPO 算法时，MoE 模型的专家激活波动性会使得 RL 训练无法正常收敛。为了解决这一挑战，过去采用了路由回放（Routing Replay）训练策略，即缓存\pi_{\theta_\text{old}}中激活的专家，并在计算重要性比率时在 \pi_\theta中“回放”这些路由模式。Routing Replay 对于 GRPO 训练 MoE 模型的正常收敛至关重要。然而，Routing Replay 的做法会产生额外的内存和通信开销，并可能限制 MoE 模型的实际可用容量。

GSPO 的一大突出优势在于彻底消除了对 Routing Replay 的依赖。其核心洞见在于：GSPO 仅关注序列级别的似然（即\pi_\theta(y_i|x)，而对个别 token 的似然（即 \pi_\theta(y_{i,t}|x,y_{i,<t})）不敏感。因此，其无需 Routing Replay 等对基础设施负担较大的手段，既简化和稳定了训练过程，又使得模型能够最大化地发挥容量与潜能。

三、从理论角度再思考 LLM + RL

从前文的讨论可以看到，与 RL 的经典场景不同，使用 RL 训练 LLM 显得比较“拧巴”，其根本原因就是语言本身的特点及其建模过程导致的。那么我们是否可以从更高的角度来思考这一问题呢？

3.1 DPO：有监督对齐

在上节中，我们详细讨论了 online 的学习过程，以 PPO 为例，即由策略产生样本，在RM（裁判）和 VM（教练）的指导下进行训练，其核心在于训练数据是实时产生的。而与之相对，DPO 则是直接从优化目标求解最优对齐模型，其核心在于融合策略和奖励的学习，使之可以在一个步骤中通过有监督方式得到。

回顾一下，奖励模型的损失函数： \max _{r_\phi}\left\{\mathbb{E}_{x, y_{\mathrm{w}}, y \sim D}\left[\log \sigma\left(x, y_{\mathrm{w}}\right)-\log \sigma\left(x, y_1\right)\right]\right\}\\ PPO 损失函数： \max _{\pi_\theta}\left\{\mathbb{E}_{x \sim D, y \sim \pi_\theta(y \mid x)}\left[r_\phi(x, y)\right]-\beta \mathbb{D}_{K L}\left[\pi_\theta(y \mid x) \| \pi_{\text {ref }}(y \mid x)\right]\right\}\\ DPO 损失函数可以看做二者的融合： \max _{\pi_\theta}\left\{\mathbb{E}_{x, y_{\mathrm{w}}, y_1 \sim D}\left[\log \sigma\left(\beta \log \frac{\pi_\theta\left(y_{\mathrm{w}} \mid x\right)}{\pi_{\mathrm{ref}}\left(y_{\mathrm{w}} \mid x\right)}-\beta \log \frac{\pi_\theta\left(y_1 \mid x\right)}{\pi_{\mathrm{ref}}\left(y_1 \mid x\right)}\right)\right]\right\}\\ 注意到 DPO 没有对策略的采样过程，而利用预先收集的比较数据，直接根据偏好比较优化策略。通过这种方法，我们可以绕过单独训练奖励模型，直接使用标注的成对偏好数据一次性训练对齐模型 \pi_\theta ，而这是典型的静态的有监督方法。​

尽管这种方式比较简单，对计算资源要求较少，但其实际效果常常不如 PPO，主要局限性体现在：

评估与生成之间的脱节

DPO 的训练过程仅让模型学会“评估”，而没有融入实际对弈所需的在线生成过程。相比之下，PPO 通过在线生成和试错来学习，将评估能力转化为生成能力。没有这种在线探索，仅用 DPO 训练的模型可能在离线数据上得分很高，但在实际生成过程中表现不佳。

离线训练的局限性

RLHF 本质上是一种在线学习方法，因为它需要持续纠正模型现有的知识。然而，DPO 完全是离线的；它迫使模型仅依赖于标注者认为“正确”的，并遵循预定的最优路径，几乎没有探索的空间。在实践中，通常使用诸如在首选响应上进行初始监督微调（SFT）或用多样化的输出增强偏好数据等技术，以引入一些在线学习和探索的元素。

要求高数据质量

由于 DPO 训练完全依赖于离线偏好数据，其有效性高度敏感于这些数据的质量和覆盖范围。如果训练数据不全面或与实际的生成分布不匹配，模型可能会生成具有正确相对比例的正负例子的响应，但绝对概率可能会被稀释，甚至可能出现训练数据中不存在的输出。

当然 DPO 也可以改造成在线的形式，即 online DPO , 即对于一个输入 x , 可以使用当前策略产生两个输出 y_1,y_2 , 然后引入一个一个标注器将样本标注为 y^+,y^- , 然后进一步训练 DPO，该过程如下图所示

3.2 统一视角理解从 SFT 到 RL

在前文中，我们主要从经验的角度去讨论和比较各种方法，那么如果从第一性原理的角度，如果更加深刻理解这一问题呢？本节将围绕此点展开，以下内容主要参考 All Roads Lead to Likelihood: The Value of Reinforcement Learning in Fine-Tuning。

从本质上来说，无论是SFT，PFT，RL，后训练的目标可以统一表述为 \pi^{\star}=\underset{\pi \in \Pi}{\operatorname{argmin}} \underbrace{\mathbb{D}_{\mathrm{KL}}\left(\mathbb{P}_{\mathcal{D}} \| \mathbb{P}_\pi\right)}_{\text {Data Likelihood }}+\beta \underbrace{\mathbb{D}_{\mathrm{KL}}\left(\mathbb{P}_\pi \| \mathbb{P}_{\pi_{\mathrm{ref}}}\right)}_{\text {Prior Regularization}} .\\ 其中 \mathbb{P}_{\mathcal{D}} ，\mathbb{P}_{\pi}, \mathbb{P}_{\pi_{ref}} 分别表示训练数据集上的均匀分布，在策略 \pi 上的分布，参考策略 \pi_{ref} 上的分布。上式中，第一项数据似然使用前向 KL 项，衡量数据集 \mathcal{D} 中的样本与策略 \pi 生成样本的相似程度，第二项先验正则使用反向 KL 项，衡量策略 \pi 与参考策略 \pi_{ref} 的近似程度。

理想情况下，如果\mathcal{D} 能够覆盖所有轨迹对，就无需第二项，但在真实场景中，由于样本数量的限制，故添加第二项作为正则化。通常情况下，会把起始模型当作 \pi_{ref} , 但随着训练的进行， \pi 的能力不断增强，此时 \pi_{ref} 反而成了限制 \pi 进一步提高能力，因此也可以使用 \pi 本身的熵 \mathbb{H}(\pi)=\mathbb{E}_{\xi \sim \pi}\left[\sum_h^H-\log \pi\left(a_h \mid s_h\right)\right] 作为正则化项 \pi^{\star}=\underset{\pi \in \amalg}{\operatorname{argmin}} \mathbb{D}_{\mathrm{KL}}\left(\mathbb{P}_{\mathcal{D}} \| \mathbb{P}_\pi\right)-\mathbb{H}(\pi)\\

按照以上理论，对于经典 SFT，其目标即在离线数据上通过最大似然估计（MLE）直接优化策略参数。而对于在线策略，该过程包括两个阶段：（1）通过 MLE 拟合奖励模型（RM，空间 \mathcal{R} ）；（2）使用 RM 为强化学习（RL，空间 \Pi ）过程提供反馈。

其中RM的训练通常采用 BT 模型，即 \mathbb{P}_r^{\mathrm{BT}}\left(\xi^+ \succ \xi^- \mid s_0\right)=\sigma\left(r\left(\xi^+\right)-r\left(\xi^-\right)\right)

如果 RM 是局部的，我们可以通过用 r 的对数概率之和替换 r 来通过 MLE 拟合策略 \begin{aligned} \hat{\pi}_{\mathrm{mle}} & =\underset{r_\pi \in \mathcal{R}(\Pi)}{\operatorname{argmin}} \mathbb{D}_{\mathrm{KL}}\left(\mathbb{P}_{\mathcal{D}} \| \mathbb{P}_{r_\pi}^{\mathrm{BT}}\right) \\ & =\underset{r_\pi \in \mathcal{R}(\Pi)}{\operatorname{argmax}} \sum_i^N \log \sigma\left(r_\pi\left(\xi_i^{+}\right)-r_\pi\left(\xi_i^{-}\right)\right) \\ & =\underset{\pi \in \Pi}{\operatorname{argmax}} \sum_i^N \log \sigma\left(\sum_h^H \log \frac{\pi\left(a_{h, i}^{+} \mid s_{h, i}^{+}\right)}{\pi\left(a_{h, i}^{-} \mid s_{h, i}^{-}\right)}\right) \end{aligned}\\ 由此可知，离线 PFT 方法如 DPO 本质上是一个在局部奖励模型上的轨迹级分类问题。

进一步可推出，当 \mathcal{R} = \mathcal{R}(\Pi) (即覆盖了相同的奖励函数集合)，此时 RLHF 等价于 MLE (DPO)

\hat{\pi}_{d p o}^{\star}=\underset{\pi \in \Pi}{\operatorname{argmax}} \mathbb{E}_{\xi \sim \pi}\left[\hat{r}_{m l e}(\xi)\right]-\mathbb{D}_{K L}\left(\mathbb{P}_\pi \| \mathbb{P}_{\pi_{\text {ref }}}\right)=\hat{\pi}_{r l h f } \\也就是说从信息论的角度，以上方法无法区分， 但是，以上假设条件过于理想，在真实情况下的经验是 online policy (如 PPO) > offline policy (如 DPO) > SFT， 那么这之间的差异是由什么导致的呢？

从计算机科学的角度来看，将策略视为生成器，将奖励模型视为验证器，奖励函数（可以用深度更低的电路表示）比最优策略更容易表示。因此反向强化学习（即从演示中学习奖励模型并通过强化学习解码）优于行为克隆（即通过最大似然估计直接学习策略）的方法，即在线学习是更优的策略学习。从端到端来看，在线微调只需要在 \Pi(\mathcal{R}_{sim}) \subset \Pi 中搜索策略，而不是像离线微调那样在整个 \Pi 中进行搜索。

总结一下，本文第一部分从经典的 RL 出发，重新审视和梳理了 RLHF 与 RLVR 建模中的各个概念，并解释了其中部分关键细节的底层原理。第二部分则分别讨论了当前主流方法的原理、联系、区别及适用场景。最后则从更加宏观的视角，分析了 RL 与 SFT 方法在本质上的区别与联系。

参考资料

[1] RLHF without RL - Direct Preference Optimization

[2] Unifying RLHF Objectives

[3] Reinforcement Learning with Verifiable Rewards: GRPO's Effective Loss, Dynamics, and Success Amplification

[4] Policy Gradient
Algorithms | RLHF Book by Nathan Lambert

[5] https://huggingface.co/blog/NormalUhr/rlhf-pipeline

[6] https://pub.towardsai.net/group-relative-policy-optimization-grpo-illustrated-breakdown-explanation-684e71b8a3f2

[7] https://openreview.net/pdf?id=w3d44iguZK

[8] AI Colleague for Research Papers

[9] TLCR: Token-Level Continuous Reward for Fine-grained Reinforcement Learning from Human Feedback

[10] Secrets of RLHF in Large Language Models Part I: PPO

[11] Secrets of RLHF in Large Language Models Part II: Reward Modeling

[12] https://huggingface.co/blog/zh/putting_rl_back_in_rlhf_with_rloo

[13] All Roads Lead to Likelihood: The Value of Reinforcement Learning in Fine-Tuning

[14] GitHub - hkproj/rlhf-ppo: Notes and commented code for RLHF (PPO)

[15] TRPO 算法

[16] From REINFORCE to Dr. GRPO

[17] Policy Gradient
Algorithms | RLHF Book by Nathan Lambert

柳青春已半，晓日初曈昽。洱波三万顷，轻舟泛长风。 —— 李元阳 《泛洱水》
