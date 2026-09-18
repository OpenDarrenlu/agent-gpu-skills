# LLM(34)：Batch size 与 Learning rate 的理论与实践

**作者**: 紫气东来​上海交通大学 工学硕士

**原文链接**: https://zhuanlan.zhihu.com/p/1995433233983231025

---

​
目录
收起
一、基于 SGD 优化器的证明
1.1 平方根缩放与线性缩放
1.2 梯度噪声缩放理论
1.3 数据量与步数的关系
二、基于 Adam 优化器的讨论
三、考虑learning rate schedulers (LRS) 的影响
参考资料

在之前的文章中，笔者比较简略讨论过 scaling law 对 Batch size 与Learning rate 超参设置的指导，当时的主要结论如下：

当计算预算增加时，应增大 Batch size；
当计算预算增加时，应减小 Learning rate;
当计算预算不变时，增大 Batch size 的同时应增大 Learning rate；

细心的读者应该会发现，按照以上的结论，Batch size 与Learning rate 既有同步变化的时候，又有反向变化的时候，看起来有些奇怪。诚然，此前的分析比较定性和简略，缺少理论和实验依据。那么本篇将试图用更严谨的方式从定量角度来分析和理解这一问题。

一、基于 SGD 优化器的证明

基于对 scaling law 的认识，当计算预算 
𝐶
 （主要是参数量 
𝑁
 和数据量 
𝐷
 ）增加时，最终的 loss 会降低，也即最终的效果会变好。现在假设模型参数不变，增加数据量时，如果Batch size 不变，训练时间会线性增加，此时增大 Batch size （主要是通过增加有效算力来实现）就变成了缩短训练时间的有效方法。

在之前的分析中我们提到，当Batch Size增大时，每个Batch的梯度将会更准，此时就可以增大 Learning rate，以实现更快梯度下降，进而缩短训练时间。那么这个过程究竟是怎么发生的呢，下面我们就试图探究一下。

1.1 平方根缩放与线性缩放

首先我们考虑简单的 SGD 优化器下的情况，将随机采样一个样本的梯度记为 \tilde{\boldsymbol{g}} ，其整体的均值和协方差分别记为 {\boldsymbol{g}} 和 {\boldsymbol{\Sigma}} ，那么当采样数目增加到B 时，有

\begin{aligned} &\tilde{\boldsymbol{g}}_B \triangleq \frac{1}{B} \sum_{i=1}^B \tilde{\boldsymbol{g}}^{(i)},\\ &\mathbb{E}\left[\tilde{\boldsymbol{g}}_B\right] = \boldsymbol{g},\\ &\mathbb{E}\left[\left(\tilde{\boldsymbol{g}}_B-\boldsymbol{g}\right)\left(\tilde{\boldsymbol{g}}_B-\boldsymbol{g}\right)^{\top}\right] = \frac{\boldsymbol{\Sigma}}{B}. \end{aligned}
即增加采样数目不改变均值，而协方差则缩小到 1/B 。对于 SGD 优化器，其参数更新过程为 w_{t+1} = w_t-\eta \tilde{\boldsymbol{g}}_B ，其变化量 \eta \tilde{\boldsymbol{g}}_B 的协方差
\mathbb{E}\left[ (\eta \tilde{\boldsymbol{g}}_B - \boldsymbol{g})(\eta \tilde{\boldsymbol{g}}_B - \boldsymbol{g})^\top \right] = \frac{\eta^2}{B} \mathbf{\Sigma} + (\eta - 1)^2 \boldsymbol{g} \boldsymbol{g}^\top
为了让增量的噪声强度即协方差矩阵保持不变，即 \frac{\eta^2}{B} 保持常数，则有
\eta \propto \sqrt{B}
这样就从方差角度证明了SGD 优化器下的Batch size 与Learning rate 的平方根变化关系。

以上分析仅考虑了单步的协方差，但是优化是一个整体的过程。下面我们将从另一个角度来审视这一过程，回到SGD 的更新过程
w_{t+1} = w_t-\eta \tilde{\boldsymbol{g}}_{B,t}=w_t-\eta (\tilde{\boldsymbol{g}}_{t}+\epsilon_t)
其中 \epsilon_t =\tilde{\boldsymbol{g}}_{B,t}-\tilde{\boldsymbol{g}}_{t} 表示梯度的噪声，这个噪声来源于批次的随机采样，显然其均值为 0 。现在从另一个角度考虑，目标不仅仅是控制噪声，还要保证训练的进度（收敛速度）。

考虑一个更合理的等价目标：保持每个epoch（遍历整个数据集一次）内，模型权重的总期望更新量（或总更新步长）大致相同。即(单步期望更新范数)×(步数)≈常数，现在举例来说明，假设总数据量为 D ， B^\prime = kB ，则

小批量中，一个epoch共更新 D/B 步，每个步骤的更新向量是 -\eta \tilde{\boldsymbol{g}}_{B,t} （期望为 \tilde{\boldsymbol{g}}_{t} ）
大批量中，一个epoch共更新 D/ kB 步，每个步骤的更新向量是 -\eta^\prime \tilde{\boldsymbol{g}}_{B^\prime,t}（期望也为 \tilde{\boldsymbol{g}}_{t} ）

现在为了保证总的更新量一致，则有
\frac{D}{B}\eta \tilde{\boldsymbol{g}}_{B,t} \approx \frac{D}{kB}\eta^\prime \tilde{\boldsymbol{g}}_{B^\prime ,t}
因此有 \eta^\prime = k\eta ，这就是线性缩放规则。

事实上，线性缩放规则应用更广，效果也更好，这是因为训练过程中更关心训练效率（用更少的epoch达到目标精度）。线性缩放确保了在扩大批量、减少迭代次数时，每一步的“力度”足够大，从而不拖慢每个epoch的收敛速度。

1.2 梯度噪声缩放理论

上一节推导出的结论显示，Learning rate 可随着 Batch size 的增大以平方根或者线性速率同步增大。但是我们不禁又产生疑问，这种增加是无限制的吗？接下来，我们将从损失函数出发，探究Learning rate 与 Batch size 的关系。

考虑目标函数 L(w) 在局部最小值附近的二次近似：

L(w) \approx L(w^*) + \frac{1}{2} (w -w^*)^T H (w -w^*)

其中 H 为正定 Hessian 矩阵。通过坐标变换到 H 的特征基，问题分解为独立的一维问题。对于第 i 个特征方向，损失函数为：

L_i(w_i) = \frac{1}{2} \lambda_iw_i^2

其中 \lambda_i > 0 为特征值。

在该方向上，梯度为 \lambda_iw_i，小批量梯度估计的噪声方差为 \sigma_i^2 / B，其中 \sigma_i^2 为单个样本梯度在该方向上的方差。定义该方向的梯度噪声尺度为：

B_{\text{noise}, i} = \frac{\sigma_i^2}{(\lambda_iw_i)^2}

它衡量了梯度平方与噪声方差的比值。

从当前点 w_i 出发，SGD 更新为：

w_i' =w_i - \eta (\lambda_iw_i + n_i)

其中 n_i 为零均值、方差为 \sigma_i^2 / B 的噪声。下一步的期望损失为：

\mathbb{E}[L_i(w_i')] = \frac{\lambda_i}{2} \mathbb{E}\left[ (w_i - \eta (\lambda_iw_i + n_i))^2 \right]

为最小化该期望损失，等价于最小化：

\mathbb{E}\left[ (w_i - \eta (\lambda_iw_i + n_i))^2 \right] = (1 - \eta \lambda_i)^2w_i^2 + \eta^2 \frac{\sigma_i^2}{B}

对 \eta 求导并令导数为零：

-2\lambda_i (1 - \eta \lambda_i)w_i^2 + 2\eta \frac{\sigma_i^2}{B} = 0

解得：

\eta = \frac{\lambda_iw_i^2}{\lambda_i^2w_i^2 + \sigma_i^2 / B} = \frac{1}{\lambda_i} \cdot \frac{1}{1 + \frac{\sigma_i^2}{B \lambda_i^2w_i^2}} = \frac{1}{\lambda_i} \cdot \frac{1}{1 + B_{\text{noise}, i} / B}

当批次大小 B \to \infty 时，噪声消失，最优学习率为 \eta_{\max, i} = 1 / \lambda_i。因此：

\eta_{\text{opt}, i} = \frac{\eta_{\max, i}}{1 + B_{\text{noise}, i} / B}

若所有特征方向具有相同的特征值 \lambda 和噪声尺度 B_{\text{noise}}，则全局最优学习率为：

\eta_{\text{opt}} = \frac{\eta_{\max}}{1 + B_{\text{noise}} / B}

其中 \eta_{\max} = 1 / \lambda， B_{\text{noise}} 表示梯度噪声尺度。根据该式可以发现：

当 B\ll B_{noise} 时， 1+{B}_{\text {noise }} / B \approx {B}_{\text {noise }} / B ，则 \eta_{opt} \approx \frac{\eta_{\max } B}{{B}_{\text {noise }}} \propto B \text {, } 此时Learning rate 与 Batch size近似呈线性关系
当 B> B_{noise} 时， \eta_{opt} 逐渐趋于饱和值 \eta_{max} ,这意味着训练成本的增加远大于训练效率的提升,也就是说当Batch Size超过这个数值时，就没必要继续投入算力去增大Batch Size了。
1.3 数据量与步数的关系

根据 scaling law, 首先假设损失 \(L\) 可以分解为以下形式：
L = L_\infty + \frac{A}{S} + \frac{B}{E},
其中 L_\infty 是无限数据和无限步骤下可达到的最佳损失， \(A\) 和 \(B\) 是与模型架构、优化算法等相关的正常数， E 表示总数据消耗量， S 表示优化步骤数。该假设基于优化理论和统计学习理论中的常见结论：对于随机凸优化，超额风险（excess risk）通常包含优化误差项 \(O(1/S)\) 和统计误差项 O(1/E) 。

定义 \(\Delta = L - L_\infty > 0\) ，则有：
\frac{A}{S} + \frac{B}{E} = \Delta
令 \(S_{\min} = A/\Delta\) 和 \(E_{\min} = B/\Delta\) 。

当 \(E \to \infty\) 时， \(S \to S_{\min}\) ；
当 \(S \to \infty\) 时， \(E \to E_{\min}\) 。

因此， \(S_{\min}\) 和 \(E_{\min}\) 分别表示达到损失 \(L\) 所需的最小步骤数和最小数据消耗量。

代入上式得：
\frac{S_{\min}}{S} + \frac{E_{\min}}{E} = 1.
通过代数变换：
\begin{aligned} &\frac{S_{\min}}{S} + \frac{E_{\min}}{E} = 1 \\ \Rightarrow &\frac{E_{\min}}{E} = 1 - \frac{S_{\min}}{S} \\ \Rightarrow &\frac{E_{\min}}{E} = \frac{S - S_{\min}}{S} \\ \Rightarrow &E = \frac{E_{\min} S}{S - S_{\min}}. \end{aligned}

类似地，有：
\frac{S_{\min}}{S} = 1 - \frac{E_{\min}}{E} = \frac{E - E_{\min}}{E} \Rightarrow S = \frac{S_{\min} E}{E - E_{\min}}.
由此可得：
\left(\frac{E}{E_{\min}}-1\right)\left(\frac{S}{S_{\min}}-1\right) = \left(\frac{E - E_{\min}}{E_{\min}}\right)\left(\frac{S - S_{\min}}{S_{\min}}\right) = \frac{E - E_{\min}}{E_{\min}} \cdot \frac{S - S_{\min}}{S_{\min}}.
但由 \(E = \frac{E_{\min} S}{S - S_{\min}}\) 可得 \(E - E_{\min} = \frac{E_{\min} S_{\min}}{S - S_{\min}}\) ，代入上式：
\left(\frac{E}{E_{\min}}-1\right)\left(\frac{S}{S_{\min}}-1\right) = \frac{E_{\min} S_{\min}}{(S - S_{\min}) E_{\min}} \cdot \frac{S - S_{\min}}{S_{\min}} = 1 \tag{1}
该式表明，在固定损失下，总数据消耗量 \(E\) 与优化步骤数 \(S\) 之间存在双曲线权衡，其渐近线为 \(E_{\min}\) 和 \(S_{\min}\) 。这一结论在很多场景得到了证实。

二、基于 Adam 优化器的讨论

在上一章中，我们尝试研究在 SGD 优化器中，Learning rate 与 Batch size 的关系，由于 SGD 优化器只有一阶矩，因此可能与存在二阶矩的 Adam 优化器并不相同，那么在 Adam 优化器中应该是什么关系呢？接下来我们将试图探究一下。

首先我们回顾一下 Adam 的更新规则为：

m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t

v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2

w_{t+1} = w_t - \eta \frac{m_t}{\sqrt{v_t} + \epsilon}

在平稳状态下，m_t 和 v_t 的期望近似为：

\mathbb{E}[m] \approx g, \quad \mathbb{E}[v] \approx g^2 + \frac{\sigma^2}{B}

其中 \sigma^2 = \operatorname{Tr}(\Sigma)/d（假设各方向噪声方差平均）。

现在考虑损失函数 L(w)，其梯度 \nabla L(w) 是随机变量，满足：

\mathbb{E}[\nabla L(w)] = g(w), \quad \text{Cov}[\nabla L(w)] = \frac{\Sigma(w)}{B}

其中 g(w) 为真实梯度，B 为批量大小，\Sigma(w) 为单样本梯度协方差矩阵。

定义 噪声批量大小 B_{\text{noise}} 为使得梯度噪声方差与信号方差平衡的批量大小，即满足：

\frac{\operatorname{Tr}(\Sigma(w))}{\|g(w)\|^2} = B_{\text{noise}}

在 B = B_{\text{noise}} 时，梯度噪声的幅度与信号幅度相当。

有效更新步长为 \Delta w = -\eta \frac{m}{\sqrt{v}}。其期望与方差近似为：

\mathbb{E}[\Delta w] \approx -\eta \frac{g}{\sqrt{g^2 + \sigma^2/B}}

\operatorname{Var}[\Delta w] \approx \eta^2 \frac{\sigma^2/B}{g^2 + \sigma^2/B}

定义信噪比（SNR）：

\text{SNR} = \frac{|\mathbb{E}[\Delta w]|}{\sqrt{\operatorname{Var}[\Delta w]}} = \frac{|g|}{\sigma/\sqrt{B}} = \sqrt{\frac{B}{B_{\text{noise}}}}

其中 B_{\text{noise}} = \sigma^2/g^2。

考虑损失函数的二阶泰勒展开：

L(w + \Delta w) \approx L(w) + \nabla L(w)^\top \Delta w + \frac{1}{2} \Delta w^\top H \Delta w

取期望并代入 \mathbb{E}[\Delta w] 和 \operatorname{Var}[\Delta w]：

\mathbb{E}[L(w + \Delta w)] \approx L(w) + \mathbb{E}[\Delta w]^\top g + \frac{1}{2} \mathbb{E}[\Delta w^\top H \Delta w]

为简化，假设 H = \lambda I（各向同性曲率），则：

\mathbb{E}[L(w + \Delta w)] \approx L(w) - \eta \frac{\|g\|^2}{\sqrt{\|g\|^2 + \sigma^2/B}} + \frac{\lambda \eta^2}{2} \left( \frac{\|g\|^2 + \sigma^2/B}{\|g\|^2 + \sigma^2/B} \right)

最后一项简化后为 \frac{\lambda \eta^2}{2}。

因此：

\mathbb{E}[L(w + \Delta w)] \approx L(w) - \eta \frac{\|g\|^2}{\sqrt{\|g\|^2 + \sigma^2/B}} + \frac{\lambda \eta^2}{2}

对 \eta 求导并令为零：

-\frac{\|g\|^2}{\sqrt{\|g\|^2 + \sigma^2/B}} + \lambda \eta_{\text{opt}} = 0

得：

\eta_{\text{opt}} = \frac{\|g\|^2}{\lambda \sqrt{\|g\|^2 + \sigma^2/B}} = \frac{1}{\lambda} \cdot \frac{1}{\sqrt{1 + \frac{\sigma^2}{B \|g\|^2}}}

代入 B_{\text{noise}} = \sigma^2 / \|g\|^2：

\eta_{\text{opt}} = \frac{1}{\lambda} \cdot \frac{1}{\sqrt{1 + \frac{B_{\text{noise}}}{B}}}

注意到上式在 B \ll B_{\text{noise}} 时 \eta_{\text{opt}} \propto \sqrt{B/B_{\text{noise}}}，而在 B \gg B_{\text{noise}} 时 \eta_{\text{opt}} \to 1/\lambda。但实际优化中，当 B > B_{\text{noise}} 时，噪声已低于信号，学习率不应继续增大，而应趋于饱和。

通过考虑更新稳定性条件（要求更新步长的相对波动不超过阈值）得到对称形式。引入最大学习率 \eta_{\max} = 1/\lambda（对应无噪声情况），并构造对称表达式：

\eta_{\text{opt}} = \frac{\eta_{\max}}{\frac{1}{2}\left( \sqrt{\frac{B_{\text{noise}}}{B}} + \sqrt{\frac{B}{B_{\text{noise}}}} \right)}

该形式满足：

当 B = B_{\text{noise}} 时，\eta_{\text{opt}} = \eta_{\max}
当 B \ll B_{\text{noise}} 时，\eta_{\text{opt}} \approx 2\eta_{\max} \sqrt{B/B_{\text{noise}}}，即之前推导出的平方根缩放关系
当 B \gg B_{\text{noise}} 时，\eta_{\text{opt}} \approx 2\eta_{\max} \sqrt{B_{\text{noise}}/B} ，最佳学习率不应该增大反而要减小
三、考虑learning rate schedulers (LRS) 的影响

尽管我们上文已经推导的比较完善了，但是还忽略了一个实际训练中存在的情况，即learning rate schedulers (LRS)，最常见的就是余弦 LRS，在余弦 LRS中，每一步的 learning rate 都会变化，所以计算出的最优 learning rate 可能也并未产生最佳的效果。

除了余弦 LRS，在当前的很多 LLM 都是使用Warmup-Stable-Decay（WSD）的 LRS，在WSD 的 stable 阶段，learning rate 是不变的，因此非常有助于分析，一下讨论过程来源于引文[2]。

在 1.3 小节中，我们发现在固定损失下，总数据消耗量 \(E\) 与优化步骤数 \(S\) 之间存在双曲线权衡，即

\left(\frac{E}{E_{\min}}-1\right)\left(\frac{S}{S_{\min}}-1\right) = 1 \tag{1}

然而在 WSD 学习率调度下，我们观察到不同批次大小的训练曲线L​(D)在练习过程中相交。具体而言，虽然在目标损失相对较高时关系式 E_1<E_2 成立，但一旦目标损失低于特定阈值，该关系式就会反转，得到 E_1>E_2 。这一观察结果与标准公式E​(S)所隐含的单调性直接矛盾。因此，这些实验结果表明，临界批次大小的基本原理在 WSD 范式的稳定阶段并不成立。

为了解决以上问题，引文[2]构建了一个专门针对 WSD 学习率调度算法的全新E​(S)理论框架。该框架将达到目标损失所需的数据消耗E建模为优化步骤S的函数，并将演化过程细致地分解为三个不同的阶段：

初始阶段：E与S−Sm​i​n成反比波动
过渡阶段：E表示为S的二次函数
渐近阶段：E 随 S 线性增加

函数表示为：
E(S)=\left\{\begin{array}{l} B_{-1} /\left(S-S_{\min }\right)+B_0, S_{\min }<S<S_1, \\ C\left(S-S_{o p t}\right)^2+E_{\min }, S_1<S<S_2, \\ A_1 S+A_0, S>S_2 . \end{array}\right.

基于新 E(S)，论文提出两个核心 Batch Size 指标，为动态调度提供理论依据：

B_{min} = A_1, \quad B_{opt} = \frac{E_{min}}{S_{opt}}

指标	核心含义	几何意义
B_{min}	达到目标损失的 最小 Batch Size 阈值：若 Batch Size < B_{min}，即使增加步数 S，也无法达到目标损失（梯度噪声过大）	线性阶段的斜率（A_1）
B_{opt}	最小化总数据消耗的 最优 Batch Size：使用 B_{opt} 训练时，数据效率最高（E=E_{min}）	原点到 E(S) 最小值点的斜率

实验发现无论模型规模（122M~1B）如何，B_{min} 和 B_{opt} 均随目标损失降低而单调增大。


物理直觉：训练后期（损失更低），模型需要更大的 Batch Size 来抑制梯度噪声，同时维持数据效率——这是动态 Batch Size 调度的核心理论依据。因此论文设计了 以数据量为基准的动态调度算法，核心思路是 “随训练数据量增加，逐步增大 Batch Size”




参考资料

[1] How Does Critical Batch Size Scale in Pre-training?

[2] How to Set the Batch Size for Large-Scale Pre-training?

[3] Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective

[4] Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective

[5] https://spaces.ac.cn/archives/10542

[6] An Empirical Model of Large-Batch Training

[7] One weird trick for parallelizing convolutional neural networks

[8] On the Generalization Benefit of Noise in Stochastic Gradient Descent

一万年来谁著史？三千里外欲封侯。 —— 李鸿章《入都》
