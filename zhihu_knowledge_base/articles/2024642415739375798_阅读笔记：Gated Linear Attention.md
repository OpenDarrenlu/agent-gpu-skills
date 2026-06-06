# 阅读笔记：Gated Linear Attention

**作者**: 半只兔子B乎兔子 https://mzeromiko.github

**原文链接**: https://zhuanlan.zhihu.com/p/2024642415739375798

---

原文链接： Gated Linear Attention Transformers with Hardware-Efficient Training
笔记链接：https://mzeromiko.github.io/blogs
声明：本文为个人阅读笔记。所有来自我自己的推导都可能存在错误，后续需要对照代码进行交叉验证。
一、动机

为线性注意力（Linear Attention）构造硬件高效的训练算法。

二、符号约定
使用 
𝑆
,
𝑄
 等粗体大写字母表示矩阵
使用 
𝑞
𝑡
,
𝑘
𝑡
 等表示列向量（即 
[
𝑑
,
1
]
 的形式），矩阵则是 
[
𝐿
,
𝑑
]
 的形式，因此会有额外的转置操作
使用 
𝑊
𝑡
 等表示可学习参数
使用 
𝑞
𝑡
 表示 
𝑄
 的第 
𝑡
 行
注意：本文与原论文不同。原文中所有向量均为行向量。因此本文所有公式都是重写版本，如有错误请指正。
三、背景知识
3.1 Self-Attention

𝑞
𝑡
,
𝑘
𝑡
,
𝑣
𝑡
	
=
𝑊
𝑄
𝑥
𝑡
,
𝑊
𝐾
𝑥
𝑡
,
𝑊
𝑉
𝑥
𝑡



𝑜
𝑡
=
∑
𝑖
=
1
𝑡
𝑣
𝑖
exp
⁡
(
𝑘
𝑖
⊤
𝑞
𝑡
)
∑
𝑖
=
1
𝑡
exp
⁡
(
𝑘
𝑖
⊤
𝑞
𝑡
)
	
⇔
𝑂
=
softmax
(
𝑄
𝐾
⊤
⊙
𝑀
)
𝑉



3.2 Linear Attention

𝑜
𝑡
	
=
∑
𝑖
=
1
𝑡
𝑣
𝑖
𝜙
(
𝑘
𝑖
)
⊤
𝜙
(
𝑞
𝑡
)
∑
𝑖
=
1
𝑡
𝜙
(
𝑘
𝑖
)
⊤
𝜙
(
𝑞
𝑡
)



𝑆
𝑡
	
=
∑
𝑖
=
1
𝑡
𝑣
𝑖
𝜙
(
𝑘
𝑖
)
⊤
∈
𝑅
𝑑
×
𝑑
,
𝑧
𝑡
=
∑
𝑖
=
1
𝑡
𝜙
(
𝑘
𝑖
)
∈
𝑅
𝑑
×
1



𝑆
𝑡
	
=
𝑆
𝑡
−
1
+
𝑣
𝑡
𝜙
(
𝑘
𝑡
)
⊤
,
𝑧
𝑡
=
𝑧
𝑡
−
1
+
𝜙
(
𝑘
𝑡
)
,
𝑜
𝑡
=
𝑆
𝑡
𝜙
(
𝑞
𝑡
)
𝑧
𝑡
⊤
𝜙
(
𝑞
𝑡
)
.

之前的工作发现，即使去掉核函数和归一化项，效果依然不错，即：

\mathbf{S}_t = \mathbf{S}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^\top , \quad \boldsymbol{o}_t = \mathbf{S}_t \boldsymbol{q}_t

3.3 分块线性注意力（Linear Attention + Chunkwise）

将序列 \mathbf{X} 切成若干互不重叠的 chunk，每个 chunk 长度为 C。定义：

\begin{aligned} \square_{[t]}^i = \square_{tC+i} ,\quad \square_{[t]} = \square_{[t]}^{1:C} \in \mathbb{R}^{C \times d} \quad \text{for } \square \in \{ \mathbf{Q, K, V, O} \} \end{aligned}

则有：

\begin{aligned} \mathbf{S}_{[t]}^{C} &= \mathbf{S}_{[t-1]}^{C} + \sum_{i=tC+1}^{tC+C} \boldsymbol{v}_i \boldsymbol{k}_i^\top \quad \in \mathbb{R}^{d \times d} \\ \\ \mathbf{O}_{[t]} &= \mathbf{Q}_{[t]} \mathbf{S}_{[t]} + \left( \mathbf{Q}_{[t]} \mathbf{K}_{[t]}^\top \odot \mathbf{M}\right) \mathbf{V}_{[t]} \end{aligned}

四、Flash Linear Attention
4.1 设计原则
需要充分利用 GPU 上的 SM
考虑 batch-size=1 的场景，因此需要在时间维度上做并行
使用 Tensor Core
采用分层显存设计，最优化利用 SRAM 和 HBM
块内并行，块间串行
4.2 算法

FLA 实现了两种分块（chunkwise）算法：

评论：纯串行方案（只在每个块内做并行，块间串行）看起来似乎也还可以？
五、Gated Linear Attention
5.1 递推模式（Recurrent Mode）

一般形式：

\begin{aligned} \mathbf{S}_t = \mathbf{G}_t \odot \mathbf{S}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^\top ,\quad \boldsymbol{o}_t = \mathbf{S}_t \boldsymbol{q}_t \end{aligned}

对于 GLA，门控矩阵具有低秩结构：

\begin{aligned} \mathbf{S}_t = ( \mathbf{1} \boldsymbol{\alpha}_t^\top) \odot \mathbf{S}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^\top = \mathbf{S}_{t-1}\text{Diag}(\boldsymbol{\alpha}_t) + \boldsymbol{v}_t \boldsymbol{k}_t^\top \end{aligned}

原文关键评论：

GLA 最核心的设计在于门控（Gate）的参数化需要在参数效率（parameter-efficiency）、状态大小（state size）和训练效率（training efficiency）三者之间取得平衡。
Mamba 中的 Gate 来自可学习矩阵 \mathbf{A} 与数据相关的 \boldsymbol{\alpha}_t 的组合，即 Gate 是一个满秩矩阵。然而这种设计无法使用 Tensor Core，因为它不能被改写为矩阵乘法的形式。为此 Mamba 设计了一种 prefix sum 算法来充分利用 SRAM。但由于 SRAM 容量有限，该方法无法扩展到更大的隐藏状态，从而在召回密集型任务（recall-intensive tasks）上表现欠佳。
5.2 分块递推模式（Chunkwise Recurrent Mode）

定义以下辅助变量：

\begin{aligned} \boldsymbol{\gamma}_{[t]}^r &= \prod_{i=tC+1}^{tC+r} \boldsymbol{\alpha}_i \in \mathbb{R}^{d \times 1} , \quad \\ \\ \mathbf{H}_{[t]}^{r} &= \sum_{i=1}^{r} (\boldsymbol{v}_{[t]}^{i} \boldsymbol{k}_{[t]}^{i\top}) \text{Diag}(\frac{\boldsymbol{\gamma}_{[t]}^{r}}{\boldsymbol{\gamma}_{[t]}^{i}}) \in \mathbb{R}^{d \times d} \\ \\ \mathbf{\Gamma}_{[t]} &= [ \boldsymbol{\gamma}_{[t]}^{1}, \boldsymbol{\gamma}_{[t]}^{2}, \dots, \boldsymbol{\gamma}_{[t]}^{C} ]^\top \in \mathbb{R}^{C \times d} \\ \\ \overleftarrow{\boldsymbol{q}_{[t]}^{i}} &= \boldsymbol{q}_{[t]}^{i} \odot \boldsymbol{\gamma}_{[t]}^{i} , \quad \overrightarrow{\boldsymbol{k}_{[t]}^{i}} = \frac{\boldsymbol{k}_{[t]}^{i}}{\boldsymbol{\gamma}_{[t]}^{i}} \\ \\ \overleftarrow{\mathbf{Q}_{[t]}} &= \mathbf{Q}_{[t]} \odot \mathbf{\Gamma}_{[t]} , \quad \overrightarrow{\mathbf{K}_{[t]}} = \mathbf{Q}_{[t]} \oslash \mathbf{\Gamma}_{[t]} \end{aligned}

那么有：

\begin{aligned} \mathbf{H}_{[0]}^{r} &= \mathbf{S}_{r} \\ \\ \mathbf{S}_{[t]}^{r} &= \mathbf{S}_{[t-1]}^{C} \text{Diag}(\boldsymbol{\gamma}_{[t]}^{r}) + \mathbf{H}_{[t]}^{r} \\ \\ \mathbf{H}_{[t]}^{r} &= \sum_{i=1}^{r} (\boldsymbol{v}_{[t]}^{i} \boldsymbol{k}_{[t]}^{i\top}) \text{Diag}(\frac{\boldsymbol{\gamma}_{[t]}^{r}}{\boldsymbol{\gamma}_{[t]}^{i}}) = \sum_{i=1}^{r} \boldsymbol{v}_{[t]}^{i}\left(\frac{\boldsymbol{k}_{[t]}^{i}}{\boldsymbol{\gamma}_{[t]}^{i}}\right)^{\top} \text{Diag}(\boldsymbol{\gamma}_{[t]}^{r}) \end{aligned}

进一步推导输出：

\begin{aligned} \boldsymbol{o}_{[t]}^{r} &= \mathbf{S}_{[t]}^{r} \boldsymbol{q}_{[t]}^{r} = \mathbf{S}_{[t-1]}^{C} \text{Diag}(\boldsymbol{\gamma}_{[t]}^{r}) \boldsymbol{q}_{[t]}^{r} + \mathbf{H}_{[t]}^{r} \boldsymbol{q}_{[t]}^{r} \\ \\ \Rightarrow \boldsymbol{o}_{[t]}^{r} &= \mathbf{S}_{[t-1]}^{C} \overleftarrow{\boldsymbol{q}_{[t]}^{r}} + \sum_{i=1}^{r} \boldsymbol{v}_{[t]}^{i} \left(\overrightarrow{\boldsymbol{k}_{[t]}^{i}}\right)^{\top} \overleftarrow{\boldsymbol{q}_{[t]}^{r}} \end{aligned}

写成矩阵形式：

\begin{aligned} \mathbf{O}_{[t]} &= \overleftarrow{\boldsymbol{Q}_{[t]}} \mathbf{S}_{[t-1]}^{C \top} + \left( \overleftarrow{\mathbf{Q}_{[t]}} \left(\overrightarrow{\mathbf{K}_{[t]}} \right)^\top \odot \mathbf{M} \right) \mathbf{V}_{[t]} \end{aligned}

其中，状态 \mathbf{S}_{[t]}^C 可以根据下式提前递推计算：

\begin{aligned} \mathbf{S}_{[t]}^{C} = \mathbf{S}_{[t-1]}^{C} \text{Diag}(\boldsymbol{\gamma}_{[t]}^{C}) + \mathbf{H}_{[t]}^{C} = \left(\mathbf{S}_{[t-1]}^{C} + \mathbf{V}_{[t]}^\top \overrightarrow{\boldsymbol{K}_{[t]}} \right) \text{Diag}(\boldsymbol{\gamma}_{[t]}^{C}) \end{aligned}

5.3 带二级分块的分块递推模式

对于较大的 chunk-size，\overleftarrow{\mathbf{Q}_{[t]}} (\overrightarrow{\mathbf{K}_{[t]}})^\top 中的衰减可能过大从而导致精度损失。GLA 提出将 chunk 进一步划分为 sub-chunk，对跨度较大的部分在 log 域计算衰减。

引入变量 \mathbf{P}_{[t][\tau]}，假设 sub-chunk 长度为 T：

\mathbf{P}_{[t][\tau]} = \overleftarrow{\boldsymbol{Q}_{[t]}} \left(\overrightarrow{\boldsymbol{K}_{[\tau]}} \right)^\top \odot \mathbf{M}_{[t][\tau]}

分为三种情况：

情况 1（粉色部分）：对角 sub-chunk。该部分对精度要求较高，因此采用逐元素全精度计算。
情况 2（橙色部分）：非对角 sub-chunk。该部分使用半精度矩阵运算，每个 sub-chunk 内单独计算。
情况 3（灰色部分）：只在并行模式下才需要计算，在 chunkwise 模式下不需要。
评论：如果需要计算情况 3，其计算方式与橙色部分相同。

对于情况 1：

\begin{aligned} (\mathbf{P}_{[t][\tau]})_{i, j} = \sum_{d} (\boldsymbol{q}_{[t]}^{i})_{d} ~(\boldsymbol{k}_{[\tau]}^{j})_{d} ~ \exp(\log \boldsymbol{\gamma}_{[t]}^{i} - \log \boldsymbol{\gamma}_{[\tau]}^{j} ) ，\quad t=\tau, i>j \end{aligned}

对于情况 2（注意这不是对角块）：

\begin{aligned} \mathbf{P}_{[t][\tau]} &= \overleftarrow{\boldsymbol{Q}_{[t]}} \left(\overrightarrow{\boldsymbol{K}_{[\tau]}} \right)^\top ,\quad t \ne \tau \\ \Rightarrow \mathbf{P}_{[t][\tau]} &= \left(\boldsymbol{Q}_{[t]} \odot \exp(\log \boldsymbol{\gamma}_{[t]}^{1:T}) \right) \left(\boldsymbol{K}_{[\tau]} \odot \exp(-\log \boldsymbol{\gamma}_{[\tau]}^{1:T}) \right)^\top ,\quad t > \tau \end{aligned}

六、GLA 反向传播
6.1 递推模式

前向：

\begin{aligned} \mathbf{S}_t = \mathbf{G}_t \odot \mathbf{S}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^\top ,\quad \boldsymbol{o}_t = \mathbf{S}_t \boldsymbol{q}_t \end{aligned}

反向：

\begin{aligned} \delta \boldsymbol{o}_t &= \frac{\partial L}{\partial \boldsymbol{o}_t} \\ \\ \delta \mathbf{G}_t &= \frac{\partial L}{\partial \mathbf{G}_t} = \delta \mathbf{S}_t \odot \mathbf{S}_{t-1} \\ \\ \delta \mathbf{S}_t &= \mathbf{G}_{t+1} \odot \delta \mathbf{S}_{t+1} + \delta \boldsymbol{o}_t \boldsymbol{q}_t^\top \\ \\ \delta \boldsymbol{q}_t &= \frac{\partial L}{\partial \boldsymbol{q}_t} = \mathbf{S}_t^\top \delta \boldsymbol{o}_t \\ \\ \delta \boldsymbol{v}_t &= \frac{\partial L}{\partial \boldsymbol{v}_t} = \delta \mathbf{S}_t \boldsymbol{k}_t \\ \\ \delta \boldsymbol{k}_t &= \frac{\partial L}{\partial \boldsymbol{k}_t} = \delta \mathbf{S}_t^\top \boldsymbol{v}_t \end{aligned}

GLA 特有的反向：

\begin{aligned} \delta \boldsymbol{\alpha}_t = \delta \mathbf{G}_t^\top \mathbf{1} \end{aligned}

6.2 数学预备知识

1. 定义

\begin{aligned} \mathbf{A} \in \mathbb{R}^{m \times n} , \quad \mathbf{B} \in \mathbb{R}^{n \times k} , \quad \mathbf{C} = \mathbf{A} \mathbf{B} , \quad y = f(\mathbf{C}) , \quad \delta \mathbf{C} := \frac{\partial y}{\partial \mathbf{C}} \end{aligned}

2. 矩阵迹的性质

\begin{aligned} \text{Tr}(ABC) &= \text{Tr}(BCA) = \text{Tr}(CAB) \\ \\ \text{Tr}(A^\top (B \odot C)) &= Tr((A \odot B)^\top C)=Tr((A \odot C)^\top B) \end{aligned}

3. 常用微分公式

\begin{aligned} d(\mathbf{A}\mathbf{B}) &= (\mathbf{A} (d \mathbf{B}) + (d \mathbf{A}) \mathbf{B}) \\ \\ d(\mathbf{A} \odot \mathbf{B}) &= (\mathbf{A} \odot (d \mathbf{B}) + (d \mathbf{A}) \odot \mathbf{B}) \end{aligned}

4. 矩阵乘法的梯度

\begin{aligned} dy = \text{Tr}\left( (\frac{\partial y}{\partial \mathbf{C}})^\top d \mathbf{C}\right) &= \text{Tr}\left( (\delta \mathbf{C})^\top (d \mathbf{C})\right) = \text{Tr}\left( (\delta \mathbf{C})^\top (\mathbf{A} (d \mathbf{B}) + (d \mathbf{A}) \mathbf{B}) \right) \\ \\ \text{while}\quad dy = \text{Tr}\left( (\frac{\partial y}{\partial \mathbf{B}})^\top (d \mathbf{B})\right) &\quad \text{so we have}\quad \delta \mathbf{B} = \mathbf{A}^\top \delta \mathbf{C} , \quad \delta \mathbf{A} = \delta \mathbf{C} \mathbf{B}^\top \end{aligned}

5. Hadamard 积的梯度

\begin{aligned} dy = \text{Tr}\left( (\delta \mathbf{D})^\top (d \mathbf{D})\right) &= \text{Tr}\left( (\delta \mathbf{C})^\top (\mathbf{A} \odot (d \mathbf{B}) + (d \mathbf{A}) \odot \mathbf{B}) \right) \\ \\ &= \text{Tr}\left( (\delta \mathbf{C} \odot \mathbf{A})^\top (d \mathbf{B}) + (\delta \mathbf{C} \odot \mathbf{B})^\top (d \mathbf{A}) \right) \\ \\ \text{while}\quad dy = \text{Tr}\left( (\delta \mathbf{B})^\top (d \mathbf{B})\right) &\quad \text{so we have}\quad \delta \mathbf{B} = \delta \mathbf{C} \odot \mathbf{A} , \quad \delta \mathbf{A} = \delta \mathbf{C} \odot \mathbf{B} \end{aligned}

6.3 分块递推模式的反向传播
评论：我觉得与其从递推模式的反向传播开始推导，不如直接从分块递推的前向公式出发推导更为简单。

回顾前向的重要结论：

\begin{aligned} \mathbf{O}_{[t]} &= \overleftarrow{\boldsymbol{Q}_{[t]}} \mathbf{S}_{[t-1]}^{C \top} + \left( \overleftarrow{\mathbf{Q}_{[t]}} \left(\overrightarrow{\mathbf{K}_{[t]}} \right)^\top \odot \mathbf{M} \right) \mathbf{V}_{[t]} \\ \\ \mathbf{S}_{[t]}^{C} &= \left(\mathbf{S}_{[t-1]}^{C} + \mathbf{V}_{[t]}^\top \overrightarrow{\boldsymbol{K}_{[t]}} \right) \text{Diag}(\boldsymbol{\gamma}_{[t]}^{C}) \end{aligned}

对于 \delta \mathbf{S}_{[t]}^{C}

\begin{aligned} \left.\delta \mathbf{S}_{[t-1]}^{C}\right|_{\text {from } \mathbf{O}_{[t]}} &= \delta \mathbf{O}_{[t]}^{\top} \overleftarrow{\mathbf{Q}}_{[t]} ,\quad \left.\delta \mathbf{S}_{[t-1]}^{C}\right|_{\text {from } \mathbf{S}_{[t]}^C} = \delta \mathbf{S}_{[t]}^C \text{Diag}(\boldsymbol{\gamma}_{[t]}^{C}) \\ \\ \Rightarrow \delta \mathbf{S}_{[t]}^{C} &= \delta \mathbf{S}_{[t+1]}^C \text{Diag}(\boldsymbol{\gamma}_{[t+1]}^{C}) + \delta \mathbf{O}_{[t+1]}^{\top} \overleftarrow{\mathbf{Q}}_{[t+1]} \end{aligned}

对于 \delta \mathbf{V}_{[t]}

\begin{aligned} \left.\delta \mathbf{V}_{[t]}\right|_{\text {from } \mathbf{O}_{[t]}} &= \left(\left(\overrightarrow{\mathbf{K}_{[t]}} \right) \overleftarrow{\mathbf{Q}_{[t]}}^\top \odot \mathbf{M}^\top \right) \delta \mathbf{O}_{[t]} \\ \\ \left.\delta \mathbf{V}_{[t]}\right|_{\text {from } \mathbf{S}_{[t]}^C} &= \left(\overrightarrow{\mathbf{K}_{[t]}} \right) \text{Diag}(\boldsymbol{\gamma}_{[t]}^{C}) \delta \mathbf{S}_{[t]}^{C \top} \\ \\ \Rightarrow \delta \mathbf{V}_{[t]} &= \left(\overrightarrow{\mathbf{K}_{[t]}} \right) \text{Diag}(\boldsymbol{\gamma}_{[t]}^{C}) \delta \mathbf{S}_{[t]}^{C \top} + \left(\left(\overrightarrow{\mathbf{K}_{[t]}} \right) \overleftarrow{\mathbf{Q}_{[t]}}^\top \odot \mathbf{M}^\top \right) \delta \mathbf{O}_{[t]} \end{aligned}

对于 \delta \overrightarrow{\mathbf{K}_{[t]}}

\begin{aligned} \left.\delta \overrightarrow{\mathbf{K}_{[t]}}\right|_{\text {from } \mathbf{O}_{[t]}} &= \left( \mathbf{V}_{[t]} \left(\delta \mathbf{O}_{[t]} \right)^\top \odot \mathbf{M}^\top \right) \overleftarrow{\mathbf{Q}_{[t]}} \\ \\ \left.\delta\overrightarrow{\mathbf{K}_{[t]}}\right|_{\text {from } \mathbf{S}_{[t]}^C} &= \mathbf{V}_{[t]} \delta \mathbf{S}_{[t]}^C \text{Diag}(\boldsymbol{\gamma}_{[t]}^{C}) \\ \\ \Rightarrow \delta\overrightarrow{\mathbf{K}_{[t]}} &= \mathbf{V}_{[t]} \delta \mathbf{S}_{[t]}^C \text{Diag}(\boldsymbol{\gamma}_{[t]}^{C}) + \left( \mathbf{V}_{[t]} \left(\delta \mathbf{O}_{[t]} \right)^\top \odot \mathbf{M}^\top \right) \overleftarrow{\mathbf{Q}_{[t]}} \end{aligned}

对于 \delta \overleftarrow{\mathbf{Q}_{[t]}}

\begin{aligned} \delta \overleftarrow{\mathbf{Q}_{[t]}} = \left.\delta \overleftarrow{\mathbf{Q}_{[t]}}\right|_{\text {from } \mathbf{O}_{[t]}} = \delta \mathbf{O}_{[t]} \mathbf{S}_{[t-1]}^C + \left( \delta\mathbf{O}_{[t]}\mathbf{V}_{[t]}^\top \odot \mathbf{M} \right) \overrightarrow{\mathbf{K}_{[t]}} \end{aligned}

对于 \delta \mathbf{K}_{[t]}

\begin{aligned} \delta \mathbf{K}_{[t]} = \delta \overrightarrow{\mathbf{K}_{[t]}} \oslash \mathbf{\Gamma}_{[t]} \end{aligned}

对于 \delta \mathbf{Q}_{[t]}

\begin{aligned} \delta \mathbf{Q}_{[t]} = \delta \overleftarrow{\mathbf{Q}_{[t]}} \odot \mathbf{\Gamma}_{[t]} \end{aligned}

对于 \delta \mathbf{\Gamma}_{[t]}

\begin{aligned} \left.\delta \mathbf{\Gamma}_{[t]}\right|_{\text {from } \mathbf{S}_{[t]}^C} &= \left[0,0,..., \text{diag}\left(\left(\mathbf{S}_{[t-1]}^{C \top} + \overrightarrow{\boldsymbol{K}_{[t]}}^\top \mathbf{V}_{[t]} \right) \delta \mathbf{S}_{[t]}^{C}\right) \right]^\top \\ \\ \delta \mathbf{\Gamma}_{[t]} &= \delta \overleftarrow{\mathbf{Q}_{[t]}} \odot \mathbf{Q}_{[t]} - \delta \overrightarrow{\mathbf{K}_{[t]}} \odot \mathbf{K}_{[t]} \oslash (\mathbf{\Gamma}_{[t]} \odot \mathbf{\Gamma}_{[t]}) + \left.\delta \mathbf{\Gamma}_{[t]}\right|_{\text {from } \mathbf{S}_{[t]}^C} \end{aligned}

或者等价地写为：

\begin{aligned} \delta \mathbf{\Gamma}_{[t]} \odot \mathbf{\Gamma}_{[t]} &= \delta \mathbf{Q}_{[t]} \odot \mathbf{Q}_{[t]} - \delta \mathbf{K}_{[t]} \odot \mathbf{K}_{[t]} + \left.\delta \mathbf{\Gamma}_{[t]}\right|_{\text {from } \mathbf{S}_{[t]}^C} \odot \mathbf{\Gamma}_{[t]} \\ \\ \left.\delta \mathbf{\Gamma}_{[t]}^C\right|_{\text {from } \mathbf{S}_{[t]}^C} \odot \mathbf{\gamma}_{[t]}^C &= \text{diag}\left(\mathbf{S}_{[t-1]}^{C \top} \delta \mathbf{S}_{[t]}^{C}\right) \odot \mathbf{\gamma}_{[t]}^C +\text{diag}\left(\overrightarrow{\boldsymbol{K}_{[t]}}^\top \mathbf{V}_{[t]} \delta \mathbf{S}_{[t]}^{C} \text{Diag}(\boldsymbol{\gamma}_{[t]}^{C}) \right) \\ &= \text{diag}\left(\mathbf{S}_{[t-1]}^{C \top} \delta \mathbf{S}_{[t]}^{C}\right) \odot \mathbf{\gamma}_{[t]}^C +\text{diag}\left(\overrightarrow{\boldsymbol{K}_{[t]}}^\top \left.\delta\overrightarrow{\mathbf{K}_{[t]}}\right|_{\text {from } \mathbf{S}_{[t]}^C}\right) \\ &= \text{diag}\left(\mathbf{S}_{[t-1]}^{C \top} \delta \mathbf{S}_{[t]}^{C}\right) \odot \mathbf{\gamma}_{[t]}^C +\text{diag}\left(\boldsymbol{K}_{[t]}^\top \left.\delta \mathbf{K}_{[t]}\right|_{\text {from } \mathbf{S}_{[t]}^C}\right) \end{aligned}

对于 \delta \boldsymbol{\alpha}_{[t]}

\begin{aligned} \mathbf{\Gamma}_{[t]} &= [ \prod_{i=tC+1}^{tC+1} \boldsymbol{\alpha}_i, \prod_{i=tC+1}^{tC+2} \boldsymbol{\alpha}_i,...\prod_{i=tC+1}^{tC+C} \boldsymbol{\alpha}_i]^\top \in \mathbb{R}^{C \times d} \\ \\ \delta \boldsymbol{\alpha}_r &= \sum_{j \ge r} \delta \mathbf{\Gamma}_{j,:} \odot (\prod_{i=tC+1}^{tC+j} \boldsymbol{\alpha}_i \oslash \boldsymbol{\alpha}_r) = \left(\sum_{j \ge r} \delta \mathbf{\Gamma}_{j,:} \odot \mathbf{\Gamma}_{j,:} \right) \oslash \boldsymbol{\alpha}_r \\ \\ \delta \mathbf{A}_{[t]} &= [\delta \boldsymbol{\alpha}_{[t]}^1, \delta \boldsymbol{\alpha}_{[t]}^2,...\delta \boldsymbol{\alpha}_{[t]}^C]^\top = \text{suffix\_sum}_{row}(\delta \mathbf{\Gamma} \odot \mathbf{\Gamma}) \oslash \mathbf{A}_{[t]} \end{aligned}

或者在 log 域下写为：

\begin{aligned} \log \mathbf{\Gamma}_{[t]} &= [ \sum_{i=tC+1}^{tC+1} \log \boldsymbol{\alpha}_i, \sum_{i=tC+1}^{tC+2} \log \boldsymbol{\alpha}_i,...\sum_{i=tC+1}^{tC+C} \log \boldsymbol{\alpha}_i]^\top \in \mathbb{R}^{C \times d} \\ \\ \Rightarrow \delta \log \boldsymbol{\alpha}_r &= \sum_{j \ge r} \delta \log \mathbf{\Gamma}_{j,:} \end{aligned}

七、网络架构
7.1 Token-Mixing

\begin{aligned} \boldsymbol{\alpha}_{t} = \sigma\left(\left(\mathbf{W}_{\alpha}^{1} \mathbf{W}_{\alpha}^{2} \boldsymbol{x}_{t} + \boldsymbol{b}_{\alpha}\right)\right)^{\frac{1}{\tau}} &\in \mathbb{R}^{d_{k} \times 1} \\ \\ \mathbf{S}_{t}^{h} = \left( \boldsymbol{\alpha}_{t}^{h}\mathbf{1} \right) \odot \mathbf{S}_{t-1}^{h} + \boldsymbol{v}_{t}^{h}\boldsymbol{k}_{t}^{h \top} &\in \mathbb{R}^{d_{v}^{\prime} \times d_{k}^{\prime}} \\ \\ \boldsymbol{o}_{t}^{h} = \mathbf{S}_{t}^{h} \boldsymbol{q}_{t}^{h} &\in \mathbb{R}^{d_{v}^{\prime} \times 1} \\ \\ \boldsymbol{o}_{t}^{\prime} = \operatorname{concat}\left(\operatorname{LN}\left(\boldsymbol{o}_{t}^{1}\right), \dots, \operatorname{LN}\left(\boldsymbol{o}_{t}^{H}\right)\right) &\in \mathbb{R}^{d_{v} \times 1} \\ \\ \boldsymbol{r}_{t} = \operatorname{Swish}\left(\mathbf{W}_{r} \boldsymbol{x}_{t} + \boldsymbol{b}_{r}\right) &\in \mathbb{R}^{d_{v} \times 1} \\ \\ \boldsymbol{y}_{t} = \mathbf{W}_{O} \left(\boldsymbol{r}_{t} \odot \boldsymbol{o}_{t}^{\prime}\right) &\in \mathbb{R}^{d \times 1} \end{aligned}

其中：

\begin{aligned} \mathbf{W}_{\alpha}^{1} \in \mathbb{R}^{d \times 16} ,\quad \mathbf{W}_{\alpha}^{2} \in \mathbb{R}^{16 \times d_{k}} , \quad \tau = 16 , \quad d_{k} = \frac{d}{2} , \quad d_{v} = d \\ \\ (\mathbf{W}_{Q}, \mathbf{W}_{K}, \mathbf{W}_{V}, \mathbf{W}_{O}, \mathbf{W}_{r}) \in \text{Full Rank} \end{aligned}

7.2 Channel-Mixing

\begin{aligned} \operatorname{SwiGLU}(\mathbf{Z}) = \left(\operatorname{Swish}(\mathbf{Z} \mathbf{W}_1) \odot \mathbf{Z} \mathbf{W}_2\right) \mathbf{W}_3 \end{aligned}

7.3 单层网络配置

\begin{aligned} \mathbf{Y}^{(l)} &= \operatorname{GLA}\left(\operatorname{LN}\left(\mathbf{X}^{(l)}\right)\right) + \mathbf{X}^{(l)} \\ \\ \mathbf{X}^{(l+1)} &= \operatorname{SwiGLU}\left(\operatorname{LN}\left(\mathbf{Y}^{(l)}\right)\right) + \mathbf{X}^{(l)} \end{aligned}

最终 GLA 层大约占用 4d^2 个参数，与标准 Attention 层对齐。

八、实验

1. 数据与分词

数据集：SlimPajama dataset
分词器：Mistral tokenizer
使用了 100B token 的子集

2. 对比方法

Transformer++：包含 RoPE、SwiGLU、RMSNorm 的 LLaMA 变体
RetNet：其中 FFN 层被替换为 SwiGLU

3. 训练配置

模型规模：340M 和 1.3B
优化器：AdamW
340M 模型：训练 15B tokens，batch-size = 0.5M tokens，warmup = 0.5B tokens
1.3B 模型：训练 100B tokens，batch-size = 2M tokens，warmup = 1B tokens
学习率 3e-5，weight-decay = 0.01，gradient clipping = 1.0

4. 评测：使用 lm-eval

5. 召回任务

召回任务通常被认为是线性注意力表现较差的方向，它要求模型回忆之前见过的精确信息。

6. 长序列训练

采用两种训练模式：

模式 a：直接用 8K 长度训练
模式 b：以 2K 长度为一个 segment，共 12 个 segment，训练 24K 长度。segment 之间梯度不反传。

测试时按分段计算 PPL：




7. GLA 消融实验
