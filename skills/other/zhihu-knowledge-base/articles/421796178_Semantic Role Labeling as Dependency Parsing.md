# Semantic Role Labeling as Dependency Parsing

**作者**: 张宇月之暗面

**原文链接**: https://zhuanlan.zhihu.com/p/421796178

---

Introduction

语义角色标注（SRL）是NLP中一个基础且重要的任务，主要涉及谓词和论元的识别，以及相应的角色标签标注等等.

最近主流的SRL方法主要分为BIO-based和span-based. 前者将SRL视为序列标注，而后者则是将SRL视为对于<谓词，论元头，论元尾>这样三元组的预测. 然而这两种方法都有一些共有的缺陷，忽视了对于论元内部结构建模.

这种内部结构在直觉上对于SRL很有效，例如在上面的图中，谓词take对应的论元「out of the market」的标签为A2，这种关系可以反映在take到论元中心词out的弧中，此外，该论元的边界也和相应的子树边界完美对应. 如果捕捉到内部结构信息，可以有效引导角色标签分类以及论元识别这两个子任务. 然而由于SRL是一个shallow parsing task，缺乏层次化的结构标注，这种内部结构还很少被前人工作利用.

基于这些观察，我们提出将平坦论元结构建模为隐式（latent）依存子树. 通过这种方式，我们可以方便地将SRL归纳成一个依存句法分析任务. 基于这种归纳，我们可以无缝利用已有的一些成熟的依存句法分析技术，例如TreeCRF、高阶建模等等，来进行全局概率推断. 我们的方法不需要预先指定谓词以及依存句法树，因此是end-to-end的. 我们的代码将开源在https://github.com/yzhangcs/crfsrl.

Semantic Role Labeling as Dependency Parsing: Exploring Latent Tree Structures Inside Arguments
aclanthology.org/2022.coling-1.370
Methodology

我们的方法主要分为两个阶段：1）通过一定的规则将SRL结构转化为依存句法树；2）基于给定的依存句法树学习一个parser，然后通过后处理过程将预测出的dependency trees恢复为SRL结构.

上图给出了我们方法的主要步骤

SRL->Tree

首先是将SRL转化成树结构，图2b给了一个例子，对于谓词take，首先我们构建一条根到谓词的弧0->take，弧标签设为PRD，接着构建谓词到论元/非论元的子树. 对于一个像「to do more」这样的论元span，我们连接一条谓词take到该论元的弧，将论元标签A1设为这条弧的标签，剩下的内部的弧「to do more」我们不做任何假设，将这个部分视为未被realize的latent tree，允许任何连接，并且不分配标签. 对于非论元span，操作类似，除了我们将谓词到span的标签设为O（例如want->.）.
通过上面这种方式，我们将一个SRL图转化为了若干个以谓词为根的partially-observed trees.

Dependency parsing with span-constrained (second-order) TreeCRF

我们使用类似于经典Biaffine Parser的架构来学习上面转化得到的树，在打分器后面我们后接了一个TreeCRF来进行全局推断，最大化树概率，并进一步提出了一个带兄弟（siblings）信息的二阶拓展. 最终训练的目标函数如下

训练时我们将最大化SRL图 g 的概率近似为最大化上述转化得到的依存树概率，并对此按谓词分解，每个谓词对应的依存树概率为

上面的公式我们通过复杂度为 O(n^3) 的TreeCRF来计算，得到相应的树概率，其中latent subtree在训练过程中会被marginalize掉. 一个主要的问题是经典的TreeCRF考虑的是所有候选树，然而在我们的场景中引入了许多span的约束，要求转化出来的依存树应当满足SRL的图结构，而这些span constraints没法被典型TreeCRF达成.

有鉴于此，在本文中我们提出了一个span-constrained的TreeCRF，并将之推广到了二阶的场景，下图给出了相应的deduction rules.

Recovery

通过上面的方法得到一个句法分析器之后，我们剩下需要做的是利用该分析器预测句法树，并恢复为SRL图结构. 恢复过程非常简单

由于弧标签的概率分布和树结构独立，因此我们首先对0->i的弧进行分类，对于标签为PRD的弧，我们认为i是谓词，并解码出剩下的树结构.
从谓词i到其他词，我们认为他们是论元span的中心词，并以他们为起始，自底向下做遍历，将子树坍缩成一个平坦的谓词
最终我们收集所有形成的谓词及其论元，得到最终的SRL预测 g^{\prime} .
Experiments

我们在CoNLL05和CoNLL12两个基准数据集上做实验，下表给出了实验结果

在不给定谓词的场景下，我们的一阶方法CRF以及二阶方法CRF2o显著的超越了前人的结构，并且优势在CoNLL05 out-of-domain Brown数据上尤为显著. 在给定谓词场景下，CRF2o使用BERT之后在CoNLL05 Test上和现有最好的结果88.8相近，并在CoNLL12上达到87.57，显著超过了他们的86.5. 使用RoBERTa之后，CRF2o在三个数据上达到了89.63，83.72以及88.32的 F_1 值，达到了新的state-of-the-art.
