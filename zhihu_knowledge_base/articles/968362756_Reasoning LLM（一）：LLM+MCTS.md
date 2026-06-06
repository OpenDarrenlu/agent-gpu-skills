# Reasoning LLM（一）：LLM+MCTS

**作者**: 紫气东来​上海交通大学 工学硕士

**原文链接**: https://zhuanlan.zhihu.com/p/968362756

---

​
目录
收起
一、从 MCTS 和 UCB 谈起
1.1 MCTS 的思想及其原理
1.2 置信上界与选择决策
二、LLM 与 MCTS 结合的实现与讨论
2.1 LLM + MCTS 的范式讨论
2.2 LLM + MCTS v.s only LLM
2.3 LLM + MCTS 推理决策逻辑
2.4 MCTS 优化 LLM 训练
参考资料

随着 OpenAI o1 的横空出世，其在数学和编码方面已经达到人类博士的水平，并且引领着 Scaling Law 从 Pretraing 到 Inference 的转变。从认知科学的角度来说，人类的思考行为模式有两种：

系统1（快思考）：快速、自动和直观的，不耗精力的，而且往往是无意识的
系统2（慢思考）：深思熟虑的，有意识的，耗费精力的，需要集中注意力分析推理

当前 LLM 的大部分问答等场景都属于系统 1 的范畴，尽管也出现了 CoT 及其衍生版本，可以一定程度增加“思考”程度，但仍然是比较简单和浅层的。如何提高 LLM 的逻辑推理和决策规划能力将是 LLM 研究及应用的重要方面，本篇及后续系列将管窥蠡测，深入探讨 LLM Inference scaling law 的原理、范式、及其实现细节。

一、从 MCTS 和 UCB 谈起

从当前学术圈和开源社区的研究成果来看，LLM + MCTS 已成为一个热点，并在一些场景中取得不错的效果。其核心思想就是将 LLM 的知识和 MCTS 的决策优化能力结合起来，在此笔者将不厌其烦地从回顾 MCTS 的原理开始，逐步剖析整个过程。

1.1 MCTS 的思想及其原理

蒙特卡洛树搜索(Monte Carlo tree search, MCTS ) 是一种用于某些决策过程的启发式搜索算法，常用于解决博弈树问题，该算法是在线的，即动作选择和动作执行交错进行。MCTS 也常常和 RL 结合在一起，用来解决序列决策问题，最典型的案例即 DeepMind 的 AlphaGo。

MCTS 基于搜索空间的随机采样扩展搜索树，其基本过程是使用模拟来构建一棵树。已评估的状态存储在搜索树中。评估状态集是通过迭代以下四个步骤逐步构建的：

选择(Selection)：从根节点开始，选择连续的子节点，直到到达叶节点
拓展(Expansion)：除非最终到达的节点是终止状态，否则通过选择一个操作并使用操作结果创建新节点来扩展所选节点的子节点
模拟(Simulation)：选择一个新节点并执行 MDP 到终止状态的随机模拟
回溯(Backpropagation)：最后将节点的值反向传播到根节点，沿途使用期望值更新各个节点的值。

以上 4 个步骤按照顺序执行一次称之为 rollout。

接下来笔者将用更加严谨的语言来描述整个过程。以马尔可夫决策过程(MDP) 为例，状态-动作值函数为 Q(s,a) , 状态值函数为 V(s) ，动作集合为 A(s) ，状态转移函数为 P_a(s^\prime|s) ，奖励值为 r 。

1.1.1 节点定义及构建

首先逐步构建搜索树，每个树节点存储：

子节点和父节点（非必须）
回报值
访问次数
探索权重（非必须）

一个最简版本的初始化定义如下：

class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight

下图是一个典型的 Tree 的结构，其中白色节点表示状态节点，a~e 表示动作，黑色节点表示概率不确定性。

1.1.2 选择(Selection)

从代表当前状态的根节点开始，选择具有最高置信上限 (UCT) 分数的子节点。重复这个选择过程，直到到达一个叶节点。如下图中的红色路径

简单的代码实现如下所示：

    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)  # descend a layer deeper

1.1.3 拓展(Expansion)

从当前状态创建一组新的子节点，这些子节点中的每一个都代表一个状态，可以通过从当前状态采取操作来达到该状态。因此，子节点的数量等于从当前状态可以采取的可能动作的数量。为子节点创建节点后，选择其中一个子节点（通常是第一个子节点）作为新状态。如下图中的蓝色节点。

如果之前访问过当前的叶子节点，那么可以跳过扩展阶段，直接进入模拟阶段。

简单的代码实现如下所示：

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

1.1.4 模拟(Simulation)

从这个阶段我们想要的是估计当前状态有多好，即从当前状态中可以获得的最佳回报（或奖励）是多少。为此从当前状态开始进行模拟，直到终止，如下图中的粉色节点。进行模拟可能有几种方法：

随机动作：更简单的选项之一是选择随机动作，直到达到最终状态。问题在于，不同的行动可能会导致截然不同的状态。
使用策略选择动作：即将政策视为近似给定状态最佳行动的函数。创建此函数的一种方法是使用神经网络。

使用随机动作的模拟的代码如下：

    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        invert_reward = True
        while True:
            if node.is_terminal():
                reward = node.reward()
                return 1 - reward if invert_reward else reward
            node = node.find_random_child()
            invert_reward = not invert_reward

1.1.5 回溯(Backpropagation)

一旦获得了状态值的近似值，就必须更新一路上访问过的所有节点。因此，我们将获得的返回值添加到每个访问状态的值中。我们还将每个状态的访问计数加 1。如下图的绿色节点表示更新过程

简单的代码实现如下所示：

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

1.1.6 rollout

即将以上4个步骤完整执行一遍，代码如下所示：

    def do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration.)"
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

在 rollout 之后，还需要选择得分最高的action，即 \operatorname{argmax}_{a \in A(s)} Q\left(s_0, a\right) ，代码如下：

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        return max(self.children[node], key=score)

使用以上MCTS可以实现一个完整的井字游戏，完整代码见 AI_analysis/mcts/mcts_test.ipynb at main · ifromeast/AI_analysis。

1.2 置信上界与选择决策

在上一节中还有一个遗留问题，即 _select() 函数中的 _uct_select() , 这也是 MCTS 中的关键部分，即对探索(exploration)和利用(exploitation)的平衡。

在介绍 UCT(Upper Confidence Trees, 置信树上界) 之前需要先介绍 UCB(Upper Confidence Bound, 置信区间上界) ，UCB 说明的是在博弈树的每个节点中应选择(UCB1) \frac{w_i}{n_i}+c \sqrt{\frac{\ln N_i}{n_i}}\\ 值最大的节点。

其中 \frac{w_i}{n_i} 表示后继节点的平均值， n_i 表示节点 i 的访问次数， N_i 表示节点 i 的父节点的访问次数， c 为探索参数，理论值为 \sqrt{2} 。

其中加号前面是该节点当前的收益均值，后面的叫做bonus，本质上是均值的标准差。这个公式反映：均值越大，标准差越小，被选中的概率会越来越大，起到了exploit的作用；同时哪些被选次数较少的节点也会得到试验机会，起到了explore的作用。

UCT 算法是 MCTS 与 UCB1 策略的结合，即 UCT = MCTS+UCB1 ，其实现过程如下所示：

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])
        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(log_N_vertex / self.N[n])
        return max(self.children[node], key=uct)

以上即置信上界的基本原理与实现。为了严谨起见，接下来将尝试推导一下 UCB 公式，不感兴趣的话可以跳过。

该问题即不确定性条件下的优化问题，最典型的就是 The Multi-Armed Bandit Problem，目的是指导反复选择看起来具有高价值的行动。为什么？

[利用] 要么该行动确实是好的，使我们获得了良好的回报（最大化 return）
[探索] 要么该行动的回报很低，更新了我们对它的知识，导致我们随着时间的推移而减少选择它（最小化累积regret）。

在The Multi-Armed Bandit Problem中，对于每个臂 a，我们的目标是在高置信度下，估计置信上限 U_t(a) 即 Q(a)\leq U_t(a) 。然后算法在每个时间步选择具有最大 UCB 的动作。则 \begin{aligned} \operatorname{regret}(U C B, T) & =\sum_{t=1}^T\left(Q\left(a^*\right)-Q\left(a_t\right)\right) \\ & =\sum_{t=1}^T U_t\left(a_t\right)-Q\left(a_t\right)+Q\left(a^*\right)-U_t\left(a_t\right) \\ & \leq \sum_{t=1}^T U_t\left(a_t\right)-Q\left(a_t\right) \end{aligned}\\

我们可以将上界表示为 U_t\left(a_t\right)=\hat{Q}\left(a_t\right)+d ，其中 \hat{Q}\left(a_t\right) 表示估计值。

接下来确定上界的置信范围，首先需要使用 Chernoff-Hoeffding Bound 如下：

让 X_1,...,X_n 是独立同分布的随机变量[0,1]，且 \bar{X}_n=\frac{1}{n} \sum_{\tau=1}^n X_\tau 是样本均值，则有 P\left[E[X]>\bar{X}_n+u\right] \leq \exp \left(-2 n u^2\right)\\

现在我们假设置信值，令 P\left[Q\left(a_t\right)>\hat{Q}\left(a_t\right)+u\right] \leq \exp \left(-2 t u^2\right)=\frac{\delta}{t^2}\\ 即可解得 u=\sqrt{\frac{1}{n\left(a_t\right)} \log \left(t^2 / \delta\right)}\\ 这样我们就得到了置信上界 U_t\left(a_t\right)=\hat{Q}\left(a_t\right)+\sqrt{\frac{1}{n\left(a_t\right)} \log \left(t^2 / \delta\right)} 其成立的概率至少为 1-\frac{\delta}{t^2} 。之所以取 \frac{\delta}{t^2} 既是为了保证置信度较高，同时也保证了一定的探索性。

二、LLM 与 MCTS 结合的实现与讨论

在 OpenAI o1 发布前后，出现了众多的尝试复杂推理和决策规划的方案，例如 rStar，ReST-MCTS，OpenR，DeepSeek-Prover，ALPHALLM ，LLM-MCTS 等，不一而足。 在众多的方案中，LLM 与 MCTS 结合是当前最为主流的方案，由此在本节中，将专注于思考和讨论以下几个方面的问题：

为什么要将 LLM 与 MCTS 结合起来？
为什么 LLM 可以与 MCTS 结合起来？
LLM 要如何与 MCTS 有效结合起来？
LLM + MCTS 是终极方案么，有何局限性？
2.1 LLM + MCTS 的范式讨论

在现有的 LLM 复杂推理方法中，可以归纳出以下6种范式：

系统分析(Systematic Analysis，SA)。 从问题的整体结构出发，首先分析输入和输出，以及约束条件，然后决定算法的选择和数据结构的使用。
方法重用(Method Reuse，MR)。对于一些可以转化为经典问题的问题（比如最短路径或者背包问题），快速复用现有的方法来解决。
分而治之(Divide and Conquer，DC)。它将一个复杂的问题分解为子问题，并通过解决子问题来构造整体解决方案。
自我完善(Self-Refinement，SR)。 在推理过程中评估其推理过程，以确定是否存在任何问题并纠正任何错误。
上下文识别(Context Identification，CI)。对于一些需要额外信息输入的数据集， 首先总结与query相关的上下文的不同方面，然后给出相应query的回复。
强调约束（Emphasizing Constraints，EC）。对于一些对生成的文本有约束的数据集，通常在推理过程中强调相应的约束。

以上几种范式使用的频率也不尽相同，在一些测试中统计的频率分布如下：

尽管不同团队提出的关于 LLM + MCTS 的方案、场景、数据、目标等各有不同，但是仍然可以在其中发现较多的共同性，即范式。本节将以此为基点开始讨论，自顶向下地开始 LLM+MCTS 方案的思考、探究与实现。

2.2 LLM + MCTS v.s only LLM

回顾一下我们通常训练并使用 LLM 的过程：

通过大量数据的 Pretraining 过程向 LLM 灌输知识；
通过特定数据的 Finetuning(SFT, RLHF) 过程让 LLM 具有执行某些任务的能力；
通过用户输入的 Inference 过程完成某项功能的执行。

在这个过程中，LLM 既是世界模型，又是决策者。这样会导致一个问题，LLM 在推理过程倾向于系统1 (快思考)，其过程比较跳跃、缺少推导过程，呈现出来的结果就是幻觉。尽管泛 CoT 类方法可以一定程度上缓解这个问题，但其作用仍然是有限的。

CoT: LLM既是世界模型，又是决策者

当然为了将两种功能分开，可以将 LLM 作为决策者，同时引入世界模型（当然也可以是 LLM）

RAP：LLM 作为决策者，同时引入世界模型

另一个自然的想法就是，将 LLM 仅作为世界知识和生成器，而采用其他方法如 MCTS 作为决策者。这样会产生至少2个效果：

在训练过程中，MCTS 可以构造出更高质量的数据以供 LLM 训练；
在推理过程中，LLM 通过与 MCTS 的多步交互与迭代，以时间换正确率。
LLM-MCTS：LLM 作为世界模型，MCTS 作为决策者

下面我们将分别从推理和训练的角度来探究其基本范式。

2.3 LLM + MCTS 推理决策逻辑

在使用 LLM 结合 MCTS 实现复杂的推理的过程中，通常需要以下模块和环节：

候选生成。这个部分是由 LLM 承担的，即生成多个或多步候选答案；
评估判断。这部分是由 MCTS 来实现，即通过LLM 或者规则等方式验证判断某个答案是否符合预期，该步骤也可以称之为过程监督；
迭代交互。这部分由 LLM 与 MCTS 共同实现，即 LLM 先产生候选结果，MCTS 进行探索和评估，并反馈给 LLM 进行下一轮迭代。

具体而言，还有一些关键问题需要解决，如：

问题1 ：动作空间该如何构建？

在交互过程中，MCTS 的作用主要是将复杂问题拆分成多个或者多步执行的子问题，其依据是设定的动作空间，即树搜索的范围。而 LLM 的作用是根据指定的动作生成候选答案或者生成评估验证的反馈，这里的问题是LLM输出通常是无限、连续的动作空间。那么如何设置动作空间以保证既充分又完备呢，以下是一个 Action 集合的案例，共有A1~A5 5个基础动作：

A1（propose a one-step-thought）：步步推理，每一步都有一些中间答案，然后在最后一步中得到最终答案
A2（propose the remaining thought steps）：一次性推理完毕，直接得出最终答案
A3 (propose next sub-question along with its answer)：将原始问题拆解成若干子问题并做相关回答。最后一个子问题的答案即是最终答案（和A1有些类似，但采取的是subquestion-subanswer这种指示方式）
A4 (Answer the sub-question again)：有时A3中某个子问题的回答不一定可信，我们尝试重新回答它。这时我们会采用A2的模版，重新回答这个子问题
A5（Rephrase the question/sub-question）：重新复述一个原始问题/子问题。例如去掉大段文字表述信息，只把关键部分提取成condition1..., condition2之类的形式，用这个形式当作新的问题。

问题2 ：该如何进行过程验证和评估？

过程监督可以说是这个方案中关键步骤，无论生成的结果是否正确，无法有效评估和判断，则同样是没有价值的，而当前的过程评估方法大概有以下几种：

专家式：使用另一个更好的（当然也可以是本身）的 LLM 作为专家来判断生成的结果是否正确、合理
集成式：类似于集成学习的方法，即同时生成多个答案，通过答案之间的一致性来进行判断
奖励式：即专门训练一个 PRM (process reward model) 来为结果进行打分，其思想即源于 RL 的 reward model.

由此我们便获得一个通用的推理范式，其流程图如下，其中每个环节的关键步骤用红框标出，这也是不同算法所能调整和修改的部分，在之后的代码解读部分会具体说明不同算法的操作方法及其效果：

2.4 MCTS 优化 LLM 训练

使用 MCTS 优化 LLM 的训练，不同于 AlphaGo 中 MCTS+RL 的 online learning 的形式，在这里是一种 offline learning 的形式，即通过产生更加高质量的数据来训练 LLM。该部分有机会将会在之后的文章中详细讨论，在此不是本篇的重点。

参考资料

[1] MCTS meets LLMs: Enabling Complex Reasoning and Strategic Planning - inovex GmbH

[2] DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search

[3] SOCIAL MEDIA TITLE TAG

[4] Large Language Models as Commonsense Knowledge for Large-Scale Task Planning

[5] GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models

[6] OpenR: An Open Source Framework for Advanced Reasoning with Large Language Models

[7] https://github.com/THUDM/ReST-MCTS

[8] http://www.incompleteideas.net/609%20dropbox/other%20readings%20and%20resources/MCTS-survey.pdf

[10] https://courses.cs.washington.edu/courses/cse599i/18wi/resources/lecture19/lecture19.pdf

[11] rStar

[12] Solving Math Word Problems via Cooperative Reasoning induced Language Models

[13] bandit_simulations/python/multiarmed_bandits/analysis/ucb.md at master · kfoofw/bandit_simulations

[14] https://wensun.github.io/CS4789_data/UCB_note_new.pdf

[15] https://users.ece.cmu.edu/~yuejiec/ece18813B_notes/lecture2-stochastic-bandits.pdf

[16] Physics of Language Models: Part 2.1, Grade-School Math and the Hidden Reasoning Process

[17] Physics of Language Models: Part 2.2, How to Learn From Mistakes on Grade-School Math Problems

[18] LLM code gen

[19] AlphaZero-Like Tree-Search can Guide Large Language Model Decoding and Training

[20] 多臂老虎机UCB1算法推导-CSDN博客

[21] Proof of sublinear regret of UCB algorithms for bandits

[22] Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing

[23] A Comparative Study on Reasoning Patterns of OpenAI's o1 Model

[24] Q*: Improving Multi-step Reasoning for LLMs with Deliberative Planning

[25] Marco-o1: Towards Open Reasoning Models for Open-Ended Solutions

欲持一瓢酒，远慰风雨夕。落叶满空山，何处寻行迹。 —— 韦应物《寄全椒山中道士》
