# [ICLR'23] Bayes risk CTC: Controllable CTC alignment in Sequence-to-Sequence tasks

**作者**: 张宇月之暗面

**原文链接**: https://zhuanlan.zhihu.com/p/617714746

---

前言

Connectionist Temporal Classification (CTC)最初被提出于自动语音识别（ASR）领域，旨在解决语音输入和文本输出难以对齐的问题. 训练时CTC无须指定一条明确的语音和文本序列的对齐路径，相反考虑所有可能的对齐，这极大提升了ASR的性能，并且由于条件独立性假设，CTC可以保证极高的效率，因此自2006年提出至今仍是非自回归ASR方面的标准模型. 近年来在其他领域，CTC同样有了广泛的应用，例如机器翻译[1] [2]、文本摘要[3]、语音翻译[4]等等. 针对CTC有大量的改进方法，例如CTC作者本人在Graves et al. 2012[5]提出的RNN-Transducer，解决条件独立性问题，这里不一一列举.
本文要介绍的是最新发表于ICLR'23的一篇文章

Bayes risk CTC: Controllable CTC alignment in Sequence-to-Sequence tasks
openreview.net/forum?id=Bd7GueaTxUz

一个非常漂亮的关于CTC的拓展. 最直接的insight是作者注意到CTC在的训练过程中会考虑所有的合法对齐，即输出序列可以对应很多个CTC路径，并且每条路径都是平等对待的，这导致了预测CTC路径的不确定性，例如下面的例子中，预测的CTC对齐事实上和reference并不是完全对应的，这限制了CTC的应用.

因此作者提出需要我们显式地对CTC对齐进行控制. 作者提出了采用一个Bayes risk function在训练时给所有的CTC路径加权，从而使得预测的CTC路径具备我们需要的特性，这也是模型名字的由来：Bayes risk CTC (BRCTC). 关于risk function的定义多种多样，可以视实际需要而定，文中作者举了两个例子：

当seq2seq输入序列过长，可以使用BRCTC对序列进行降采样，缩短输入的长度，提升解码效率；
对于实时系统，可以定义关于latency的risk function，达成quality-latency的tradeoff；

问题的关键在于CTC所有可能的路径是指数级别的，如何对每条路径进行加权是一项挑战. 文章中作者非常机智的提出可以首先按照期望特性的value（例如latency是多少）对path先分组，然后将risk function与forward-backward算法相结合，非常高效地利用了动态规划进行计算.

Reviews on CTC

首先简单回顾一下CTC算法. 关于CTC的介绍文章已经很多了，对更多细节感兴趣的可以看看Hannun et al. 2017的经典博文，知乎上也有很多介绍，例如这篇，这里主要介绍文章的一些notation，以及CTC用到的forward-backward算法.

对于一个seq2seq任务（包括但不限于ASR、TTS、MT、ST），给定输入长度为T的序列\mathbf{x} = [x_1,\cdots,x_T]，我们的目标是得到长度为U的输出序列\mathbf{l} = [l_1,\cdots,l_U]. CTC尝试最大化P(\mathbf{l}\mid\mathbf{x}). CTC不对该式子直接建模，而是转而最大化\mathbf{l}对应的所有合法路径概率和，其中每条路径\pi=[\pi_1,\cdots,\pi_T]的长度都是T. CTC引入了一个空标签\emptyset，并定义了一个mapping函数\mathcal{B}，使得\mathcal{B}(\pi)=\mathbf{l}，即去除所有的\emptyset和连续重复非空字符，可以得到\mathbf{l}，e.g.，\mathcal{B}(\emptyset aa\emptyset abb)=aab. 相应地，最大化目标序列概率被转化为

P(\mathbf{l}\mid\mathbf{x})=\sum_{\pi\in\mathcal{B}^{-1}(\mathbf{l})}P(\pi\mid\mathbf{x})

其中\mathcal{B}^{-1}对应路径集合. CTC引入了条件独立性假设，也就是P(\pi\mid\mathbf{x})=\prod P_t(\pi_t\mid\mathbf{x})=\prod y_{\pi_t}^t.

前向后向算法

我们知道CTC包含了指数级的搜索空间，因此直接进行遍历是不现实的. 因此CTC通常借助于前向后向算法来完成高效的计算.

计算前首先讲目标序列进行转换，两两token间插入一个空标签\emptyset（细节参考Hannun et al. 2017）得到\mathbf{l}'=[\emptyset,l_1,\emptyset,l_2,\emptyset,\cdots,\emptyset,l_U,\emptyset]. 可以看到\mathbf{l}'的长度为2U+1.





定义1\leq t \leq T, 1 \leq v \leq 2U+1，我们可以分别得到前向和后向算法的递推式（示例图如上）：

\alpha(t, v) = \sum_{\pi:\mathcal{B}(\pi_{1:t})=\mathcal{B}(\mathbf{l}'_{1:v}),\pi_t=l'_v}\prod_{t'=1}^{t}y_{\pi_t'}^{t'}

\beta(t, v) = \sum_{\pi:\mathcal{B}(\pi_{t:T})=\mathcal{B}(\mathbf{l}'_{v:2U+1}),\pi_t=l'_v}\prod_{t'=t}^{T}y_{\pi_t'}^{t'}

很容易得到限定\pi_t为l'_v的边缘概率

P(\pi_t=l'_v\mid\mathbf{x})=\sum_{\pi\in\mathcal{B^{-1}(\mathbf{l}),\pi_t=l'_v}} P(\pi\mid\mathbf{x})=\frac{\alpha(t, v)\cdot\beta(t, v)}{y^t_{l'_v}}

Bayes risk CTC

对于CTC而言，目标序列\mathbf{l}从所有合法的路径\pi\in\mathcal{B}^{-1}\mathbf{l}得到，我们希望在获得\mathbf{l}的同时，也会得到一个合理的路径/对齐. 这在vanilla CTC中是无法办到的，因为每条合法路径都被视为等价，不被区分. 为了引入区分性，让预测路径可控，Bayes risk CTC定义了一个风险函数r(\pi)来定义希望得到的路径的特征，这样目标函数就变成了

J_{brctc}(\mathbf{l},\mathbf{x})=\sum_{\pi\in\mathcal{B}^{-1}(\mathbf{l})}[P(\pi\mid\mathbf{x})\cdot r(\pi)]

如何计算上式是一个关键的挑战. BRCTC提出首先对路径分组，定义f(\pi)=\tau为我们关心的路径属性的值，可以先将路径集合按照\tau划分，然后应用前向后向算法对每个子集求值，定义r_g(\tau)为对应属性值的风险，上式可以改写为

J_{brctc}=\sum_{\tau}\sum_{\pi: f(\pi)=\tau}[P(\pi\mid\mathbf{x})\cdot r(\pi)]=\sum_{\tau}[r_g(\tau)\cdot\sum_{\pi: f(\pi)=\tau}P(\pi\mid\mathbf{x})]

可以看到相同子集的path共享相同的风险r_g(\tau)，因此我们可以在上面将共有的r_g(\tau)提前.

路径划分的例子

一个路径划分的例子是根据某个非空的token l_u的终点划分，也就是该token最晚在哪个位置完成预测.
定义\tau = f_u(\pi) = \arg\max_t(\pi_t = l_u)，l_u对应的插入了\emptyset后新序列的token位置为l'_{2u}，注意到一个事实：当2u为终止点，那么必然有l'_{2u}\neq l'_{2u+1}，因此值为f_u(\pi)的路径概率可以这样推导

\sum_{\pi\in\mathcal{B}^{-1}(\mathbf{l})} P(\pi\mid\mathbf{x})= \sum_{\pi\in\mathcal{B}^{-1}(\mathbf{l}),\pi_{\tau}=l'_{2u},\pi_{\tau+1}\neq l'_{2u}} P(\pi\mid\mathbf{x})\\ =\sum_{\pi\in\mathcal{B}^{-1}(\mathbf{l}),\pi_{\tau}=l'_{2u}} P(\pi\mid\mathbf{x})-\sum_{\pi\in\mathcal{B}^{-1}(\mathbf{l}),\pi_{\tau}=l'_{2u},\pi_{\tau+1}=l'_{2u}} P(\pi\mid\mathbf{x})\\ =\frac{\alpha(\tau, 2u)\cdot\beta(\tau, 2u)}{y^{\tau}_{\pi_{\tau}}} - \alpha(\tau, 2u)\cdot t\beta(\tau+1, 2u)

更多变幻请参考Appendix C，最终的公式为

\sum_{\pi\in\mathcal{B}^{-1}(\mathbf{l})} P(\pi\mid\mathbf{x})= \frac{\alpha(\tau, 2u)\cdot\hat{\beta}(\tau, 2u)}{y^{\tau}_{\pi_{\tau}}}\\\mathrm{s.t.,}\quad\hat{\beta}(\tau, 2u) = \beta(\tau, 2u) - \beta(\tau + 1, 2u)\cdot y^{\tau}_{\pi_{\tau}} \quad\mathrm{if}\quad \tau < T\quad \mathrm{else}\quad \beta(\tau, 2u)

代入目标函数可以得到

J_{brctc}(\mathbf{l},\mathbf{x})=\sum_{\tau=1}^{T}r_g(\tau)\cdot \frac{\alpha(\tau, 2u)\cdot\hat{\beta}(\tau, 2u)}{y^{\tau}_{\pi_{\tau}}}

上式可以看到我们可以通过前向后向算法的结合算出特定属性的path子集概率和，遍历所有属性值就是最终的目标函数. 下面以降采样（down-sample）为例，说明如何设计的risk function.

对于ASR、ST而言，输入长度通常远大于输出长度，我们希望解码前先对序列进行裁剪以提高计算效率，这要求对结束token l_U的预测越早越好，因此定义r_g(\tau) = e^{-\lambda\cdot \tau/T}，表示希望\tau越小越好，相应的risk也就越大.
对应的目标函数为

J_{brctc}(\mathbf{l},\mathbf{x})=\sum_{\tau=1}^{T}e^{-\lambda\cdot \tau/T}\cdot \frac{\alpha(\tau, 2u)\cdot\hat{\beta}(\tau, 2u)}{y^{\tau}_{\pi_{\tau}}}

总结

本文提出了一个新颖的CTC拓展，通过设计risk function来控制CTC预测对齐的特征. 为了方便计算，本文提出将CTC路径按照特征划分，并通过前向后向算法合作，高效地计算出了期望风险函数. 作者举了两个例子：1）降采样；2）性能-效率权衡，设计了相应的risk function来说明了BRCTC的有效性.

参考
^Jindřich Libovický and Jindřich Helcl. 2018. End-to-End Non-Autoregressive Neural Machine Translation with Connectionist Temporal Classification. In Proceedings of EMNLP, pages 3016–3021, Brussels, Belgium. https://aclanthology.org/D18-1336/
^Chitwan Saharia, William Chan, Saurabh Saxena, and Mohammad Norouzi. 2020. Non-Autoregressive Machine Translation with Latent Alignments. In Proceedings of EMNLP, pages 1098–1108, Online. https://aclanthology.org/2020.emnlp-main.83/
^Puyuan Liu, Chenyang Huang, and Lili Mou. 2022. Learning Non-Autoregressive Models from Search for Unsupervised Sentence Summarization. In Proceedings of ACL, pages 7916–7929, Dublin, Ireland. https://aclanthology.org/2022.acl-long.545/
^Yan Brian, Dalmia Siddharth, Higuchi Yosuke, Neubig Graham, Metze Florian, Black Alan, W, and Watanabe Shinji. 2022. CTC alignments improve autoregressive translation. https://arxiv.org/abs/2210.05200
^Alex Graves. 2012. Sequence transduction with recurrent neural networks. https://arxiv.org/abs/1211.3711
