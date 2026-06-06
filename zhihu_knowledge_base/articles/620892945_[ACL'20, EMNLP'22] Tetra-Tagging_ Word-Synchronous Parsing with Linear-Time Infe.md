# [ACL'20, EMNLP'22] Tetra-Tagging: Word-Synchronous Parsing with Linear-Time Inference

**作者**: 张宇月之暗面

**原文链接**: https://zhuanlan.zhihu.com/p/620892945

---

These things always come and go. The cool kids always care about structure :-)
-- Ryan Cotterell
前言

介绍一篇由Kitaev and Klein. (2020)提出的比较出彩的将成分句法分析（Constituency Parsing）转化为序列标注的文章：
Tetra-Tagging: Word-Synchronous Parsing with Linear-Time Inference.[1]

Parsing as Tagging的文章很多了，不同于CKY算法，序列标注的好处在于在解码时仅需要线性复杂度就可以解析出合法树结构，但是以前工作的缺点在于树结构转换出的标签空间比较大，并不是很simple，结果也比较差.
Kitaev and Klein. (2020)这篇的优点在于其Tagger的标签空间大小仅为4（因此命名为Tetra Tagger），与此同时，以BERT-large为底座训练的模型在PTB test上的性能达到了95.44，真正做到了Simple yet effective.

这篇ACL是一篇短文，空间限制有些地方比较语焉不详（比如解码，训练超参等等），后来Amini and Cotterell. (2022)[2]将Tetra Tagger与传统的In-order shift-reduce parser联系了起来，里面也提供了Tetra Tagger更多的细节，感兴趣的读者可以看这篇.
笔者已经将Tetra Tagger集成到了SuPar中，欢迎复现~

方法
树 
→
 标签

事实上方法部分用一幅图就可以说明，首先以二叉化的无标签句法树为例，后面会拓展到N-ary和有标签情况：

我们可以用叶子结点和非叶子结点对应的孩子方向表示树结构，将方向用标签表达，总结起来就是下面四种情况：

↗
: 叶子结点是一个左孩子
↖
 : 叶子结点是一个右孩子
⇒
 （方向右上）: 非叶子结点是一个左孩子
⇐
 （方向左上）: 非叶子结点是一个右孩子

一棵二叉树有
𝑁
个叶子结点和
𝑁
−
1
个非叶子结点，因此可以用
2
𝑁
−
1
个方向标签标识一棵树，也就是句子里每个词对应两个标签，除了最后一个，最后一个词只对应一个向右的标签.
下面是一个树转化成标签序列的例子

模型

Tetra Tagger采用BERT-large为底座训练模型，参数部分除了BERT只多了两个线性层，分别用于对叶子结点的标签和非叶子结点的标签分类.

训练时，最大化树概率的目标被转化成了最大化2N-1个标签的概率，假设P(l^{leaf}_i \mid\bf{x})和 P(l^{node}_i \mid \bf{x}) 分别表示第i个位置叶子结点和非叶子结点的概率，那么损失函数为

\mathcal{L} = -\log P(\bf{y}\mid\bf{x}) = -\left[\left(\sum_{i=1}^{N} \log P(l^{leaf}_i \mid\bf{x}) + \log P(l^{node}_i \mid \bf{x})\right) + \log P(l^{leaf}_N \mid\bf{x})\right]

标签 \rightarrow 树

训练完一个Tagger，解码时需要将标签转换回树结构. 和shift-reduce parser类似，这里需要引入stack/buffer结构，根据四个方向标签执行动作




以上图为例：

\nearrow ：buffer中取出一个叶子，结点插入到stack
\nwarrow ：buffer中取出一个叶子，和stack最后一个元素的空槽合并（图中用\emptyset表示）
\Rightarrow （方向右上）：取出stack中最后一个元素，成为一个新的非叶子结点左孩子，重新放入stack中，注意非叶子结点还没找到右孩子，因此有一个空槽\emptyset
\Leftarrow （方向左上）：取出stack中最后一个元素，成为一个新的非叶子结点的孩子，再将这个非叶子结点作为stack剩下的最后一个元素的右孩子，同样，非叶子结点还没找到右孩子，因此有一个空槽\emptyset

问题在于，这里没法直接贪心预测，因为一棵树可以对应一个唯一的标签序列，但是一个标签序列却并不一定能恢复成一棵合法的树. 最简单的例子，如果一直预测左向的标签，那么就没有右孩子出现.

作者观察到要得到合法的树，首先第一个动作必须是左向标签，插入一个叶子，并且执行完最后一个动作后，栈中必须只剩下1个元素，表示所有叶子已插入，并且已经和所有其他非叶子结点组成了一棵完整树了.
有鉴于此，作者引入了一个带栈结构的动态规划算法处理解码，如下图（Kitaev的文章图的审美一直很在线）.

插入左孩子叶子节结会往栈中增加元素，插入右孩子非叶子结点会缩减栈中元素，其余两个动作保持栈深度不变，图中红色或非实线部分都是不会被合法结果访问的，因为他们无法到达目标结点G.
一条合法的路径最终必然会走到G，表示栈中最后只剩下一个元素.
作者观察到栈最大深度为8时就可以覆盖所有PTB里的句子结构，因此算法里深度取固定值8，算法复杂度为O(N).

N-ary和有标签情况

作者对N-ary先进行了二叉化，然后再后处理恢复.
对于有标签情况，作者选择将方向与标签组合，扩大标签空间.
例如下面的句子

转化出来的标签序列就是

(['l/NP', 'l/', 'l/', 'r/NP', 'r/'], ['L/S', 'L/VP', 'R/S::VP', 'R/'])

其中l/r和L/R分别表示叶子和非叶子结点的方向，后缀则是树标签（这么一看就跟left-corner parsing更像了）.

实验部分

Tetra Tagger用BERT-large在PTB Test上达到了95.44的结果，在parsing-as-tagging这一类的工作中算是SOTA，实验平台用的TPU，和前人工作比较不太公平，但是应该也挺快的了.

总结

作者提出了一个只需要4个标签空间的序列标注系统来处理Constituency Parsing问题，解码仅需线性时间复杂度，在BERT上的结果为95.44，和SOTA结果相当，大幅超越了前人的Parsing as Tagging结果.

参考
^Kitaev and Dan Klein. 2020. Tetra-Tagging: Word-Synchronous Parsing with Linear-Time Inference. In Proceedings of ACL, pages 6255–6261, Online. https://aclanthology.org/2020.acl-main.557/
^Amini and Ryan Cotterell. 2022. On Parsing as Tagging. In Proceedings of EMNLP, pages 8884–8900, Abu Dhabi, United Arab Emirates. https://aclanthology.org/2022.emnlp-main.607/
