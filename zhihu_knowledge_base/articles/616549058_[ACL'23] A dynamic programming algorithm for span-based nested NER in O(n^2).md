# [ACL'23] A dynamic programming algorithm for span-based nested NER in O(n^2)

**作者**: 张宇月之暗面

**原文链接**: https://zhuanlan.zhihu.com/p/616549058

---

前言

得益于结构上的相似性，最近信息抽取任务中，涌现出了一大批xxx as parsing的工作，将原任务归纳为Dependency/Constituency Parsing[1] [2] [3] [4] [5] [6]，并取得了不错的性能. 其中最具代表性的是(Nested) NER，Fu et al. (2021)[3]和Lou et al. (2022)[4]创造性的提出了将Nested NER转换成了一棵带隐式节点的成分句法树，相应地使用修改的CKY算法来处理.

本文介绍的是Caio Corro最近的一篇文章，和上述两篇工作一样，作者仍然是采用类CKY算法来处理Nested NER结构，但是，不同于Fu et al. (2021)[3]和Lou et al. (2022)[4]必须要依赖于隐变量才能使Nested NER结构和句法树完全适配，本文提出的算法最大的亮点在于无结构性歧义，即搜索空间和所有可能的NER组合是一一对应的. 本文提出的算法可以视为CKY与Semi-Markov CRF的结合版，其基础版本的时间复杂度是 
𝑂
(
𝑛
3
)
 ，与CKY等同，在此基础上，作者进一步对搜索空间进行裁剪，并提出了一个 
𝑂
(
𝑛
2
)
 的版本. 由于Nested NER任务和Constituency Parsing的相似性，笔者认为本算法同时也可以对Constituency Parsing有所借鉴，例如一个直接的拓展是对N-ary Parsing更高效的处理[6].

A dynamic programming algorithm for span-based nested NER in O(n^2)
aclanthology.org/2023.acl-long.598.pdf
基本方法

首先引入本文用到的术语，用 
𝑠
=
𝑠
1
,
…
,
𝑠
𝑛
 表示一个句子， 
𝑠
𝑖
:
𝑗
 表示一个span，那么一个命名实体（NE）可以表示为 
<
𝑡
,
𝑖
,
𝑗
>
 . 例如下面的句子第一个实体可以表示为 
<
𝑃
𝐸
𝑅
,
0
,
1
>
 .

对于Non-nested NER而言，句子中的NE结构是互不相交的，但是对于Nested NER，一个实体可以内嵌于另一个实体，作为孩子，例如下面的例子， 
<
𝑃
𝐸
𝑅
,
2
,
3
>
 以及 
<
𝑃
𝐸
𝑅
,
5
,
6
>
 都是 
<
𝑃
𝐸
𝑅
,
0
,
8
>
 的孩子. 我们称像 
<
𝑃
𝐸
𝑅
,
0
,
8
>
 这样不在其他任何实体内部的实体为first-level mentions.

和前人工作一样，给定一个例子 
𝑦
=
{
<
𝑡
,
𝑖
,
𝑗
>
}
 ，本文的训练目标是最大化概率

最大的难点在于计算 
𝑍
(
𝑤
)
:=
∑
𝑦
′
exp
⁡
(
𝑤
⊤
𝑦
′
)
 ，需要遍历所有可能的组合. 这里用到了作者提出的类CKY动态规划算法来完成. 推断的情形类似，只不过将sum-product变成了max-product

下面开始解释本文提出算法的细节. 本文所有的算法描述都是基于parsing-as-deduction框架.

Non-nested NER

先从简单的Non-nested NER情形开始描述，本文对于Non-nested NER是基于Semi-Markov CRF，作者将其转化成了parsing-as-deduction语言. 定义两种结构：

[t, i, j] ：表示一个实体 <t, i, j>
[\rightarrow, i] ：表示已经覆盖到部分句子 s_{0:i} 的局部分析

算法会对所有可能的 [t, i, j] 实体结构打分，分值为 w_{<t,i,j>} . 算法对应的推导规则如下

分为两种情况：

(a) 局部分析 [\rightarrow, i] 后面遇到了一个实体 [t, i, j] ，合并形成一个新的 [\rightarrow, j]

(b) [\rightarrow, i-1] 下一个不是实体，因此消耗一个词，形成新的局部分析 [\rightarrow, i] .

试举一例

从起点开始，局部分析 [\rightarrow, 0] 会与实体 [PER, 0, 1] 合并，形成新的局部分析 [\rightarrow, 1] ，下面几步都没有实体，一直消耗句子，走到5形成 [\rightarrow, 5] ，接着遇到实体 [PER, 5, 8] ，合并形成 [\rightarrow, 8] ，走到句尾，过程结束. 整体算法的复杂度为 O(n^2|T|) .

Nested NER

下面是正餐，作者在Semi-Markov基础上提出一个新的类CKY算法处理Nested NER：无需二叉化；很容易处理不被实体覆盖的词；自然的N-ary结构. 新算法定义了如下结构：

[t, i, j] ：和之前一样，表示一个实体 <t, i, j>
[\rightarrow, i] ：和之前一样，表示已经覆盖到部分句子 s_{0:i} 的局部分析
[\mapsto, i, j] （个人感觉 [i, \mapsto, j] 这样子的表示更自然:-)）：代表自 i 开始的一个实体局部分析，它可能继续扩展
[\leftrightarrow, i, j] ：表示对span s_{i,j} 的所有分析已完成，找到了对应这个范围的实体，以及所有的孩子

下面是相应的推导规则

乍一看挺复杂的，实际上都是对Non-nested情况的自然拓展：

(c) 将局部分析 [\mapsto, i,k] 和一个分析好的实体结构 [\leftrightarrow,k,j] 合并成新的局部分析 [\mapsto, i, j]

(d) [\mapsto, i, j-1] 向后走一格成为 [\mapsto, i, j]

(e) 实体结构 [\leftrightarrow,i,k] 和实体结构 [\leftrightarrow,k,j] 合并成局部分析实体结构 [\mapsto,i,j]

(f) 实体结构 [\leftrightarrow,i,j-1] 走一格，变成未完成的局部分析 [\mapsto, i, j]

(g) 局部分析 [\mapsto, i, j] 本身是标签为 t 的实体，因此直接成为实体结构 [\leftrightarrow,i,j] （加上实体score）.

到规则(g)为止，我们终于可以完成将Nested结构转换成一个分析好的实体结构（即，成为了一个找到了所有孩子的整体），这使得我们可以自然地接入Non-nested的规则，稍微的改写

这就是算法的全貌，下图给了一个例子

整体算法复杂度是 O(n^3|T|) ，其中 |T| 是标签规模.

O(n^2) Nested NER

进一步对搜索空间进行裁剪，我们还可以得到一个更高效的 O(n^2) 的算法. 这里作者限制一个实体最多包含一个长度超过1的孩子，从下表中看到加入这样的限制在经典数据集中的覆盖度还可以.

同样地，开始定义算法归纳项：

[t, i, j]：和之前一样，表示一个实体 <t, i, j>
[\rightarrow, i]：和之前一样，表示已经覆盖到部分句子 s_{0:i} 的局部分析
[\mapsto, i, j] ：和之前一样，代表自 i 开始的一个实体局部分析，它可能继续扩展
[\leftrightarrow, i, j]：和之前一样，表示对span s_{i,j} 的所有分析已完成，找到了对应这个范围的实体，以及所有的孩子
[\Leftarrow, i, j] （实际还是 \mapsto 的相反形式，Latex没法打，将就看）：代表自 i 开始的一个实体局部分析，它可能从左边继续扩展

加入了上述的限制之后，(d)(f)(g)(h)(i)规则都是可以复用的，以及需要两条新规则

(j): [\mapsto, i, j-1] 遇到长度为1的实体结构 [\leftrightarrow, j-1, j] ，合并成 [\mapsto, i, j]

(k): 实体结构 [\leftrightarrow, i, j-1] 和长度为1的实体结构 [\leftrightarrow, j-1, j] 合并成 [\mapsto, i, j]

这里可以看到(j)(k)与(c)(e)最大的区别在于丢掉了一个自由遍历变量k，因此可以下降到平方复杂度.

下面是剩余的规则

(n)对应(d)，(o)对应(j)，(l)和(m)分别对应(k)和(f)，情况类似，这里不赘述. 最后当所有左邻居找到，则转而去处理右邻居

下表给了一个运行的例子

实验部分

直接贴表.

总结

作者提出了一个非常聪明的 1) 无结构歧义 2) 方便处理N-ary 3) 无需二叉化的算法来处理Nested NER. 对搜索空间进行裁剪，在保留很好的覆盖度的同时，进一步拓展到了二次方复杂度. Very Cool!

参考
^Juntao Yu, Bernd Bohnet, and Massimo Poesio. 2020. Named Entity Recognition as Dependency Parsing. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 6470–6476, Online. Association for Computational Linguistics. https://aclanthology.org/2020.acl-main.577.pdf
^Songlin Yang and Kewei Tu. 2022. Bottom-Up Constituency Parsing and Nested Named Entity Recognition with Pointer Networks. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 2403–2416, Dublin, Ireland. Association for Computational Linguistics. https://aclanthology.org/2022.acl-long.171
^abcYao Fu, Chuanqi Tan, Mosha Chen, Songfang Huang, Fei Huang. 2021. Nested Named Entity Recognition with Partially-Observed TreeCRFs. *Proceedings of the AAAI Conference on Artificial Intelligence*, *35*(14), 12839-12847. https://ojs.aaai.org/index.php/AAAI/article/view/17519
^abcChao Lou, Songlin Yang, and Kewei Tu. 2022. Nested Named Entity Recognition as Latent Lexicalized Constituency Parsing. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 6183–6198, Dublin, Ireland. Association for Computational Linguistics. https://aclanthology.org/2022.acl-long.428
^Yu Zhang, Qingrong Xia, Shilin Zhou, Yong Jiang, Guohong Fu, and Min Zhang. 2022. Semantic Role Labeling as Dependency Parsing: Exploring Latent Tree Structures inside Arguments. In *Proceedings of the 29th International Conference on Computational Linguistics*, pages 4212–4227, Gyeongju, Republic of Korea. International Committee on Computational Linguistics. https://aclanthology.org/2022.coling-1.370
^abXin Xin, Jinlong Li, and Zeqi Tan. 2021. N-ary Constituent Tree Parsing with Recursive Semi-Markov Model. In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 2631–2642, Online. Association for Computational Linguistics. https://aclanthology.org/2021.acl-long.205
