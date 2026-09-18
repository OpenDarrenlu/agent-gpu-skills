# 菜鸟学cute layout-coordinate

**作者**: xieym我是little菜鸟-菜菜菜没有敌

**原文链接**: https://zhuanlan.zhihu.com/p/1964717254076789209

---

文章内容取自《Categorical Foundations for CuTe Layouts》

coordinate：A note on A note on the algebra of CuTe Layouts

首先声明：小菜鸟水平有限，吐槽comments随便写。不要被误导[憨笑]

cute表达的多维数组，包括两个索引概念

coordinate
co-domain index（memory offset）

我们熟悉二维矩阵形式。从C/C++这种procedure programming language学编程，会默认二维数组就是row major；像科学计算语言可能采用column major编程语言。后面会谈到column major在tiling场景下更常见。其实在cute世界里，一个矩阵可以由layout即(shape: stride)，无论row major还是column major，shape是相同的，就是矩阵在逻辑上表示形式。stride表示矩阵在物理世界（memory）是如何存储的。为什么可以这么说呢？（我理解的）对于一个layout L不管是row major还是column major, sort(L)是同构的；所以，矩阵的permutation只改变矩阵存储数据的方式。当shape确定之后，有没有唯一或者说确定表达矩阵元素的方式？这就是coordinate。cute采用了colexicographical表达多维索引的变换关系。具体可以见下图

我这里通俗解释下：

一个m维矩阵，每个维度大小分别为s1,s2,...,sm。layout定义域S=s1 * s2 * ... * sm

任何0<=x<=m-1数值都是一个1-D coordinate，通过类似column major方式（就是prefix product从d1,d2,到dm)去计算x落在每个维度上的索引(x1,x2,...,xm)，也就是m-D coordinate，也是我们直觉上坐标含义。

2. 如何通过(x1,x2,..,xm)计算co-domain的index呢？(x1,x2,..,xm) dot product (d1,d2,...dm)

所以，2表示coordinate function




1+2表示layout function

layout function和coordinate function计算结果都是co-domain 的index

为啥要说这个呢？我们可以看下composition的定义

R = B◦A= B(A)

ΦA size(B) ==>使用size（B）限制A的layout function输出 ==> A在co-domain上index

B(A) ==>把A在co-domain上的index当作B的1-D coordinate进行layout function变化得到原本A的1-D coordinate映射到B的co-domain上了（媒介就是A的co-domain的index）。

当然，composition需要满足shape不变，也就是domain不变。
