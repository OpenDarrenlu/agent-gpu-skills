# [ACL'23] Hexatagging: Projective Dependency Parsing as Tagging

**作者**: 张宇月之暗面

**原文链接**: https://zhuanlan.zhihu.com/p/640952220

---

TL;DR

Tetra-tagger在依存句法上的简单拓展.

yzhangcs：[ACL'20, EMNLP'22] Tetra-Tagging: Word-Synchronous Parsing with Linear-Time Inference
10 赞同 · 2 评论 文章

增加两个标签L和R表示一个span中左右子树哪个是head，以此将依存树转化为成分树，再继续Tetra-tagging的方法.


Hexatagging: Projective Dependency Parsing as Tagging

[1]

这篇文章获得了ACL2023的Outstanding Paper Reward.

方法

如图，作者采用了一个简单的办法，完成了依存到成分的转换，以reads 
→
 she举例来说，两者组成一个span，因为reads是head，因此constituent的标签是R，表示这个headed span的方向.

其他情况，向上生长合并，以此类推.
下面给了个完整转换例子，算法实现是top-down的，一直递归到最低的元素，然后向上两两合并，最终组成一棵二叉成分树.

训练模型用的BERT-like encoder，加3层LSTM，然后和Tetra一样，独立预测每个位置标签.
解码采用了stack size为8的动态规划，自左向右，保证树的合法性.

总结

一个非常simple yet effective的parsing as tagging系统，仅通过两个标签L/R，将Tetra-tagging拓展到依存句法，最终的结果和速度都很优秀.

一些limitations：目前还只能在projective tree上work，需要更多训练迭代次数结果才可以和Yang and Tu [2] Comparable.

参考
^Afra Amini, Tianyu Liu and Ryan Cotterell. 2023. Hexatagging: Projective Dependency Parsing as Tagging. In Proceedings of ACL. https://arxiv.org/pdf/2306.05477.pdf
^Songlin Yang and Kewei Tu. 2022. Headed-Span-Based Projective Dependency Parsing. In Proceedings of ACL, pages 2188–2200, Dublin, Ireland. https://aclanthology.org/2022.acl-long.155
