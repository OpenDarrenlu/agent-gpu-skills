# Tournament分支预测器简介

**作者**: SnoopyBug划水划呀划~

**原文链接**: https://zhuanlan.zhihu.com/p/489187295

---

一、思路

基于已有的Gshare分支预测器，希望对其优化来提高预测精度。能够优化的方面可能有3个：

增加GHR（global history register）的长度，使用更多的全局历史信息；
增加PHT（pattern history table）中状态机的状态编码长度，使其拥有更多状态；
优化Hash函数，或者使用其他方法进行预测。

前期通过对上面提到的方法进行测试，发现前两种方法的影响不如第3种方法带来的影响大，并且优化预测方法更具彻底性。因此决定：先优化预测方法，再对GHR长度和状态编码长度进行调参，得到最优方法。

二、分析
（1）预测方法
①使用局部历史进行预测

已有的Gshare分支预测器：使用有限个全局历史编码串和分支地址的一部分进行异或，将结果作为索引在PHT中寻找状态，通过状态值来预测是否跳转。

UINT32 phtIndex = (PC ^ ghr) % (numPhtEntries);    // numPhtEntries 为PHT中的项数
UINT32 phtCounter = pht[phtIndex];

但只基于全局历史的预测是有风险的，当全局历史不那么规律时，预测器的表现将十分糟糕。但可能这种情况下，对于同一指令，它的局部历史是比较规律的。考虑：当PC相同时，可以认为是同一条指令。因此对于每一个PC，使其拥有独立的BHR（branch history register）来代替GHR来进行上述操作可能会提高预测精度。

// branch history register table
bhrt = (UINT32 *)malloc(numPhtEntries * sizeof(UINT32));

// HIST_LEN 为GRH长度
UINT32 bhrtIndex = (PC & ~(1<<HIST_LEN)) % (numPhtEntries);
UINT32 phtIndex = (PC ^ bhrt[bhrtIndex]) % (numPhtEntries);
UINT32 phtCounter = pht[phtIndex];
②融合全局与局部历史

使用局部历史相比使用全局历史的预测精度确实提高不少，但只基于局部历史的话，GHR就失去作用了。希望将全局历史信息与局部历史信息融合，采用了Tournament预测器：

Tournament预测器基于全局预测和局部预测进行综合考虑；
全局预测与局部预测独立进行，由选择器选择其中一个方法的预测结果作为最终预测结果；
和PHT类似，选择器实质上也是一个长度为 2^k 的状态表，使用GHR的低k位进行索引。（也可以用PC或者PC^GHR来索引，效果还不错）；
对于选择器中的一个状态，如果全局预测和局部预测结果一致，状态不变；否则：对于该状态，如果局部预测正确就+1，全局预测正确就-1；
Tournament示意
优点：综合全局和局部历史信息，精度较高；
缺点：Tournament预测器的空间占用是比较大的，因为不仅有两个预测器，还有一个选择器。
（2）状态编码长度
长度太短：每个状态蕴含的历史信息较少，使得历史信息在状态机中发挥的作用较小；
长度太长：信息冗余，且对时间上较近的跳转结果不灵敏，无法及时改变预测值。
（3）GHR长度
长度太短：全局历史信息太少，精度可能不高；
长度太长：虽然精度可能有所提高，但空间代价是巨大的。
三、结果

预测器在不同测试集（L1, L2, ...)下的测试结果（每执行1k条指令的错误预测数）如下：

Result

可以看出，当GHR取低26位时，Tournament预测器表现较好。但考虑到空间消耗，退而求其次，取22位或者18位时的平均精度相较初始的Gshare预测器依然大幅提升了。
