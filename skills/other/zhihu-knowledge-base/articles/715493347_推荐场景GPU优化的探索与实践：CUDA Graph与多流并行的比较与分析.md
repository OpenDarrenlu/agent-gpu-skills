# 推荐场景GPU优化的探索与实践：CUDA Graph与多流并行的比较与分析

**作者**: djy123456​阿里巴巴 员工

**原文链接**: https://zhuanlan.zhihu.com/p/715493347

---

（插一个视频：【利用多 CUDA stream 加速推荐系统-哔哩哔哩】 https://b23.tv/nYZ3g7U）

背景&问题




RTP 系统(即 Rank Service)，是一个面向搜索和推荐的 ranking 需求，支持多种模型的在线 inference 服务。在过去的几年里，我们对于RTP业务的GPU性能优化已经做了不少尝试，包括kernel fusion，CUDA Graph等。在此基础上，今年我们又在RTP上集成了Multi Stream，改变了TensorFlow的单流机制，让多流的执行并行，作为增加GPU并行度的另一种选择。

RTP执行的backend是TensorFlow。TensorFlow作为调度算子的框架应用在推荐业务的推理场景中，对GPU是比较不友好的。这是因为TensorFlow总是倾向于用大量的operator构建一个graph，每个operator在GPU上又会调用一个或者多个kernel；而同时，TensorFlow本身是一个单流模型，在一个进程内只有一个Stream group（包含一个Compute Stream，一个H2D Stream，一个D2H Stream和一个D2D Stream），kernel间的执行很难并行，可以说调度的效率是偏低的。







图1 TensorFlow Stream group

在GPU上，kernel的调用分为kernel launch和kernel执行两步，kernel launch负责准备好执行kernel需要的资源，通常在us级别；kernel执行则是实际上GPU完成的计算。一些比较简单的kernel执行时间可能也在us级别，但却不得不等待数us的kernel launch，这就是所谓的kernel launch瓶颈。在我们的推理场景中，由于graph包含的kernel数量极多，单流模型的调度效率又低，几乎一定是kernel launch bound。







图2 典型的搜推模型运行timeline

缓解kernel launch瓶颈主要有两个思路，一个就是kernel fusion，通过减少kernel数量减少launch数量，同时也会带来访存和计算上的优化；另一个思路就是提高kernel launch的效率，减少每一次kernel launch的代价或者并行launch kernel。

CUDA Graph

熟悉RTP的同学可能了解到，在2020年，我们已经在RTP分支的TensorFlow里集成了CUDA Graph。CUDA Graph通过预先create或者capture一个graph（我们希望这尽可能是一个完整的GPU子图），将graph里数量众多的kernel launch转化成一次graph launch，以降低launch在device和host上的开销，几乎可以说是解决了kernel launch瓶颈的问题。







图3 kerne launch转化为graph launch

但实际应用CUDA Graph需要满足比较高的要求：

一个是CUDA Graph并不支持动态shape，而搜推场景的batch size大部分都是动态的。为了满足这个条件，我们的方案是预先capture多张不同batch size的子图供运行时的请求选择；要是请求的batch size超过预先capture的最大值，就back up到TensorFlow的实现。

这已经是一个相对合理的方案，但实际应用的时候还是会有不少问题。一个问题是，经过预先的填充，当前graph里只能有唯一一个动态的维度，且它的值必须是batch size，这也意味着，子图里一些诸如Concat，Gather，Split等可能会导致破坏这一条件的操作应当要被谨慎的排除出去。另一个问题是，对于batch size的选择依赖于模型输入的分布和实际硬件的显存（因为多份图当然占用了多份存储），这就依靠经验，或工具层自动的根据历史流量分布选择参数。







图4 动态batch size支持

第二个要求是对于CUDA Graph来说，必须保证Graph的输入输出地址固定。针对这个限制，当前我们的方案是将来自CPU的输入先放到GPU上，然后和在GPU上完成了一些计算的tensor一起作为CUDA Graph的输入，通过D2D copy到Graph里面。增加了一层数据传输必然会带来延时的增长。当然还可以有另一个方案，先保证整个GPU子图都可以被capture，然后将CPU输入拷贝到固定的host地址上，将原方案里的H2D+D2D转换成H2H+H2D。但无论如何，多一级Memcpy是不可避免的。

在这些限制下，我们对CUDA Graph的用法就变成了，先通过kernel fusion将整个GPU子图整理成一张结构干净，shape“固定”的子图，然后再capture整理完的子图，让CUDA Graph照顾一些手工的kernel fusion难以整理到位，但实际计算又很轻的计算，比如常见的elementwise操作等，让这些本身计算开销小的kernel的launch开销也几乎可以忽略不计。基于这种比较精细的用法，CUDA Graph的收益主要有：

将大量的kernel launch转化为一次graph launch，从而极大的节省了host和device开销；
多个CUDA Graph的执行是完全独立、可并行的，因此会直接被分配到多个Stream上，这种多Stream的并行也极大的提升了吞吐，很好的增强了单机服务能力。

不过这种能够保证CUDA Graph优化效果的用法事实上对工程同学提出了不低的要求，需要用户既熟悉模型结构（且能做一定程度的图优化），也熟悉模型流量分布，还要简单了解device arch（至少是不同型号的GPU memory大小）。这些要求稍不满足，便很容易得出一个效果不佳，提升有限的结论。但不得不承认的是，CUDA Graph可以称得上我们过去几年最重要的优化，在重要场景实现了平均2倍左右的吞吐提升，帮助千卡集群平稳度过多轮大促的流量洪峰。

Multi Stream

为了降低优化的应用门槛，提高适用性，今年我们的一个主要工作就是在TensorFlow里集成了Multi Stream。

MultiStream的实现来自Nvidia DevTech团队。它的基础思路非常简单：一个Stream的device利用率低，就分配多个Stream，并且是把整个GPU子图放到不同Stream上，让请求和请求并行。







图5 多Stream group并行

直接创建多个Stream group的性能提升是比较有限的。通过分析GPU timeline，会发现在每个Stream group内，都存在大量的cuEventRecord和cuEventQuery，这些Event大部分都来源于Compute Stream和 Memcpy Stream间的同步。在整个进程只有一个Stream group时，通过将计算和传输行为分配到多个Stream上以尽可能overlap，并通过必要的同步来保证行为当然是非常合理的。但当我们有多个Stream group后，是不是Stream group间的overlap就足以提升device利用率了呢？我们的实验证明，当整个device存在多个Compute Stream时，把相对应的Memcpy Stream合并到comput Stream中，可以有效减少Stream间的同步行为，提高GPU利用率。

此外，我们在GPU timeline中看到层出不穷的pthread_rwlock_wrlock，阻碍了kernel launch。这是因为GPU driver对cuda context有读写保护。当一个cuda context向多个Stream launch kernel时，driver会给kernel launch上比较重的锁。事实上这层锁随着driver更新在逐步减轻，driver510已经将读写锁改成读锁，这层限制大概率会随着驱动的升级进一步被弱化。但当前我们最好的方法还是直接把合并后的每个Stream都放到各自不同的context中去，并通过MPS实现context间的并行。MPS是Nvidia对于多process/context的优化方案，将多个process/context放到同一个control daemon下，共享一个context，是一个比较成熟，且相对易用的方案。这里提供一个文档（Nvidia MPS）。

还有一个相对简单的点是，开启多流后，为了避免多个thread向同一个Stream launch kernel的pthread_mutex_lock，我们也给每个Stream配了一个私有的CPU thread，让这一个thread去完成对应Stream上的所有kernel launch。当然这种做法依然无法避免多个thread一起H2D的launch竞争。我们也做了一点尝试，但都不是很成功，就不赘述了。







图6 Multi context + MPS在推荐业务上的timeline

到这里，我们能做的就基本上做完了。简单的在推荐的几个场景做了验证，测试下来多流的性能提升大概能够接近CUDA Graph的性能，如图6所示，创建了4个context，每个context各一个Stream，且对应一个thread，Stream与Stream间，计算与传输间，都可以比较好的overlap。在FY24双十一，多流的优化已经推广到了大部分的RTP业务上，非常显著的提升了性能，在大部分业务场景上取得了1-3倍的性能提升，降低了业务运行的硬件成本。

总结&致谢

最后我们简单的比较一下这两种方案：

CUDA Graph作为有硬件支持的方案，将大量kernel launch转换为一次graph launch，可以同时节省host和device开销，在应用得当的前提下应当是最优性能的最佳选择；
Multi Stream主要是通过创建多个Stream的做法增加了kernel执行的并行，从而更好的利用资源，在易用性上远超CUDA Graph。

我们的工作都是在与Nvidia DevTech团队的密切合作下完成的，非常感谢Nvidia DevTech团队对我们的技术支持。
