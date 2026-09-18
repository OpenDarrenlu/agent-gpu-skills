# tensorflow 和 caffe2

**作者**: 刘侃呵呵

**原文链接**: https://zhuanlan.zhihu.com/p/26512099

---

一定是我太无聊了

caffe2 的代码看起来都透露着 tensorflow 的阴影，比如说 op，opdef，opregistry 等等，矩阵运算也用了 eigen，不过整体代码简洁了 （删除线开始）feature 少了（删除线结束）许多。

1. python 及语言 wrapper 方面

tensorflow 正常能用的语言 wrapper 就 python 了，其他的只能跑 predict 不能再多了，用 c++ 手动构造 train 网络不要太累。caffe2 也就 python。功能上来说该有的都有（帮忙算 grad），tensorflow 的功能会丰富一些，代码看起来也简洁点（个人感觉）。

2. op 的实现与构建

两者实现方式如出一辙，从cpu/gpu 异构的支持（tf 的 device 和 caffe2 的 context），到 input()，output() 方法，以及 type/shape inference。有个区别，caffe2 是用 blob 封装 tensor (或者其他T[]等) 来做输入输出，而 tensorflow 输入输入则直接是 tensor。

矩阵运算 caffe2 是用 eigen 来做的，不过没看到多线程（目前 github 上开源版本看起来，或者我眼瞎了）。。。tensorflow 则开了线程池。

3. 执行框架与线程模型

caffe2 提供了了两种执行框架，普通版和 dag 并行版。普通版的执行框架会顺序执行所有 op（因为大家都是读写 blob 嘛，所以只要依赖顺序对就好），dag 版会计算出图的依赖顺序，计算出互不依赖的 op 链交给不同线程执行。

tensorflow 直接用 tensor 做输入输出，并没有做顺序的编排。目前都是每做一个 op 存一下结构，然后通过 edge activate 一下后面的节点，看看哪些能做，再根据 time cost 决定是否丢线程池，充分并行起来。一方面比较自由，但某些时候 overhead 并不能忍。

而且，tensorflow 内置支持了控制流（If，loop 等），caffe2 没有这些。但 tensorflow 这些控制流生成图之后都是很碎的 op。。。当然这是有优化空间的。

4. gpu ，还有图优化

gpu 实现细节没细看，大体上都是 stream 起来似乎差别不大。而 tensorflow 有 xla，减少 launch kernel 的 overhead，顺便为其他设备优化提供更多可能（你家 int8 tpu）。不过想想看 xla 可能对于绝大多数 cpu 的东西应该效果没啥改观，除非说正好是 element-wise 的 x*a+b 被向量化了。。

tensorflow 为图优化提供了充分的机制，xla 只是其中的一种。还有变量折叠，死分支消除等等，并且允许用户自定义优化方法。caffe2 作为轻量级框架则没有这些功能。

5. 分布式

tensorflow 中是有个 VariableOp 来存用户会 update 的变量（比如说 weight），tensorflow 本身就是一个兼职 ps，之间通过 grpc 通信。在集群模式下，通过 master，tensorflow 会将 graph partition 到各个 device（不同机器的不同 cpu 和 gpu）（当然你要在图里面提前说好），然后 run。

对于 caffe2 来说，weight 是 blob，blob 在 workspace 上，是可以被更新的。而在分布式运行的过程中，则借助 store op （有 redis，file 等 handler）来 get 或者 set 这个局部的 blob。讲道理这样是可以分布式的，但是运行起来似乎久没有 tensorflow 那么轻松了。毕竟 caffe2 只是通过 data_parallel_model.py 来分配下单机 gpu 和然后带上 broadcast/reduce，没有 master 时候就要手动再多做些事情，比如说 ssh 到第二台机器上启动 resnet50_trainer.py 噗。换句话说，需要手动造个简单的 master。

目前 op 的 placement 都是用户分配的（variable 放哪，用 gpu 还是 cpu 算），或许以后能自动优化分配是不是更屌？


6. 其他

想想也没啥了，毕竟关注点主要在西加加 core 的实现上。顺便吐槽下 caffe2 要求用户先把数据导入 db 。。。当然作者都说 unframework 了，那肯定比 framework 功能会少很多。tensorflow serving 和 tensorflow board 等等。毕竟人少代码写不过来，功能不丰富生态不健全也正常。。cpu 性能方面，个人感觉对于比较纯洁的网络来说框架 overhead 不会太高，简单的不如好用的。内存方面，gpu 我没测过 lol，cpu 谁管那么多。。

（丸）
