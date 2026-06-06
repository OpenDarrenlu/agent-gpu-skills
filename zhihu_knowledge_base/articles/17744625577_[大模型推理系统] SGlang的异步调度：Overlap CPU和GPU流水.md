# [大模型推理系统] SGlang的异步调度：Overlap CPU和GPU流水

**作者**: JoeNomad​​新南威尔士大学 信息技术硕士

**原文链接**: https://zhuanlan.zhihu.com/p/17744625577

---

Prologue：

在自回归模型的语义下，下一步的input是当前的推理结果，serving系统会在每次推理之间进行调度工作，目前有一些工作是面向SLO的，如chunked prefill，PD分离，ranking等等，这些工作不在今天的讨论范围，且是正交的关系，可以叠加使用。今天介绍的内容主要解决的是CPU&GPU之间的气泡问题，调度器是执行在cpu上的(负责规划新请求,分配block,驱逐完成的请求...etc)，vanilla的实现是同步的，由于schedule和forward是串行的逻辑，那么在schedule的同时gpu是空转的，此时浪费了一些GPU资源。当然，也有工作试图解决这个问题如multistep，通过一次规划多个decode step来减少schedule的次数，从而减少空泡，但我认为解决的不够干净，并且也会影响到SLO。

本文将深入探讨SGlang的异步调度机制，分析其如何实现CPU和GPU流水线的重叠。

MAIN
问题描述
vanilla schedule GPU Bubble

在vanilla的实现中，每个iteration都需要进行同步的规划，主要是为了取到这一轮sample的token，去识别eos来驱逐完成的请求和传给下次推理做input，同步后到schedule结束，GPU上是没有任务在进行的，所以空泡的大小就是scheudle执行的latency。

设计&&实现细节

对与这个问题，SGlang的解法的idea跟多级流水线一样，overlap CPU和GPU的执行，在GPU进行forward的同时去schedule下一个step，但我们知道下一个step跟当前forward的结果是有数据依赖关系的，这里SGlang的解法就很巧妙了。一图胜千言：


vanilla和async schedule的对比(精简示意图)

在async schedule中，下一个step的开始时间不是当前step执行完，而是当前step发射完，发射完之后会立刻返回一个future token的下标list和下标map，从这里打破了step之间在cpu上的数据依赖，cpu不需要知道下一个token是什么就可以发射gpu kernel。当发射完当前step之后，next token的地址我们就知道了，那么立刻发射一个d2d的copy，然后再发射下一个step的kernel。GPU的stream是保序的，所以计算结果不会受影响。另外，scheudle会在发射完下一轮kernel的后等待上一轮的结果，因此在整个执行流上，cpu和gpu只差一个step，不会超前太多。

因为async schedule会在当前step等待上一轮的结果，反馈给用户，因此需要对第一个step进行特殊处理，如图缩减，我在第一个wait地方标注了红色虚线，这里是说其实并没有真的等待，因为没有东西可等。在实现上是在第一个step前插入了一个dummy batch，它本身就是空的，所以cpu执行waiting的时候就跳过了。

代码指路

主要涉及的文件:

sglang/python/sglang/srt/managers/tp_worker_overlap_thread.py
sglang/python/sglang/srt/managers/scheduler.py
schedule next batch的逻辑

在worker中启动了一个线程non blocking的去做kernel forward，205行是一个异步下发任务的过程，这里给了一个future_token_ids_ct是告诉forward线程output token可以从哪块地址开始写入。下发完任务之后立刻返回了一个future_next_token_ids，这里的值不是真正的output token，而是output token将要写入的下标。

forward线程的逻辑

waiting上一轮forward结束是通过cuda event来实现的，也就是132行的copy done。135-138行实现的就是取input id的下标并且把真的token从map中拿出来。

处理下标

值得一提的是，这里实现取的逻辑是把index先设成负值，这样负数就是下标，正数是token，用clamp + where筛选并赋值做完的token，这里的写法还是很巧妙高效的。

如上述图中我在代码中加了log，执行单条req的结果如下，逻辑就很清楚了

Epilogue

这篇文章的写作初衷源于我们在优化自研推理引擎时，发现GPU存在空闲间隙，进而接触到SGlang这一方法。文章到这里就告一段落了，如果以后有机会的话也会写写我们自研的一些成果，最后特别感谢 @Uranus 的亲情科普，也希望本文能够帮助到有同样需求的同学~

提前祝大家新春愉快！
