# PyTorch CPU性能优化（二）：并行化优化

**作者**: MingfeiPyTorch CPU Perf Maintainer

**原文链接**: https://zhuanlan.zhihu.com/p/495278415

---

​
目录
收起
Dimension Collapse
逻辑语义连续并且物理存储连续
逻辑语义连续但物理存储不连续
逻辑语义不连续但物理存储连续
Dimension Blocking
Parallel Dim Reduce
BatchNorm2d计算stats：（CF）
BatchNorm2d计算stats：（CL）

本篇是关于PyTorch CPU性能优化相关的简单入门教程的第二篇。

另外三篇：

马鸣飞：PyTorch CPU性能优化（一）：Memory Format 和 Channels Last 的性能优化
126 赞同 · 13 评论 文章
马鸣飞：PyTorch CPU性能优化（三）：向量化
22 赞同 · 2 评论 文章
马鸣飞：PyTorch CPU性能优化（四）：BFloat16
60 赞同 · 22 评论 文章

本章节对应英文版本在：

Part II: Parallelization Techniques
gist.github.com/mingfeima/664a065cc994318681f6a632c849e1fa
Dimension Collapse

Dimension Collapse是帮助我们简化一个ND Array遍历问题的一种手段，具体来说适用于以下几种场景：

逻辑语义连续并且物理存储连续

假设我们有一个shape是[M, N]的2D tensor，想要做一个inplace的加法。那么，最直观的想法就是写两个循环，一个for处理M，一个for处理N，这样来遍历整个tensor。或者我们也可以把它当作一个shape是[M * N]的1D tensor，只写一个循环，结果是一样的。原因在于：add这个操作在语义上并不区分维度，或者说逻辑语义上这个操作是连续的；另外，物理上这个tensor也是连续存储的。所以这个tensor是1D还是2D亦或是xD并不关键。

在并行化的时候，只需要把整个tensor平均分给每个thread即可，如图Fig-1(a)所示：

这个把2D的index映射到1D index的过程就是dimension collapse。当然这是最简单的情况。

逻辑语义连续但物理存储不连续

还是上面这个问题，但这次tensor是非连续的，stride[1]是N + N'。这个时候如果我们想并行化，有以下几种手段：

1. 在outer dimension上parallel:如Fig-1(b)所示，直接切分最外围的M，并在N上做向量化。如果M足够大可以用满所有的CPU核心，这种实现是比较合理的；

2. 在inner dimension上parallel: 一般情况不会这么切分，因为这种切分方式会导致每个thread上的访存不连续，跨行的stride是N + N'；

3. 映射到1D index：如果M的大小不足以用满所有CPU的核心，比如说M=4，那么在一个20核的CPU上就有16/20的CPU资源没有利用到。这种情况需要collapse M和N来增加算法并发度。同样我们可以把2D的index （m, n）映射到一个1D的index (i)，这时(i)是一个global index，用于任务划分。在寻址的时候还是需要根据(i)算出(m, n)，然后根据stride计算 offset = m * (N + N') + n 。

从 (i) 求取 (m, n)也有多种方法：

直接除法+取余，这是最直观的想法，就不写了。这个方法有个问题是整数的除法很慢，编译器会把div和mod做一个操作优化，不过对每个元素都做一次idiv有的时候开销是难以接受的；
增量法，思想其实很简单，就是实际上我们只需要确定每个thread起始的global index (i) 对应的 (m, n)即可，剩下的可以++n （不换行），和++m (换行)。上篇提到的data_index_init和data_index_step就是做这个工作的，好处就是可以减少idiv的数量，如Fig-1(c)所示。

4. 增加Blocking，从而方便向量化。在N上做blocking，把tensor当成是[M, K, block_size]的。block_size一般设成vector length的整数倍，之后对M和K collapse，在MK上并行，整个过程如Fig-1(d)所示。

逻辑语义不连续但物理存储连续

多数的带kernel的OP都属于这个范畴，比如上篇中MaxPool2d在CF上的实现。回顾一下MaxPool2d这个例子，在我修改之前，ATen中的实现大致是长这个样子的：

// pseudo on max_pool2d channels first
  
  void max_pool2d_update_output_frame() {
    // parallel on C
    at::parallel_for(0, channels, 0, [&]() {
      // do the job
    });
  }
  
  void max_pool2d_update_output() {
    // parallel on N
    at::parallel_for(0, nbatch, 0, [&]() {
      max_pool2d_update_output_frame();
    });
  }

这是个nested omp loop，先对N展开，然后对C展开。这个实现有这么几个问题：

一般来讲，尽量不用nested omp loop。原因是omp是没有全局资源管理这个概念的，不同的omp thread pool并不知道对方的存在，可能会抢核，这对CPU性能来说是灾难性的，有个专门的术语，e.g. over-subscription。PyTorch为了避免这种情况做了限定：内层的omp loop会被强制sequential执行。所以正常情况下PyTorch CPU上是不会over-subscription的，当然不正常的情况也很多，一般都是用户加了自己的threading机制做什么工作；
如此一来，这个写法就很不灵活，适配性很差。比如input shape [4, 64, 112, 112]就只会用4个核，而[1, 3, 224, 224]只会用3个核。

所以为了增加并行度，提高算法灵活性，采取了使用dimension collapse的实现，NCHW都会被用来parallel。

Dimension Blocking

Blocking可以看作是Collapse的逆过程，就是指把2D tensor切成3D，3D tensor切成4D，等等。引入blocking的原因有很多，可能是因为：1）增加并行度；2）方便向量化。要具体问题具体分析。上一章节中Fig-1(d)就是为了向量化。

这里举另外一个例子，torch.cumsum()，也被称为prefix_sum，#74899和#74900是专门优化cumsum的CPU性能的，PR中的parallel scheme很简单，对于dim=-1的情况直接切分outer dimension，然后对inner most dimension做向量化。

这个PR没有针对1D tensor的情况做特殊处理，1D的input会走sequential path。但prefix_sum这个算子本身是可以并行化的，虽然它本质上是个串行的算子。并行化算法分三个step，如Fig-2(a)所示:

在每个thread上面，按照offset = 0，各自做prefix_sum;
更新每个thread的offset，t0的offset是0, t1的offset是sum(t0)，以此类推；
在每个thread上面，加上各自更新后的offset。

上述并行化算法有个问题，就是可能cache利用率不高。原因在于我们需要在step-1和step-3读两遍input data，如果tensor size很大很大，有可能在step-3去读的时候，需要的数据已经被flush掉了。这个时候，我们需要做blocking，以改变数据读取的顺序。可以把input当成是[M, N]的，

 N = T * block_size
 M = numel / N

block_size对应每个thread一轮处理的数据量，选择可以L2命中的尺寸（L2是per core的cache）。这样，每次并行处理T * block_size这么多数据，在每个thread上串行执行M次，整个过程如Fig-2(b)所示。

做blocking的方式非常灵活，一般需要根据input shape具体问题具体分析。这里还有一个复杂一点的例子，是优化log_softmax对于dim != -1的情况，有兴趣可以看一下：

Parallel Dim Reduce

另一种很常见的情况是parallel reduction，分两种：AllReduce和DimReduce。

AllReduce指的是把一个ND tensor reduce成一个scalar，这个涉及的主要问题是数值稳定性，这里PASS 。本章主要介绍DimReduce，就是把ND tensor沿着某个dimension做reduction。BatchNorm2d计算stats的过程就是个DimReduce：input的shape是[N, C, H, W]，mean, rstd的shape是[C]。在channels first上面，是horizontal reduce；在channels last上面是vertical reduce。

BatchNorm2d计算stats：（CF）

CF上面从[N, C, HW] reduce到 [C]的过程非常直白，按照rowwise一直累加就行了，如图Fig-3(a)所示：

这里不再赘述，唯一值得一提的，从vector到scalar的reduce过程（有人叫hsum或者horizontal_sum），需要多个cycle才能完成，所以一般要尽量减少hsum的数量，就是一直用vector累加，到最后才做hsum到scalar。

BatchNorm2d计算stats：（CL）

CL上面的实现要麻烦一点：如果是串行执行，那么直接在垂直方向一行一行累加下来就行了，如Fig-3(b)中的One Path实现。如果要并行话，要借助另一块buffer，做个Two Path Reduction，不然直接切分NHW会导致写冲突。

Two Path Reduction: 要申请一块临时的buffer，大小是[T, C]：

Path-1: 切分NHW，从[NHW, C] reduce 到[T, C]；
Path-2: 从[T, C] reduce到[C]。

PyTorch中相应的kernel如下：

  int num_threads = at::get_num_threads();
  Tensor buffer = at::empty({num_threads, n_channel}, input.options()).zero_();
  scalar_t* buffer_data = buffer.data_ptr<scalar_t>();

  // compute mean per input
  at::parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    scalar_t* buffer_ptr = buffer_data + tid * n_channel;
    for (const auto i : c10::irange(begin, end)) {
      const scalar_t* x_ptr = input_data + i * n_channel;
      vec::map2<scalar_t>(
          [](Vec x, Vec y) { return x + y; },
          buffer_ptr,
          x_ptr,
          buffer_ptr,
          n_channel);
    }
  });

  at::parallel_for(0, n_channel, 1, [&](int64_t begin, int64_t end) {
    for (const auto c : c10::irange(begin, end)) {
      accscalar_t sum = 0;
      for (const auto t : c10::irange(num_threads)) {
        sum += buffer_data[t * n_channel + c];
      }
      scalar_t mean = sum / N;
      mean_data[c] = mean;
    }
  });

注释：

vec::map2<>() 是个向量化的语法糖，功能是把X的一行加到buffer的一行上；
这个实现针对一般性的BatchNorm2d input shape比较有效，如果HW很小并且C非常大，效率会较低。
