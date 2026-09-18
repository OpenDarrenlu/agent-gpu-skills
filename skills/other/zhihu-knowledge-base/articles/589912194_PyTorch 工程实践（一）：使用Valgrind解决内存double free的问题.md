# PyTorch 工程实践（一）：使用Valgrind解决内存double free的问题

**作者**: MingfeiPyTorch CPU Perf Maintainer

**原文链接**: https://zhuanlan.zhihu.com/p/589912194

---

​
目录
收起
前言
使用 Valgrind 定位问题
重新编译 PyTorch
使用 Valgrind memcheck
分析 log
问题确认
问题处理

这个系列会记录一些在 PyTorch 优化工作中遇到的各种各样的问题，会挑一些比较简单的案例，不需要很深的背景知识也能理解，保证能学会。也算是介绍一下底层工程师的日常工作。

前言

这里介绍的是怎么通过检测工具 Valgrind来分析处理内存 Double free 的问题。问题报在 #89677，是EmbeddingBag 在CPU device 上面出现的 double free 问题：

self = torch.tensor([2, 4, 8, 8, 7, 3, 1, 8, 7, 8, 1, 1, 0, 0, 3, 1, 0, 2, 1, 5, 8, 7, 9, 7, 0, 0, 7, 6, 5, 5, 9, 1, 6, 2, 9, 1, 4, 3, 2, 3, 1, 1, 0, 6, 3, 9, 3, 9, 6, 6, 9, 2, 8, 5, 7])
weight = torch.rand([100, 5], dtype=torch.float32)
offsets = torch.tensor([0,  6, 12, 15, 25, 32, 40, 42, 46, 53, 53])

res = torch.nn.functional.embedding_bag(
    self, weight, offsets,
    norm_type = 2.0,
    scale_grad_by_freq = False,
    mode = 'mean',
    sparse = True,
    include_last_offset = True,
    padding_idx = 61
)

一般按照经验，double free 大多数都是由于“读越界”造成的，不过不知道这点也没关系，下面我们通过 Valgrind 可以很快就定位这个问题。

[TIP] 这里最好知道 EmbeddingBag的基本语义是什么。

使用 Valgrind 定位问题
重新编译 PyTorch

首先，需要重新编译 PyTorch，加入调试信息：

DEBUG=1 python setup.py install

开了 DEBUG=1 之后编译会慢很多，我的机器上差不多要五分多钟了。

使用 Valgrind memcheck

可以把上面出问题的那个脚本命名为 test_embedding_segfault.py，按下面的命令调用 Valgrind：

valgrind python test_embedding_segfault.py 2>&1 | tee embedding_segfault.txt

valgrind的默认选项就是memcheck，所以--tool=memcheck 加不加都行。Valgind 运行也比较慢，上面这个例子也要差不多五六分钟能跑完，会输出很多 log，绝大多数都没用，所以先用 tee 记录下来。

从 valgrind 成千上万行的 log 里面定位到真正出问题的地方是个很麻烦的过程，这里有个小技巧：可以在你所关心的函数前面加打印，比如在 EmbeddingBag.cpp#L1090 这个函数里面加一句打印。回头再查找 log的时候直接搜索，就可以很快定位问题。这个算是“刻舟求剑”的“剑”。

分析 log

这个 case 的 log 有一万七千多行，绝大多数都是 PyObject 报出来的没有用的信息。关键内容摘抄如下：

### _embedding_bag_cpu_impl_out: 0 
==450738== Invalid read of size 4
==450738==    at 0x202C1A64: at::native::cpublas::(anonymous namespace)::cpublas_axpy_impl(c10::ScalarType, long, c10::Scalar const&, void const*, long, void*, long)::{lambda()#1}::operator()() const::{lambda()#14}::operator()() const (BlasKernel.cpp:220)
==450738==    by 0x202C246B: at::native::cpublas::(anonymous namespace)::cpublas_axpy_impl(c10::ScalarType, long, c10::Scalar const&, void const*, long, void*, long)::{lambda()#1}::operator()() const (BlasKernel.cpp:220)
==450738==    by 0x202C2890: at::native::cpublas::(anonymous namespace)::cpublas_axpy_impl(c10::ScalarType, long, c10::Scalar const&, void const*, long, void*, long) (BlasKernel.cpp:220)
==450738==    by 0x1A10F0F4: void at::native::DispatchStub<void (*)(c10::ScalarType, long, c10::Scalar const&, void const*, long, void*, long), at::native::cpublas::axpy_stub>::operator()<c10::ScalarType const&, long&, float&, float const*&, long&, float*&, long&>(c10::DeviceType, c10::ScalarType const&, long&, float&, float const*&, long&, float*&, long&) (DispatchStub.h:158)
==450738==    by 0x1A1D9958: void at::native::cpublas::axpy<float>(long, float, float const*, long, float*, long) (CPUBlas.h:133)
==450738==    by 0x1A1C5231: std::enable_if<std::is_same<float, float>::value, void>::type at::native::(anonymous namespace)::index_select_add<float, long>(at::Tensor const&, at::Tensor const&, at::Tensor const&, at::Tensor&, at::Tensor const&, bool, at::Tensor&, long, at::native::_EmbeddingBagKernelCacheImpl<at::native::_CallbackAndBlockSize<true, int, float>, at::native::_CallbackAndBlockSize<false, int, float>, at::native::_CallbackAndBlockSize<true, long, float>, at::native::_CallbackAndBlockSize<false, long, float>, at::native::_CallbackAndBlockSize<true, int, unsigned short>, at::native::_CallbackAndBlockSize<false, int, unsigned short>, at::native::_CallbackAndBlockSize<true, long, unsigned short>, at::native::_CallbackAndBlockSize<false, long, unsigned short> >*) (EmbeddingBag.cpp:448)

上面那个 "_embedding_bag_cpu_impl_out" 就是那把“剑”，我们可以看到出现 Invalid read 的地方是 cpublas_axpy_impl，是从 EmbeddingBag.cpp#L446 这个地方调进去的。另外可以猜出来应该是 src 或者 output 的越界，因为这两个的 data type 是 float。如果是 index 越界，应该是 8 个 bytes (data type int64_t)。

问题确认

接下来需要确定一下是不是这个问题，“读越界”的问题一般都是对指针访问造成的，那么很简单，我们把访问指针的 index 都打印出来，看看有没有超过 size 的就行了。比如，对于 output 的访问是通过 add_indices 按行访问的：

### when `include_last_offset` is true, output tensor size is {offsets.size(0) - 1, weight.sizes()[1]}
### so the output size is {10, 5}

### add_indices is a 1D tensor size of {55}
### values in add_indices are the row index to access output tensor
{ 0 0 0 0 0 0
  1 1 1 1 1 1
  2 2 2
  3 3 3 3 3 3 3 3 3 3
  4 4 4 4 4 4 4
  5 5 5 5 5 5 5 5
  6 6
  7 7 7 7
  8 8 8 8 8 8 8
  10 10 }

可以看到最后有两个 10 (index == 10 是访问第 11 行)，问题就出在这里，因为 output 本身只有 10 行，所以发生了读越界。

一般对于 segmentation fault 能定位问题之后，解决就简单很多了。我们可以看到通过 Valgrind 来定位这个问题非常方便，不需要很深厚的背景知识也可以轻松完成。

问题处理

下面是问题的处理过程，需要对相关代码有一定了解，不感兴趣的可以省略，重点是上面问题的定位过程。

EmbeddingBag 的语义是通过 offsets 来确定把 weight 和 output 之间的对应关系，比如这个案例中

offsets = {0,  6, 12, 15, 25, 32, 40, 42, 46, 53, 53}

语义是把 weight 的 [0, 6) 行累加到 output 的第 0 行；把 weight 的 [6, 12) 行累加到 output 的第 1 行，以此类推。为了完成这个功能，需要计算每个 weight index 对应的 output 的行号，这个功能是这个函数完成的：

static void make_offset2bag(const Tensor &offsets, Tensor& offset2bag) {
  offset2bag.index_add_(
      0, offsets, at::ones_like(offsets, LEGACY_CONTIGUOUS_MEMORY_FORMAT)); // offset2bag = [1 0 1 0 1]
  offset2bag[0] -= 1;                     // offset2bag = [0 0 1 0 1]
  offset2bag = offset2bag.cumsum(0, offset2bag.scalar_type());     // offset2bag = [0 0 1 1 2]
}


在进行 cumsum 之前，offset2bag （也就是 add_indices）的值是这样的：

### offset2bag is a 1D tensor size of {55 + 1}
0 0 0 0 0 0 1
0 0 0 0 0 1
0 0 1
0 0 0 0 0 0 0 0 0 1
0 0 0 0 0 0 1
0 0 0 0 0 0 0 1
0 1
0 0 0 1
0 0 0 0 0 0 2
0 0

问题就是 offset2bag[53] = 2。处理方式非常简单，在 include_last_offset 为真的时候把 offsets[-1]替换一下就好了。这样 CPU 的行为也能和 CUDA 一致。

这里，提出一个问题：如果需要优化这个 operator 的性能，应该采用什么样的算法呢？
