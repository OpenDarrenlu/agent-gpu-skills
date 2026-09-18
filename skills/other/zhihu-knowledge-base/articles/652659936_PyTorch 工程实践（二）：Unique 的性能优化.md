# PyTorch 工程实践（二）：Unique 的性能优化

**作者**: MingfeiPyTorch CPU Perf Maintainer

**原文链接**: https://zhuanlan.zhihu.com/p/652659936

---

​
目录
收起
介绍
问题定位
算法分析
优化版的 Unique Kernel
结果

前些天有人在提了一个 issue 抱怨 pytorch 上面 cpu 的 Unique 的性能太慢，比 NumPy 还慢，#107098。我第一反应是挺好奇的，因为印象中 NumPy 对性能这一块不是最看重的，但我测了一下还真是。查看了一下 NumPy 的源代码发现使用的算法设计是比较精巧的，但从实现层面讲性能优化做得并不到位。于是也在 PyTorch 里面实现了同样的算法，同时也增加了更多 parallel 的优化。

我觉得这个算法挺有意思的，这篇就介绍一下 Unique 的优化，整体思路和前缀和 （Prefix Sum）的优化过程是很相似的。

介绍

Unique 这个算子的语义是找出一个 tensor 中的“独特”的元素，返回值可以有三个 tensor：

output: 输入中的“独特”元素；
inverse_indices: 输出的元素在输入序列中的位置；
count：独特元素的个数。

后两个输出是 optional 的，API 的相似定义参考 Unique，这里不过多赘述。

按着 issue reporter 的 benchmark 来测量，在我的机器上实测的原始结果如下，CPU 型号是 Intel(R) Xeon(R) Gold 6248 CPU @ 2.5GHz，测量采用 single socket (20 cores)：

Numpy just sort: 0.41129398345947266 s
Numpy sort + indexes: 6.422696590423584 s
Torch just sort: 9.109549283981323 s
Torch sort + indexes: 37.59021711349487 s

可以看到确实 torch 比 NumPy 还慢，这里 sort 指仅仅输出 output；而 sort + index 指输出（ouput, inverse_indices, count）。

问题定位

其实在定位问题，也就是为什么慢之前，往往先要确定“是不是慢？”，或者换一种说法，目前这个 kernel 还有没有继续提升的空间？我们有很多种手段可以做出判断，比如：

与其他框架对比，就像这个问题所述，是将 torch 和 numpy 作对比；有时候还可以比 TF，OpenCV，等等；
类比其他经过充分优化的算子，memory bandwidth bound 的算子可以类比 copy；compute bound 的可以类比 matmul；
计算理论性能数据，比如算 HW 提供的带宽和算力，折算出对应算子的理论执行时间，一般我们定义 performance roofline 的时候会做一个粗略的 analytical performance model；
查看 profiler，比如在 GPU 上我们用 nvprof；在 CPU 上我一般用 VTune，这个需要有一定专业背景知识。

当然最重要的还是直接看源代码，这个问题看一眼代码问题就很清楚了，但为了说明一下 VTune 的简单使用方法，我还是截了一份 VTune 的 log，如下：

Fig-1: VTune log on torch unique with input size of (1000, 1000, 32)

Summary 这个 Tab 里面可以看到并行化好不好；在 Bottom-up 这个 tap 一般是看 CPI Rate 这个指标，运行比较好的情况应该在 0.5 左右；这里面的 >5 肯定是很差的。从函数的名字能判断出事 std 里面的哈希表占了大头。

算法分析

通过源代码的分析，可以看到 NumPy 和 PyTorch 的 Unique 实现的思路是完全不一样的：

NumPy：通过 sort 讲 unique 的问题转换为 consecutive_unique，源代码位置在这里：link1
PyTorch：做哈希表，源代码位置在这里：torch/aten/src/ATen/native/Unique.cpp

用 Hash 的方式来处理这个算子很容易理解，不过性能就难以保障了，std::unordered_set 是串行的，另外对于内存的多次申请释放都是非常耗时的。相比之下，NumPy 的方式所有操作都可以有效并行，并且不存在内存的多次申请释放（也就是在预先计算出 output size 才会去申请 output tensor）。

优化版的 Unique Kernel

这里所实现的算法过程和 NumPy 完全一致，并从 performance optimization 的角度对实现的具体细节做了更多优化。代码都在这面这个 PR 里面：

Step 1，我们对 input 进行排序，范例结果如 Fig-2 所示：

  // original behavior with unique on scalar tensor
  // is to return a output size of ([1]), `flatten` here will do the job
  auto input_flattened = input.flatten();

  Tensor input_sorted, indices;
  std::tie(input_sorted, indices) = input_flattened.sort();

Fig-2: Sort on input sequence

这里有个小问题是：torch 目前没有对 float 的 1D tensor 做 parallel sort；但是 int 1D tensor 是有 parallel sort 的；所以最终的数据我测了 float 和 int 两组。

Step 2，经过排序之后，相同的元素都会被排在一起（NaN 会被 propagate 到最后面），接下来通过一个 mask 来记录每个相同元素集合中的“第一个”值，没错，他就是那个仔，这就是会被放到 output 中的那个元素。

  // `mask` keeps track of whether it is the first unique
  // in the sorted input sequence
  Tensor mask = at::empty({numel}, self.options().dtype(kBool));
  auto mask_acc = mask.accessor<bool, 1>();

  int num_threads = at::get_num_threads();
  std::vector<int64_t> unique_count_thread(num_threads, 0);
  std::vector<int64_t> offset_thread(num_threads, 0);

  // the first element is always true
  mask_acc[0] = true;

  // we can parallel on [1, numel) but we need to make sure it has
  // the same parallel scope with the next loop
  at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
    if (begin == 0) { begin += 1; }
    for (const auto i : c10::irange(begin, end)) {
      mask_acc[i] = input_sorted_data[i] != input_sorted_data[i - 1];
    }
  });


这一步也是可以并行的，另外需要两个额外的 vector 来记录每个 thread 上面遇到的 unique 元素的个数，就是 unique_count_thread。offset_thread 就是 unique_count_thread 的累计 (cumsum) 结果，这个是用来确定每个 thread 在 output tensor 中写入的 offset。

这个思路和 prefix sum 的并行化算法是一模一样的。

Fig-3: mask the 1st unique in each consecutive session

Fig-3 中的示意图为了简单只画了两个 thread。接下来就可以算出所有 unique 元素的个数，进而为 output 申请内存了。所以这个算法中不会对 output 进行 resize。

Step-3，在拿到每个 thread 在 output 中的 offset 之后根据 mask 对 output 进行赋值。

  at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    int64_t offset = offset_thread[tid];

    for (const auto i : c10::irange(begin, end)) {
      if (mask_acc[i]) {
        output_data[offset] = input_sorted_data[i];
        if (return_counts) {
          unique_index_data[offset] = i;
        }
        offset++;
      }

      if (return_inverse) {
        int64_t inverse_index = offset - 1;
        int64_t perm = indices_data[i];
        inverse_indices_data[perm] = inverse_index;
      }
    }
  });


这一步也是可以充分并行的，另外 offset - 1 就是 inverse_indices 的值。所以inverse_indices 和 count 的计算几乎是不花时间的，NumPy 在多了这两个额外输出的时候慢了那么多，是因为没优化好。。。

Fig-4: copy output per thread
结果

测试结果依旧是用 20 core 的 Cascade Lake，前面说过目前 torch 的 float 1D tensor 没有 parallel sort，所以把int 和 float 分开测，unit 都是 sec （the lower the better）:

Table-1: Performance on Int Tensor
Table-2: Performance on Float Tensor

可以看到在 fix parallel sort 之后，torch 可以做到在全部情况都超过 numpy；对于最后一种情况有超过 100 倍的提升。

Additional job: 性能应该还有进一步压榨的空间，目前 torch 对 CPU 的并行化实现方式是有一定限制的，无法精细地控制 parallel-sync-parallel 这种 loop。另外上述写法没有考虑每个 parallel loop 之间的 data cache 问题，我们看到这个算法里面对 input 和 mask 都有多次访问，如果尺寸太大的话，会因为 cache miss 导致性能下降的。在追求极限性能的时候这些都是要考虑的。
