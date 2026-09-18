# PyTorch CPU性能优化（一）：Memory Format 和 Channels Last 的性能优化

**作者**: MingfeiPyTorch CPU Perf Maintainer

**原文链接**: https://zhuanlan.zhihu.com/p/494620090

---

​
目录
收起
Memory Format：Logical Order 和 Physical Order
Channels First 和 Channels Last
通过strides访问数据
Conv2d中存Memory Format的传递
范例：MaxPool2d
特例 I：Upsampling Kernel (CF) 的优化
特例 II: VGGM 中 AvgPool3d 的优化

本篇是关于PyTorch CPU性能优化相关的简单入门教程的第一篇。

另外三篇：

马鸣飞：PyTorch CPU性能优化（二）：并行化优化
54 赞同 · 5 评论 文章
马鸣飞：PyTorch CPU性能优化（三）：向量化
22 赞同 · 2 评论 文章
马鸣飞：PyTorch CPU性能优化（四）：BFloat16
60 赞同 · 22 评论 文章

本章节对应英文版本在：

Part I: Memory Formats and Channels Last Optimization
gist.github.com/mingfeima/f040ede25b4797740634ab778b2f5888




首先，关于这篇教程的一些基本说明：

该教程主要目标是介绍CPU性能优化的基础概念；
主要范例是PyTorch ATen native kernel的优化 （因为我大部分时间在优化ATen）;
文中的代码段如果当前还没有upstream，会贴上PR链接；
PyTorch的CPU kernel会用一些wrapper，比如at::parallel_for和Vectorized，仅仅是个wrapper，理解为omp parallel 和intrinsics就好；
oneDNN以前的名字叫"mkldnn"，目前PyTorch与其相关的函数名还是以mkldnn_开头；
CL是"Channels Last"的缩写，CF是"Channels First"缩写；
一般我们说OP (operator)指语义层面的模块，而kernel指特定device下OP的对应实现，implemetation 对应某个kernel下的不同版本。
Memory Format：Logical Order 和 Physical Order

在算子性能优化中，memory format是一个非常重要的概念。memory format指的是物理内存中线性存储的1D寻址空间，到逻辑上的一个ND Array的映射关系。有时候，也表达成data format或者layout。值得注意的是，layout在PyTorch中有不同的语义，指的是tensor的storage是dense还是sparse。PyTorch中还有另一种layout是torch._mkldnn，这个是mkldnn blocked格式。

所以，memory format的概念包含两方面含义：1) 物理上数据如何存储；2）逻辑上怎么解读这块memory。

Physical Order 就是指物理上数据如何存储。在CV领域，我们常说的NCHW或者NHWC指的就是物理内存的存储格式，NCHW也可以称为"channels first"，NHWC也可以成为"channels last"。选择不同的memory format的主要考量就是性能，有的OP对CL友好，也有的OP对CF友好，具体形况和OP的语义相关；
Logical Order是一个如何记录tensor shape和stride顺序的规则。PyTorch的logical order是NCHW，也就是说，无论pysical order是什么，shape, stride和index的顺序永远都是NCHW。
Channels First 和 Channels Last

在CV领域，两种比较常见的memory format是channels first (NCHW) 和channels last (NHWC)。以前做过的一个项目是优化一个叫Neon的AI框架，这个框架比较小众，它的memory format很特殊，是CHWN，这个格式对training很友好（N = 64, 128, 256 ... )。

Fig-1是CF和CL的一个示意图，假设Tensor 'A'的shape是[2, 3, 4, 4]，如何去访问到A[1][1][2][3]这个元素：

通过strides访问数据

Tensor的shape/strides和memory format有相应的对应关系，可以根据memory format来求出对应dimension的strides，同样根据strides也可以反推出memory format:

/*
 * (n, c, h, w) is the index
 * (N, C, H, W) is size of dimension
 *
 * Channels First (NCHW) strides:  (CHW, HW, W, 1)
 * Channels Last  (NHWC) strides:  (HWC, 1, WC, C)
 *                (CHWN) strides:  (1, HWN, WN, N)
 */

/* value for index n,c,h,w under memory format of NCHW */
scalar_t v = X[n * C * H * W + c * H * W + h * W + w];

/* value for index n,c,h,w under memory format of NHWC */
scalar_t v = X[n * H * W * C + h * W * C + w * C + c];

/* value for index n,c,h,w under memory format of CHWN */
scalar_t v = X[c * H * W * N + h * W * N + w * N + n];

事实上，PyTorch的Tensor里面并没有存一个叫memory format的attribute，而是记录strides，通过shape和strides的关系来判断 memory format。记录strides这个办法比较灵活，还可以完成很多其他功能，比如memory view, 非连续存储，等等。

Conv2d中存Memory Format的传递

在channels first上面直接优化Conv，无法达到最佳性能（那种可以退化成gemm的除外），主要原因是C上的访存不连续，而一般情况下我们需要在C上做vectorization。所以在channels first上面，input和weight首先需要转化成对CPU性能友好的blocked format (例如nChw16c, OIhw16i16o），这个过程称为"reorder"；再送给onednn的primitive计算；最后output还需要从blocked format转换回NCHW。这些reorder对整体性能来说是个累赘，占用了宝贵的memory带宽。

onednn的primitive可以直接在NHWC上面计算并达到很高的效率。所以在channels last上面，input和output不需要额外的reorder转换。但是weight还是需要使用blocked format，但对于inference的场景，可以通过weight prepack的方式（也就是提前做reorder并cache下来）的方式消除weight reorder对性能的影响。

Fig-2是PyTorch CPU上Conv2d memory format的传递方式:

一般来说，CL的性能要优于CF，因为可以省掉activation的reorder，这也是当初去优化channels last的最大动因。

另外，PyTorch上面的默认格式是CF的，对于特定OP来说，如果没有显示的CL支持，NHWC的input会被当作non-contiguous的NCHW来处理，从而output也是NCHW的，这带来的一个问题就是整个memory format传递的链条会被打断。也就是一开始虽然有做了to channels last的操作，但是跑着跑着就变回NCHW了。所以，让所有的对memory format敏感的OP都有CL的显示支持很重要，这方面的工作都记录在下面这个gist里面：

范例：MaxPool2d

对于memory format敏感的OP，有的是CL性能友好，有的是CF性能友好，这个和OP的具体操作有关：

CL性能优于CF的OP，如Conv2d，ConvTransposed2d，MaxPool2d，UpsampleNearest2d等等；
CL和CF性能相当的OP，如BatchNorm2d等等；
CL的性能略低于CF的OP，如GroupNorm2d，ChannelShuffle，PixelShuffle等等(这些OP在C上向量化比较麻烦)。

下面介绍一个简单的例子：MaxPool2d。对于channels first，一般情况这个kernel是无法向量化的，因为访存方式决定。我们可以在所有的dimension上做并行化，即NCHW。CF的kernel如下：

  // parallel on dim N, C, H, W
  at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
    int64_t c = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    data_index_init(begin, c, channels, oh, output_height, ow, output_width);

    for (const auto i : c10::irange(begin, end)) {
      int64_t ih0 = oh * dH - padH;
      int64_t iw0 = ow * dW - padW;
      int64_t ih1 = std::min(ih0 + (kH - 1) * dilationH + 1, input_height);
      int64_t iw1 = std::min(iw0 + (kW - 1) * dilationW + 1, input_width);
      while(ih0 < 0) { ih0 += dilationH; }
      while(iw0 < 0) { iw0 += dilationW; }

      // local pointers
      scalar_t* input_ptr = input_data + c * input_height * input_width;

      // compute local max
      int64_t maxindex = ih0 * input_width + iw0;
      accscalar_t maxval = -std::numeric_limits<accscalar_t>::infinity();
      for (int64_t ih = ih0; ih < ih1; ih += dilationH) {
        for (int64_t iw = iw0; iw < iw1; iw += dilationW) {
          int64_t index = ih * input_width + iw;
          accscalar_t val = accscalar_t(input_ptr[index]);
          if ((val > maxval) || std::isnan(val)) {
            maxval = val;
            maxindex = index;
          }
        }
      }

      // set output to local max and store location of max
      output_data[i] = scalar_t(maxval);
      indices_data[i] = maxindex;

      // move on to next output index
      data_index_step(c, channels, oh, output_height, ow, output_width);
    }
  });

注释：

上面这个kernel直接把NC作为一个dimension来计算（channels = nbatch * channels）,因为逻辑上NC之间操作是一致的；
at::parallel_for就是一个parallel runtime的wrapper，PyTorch上面可以选OpenMP或者TBB，默认是OpenMP；可以简单理解为#pragma omp parallel；
at::parallel_for一共4个参数，前两个定义问题大小（e.g. [0, numel)代表整个tensor），第三个参数是grain_size（就是parallel的切分力度，也就是每个thread上分到的payload最小值，默认是32K，有时需要根据具体情况调整）；最后一个参数是个lambda函数，也定义每个thread上做什么任务，[begin, end）是切分到对应thread的global index；
data_index_init和data_index_step是做indexing的util，后面章节会讲，这里当成parallel版本的for_each就行了。

对于channels last的情况，我们可以在C上做向量化，因为C刚好是inner most dimension，在剩下的维度上做并行化，即NHW：

  // parallel on dim N, H, W
  at::parallel_for(0, nbatch * output_height * output_width, 0, [&](int64_t begin, int64_t end) {
    int64_t n = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    data_index_init(begin, n, nbatch, oh, output_height, ow, output_width);

    int64_t size = channels;
    int64_t len = size - (size % Vec::size());
    // temp buffer holding index with integer_t
    std::unique_ptr<integer_t []> index_buffer(new integer_t[len]);

    for (const auto i : c10::irange(begin, end)) {
      int64_t ih0 = oh * dH - padH;
      int64_t iw0 = ow * dW - padW;
      int64_t ih1 = std::min(ih0 + (kH - 1) * dilationH + 1, input_height);
      int64_t iw1 = std::min(iw0 + (kW - 1) * dilationW + 1, input_width);
      while(ih0 < 0) { ih0 += dilationH; }
      while(iw0 < 0) { iw0 += dilationW; }

      scalar_t* out = output_data + i * channels;
      int64_t* ind = indices_data + i * channels;

      // Pass I: init out lane
      iVec index0_vec = iVec(ih0 * input_width + iw0);
      Vec out_vec = Vec(-std::numeric_limits<scalar_t>::infinity());
      int64_t d1 = 0;
      for (; d1 < len; d1 += Vec::size()) {
        index0_vec.store(index_buffer.get() + d1);
        out_vec.store(out + d1);
      }
      for (; d1 < size; d1++) {
        ind[d1] = ih0 * input_width + iw0;
        out[d1] = -std::numeric_limits<scalar_t>::infinity();
      }
      // Pass II: compute local max
      for (int64_t ih = ih0; ih < ih1; ih += dilationH) {
        for (int64_t iw = iw0; iw < iw1; iw += dilationW) {
          scalar_t* in = input_data + n * input_height * input_width * channels +
              ih * input_width * channels + iw * channels;

          int64_t d2 = 0;
          for (; d2 < len; d2 += Vec::size()) {
            iVec index_vec = iVec(ih * input_width + iw);
            Vec val_vec = Vec::loadu(in + d2);
            iVec maxindex_vec = iVec::loadu(index_buffer.get() + d2);
            Vec maxval_vec = Vec::loadu(out + d2);

            // true = all ones, false = all zeros
            Vec mask = (val_vec > maxval_vec) | val_vec.isnan();
            iVec imask = vec::cast<integer_t>(mask);
            Vec out_vec = Vec::blendv(maxval_vec, val_vec, mask);
            iVec ind_vec = iVec::blendv(maxindex_vec, index_vec, imask);

            out_vec.store(out + d2);
            ind_vec.store(index_buffer.get() + d2);
          }
          for (; d2 < size; d2++) {
            int64_t index = ih * input_width + iw;
            scalar_t val = in[d2];
            int64_t maxindex = ind[d2];
            scalar_t maxval = out[d2];

            bool mask = (val > maxval) || std::isnan(val);
            out[d2] = mask ? val : maxval;
            ind[d2] = mask ? index : maxindex;
          }
        }
      }
      // convert indice data type
      vec::convert<integer_t, int64_t>(index_buffer.get(), ind, len);

      // move on to next output index
      data_index_step(n, nbatch, oh, output_height, ow, output_width);
    }
  });

注释：

PyTorch规定index的dtype是int64，为了方便向量化，上面这个kernel是拿int32来算的，当然最后需要转一下；
Vec = at::vec::Vectorized<scalar_t>是个PyTorch上做向量化的wrapper，在不同的架构上编译成相应的汇编，如avx2, avx512，也可以在mobile平台编译；
上面的逻辑算是手动向量化，如果写成普通的for循环加#pragma omp simd，用ICC是可以自动向量化的（GCC如果要想自动向量化上述逻辑需要做点特殊处理，e.g. masking）。

抛开PyTorch的parallel wrapper和vectorization wrapper，上面的kernel其实非常简单，无外乎就是下面这两个图：

特例 I：Upsampling Kernel (CF) 的优化

下面介绍两个很有意思特例：一个是Upsamping在CF上的优化，另一个是AvgPool3d在CF上的优化。

一般CF上面我不会花太多力气，但也有例外，比如upsampling。主要两方面原因：1）upsampling很多时候是当作interpolate来用的；2）GAN的模型里面有个upsampling在conv之前，而我们在quantized model上面一般不加to channels last。两种情况下都无法保证输入是CL的，所以要对CF也做优化。

这个kernel主要的bottleneck在于input indice的计算，也就是对于每个output元素，都需要找出它对应的input在feature map上的offset。这个过程要做很多dtype转换，还有scale，最终只是copy一个float，如Fig-4(a)所示：

impl-1：最简单的是实现是像MaxPool2d CF那样写，直接在NCHW上并行，但这个时候多做了很多input indice的计算，因为对于每个output的HW平面，其对应的input feature map offset在不同的NC之间是不变的，所以input indice被多算了NC那么多倍。
impl-2：一个改进的想法是先把output对应的input indice算出来，做一个thread local的cache，这样没有input indice的重复计算；然后在NC上并行。这种实现比较挑input的shape，比如[1, 2048, 7, 7]就非常合适，但[1, 3, 768, 1024]就比impl-1还慢。原因两点：1) HW如果太大，那读写indice buffer的开销太大；2) NC太小，也就是问题规模太小，无法利用所有的CPU核心。
impl-3：算是1和2的一个折中方案。对于W cache住input indice，然后在NCH上并行化，如图Fig-4(b)所示。

这段code在这个PR里面#69600，是优化qupsample_nearest2d CF的：

    std::unique_ptr<int64_t []> input_offset_arr(new int64_t[output_width]);
    int64_t* input_offset = input_offset_arr.get();
    
    for (const auto w2 : c10::irange(output_width)) {
      const int64_t w1 = nn_compute_source_index_fn(width_scale, w2, input_width);
      input_offset[w2] = w1;
    }
    
    int64_t grain_size = internal::GRAIN_SIZE / std::max(int64_t{1}, output_width);
    at::parallel_for(0, channels * output_height, grain_size, [&](int64_t begin, int64_t end) {
      int64_t nc{0}, h2{0};
      data_index_init(begin, nc, channels, h2, output_height);
      
      for (const auto i : c10::irange(begin, end)) {
        const int64_t h1 = nn_compute_source_index_fn(height_scale, h2, input_height);
        const auto* pos1 = &i_p[nc * input_height * input_width + h1 * input_width];
        auto* pos2 = &o_p[i * output_width];
        
        for (const auto w2 : c10::irange(output_width)) {
          const int64_t w1 = input_offset[w2];
          pos2[w2] = pos1[w1];
        }
        
        data_index_step(nc, channels, h2, output_height);
      }
    });

实际上，我们还可以更进一步：GAN模型及其变种用到的upsampling很多时候scale factor都是2。一般情况upsampling CF是做不了向量化的，但是如果scale factor是2就可以。在这种情况下，我们可以省掉input indice的计算，直接做个interleave copy就行了（使用vec::interleave2<>即可）：

//interleave copy
Vec o1, o2;
std::tie(o1, o2) = interleave2(a, a);
 
// interleave2
template <>
std::pair<Vectorized<float>, Vectorized<float>>
inline interleave2<float>(const Vectorized<float>& a, const Vectorized<float>& b) {
  // inputs:
  //   a = {a0, a1, a2, a3, a4, a5, a6, a7}
  //   b = {b0, b1, b2, b3, b4, b5, b6, b7}

  // swap lanes:
  //   a_swapped = {a0, a1, a2, a3, b0, b1, b2, b3}
  //   b_swapped = {a4, a5, a6, a7, b4, b5, b6, b7}
  // TODO: can we support caching this?
  auto a_swapped = _mm256_permute2f128_ps(a, b, 0b0100000);  // 0, 2.   4 bits apart
  auto b_swapped = _mm256_permute2f128_ps(a, b, 0b0110001);  // 1, 3.   4 bits apart

  // group cols crossing lanes:
  //   return {a0, b0, a1, b1, a2, b2, a3, b3}
  //          {a4, b4, a5, b5, a6, b6, a7, b7}
  const __m256i group_ctrl = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
  return std::make_pair(_mm256_permutevar8x32_ps(a_swapped, group_ctrl),
                        _mm256_permutevar8x32_ps(b_swapped, group_ctrl));
}

这样就能达到最佳性能，应该与tensor copy性能相仿或稍微慢一点，不过肯定远远快于上述任何一种实现。

特例 II: VGGM 中 AvgPool3d 的优化

一般情况AvgPool3d在CF上也是没法向量化的，但也有特例，比如VGGM中的用法。这个模型中AvgPool3d的用法很奇特，其实是为了在C上做average，特意把4D tensor view成了5D，之后走一个kernel size是[K, 1, 1]的AvgPool3d。因为HW上kernel都是1，就有了操作空间，可以直接在HW上向量化，因为访存连续；对于并行化有两个选项：

如果NC够大，比如 NC=64，那就直接在NC上并行化；
如果NC不够大，比如 NC=3，那需要在NCD上并行化；

整个过程如Fig-5所示：

在这个模型中，以AvgPool3d为核心，其实还可以把前前后后几个mul，add的操作都fuse成一个kernel，进一步提高性能。
