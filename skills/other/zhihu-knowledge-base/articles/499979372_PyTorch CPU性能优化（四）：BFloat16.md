# PyTorch CPU性能优化（四）：BFloat16

**作者**: MingfeiPyTorch CPU Perf Maintainer

**原文链接**: https://zhuanlan.zhihu.com/p/499979372

---

​
目录
收起
BFloat16基础
优化PyTorch原生算子的原则
使用向量化
bf16/fp32数据类型转换
与Memory Format之间的关系
让BFloat16能跑起来
让BFloat16跑得快
减少冗余的dtype转换
用Float32做acc type
用float32存储中间变量
缓存输入数据

本篇是关于PyTorch CPU性能优化相关的简单入门教程的第四篇。

另外三篇：

马鸣飞：PyTorch CPU性能优化（一）：Memory Format 和 Channels Last 的性能优化
126 赞同 · 13 评论 文章
马鸣飞：PyTorch CPU性能优化（二）：并行化优化
54 赞同 · 5 评论 文章
马鸣飞：PyTorch CPU性能优化（三）：向量化
22 赞同 · 2 评论 文章

本章节对应英文版本在：

PyTorch CPU Performance Optimization Tutorial - Section IV
gist.github.com/mingfeima/fcd3c89e32983c6d7033693cea046e4a
BFloat16基础

BFloat16 (Brain Floating Point)[1][2]是一种16bit的浮点数格式，动态表达范围和float32是一样的，但是精度低。下一代的Xeon Sapphire Rapids上面可以使用AMX(Advanced Matrix Extensions)对卷积和矩阵乘的操作在BFloat16上进行加速，吞吐量比Float32高一个数量级。

这里主要介绍在PyTorch上面优化BFloat16原生算子的一些小技巧，侧重性能优化方面，不介绍BFloat16训练中涉及的调参问题。

优化PyTorch原生算子的原则

首先，很重要的一点是理解BFloat16其实更像是个“存储类型”，而不是真正的“数据类型”。硬件上面，而且是要支持AVX512-BF16或则AMXBF16的硬件上，只能计算dot producet或者是tile dot product，相关的指令是_mm512_dpbf16_ps和_tile_dpbf16ps，主要是用于计算密集型的OP，就是卷积和矩阵乘。对于其他的数学运算，比如加减乘除，BFloat16无法直接计算，需要转成float32然后再计算。

在PyTorch上面，BFloat16的优化是这样的：

nn.ConvNd 和 nn.Linear 使用oneDNN，也就是mkldnn；
对于其他的 nn OP 和tensor的OP，直接优化 PyTorch native kernel。

native kernel包括：

nn.BatchNorm - support mixed dtype
nn.LayerNorm - support mixed dtype
nn.GroupNorm
nn.{Max|Avg}PoolNd
nn.Adaptive{Max|avg}PoolNd
nn.ChannelShuffle
nn.PixelShuffle
nn.UpSample - 'nearest', 'bilinear', 'bicubic', 'trilinear'
Activations - ReLU, Silu, Prelu, etc.
Advanced Indexging - gather, scatter, etc.
ROIAlign, ROIPool (TorchVision)

等等等等，还有很多很多...

这是个非常庞大的工程...

从优化原生算子的角度出发，BFloat16和int8的优化思路其实非常相似：

	BFloat16	Int8
data type conversion	cvtbf16_fp32/cvtfp32_bf16	dequantize/quantize
arithmetic	convert to fp32	convert to fp32
accumulation	fp32	int32
non-arithmetic	copy as uint16_t	copy

“non-arithmetic”指的是哪些不需要计算的操作，比如tensor copy，transpose，shuffle等等。

使用向量化
bf16/fp32数据类型转换

BFloat16和float32之间的数据类型转换是比较慢的，目前PyTorch上面没有加入原生指令的支持，数据类型转换是多条指令完成的，bf16到fp32只需要移位填零，所以还好，fp32到bf16因为要实现Rounding的功能所以差不多需要20条指令才能完成。

这就对BFloat16的性能优化提出了两个要求：

尽量减少数据类型转换，有助于提升性能，同时也有助于提高计算精度；
尽量使用向量化的逻辑。
与Memory Format之间的关系

BFloat16的性能优化工作依赖于Channels Last上的优化工作，主要因为有些常用OP在Channels First (NCHW)上面是做不了向量化的，只能走Scalar逻辑，比如MaxPool2d。这个时候BFloat16会比float32还慢，对终端用户很不友好。

让BFloat16能跑起来

在PyTorch上面BFloat16是按照uint16_t来存储的，并重载了scalar和vector上的相关所有操作。也就是说BFloat16的加法被转义了，先convert成float32，然后加法，最后再convert回BFloat16。这样，利用Vectorized<BFloat16>我们可以随意构造vectorized kernel，也可以直接构造scalar的kernel，比如下面这两个例子：

/* 
 * Example-1: Use scalar overload
 */
for (int64_t i = 0; i < 16; ++i) {
  float input_val = BFloat16(input_data[i]);
  output_data[i] = BFloat16(input_val * 2.0);
}

/*
 * Example-2: Use vector overload
 */
using bVec = vec::Vectorized<BFloat16>;
using fVec = vec::Vectorized<float>;

bVec data_bvec = bVec::loadu(input_data);
fVec data_fvec0, data_fvec1;
std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
fVec out_fvec0 = data_fvec0 * fVec(2.0);
fVec out_fvec1 = data_fvec1 * fVec(2.0);
bVec out_bvec = convert_float_bfloat16(out_fvec0, out_fvec1);
out_bvec.store(output_data + d);

Example-2会比Example-1速度快很多！

int8的native kernel情况也是类似的，也要尽量使用向量化。

但是仅仅做到scalar和vector的重载，是远远不够的，这只能让BFloat16在PyTorch上跑起来。如果只停留在这个层面，BFloat16几乎是慢得没法用的。

让BFloat16跑得快
减少冗余的dtype转换

如果算子中包含一连串的数学运算，那么只需要2次dtype转换：首先，input从bf16转到fp32；然后进行多个数学运算；最后，output从fp32转回bf16。比如Sigmoid得计算：

/*
 * Example-3: sigmoid
 *
 * sigmoid will compute -, exp, +, /.
 * the code will also compile 2. with Vectorized<BFloat16>
 * but it will do bf16/fp32 dtype conversion for 4 times instead of 1.
 */
 
 // 1. BFloat16 vectorized path
 Vectorized<float> a0, a1;
 std::tie(a0, a1) = convert_bfloat16_float(a);
 a0 = (Vectorized<float>(static_cast<float>(1)) + a0.neg().exp()).reciprocal();
 a1 = (Vectorized<float>(static_cast<float>(1)) + a1.neg().exp()).reciprocal();
 return convert_float_bfloat16(a0, a1);
 
 // 2. float32 vectorized path
 a = Vectorized<scalar_t>(static_cast<scalar_t>(0)) - a;
 a = a.exp();
 a = Vectorized<scalar_t>(static_cast<scalar_t>(1)) + a;
 a = a.reciprocal();
 return a;
用Float32做acc type

在需要做accumulation时候，要用Float32做acc type，这么做不仅是为了提高性能，同样也是为了保障数值稳定性。比如Softmax中的accumulation操作是通过vec::reduce_all这个util来完成的，当输入数据是BFloat16的时候，vec::reduce_all会在float32上做累积：

/*
 * Example-4: reduction
 *
 * when scalar_t is BFloat16, the acc type is Float32.
 */
scalar_t max_input = vec::reduce_all<scalar_t>(
              [](Vec& x, Vec& y) { return vec::maximum(x, y); },
              input_data,
              dim_size);
用float32存储中间变量

计算过程中的中间变量要用float32来存储。比如MaxPool2d在channels last上的kernel，每个thread需要申请一个长度为‘channels’的临时buffer，用于存储计算max过程中的临时结果。这个buffer在不同的iteration之间是可以复用的，不需要反复申请。kernel位置在aten/src/ATen/native/cpu/MaxPoolKernel.cpp

缓存输入数据

如果我们需要多次访问输入数据，有个时候可以把输入数据先缓存成float32，这样可以省掉后续的dtype转换。一般要注意缓存数据的大小，要能对L1命中。

下面这个例子是优化LayerNorm在BFloat16上的性能的，采取了这个策略：#71376

/* Example-5: cache input and parameter in float32
 *
 * temp buffer holding input, gamma/beta (if defined) in float
 *
 * pre convert input slice to float has 2 benefits:
 *   a. Welford algorithm involves more arithmetic operations,
 *      this will reduce rounding error and improve performance.
 *   b. The input slice (float) can be reused when updating
 *      corresponding output slice.
 */
int64_t buffer_size = pre_convert_gamma_beta ? 3 * N : N;
std::unique_ptr<float []> buffer(new float[buffer_size]);
float* input_buffer_ptr = buffer.get();
float* gamma_buffer_ptr = nullptr;
float* beta_buffer_ptr = nullptr;
if (pre_convert_gamma_beta) {
  gamma_buffer_ptr = buffer.get() + N;
  beta_buffer_ptr = buffer.get() + 2 * N;
  vec::convert(gamma_data, gamma_buffer_ptr, N);
  vec::convert(beta_data, beta_buffer_ptr, N);
}

[TBD] Softmax BFloat16 optimization example。

参考
^https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
^https://cloud.google.com/tpu/docs/bfloat16?hl=en
