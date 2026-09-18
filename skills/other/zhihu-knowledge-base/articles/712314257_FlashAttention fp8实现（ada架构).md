# FlashAttention fp8实现（ada架构)

**作者**: weishengying

**原文链接**: https://zhuanlan.zhihu.com/p/712314257

---

​
目录
收起
主要改动：
MMA 指令集的选择
SM80_16x8x16_F16F16F16F16_TN
SM89_16x8x32_F32F8F8F32_E4M3_TN
gemm-I 输出结果的转换
讨论

本 blog 是在 shengying.wei：FlashAttention 笔记 的基础上进一步的尝试，在 ada（sm_89）架构上实现 flash attention 的 fp8 版本，适合学习。代码在：

GitHub - weishengying/cutlass_flash_atten_fp8: 使用 cutlass 仓库在 ada 架构上实现 fp8 的 flash attention

该代码也是在: @66RING https://github.com/66RING/tiny-flash-attention 的基础上做的修改。

benchmark如下：

fp8 和官方 fp16 实现的性能对比
主要改动：

核心改动只有两点：

选择在 sm89 上的 fp8 mma 指令集，对应调整一些 shared memory 的layout， swizzle 定义等
调整第一个 gemm 结果 C的 Layout，使其符合第二个 gemm 对 A 的 layout 要求，包括线程内部的数据交换和 layout 定义




MMA 指令集的选择
SM80_16x8x16_F16F16F16F16_TN

在cute中，目前没有封装在 sm89 架构上的 fp8 的mma 指令集，因此需要封装一下。

封装之前先学习 SM80 上的 MMA 指令，以 SM80_16x8x16_F16F16F16F16_TN 为例。

代码如下：

// MMA 16x8x16 TN
struct SM80_16x8x16_F16F16F16F16_TN
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0,  %1},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  %9};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM80_16x8x16_F16F16F16F16_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};


这里需要注意一下，寄存器的个数，查看 mma.sync.aligned.m16n8k16 ptx 指令集文档：

每个线程需要处理A（m=16， k=16）中的元素个数为： 16*16/32 =8，8个半精度元素用 4 个 uint32_t即可储存，故 :

ARegisters =uint32_t[4];

同理，每个线程需要处理B和C中的元素个数为：16*8/32 =4，故其他三个寄存器的大小为2。

此外，该 MMA 指令集还有对应的 Traits，描述了该指令集对输入输出的数据类型要求以前期望的Layout，

代码如下：

// (T32,V4) -> (M16,N8)
using SM80_16x8_Row = Layout<Shape <Shape < _4,_8>,Shape < _2,_2>>,
                             Stride<Stride<_32,_1>,Stride<_16,_8>>>;

template <>
struct MMA_Traits<SM80_16x8x16_F16F16F16F16_TN>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using Shape_MNK = Shape<_16,_8,_16>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _2,_2,  _2>>,
                         Stride<Stride<_32,_1>,Stride<_16,_8,_128>>>;
  using BLayout = Layout<Shape <Shape < _4,_8>,Shape <_2, _2>>,
                         Stride<Stride<_16,_1>,Stride<_8,_64>>>;
  using CLayout = SM80_16x8_Row;
};


其中 ValTypeD 等定义了输入输出的dtype，Shape_MNK 定义了该 MMA atom 能够处理的问题大小，安培架构上，MMA 指令在 wrap level 执行，故 ThrID layout 的 size 为 32。

ALayout， BLayout，CLayout，描述了（wrap_thread_id, value_id） 到 A,B,C 中对应元素的坐标（m, n）的映射关系。

同时对坐标信息(m, n)编码一下，编码方式如下：

SM80_16x8x16_F16F16F16F16_TN 对应的 MMA latex 图如下：

以 CLayout 为例子，已知：

 using CLayout = SM80_16x8_Row = Layout<Shape <Shape < _4,_8>,Shape < _2,_2>>,                              
                                               Stride<Stride<_32,_1>,Stride<_16,_8>>>;

不妨打印出来(部分截图)：

可以看出，对应关系如下：

ALayout， BLayout 的意义和上述一致。

SM89_16x8x32_F32F8F8F32_E4M3_TN

在 sm89 上支持的 fp8 指令集如下：

mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32


该 ptx 指令集对应的文档如下：

注意该指令输入是 fp8，输出是 fp32。

处理 A 的大小为（m=16， k = 32），每个线程处理16*32/32 = 16 个 fp8 数据，对应内存16 byte，即四个 uint32_t，

处理 B 的大小为（n=8， k = 32），每个线程处理8*32/32 = 8个 fp8 数据，对应内存 8 byte，即两个uint32_t，

处理 C\D 的大小为（m=16， n = 8），每个线程处理16*8/32 = 4 个 fp32 数据，即四个 float。

故该 ptx 指令集封装如下：

// 添加FP8相关的SM89指令支持
// MMA 16x8x32 TN
struct SM89_16x8x32_F32F8F8F32_E4M3_TN
{
    using DRegisters = float[4];
    using ARegisters = uint32_t[4];
    using BRegisters = uint32_t[2];
    using CRegisters = float[4];

    CUTE_HOST_DEVICE static void
    fma(float    & d0, float      & d1, float      & d2, float      & d3,
        uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
        uint32_t const& b0, uint32_t const& b1,
        float const& c0, float const& c1, float const& c2, float const& c3)
    {
        asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
            "r"(b0),  "r"(b1),
            "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));

    }
};


MMA_Traits 定义如下：

template <>
struct MMA_Traits<SM89_16x8x32_F32F8F8F32_E4M3_TN>
{
     using ValTypeD = float;
     using ValTypeA = cutlass::float_e4m3_t;
     using ValTypeB = cutlass::float_e4m3_t;
     using ValTypeC = float;

     using Shape_MNK = Shape<_16,_8,_32>;
     using ThrID   = Layout<_32>;
     using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _4,_2,  _2>>,
     Stride<Stride<_64,_1>,Stride<_16,_8,_256>>>;
     using BLayout = Layout<Shape <Shape < _4,_8>, Shape <_4,  _2>>,
     Stride<Stride<_32,_1>, Stride<_8,_128>>>;
     using CLayout = SM80_16x8_Row;
};


可以使用下面代码打印一下刚刚定义的 MMA_Atom。

int main(int argc, char** argv)
{
  using namespace cute;
  using MMA_Atom_Arch = MMA_Atom<SM89_16x8x32_F32F8F8F32_E4M3_TN>;

  using TiledMma = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<_1,_1,_1>>,
        Tile<_16, _8, _32>>;

  print_latex(TiledMma{});
}





SM89_16x8x32_F32F8F8F32_E4M3_TN

读者可自行打印验证 ALayout 和 BLayout 与 SM89_16x8x32_F32F8F8F32_E4M3_TN 要求的各个线程负责的数据排布一致。

gemm-I 输出结果的转换

代码中，定义的 TileMMA 如下，在 M 方向上做了 thread repeat，在 N 方向上做了 value repeat:

using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        Tile<Int<16 * kNWarps>, _16, _32>>;


对用的 MMA latex 图如下（只看一个 wrap 内的 LayoutA 和 LayoutC 部分）：

左边是 layoutA， 右边是 layoutC

可以看出，gemm-I 的输出（寄存器中） LayoutC 不符合 gemm-II 对 LayoutA 的要求。

对于 gemm-II 来说，如 T0 线程需要的四个数据在 T0和T1中，T1线程需要的四个数据在 T2和T3中，因此需要线程内部做数据交换，这部分对应代码中：

https://github.com/weishengying/cutlass_flash_atten_fp8/blob/main/csrc/flash_attention.cu#L525

https://github.com/weishengying/cutlass_flash_atten_fp8/blob/main/csrc/reg2reg.h#L31

原本的 fp16 的实现中，使用的 SM80_16x8x16_F16F16F16F16_TN 指令集，LayoutA 和 LayoutC 对应的 latex 如下，

可以看出，gemm-I 的输出（已经在寄存器中了）可以直接被 gemm-II 作为输入A，不需要做线程内部的数据交换。https://github.com/weishengying/tiny-flash-attention/blob/main/csrc/flash_attention.cu#L556

讨论

上面说的第二点可能可以通过修改 TileMMa 的定义让 LayoutA 和 LayoutC 一致，如下面定义：

  using namespace cute;
  using MMA_Atom_Arch = MMA_Atom<SM89_16x8x32_F32F8F8F32_E4M3_TN>;
  using TiledMma = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<_1,_1,_1>>,
        Tile<_16,
            Layout<Shape <_2,_4,_2>, 
                    Stride<_1,_4,_2>>, // Permutation on N, size 16                     
            _32>>;

//   using TiledMma = TiledMMA<
//         MMA_Atom_Arch,
//         Layout<Shape<_1,_1,_1>>,
//         Tile<_16, _16, _32>>;

  print_latex(TiledMma{});


这种定义的意义可以参考官方 git 上的讨论：

https://github.com/NVIDIA/cutlass/discussions/1345

对应的 latex 中的LayoutA 和 Layout C 如下：

此时，gemm-I 的输出结果 LayoutC 符合 gemm-II 对输入 A 的Layout 要求，因此可能不需要线程内部进行数据交换了。
