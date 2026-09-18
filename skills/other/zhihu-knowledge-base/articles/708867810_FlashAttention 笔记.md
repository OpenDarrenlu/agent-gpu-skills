# FlashAttention 笔记

**作者**: weishengying

**原文链接**: https://zhuanlan.zhihu.com/p/708867810

---

​
目录
收起
参考
1. online softmax
1.1) safe softmax
1.2) algorithm 3-pass safe softmax
1.3) 2-pass online softmax
2. Multi-Pass Self-Attention
3. One-pass Self-Attention
4. FlashAttention V1
4.1）flash-attention-minimal
5. FlashAttention V2
6. 源码解读 —— tiny-flash-attention
6.1）TiledMMA
6.2）SmemLayoutQ、SmemLayoutKV
6.3）TiledCopy（gmem --> smem）
6.4）Ldmatrix (smem --> rgister)， warp level load instruction
6.5）causal mask
6.6）Softmax、rescale, P_j^j @ V_j 等
参考

From Online Softmax to FlashAttention

FlashAttention: Fast and Memory-Eﬃcient Exact Attention with IO-Awareness

FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning

1. online softmax
1.1) safe softmax

普通的softmax计算公式如下：
𝑠
𝑜
𝑓
𝑡
𝑚
𝑎
𝑥
(
{
𝑥
1
,
.
.
.
,
𝑥
𝑛
}
)
=
{
𝑒
𝑥
𝑖
∑
𝑗
=
1
𝑁
𝑒
𝑥
𝑗
}
𝑖
=
1
𝑁
为了避免数值溢出，根据指数运算的性质，实际计算如下：

𝑒
𝑥
𝑖
∑
𝑗
=
1
𝑁
𝑒
𝑥
𝑗
=
𝑒
𝑥
𝑖
−
𝑚
∑
𝑗
=
1
𝑁
𝑒
𝑥
𝑗
−
𝑚
其中， 
𝑚
=
𝑚
𝑎
𝑥
𝑗
=
1
𝑁
(
𝑥
𝑗
)

1.2) algorithm 3-pass safe softmax

因此，对一个向量 
𝑥
 计算 softmax 的最基本的流程如下：

𝑚
𝑖
:
𝑚
𝑎
𝑥
𝑗
=
1
𝑖
{
𝑥
𝑗
}
 ，初始值 
𝑚
0
=
−
∞
 ， 
𝑚
 是一个标量

{
𝑙
𝑖
}
:
∑
𝑗
=
1
𝑖
𝑒
𝑥
𝑗
−
𝑚
𝑁
 ，每个元素减去最大值求EXP然后累计求和，初始值 
𝑙
0
=
0
 ， 
𝑙
 是一个标量

{
𝑎
𝑖
}
:
 最后的 softmax 值，得到结果向量 
𝑎

for 
𝑖
 1,N do:
𝑚
𝑖
=
𝑚
𝑎
𝑥
(
𝑚
𝑖
−
1
,
𝑥
𝑖
)
end

for 
𝑖
 1,N do:
𝑙
𝑖
=
𝑙
𝑖
−
1
+
𝑒
𝑥
𝑖
−
𝑚
𝑁
end

for 
𝑖
 1,N do:
𝑎
𝑖
=
𝑒
𝑥
𝑖
−
𝑚
𝑁
𝑙
𝑁
end

这种算法需要三次 for 循环，第一个为了得到最大值，第二个为了对每个元素求指数然后累计求和，最后一个for循环求目标值。效率比较低，它的改进版可以减少一次for 循环。

1.3) 2-pass online softmax

首先，我们修改下 l_i 的定义，在循环到第 i 次时，我们只能够知道当前的最大值 m_i = max_{j=1}^{i}(x_j) ，并不知道全局的最大值，我们只能根据当前的最大值来对每个元素做 EXP 处理并求和，故 l_i = \displaystyle\sum_{j=1}^i e^{x_j - m_i} ，很显然，当循环结束时， i=N， l_N 就是根据全局最大值对每个元素做EXP处理并求和的值。

算法流程如下：

for i 1,N do: m_i = max(m_{i-1}, x_i)\\ l_i = l_{i-1} e^{m_{i-1}-m_i}+ e^{x_i - m_i}\\end

for i 1,N do: a_i = \frac{e^{x_i-m_N}}{l_N}\\end

这里重点说明下l_i = l_{i-1} e^{m_{i-1}-m_i}+ e^{x_i - m_i} 这一步的意义。首先 e^{x_i - m_i} 是循环到第 i 次时，根据当前最新最大值 m_i 对元素 x_i 求指数，还需要根据当前最新最大值 m_i 对前 （i-1）个数求指数然后求和，但是 l_{i-1} 是上一次循环时，根据上一次的最大值 m_{i-1} 对前 （n-1）个数求指数并求和的结果。因此需要将 l_{i-1} 更新为是利用当前最新最大值 m_i 进行求指求和的结果（个人认为这是全文最关键的一点，在论文中叫做 rescale）。l_{i-1}^{new} = \displaystyle\sum_{j=1}^{i-1}e^{x_j - m_i} = \displaystyle\sum_{j=1}^{i-1}e^{x_j - m_{i-1}}e^{m_{i-1}-m_i} =l_{i-1}e^{m_{i-1}-m_i}\\ 这样，只需要两个 for 循环就可以完成 softmax 计算。

2. Multi-Pass Self-Attention

先定义一下基础符号：

Q[k,:]: 矩阵 Q 的第 k 行

K^T[:,i]: 矩阵 K 转置后的第 i 列

O[k,:]: 输出矩阵 O 的第 k 行

V[j,:]: 矩阵 V 的第 j 行

\{ \bm{o} \}:\sum_{m=1}^{j}a_mV[m, :] = o_{j-1} + a_jV[j,:]: 表示求 O 的第 k 行 O[k,:] 时中间第 j 步的中间结果，这个向量是一步一步累积求得的， o_j 表示累积求和过程中的第 j 步。

用图形表示计算过程如下：

对应公式的计算过程如下：

for i 1,N do: x_i =Q[k,:]K^T[:,i] \\ m_i = max(m_{i-1},x_i) \\ l_i = l_{i-1}e^{m_{i-1}-m_i}+e^{x_i-m_i}\\end

for j 1,N do: a_j = \frac{e^{x_j-m_N}}{l_N}\\ \bm{o}_j = \bm{o}_{j-1} + a_jV[j,:]\\end

O[k,:] = o_N\\

这种算法需要两次for循环，可以将 online-softmax 的思想拓展一下，计算 a_j 的时候，并不一定非要知道 l_N和 m_N ，而是使用 l_i和 m_i ， a_i = \frac{e^{x_i - m_i}}{l_i} ，然后该元素和 V的第 i 行做乘法，注意这时候，最新的最大值已经是 m_i 了，根据最新最大值进行求指求和的结果是 l_i，那么 o_j = \displaystyle\sum_{j=1}^{i}\frac{e^{x_j-m_i}}{l_i}V[j,:] 。

3. One-pass Self-Attention

fori 1,N do: x_i =Q[k,:]K^T[:,i] \\ m_i = max(m_{i-1},x_i) \\ l_i = l_{i-1}e^{m_{i-1}-m_i}+e^{x_i-m_i}\\ a_i = \frac{e^{x_i - m_i}}{l_i}\\ \bm{o}_i = \displaystyle\sum_{j=1}^{i}\frac{e^{x_j-m_i}}{l_i}V[j,:]\\end O[k,:] = \bm{o}_N\\我们将 \bm{o_i} 改为迭代的形式，

\bm{o}_i = \displaystyle\sum_{j=1}^{i}\frac{e^{x_j-m_i}}{l_i}V[j,:]=\displaystyle\sum_{j=1}^{i-1}\frac{e^{x_j-m_i}}{l_i}V[j,:] + a_iV[i,:]\\其中：

\displaystyle\sum_{j=1}^{i-1}\frac{e^{x_j-m_i}}{l_i}V[j,:] = \displaystyle\sum_{j=1}^{i-1}\frac{e^{x_j-m_i}l_{i-1}}{l_il_{i-1}}V[j,:] = \displaystyle\sum_{j=1}^{i-1}\frac{e^{x_j-m_{i-1}}e^{m_{i-1}-m_i} l_{i-1}}{l_{i-1}l_i}V[j,:]= \bm{o}_{i-1}\frac{l_{i-1}e^{m_{i-1}-m_i}}{l_i}

故得到 one-pass 的 self-attention 的形式为：

fori 1,N do: x_i =Q[k,:]K^T[:,i] \\ m_i = max(m_{i-1},x_i) \\ l_i = l_{i-1}e^{m_{i-1}-m_i}+e^{x_i-m_i}\\ \\ \bm{o}_i = \bm{o}_{i-1}\frac{l_{i-1}e^{m_{i-1}-m_i}}{l_i} + \frac{e^{x_i - m_i}}{l_i}V[i,:]\\end O[k,:] = \bm{o}_N\\

4. FlashAttention V1

flashatten 在上面的基础上，做一下 Tiling，其核心思想与上述 one-pass self-attentio完全一致。

Q,K,V三个矩阵的shape 为：Q,K,V:(N,d) ，Br 行为一块，对Q，O 进行分块，Bc 列（行）为一块，对K^T和V进行分块。如下图所示：

flash-attention

图中的 m{ij} 对应公式中的上面带有波浪线的 mij。这里把上面推导出 one-pass self-attention 的计算公式也放在这里和 flash-attention 的进行对比。

flash-attention 图中第 10 步，计算完 m{ij}后，这是一个 block 的最大值，是局部最大值。后面然后用这个局部最大值去计算 EXP然后求和，即 P{ij} 和 l{ij}，这是不太符合直观的，根据当前局部的最大值，和上一次循环的最大值 m_i， 可以先算出最新的最大值 m_i_new, 即在“当前环境下”的最大值，用 m_i_new 来计算 EXP 然后求和才更符合直觉。所以在第11步计算 l_i^{new} 时，加法后面那一项可以理解为对这个“失误”的更正。

e^{\tilde{m}_{ij} - m_i^{new}}\tilde{l}_{ij} = e^{\tilde{m}_{ij} - m_i^{new}}\sum exp(S_{ij}-\tilde{m}_{ij})=\sum exp(S_{ij}-m_i^{new})\\ 第12 步中也有一样的操作，这样 flash-attention 背后的数学逻辑和 one-pass self-attention 完全一致。（这样看起来 flash-attention 有一些冗余的计算）

4.1）flash-attention-minimal

这里有一个很有教育意义的最小的 flash-attention demo，

我在代码里做了一些注释，kernel代码如下，可以用这个demo来熟悉 one-pass self-attention 和 flash-attention v1的算法逻辑。

__global__
void forward_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float *m, float* O) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for l and m

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];

    for (int j = 0; j < Tc; j++) {

        // Load Kj, Vj to SRAM
        for (int x = 0; x < d; x++) { // Kj, Vj: (Bc, d)
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x]; // 一共Bc个线程，每个线程负责一列，每个线程负责的列的起始地址为：tx*d，然后行方向上循环,注意，转置之后，该矩阵列优先
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x]; // 一共Bc个线程，每个线程负责一行，每个线程负责的行的起始地址为：tx*d，然后列方向上循环，该矩阵行优先
        }
        __syncthreads();  // such that the inner loop can use the correct Kj, Vj

        for (int i = 0; i < Tr; i++)  {

            // Load Qi to SRAM, l and m to registers
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            }
            float row_m_prev = m[lm_offset + (Br * i) + tx]; //m 和 l 在 gobal mem 中，对于每个batch的每个 head 来说，m和l的shape 为(N,1)
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            // S = QK^T, row_m = rowmax(S)
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) { //Qi:(Br, d), K^Tj: (d, Bc)
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x]; //注意，设置的时候设置的 Bc = Br, CTA有Br个线程，因此每个线程负责 S:(Br, Bc) 中一行的计算，每个thread访问的Qi对应行的起始地址为 tx*d
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum; // tx 行 y 列

                if (sum > row_m)
                    row_m = sum;
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0;
            for (int y = 0; y < Bc; y++) { // S:(Br, Bc)
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            // Compute new m and l
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // Write O, l, m to HBM
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                // O = S:(Br, Bc) * (Bc, d)
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];// 依然是每个 thread 负责一行
                }
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / row_l_new) \
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tx * d) + x]) \
                    + (__expf(row_m - row_m_new) * pv));
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads();  // otherwise, thread can use the wrong Kj, Vj in inner loop
    }
}

5. FlashAttention V2

长话短说，FA2 的更新主要在以下三个方面：

1）减少一些非 matmul 计算

前面推导过，FA1的理论计算如下形式：

one-pass self-attention

其中 \bm{o_i} 是根据 \bm{o_j} = \displaystyle\sum_{j=1}^{i}\frac{e^{x_j-m_i}}{l_i}V[j,:] 推导而来的，计算过程中，可以不除以分母项，等循环结束后，除以 l_N 即可。

故： \bm{o_j} = \displaystyle\sum_{j=1}^{i}e^{x_j-m_i}V[j,:] = \displaystyle\sum_{j=1}^{i-1}e^{x_j-m_{i-1}}V[j,:] *e^{m_{i-1}-m_i}+e^{x_i-m_i}V[i,:] = \bm{o_{j-1}}e^{m_{i-1}-m_i} + e^{x_i-m_i}V[i,:]

所以 flash-attention2 的核心逻辑如下：

for i 1,N do: x_i =Q[k,:]K^T[:,i] \\ m_i = max(m_{i-1},x_i) \\ l_i = l_{i-1}e^{m_{i-1}-m_i}+e^{x_i-m_i}\\ \\ \bm{o}_i = \bm{o}_{i-1}{e^{m_{i-1}-m_i}} + {e^{x_i - m_i}}V[i,:]\\end O[k,:] = \frac{\bm{o}_N}{l_N}\\

对照论文里的流程：

（ L_i 是为了反向计算时用的，只关注推理的话可以忽略）

后面两点优化是实现时的优化。

2) Q_i 方向上的并行

算法逻辑上有两层循环，外层Q，内层KV，实现时外层Q得循环可以用 thread block 并行执行替换。用论文中的图表示如下：

we parallelize the workers (thread blocks) where each worker takes care of a block of rows of the attention matrix.

worker1-5 实现中表示不同的 thread block。由于因果 mask 的原因，所以是阶梯状。

3）warp内部的划分

矩阵乘法 S_{ij}=Q_iK_{j}^{T} 中，一个 CTA（thread block）会处理一个shape 为（kBlokckM， kBlockN）的块，CTA 内部有多个 wrap，每个 warp 的处理逻辑划分如下：

这样做有什么好处呢。

后续代码中读者可以感受到这种的好处。在实现上，为了实现这种方式，在 TiledMMA 定义时，只在 M 方向上做线程拓展，N不做线程拓展。这样保证了一个 warp 可以处理 S_{ij} 中所有的列，在求 rowsum 和 rowmax 时，只需要warp内部调用线程束洗牌指令即可完成对应操作，不需要通过shared memory进行warp之间的数据交换，减少了shared memory的读写。

6. 源码解读 —— tiny-flash-attention

tridao 版本的 flash-attention 内容太多（考虑了各种场景），这里可以先从 https://github.com/66RING/tiny-flash-attention/blob/main/flash_attention_cutlass/csrc/flash_attention.cu 的代码开始看，

我在作者的基础上去掉了一些冗余的代码、简化一些 cute layout 的定义，整体逻辑完全一致，适合学习，性能上略低于官方库的性能（没有做tile调整等原因）。

6.1）TiledMMA
using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>
    >;
using ValLayoutMNK = Layout<Shape<_1, _2, _1>>;
using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // kNWarps = kNWarps, 4x1x1
        // NOTE: cutlass v3.3
        // typename Base::ValLayoutMNK>; // 1x2x1 for 16x16x16 MMA and LDSM
        // cutlass v3.4
        Tile<Int<16 * kNWarps>, _16, _16>>;


TiledMMA 定义了一个 CTA 如何协同处理 shape 为（MNK）的矩阵乘法，里面有 4 个 warp （kNWarps），即128个线程，一次 thread block level 循环可以处理 MNK 为（64, 16, 16）的矩阵乘法。

6.2）SmemLayoutQ、SmemLayoutKV
    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{}, // kSwizzle = 2 or 3
                    Layout<Shape<_8, Int<kBlockKSmem>>, // kBlockKSmem = 32 or 64
                           Stride<Int<kBlockKSmem>, _1>>{}));

    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));


定义 Q_i 和 K_j 的 shared memory，大小分别为(Br, d)和（Bc, d)，(其中Br = kBlockM, Bc = kBlockN, d = kHeadDim)使用 Swizzle 语义来避免 bank confilct。

6.3）TiledCopy（gmem --> smem）

TiledMMA 中确定了 thread block 中线程的数量（kNThreads），TiledCopy 需要组织这些线程按照一种方式将 global memory中的数据拷贝到 shared momory 中，如 TileMMA 定义一个 CTA 计算的方式一样，TileCopy 定义了一个 CTA做内存传输的方式。（同时使用向量化异步拷贝指令）。

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");

    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;

    // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
    // from the same address by the same threadblock. This is slightly faster.
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read
    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store


kNThreads 个线程被组织为 （kNThreads / kGmemThreadsPerRow>, kGmemThreadsPerRow）形式的layout，每个线程拷贝一行的 8 个数据。

假设是半精度数据，则GmemTiledCopyQKV 一次 thread block level 循环可以拷贝 (16, 8x8) 这么多个元素。GmemTiledCopyQKV 在SRC:(64,64) tensor 和 DST:(64,64) tensor 上周期性平铺。

虽然 TiledCopy 在 thread block level 层面定义了一个 CTA 如果做内存传输的行为，但是具体的行为还是每个 thread 来执行的。通过 partition_S 和 partition_D api 得到每个线程需要copy 的元素个数，即下面代码中tQgQ， tQsQ 等shape 为（（8,1),4,1)。第一个（8,1)是 TiledCopy 定义的Val layout，8 vals per read，后面4，1 是Gme4mTiledCopyQKV在目标tensor上的周期性平铺。（4=64/16， 1=64/(8x8))

  Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ(_, _, 0)); // gQ(_, _, 0):(kBlockM, kHeadDim) tQgQ:()
  Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ); // sQ:(kBlockM, kHeadDim).   tQsQ:((8,1),4,1)
  Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
  Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);


和上述类似，TiledMMA 在 thread block level 层面定义了一个 CTA 如果做矩阵乘的行为，但是具体的执行还是每个线程，通过 partition_fragment_A， partition_fragment_B api 可以获得每个线程需要处理的寄存器数据。

  Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
  Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)


sQ:(64, 64), TiledMMA:MK (64,16),根据 tensor core 的特性，可以推断出 tSrQ 的shape 为((2,2,2),1,4)

sK:(64, 64), TiledMMA:NK:(16/2,16),根据 tensor core 的特性，可以推断出 tSrQ 的shape 为((2,2),8,4)

（TiledMMA:NK:(16/2,16) 中除以2的原因是，在N方向上没有线程的拓展，只是寄存器的拓展）

6.4）Ldmatrix (smem --> rgister)， warp level load instruction

根据 tensor core （mma）和 ldmatrix 指令的特点，这是 warp level 层面的行为，每个线程 copy 8 个元素到线程的寄存器中，然后线程之间交换数据，每个线程得到自己想要的数据，最后符合mma指令对数据的排布要求。

https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-fragments-for-mma-m16n8k16-with-floating-point-type

  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
  auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
  Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

  auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
  Tensor tSsK = smem_thr_copy_K.partition_S(sK);


SM75_U32x4_LDSM_N 里面封装了 ldmatrix 指令集，因为拷贝行为和选择的 mma 指令集相关，所有建构 tiled_copy_Q 对象时也需要传入 tiled_mma 对象。最后利用 partition_S 获得每个线程需要拷贝的数据（src）。

上述让每个线程知道了这次传输的 src，但是还没有des。 des 是寄存器，在 TieldCopy 中已经描述过了：

  Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
  Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)


与gmem to smem 不同， mem -> register 这个过程中，thread level 的 src tensor 和 des tensor 根据不同的需求划分的 —— smem_tiled_copy_Q 和 TiledMMA，gmem to smem 都是用 GmemTiledCopyQKV 划分src 和 dst tensor，因此在 mem -> register 这个过程中，需要 retile 一下 dst tensor，保证src tensor 和 dst tensor 的layout 一致，如。

  tSrQ_copy_view = smem_thr_copy_A.retile_D(tSrQ);


用一张图总结下上面的大致流程。

6.5）causal mask

前面四个部分的操作相对来说比较宏观，后面的操作全部都是在 thread 寄存器层面操作数据。

通过 MMA 指令求得 Q_i * K_j^T 得到 S_{ij} 后，需要添加 causal mask，简单来说 S_{ij} 中，行下标表示 Q_i 中对应token 序列，列下标表示 K_j^T 中的 token序列，当 kBlockN * nbi + j > kBlockM * m_block + i 时，对应的值就要 mask 掉。

S_{ij} 是分散存储在一个 CTA 中的所有线程的寄存器中。需要从每个 thread 的角度上分析，当前线程负责的元素对应的 row_idx 和 col_idx 分别是多少。根据 MMA 指令集的特性，可以根据下图知道，比如 T0 线程，在一个 Wrap level 循环中，每次会处理的 col index 为 0,1， row idex 为 0,8， 可以根据 MMA 的 latex 图以此类推知道，T32 线程每次会处理的 col index 为 0,1， row idex 为 16,24，即

    const int lane_id = threadIdx.x % 32;
    const int col_idx_offset = kBlockN * nbi + (lane_id % 4) * 2;
    const int nrow_group = threadIdx.x / 32;
    const int row_idx_offset = kBlockM * m_block + lane_id / 4 + nrow_group * 16 /* 2*8 */;


同时注意到每个线程寄存器C被reshape为(nrow=(2, MMA_M), ncol=(2, MMA_N))，MMA_M，MMA_N为周期性平铺后的数值。

完整代码如下，细节部分可以参考代码注释：

template <int kBlockM, int kBlockN, int kNWarps,typename Engine, typename Layout>
inline __device__ void mask_within_nblock(Tensor<Engine, Layout> &tensor, const int m_block, const int nbi) {
    // tensor has shape (nrow=(2, MMA_M), ncol=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    // NOTE: 根据 mma_tile 的示意图来确定每个线程处理的是第几个 token

    // NOTE:
    // 计算thread的处理范围, mask掉超出范围的部分

    const int lane_id = threadIdx.x % 32;
    const int col_idx_offset = kBlockN * nbi + (lane_id % 4) * 2;

    const int nrow_group = threadIdx.x / 32;
    const int row_idx_offset = kBlockM * m_block + lane_id / 4 + nrow_group * 16 /* 2*8 */;
    // (2, nrow), 2*8 for each
    const int group_stride = kNWarps * 16;

    #pragma unroll
    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
        // SM80_16x8x16_F32F16F16F32_TN中的一组中, 一行4个线程处理8个value
        const int col_idx_base = col_idx_offset + nj * 8;
        #pragma unroll
        for (int j = 0; j < size<1, 0>(tensor); ++j) {
            // j用于计算value 1和value 2对应col
            // col_idx最终表示当前thread所处理的value的列号
            const int col_idx = col_idx_base + j;

            // mask掉scores中(QK后的结果)超出范围的部分
            // 列号和行号对比

            // Without the "make_coord" we get wrong results
            // for nrow(2, MMA_M)
            #pragma unroll
            for (int mi = 0; mi < size<0, 0>(tensor); ++mi) {

              #pragma unroll
              for (int mj = 0; mj < size<0, 1>(tensor); ++mj) {
                const int row_idx = row_idx_offset + mi * 8 + mj * group_stride;
                if (col_idx > row_idx) {
                  tensor(make_coord(mi, mj), make_coord(j, nj)) = -INFINITY;
                }
              }

            }

        }
    }
}


至此，完成了 Si^{(j)} 的计算。

6.6）Softmax、rescale, P_j^j @ V_j 等

接下来就是 rowmax，rowsum，softmax，rescale 上一次的 l_i , \bm{o_i} 等。

需要注意的是，需要从 CTA level 和算法的角度来考虑代码，同时在 thread level 层面注意代码细节（每个线程只能处理它自身的寄存器）。

这部分对应代码的 softmax_rescale_o，下面是较为详细的注释。

// scores:((2, MMA_M),(2, MMA_N))，经过了 causal 之后的 Q_i 和 k_j^T 的乘积，
// scores_max:(2 * MMA_N), rowmax 的结果
// scores_sum:(2 * MMA_N)， rowsum 的结果
// acc_o:((2, 2),(MMA_M, MMA_N))， 最后的计算结果
template<bool Is_first, typename Tensor0, typename Tensor1, typename Tensor2>
inline __device__ void softmax_rescale_o(Tensor0 &scores, Tensor1 &scores_max, Tensor1 &scores_sum,
                                         Tensor2 &acc_o, float softmax_scale_log2) {
    if (Is_first) {
        // NOTE: 第一次softmax不需要rescale, 只需要记录 Sij(kblockM, kblockN) 的 rowmax 和 rowsum
        reduce_max</*zero_init=*/true>(scores, scores_max);
        flash::scale_apply_exp2(scores, scores_max, softmax_scale_log2);
        reduce_sum(scores, scores_sum);
    } else {
        // 记录上一次的 rowmax
        Tensor scores_max_prev = make_fragment_like(scores_max); // 相当于公式中的 m_i^{j-1}
        cute::copy(scores_max, scores_max_prev);
        // NOTE: 计算最新的 max 
        // reduce_max包含步:
        //  1. 求当前thread内max: 遍历
        //  2. reduce thread间的max: 使用线程数洗牌指令做 all reduce，每个线程都获得了最大值
        reduce_max</*zero_init=*/false>(scores, scores_max); // scores_max 变成最新的最大值，相当于公式中的 m_i^{j}
        // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
        // 将acc_o转换成符合2D直觉的(nrow, ncol)的形状
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        #pragma unroll
        for (int mi = 0; mi < size(scores_max); ++mi) { // 遍历每一行
            // NOTE: 辅助变量: 当前行max
            float scores_max_cur = scores_max(mi); // 当前行的最大值
            // NOTE: 计算上一次 score_sum 的 rescale 值
            float scores_scale = expf((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2); // 想当于公式中的 e^{m_i^{j-1} - m_i^{j}}.
            scores_sum(mi) *= scores_scale; // 想当于公式中的  e^{m_i^{j-1} - m_i^{j}}l_i^{j-1}
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; } // 想当于公式中的 e^{m_i^{j-1} - m_i^{j}}O_i^{j-1}
        }
        // NOTE: Apply the exp to all the elements with new max value， 这里相当于论文公式里的 P_i^_j
        flash::scale_apply_exp2(scores, scores_max, softmax_scale_log2);

        Tensor scores_sum_cur = make_fragment_like(scores_sum);  // l_i^{j} = e^{m_i^{j-1} - m_i^{j}}O_i^{j-1}
        // NOTE: 累计求和
        reduce_sum(scores, scores_sum_cur); // rowsum(P_i^_j)
        // NOTE: 新分母累加到旧分母
        #pragma unroll
        for (int mi = 0; mi < size(scores_sum); ++mi) { scores_sum(mi) += scores_sum_cur(mi); } // l{ij} = e^{m_i^{j-1} - m_i^{j}}O_i^{j-1} + rowsum(P_i^_j)
    }
};


这一步计算完之后， score tensor 中存的是 P_i^{(j)} 的值，然后需要和 V_j 做乘法，此时， P_i^{(j)}已经在寄存器中，只需要将 V_j load 到寄存器中即可，对应代码中的 flash::gemm_A_in_regs，循环结束后，结果除以soft Max的分母部分即可，最后将寄存器结果拷回到global memory中。
