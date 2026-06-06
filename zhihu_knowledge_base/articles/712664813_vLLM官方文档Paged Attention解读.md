# vLLM官方文档Paged Attention解读

**作者**: HongzhuzhuOptimistic Pessimist

**原文链接**: https://zhuanlan.zhihu.com/p/712664813

---

​
目录
收起
输入和输出
Kernel的一些常量定义
主要代码步骤
load query
load key
QK
Softmax
Load V
LV
Output
Paged Attention vs. Flash Attention

vLLM是一款性能优越的LLM推理框架，现在也成为开源社区的主流。关于其框架和整体介绍的文章相对较多，而vLLM的核心是围绕KV Cache的分块架构起来的，所以这篇主要从核心的算子 page attention讲起。

这篇主要围绕vLLM的官方文档的描述展开，可以配合使用。另外更深入的探索可以参考另一篇文章聊聊CUDA编程中线程划分和数据分块 之 PagedAttention（V1/V2）分析 - 知乎

参考vLLM Paged Attention — vLLM

代码位置：paged_attention

关于原理部分有大量的解读，这里不做过多赘述，直接从代码开始：

call stack（以v1 为例）:

paged_attention_v1 -> CALL_V1_LAUNCHER_BLOCK_SIZE -> CALL_V1_LAUNCHER (paged_attention_v1_launcher) -> LAUNCH_PAGED_ATTENTION_V1 -> vllm::paged_attention_v1_kernel -> paged_attention_kernel

输入和输出
// Grid: (num_heads, num_seqs, max_num_partitions).
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
          int PARTITION_SIZE = 0>  // Zero means no partitioning.
__device__ void paged_attention_kernel(
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                     // max_num_partitions]
    scalar_t* __restrict__ out,  // [num_seqs, num_heads, max_num_partitions,
                                 // head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads,               // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float kv_scale)
scalar_t表示query，key，value的数据类型，例如FP16.
HEAD_SIZE表示每个head中的element数；
BLOCK_SIZE表示每个block中的token 数；
PARTITION_SIZE表示张量并行的gpu数目。

参数：

out：[num_seqs, num_heads, head_size]
q：[num_seqs, num_heads, head_size]
k_cache：[num_blocks, num_kv_heads, head_size/x, block_size, x] x表示一个向量化的大小，如float16 -> 16 / sizeof(float16) = 8
v_cache：[num_blocks, num_kv_heads, head_size, block_size]
head_mapping：[num_heads] 用于MQA, GQA，确定用的KV_head
block_tables：[num_seqs, max_num_blocks_per_seq] block_tables映射表，表示每个sequence映射到哪几个block上
context_lens：[num_seqs] 当前seq，已经推理出来key cache，value cache的长度
Kernel的一些常量定义
Sequence: 表示一个请求的句子。例如 q 的shape 是[num_seqs,num_heads,head_size]. 一共有个 num_seqs 个query sequence 通过 q指针来表示。由于这里我们考虑的是decoding阶段的single query kernel，所以每个sequence只有一个token，这里的 num_seqs就等于当前batch中token的总数。
Context: 包括生成token的文本，例如["What","is","your"] 是 context tokens, input query token 是 "name"。模型生成的token是 "?".
Vec: 一起读取和计算的elements list，也就是向量化读取的vector。
对于 query 和 key，vec size (VEC_SIZE) 取决于每个thread group能够一次读取16 bytes数据的data数量。例如，对于FP16（2 bytes），and THREAD_GROUP_SIZE = 2，那么 VEC_SIZE=16/2bytes/2threadgroup = 4，也就是一个thread一次读取4个query 和key data。
对于value，vec size (V_VEC_SIZE) 取决于每个thread能够一次读取16 bytes数据的data数量。FP16数据，V_VEC_SIZE= 16/2bytes = 8。
Thread group:
是一组thread，其大小通过THREAD_GROUP_SIZE来设定。
定义了一次可以读取和计算一个query token和一个key token的单元。每个thread只处理一个token data的部分数据。每个thread group处理的元素数量定义为 x。
例如，一个thread group包含了2个thread，head size=8，thread 0 处理index为 0, 2, 4, 6 的head， thread 1处理 index 1, 3, 5, 7的head。
Block: kv cache的存储单元。对于一个head，每个 block 存储固定大小为BLOCK_SIZE数量的tokens。每个block可能只会包含整个context的部分token。例如，block size = 16，head size = 128, 对于一个head，一个 block 存储了16 * 128 = 2048个元素。
Warp: 在一个流多处理器（SM）上，同时可以调度的一组thread数量，也就是WARP_SIZE，为32。每个warp一次迭代处理一个query token和一个block的key token之间的计算，如果是多轮迭代，就可以处理多个block。例如，如果一个context由4个warp和6个block，warp 0 处理 block0，block4，warp 1 处理 block1, block5；warp 2 处理block2，warp 3 处理block3。
Thread block: 硬件上的block。是一组NUM_THREADS可以访问相同shared memory的thread。每个thread block包含了NUM_WARPS个warp，每个thread block处理一个query token和整个context key之间的计算。
Grid: kernel启动的所有线程。其shape定义为 (num_heads,num_seqs,max_num_partitions).
主要代码步骤
load query

多个thread group同时load，一个thread group处理一个token

q处理问题的规模：一个thread group对应处理一个token，现在query只有一个token，这个token一个head，head_dim=128，vec_size=4，那么一共32个vec。

加载方式：可以让某一个thread group完成加载；但是把这些vecs分给不同的thread groups 同时执行会更快

比如一共有4个thread group来完成，也就是NUM_RHREAD_GROUPS=4，那么会通过循环来加载；

每个thread group内有2个thread，NUM_VECS_PER_THREAD=2

例如对于idx为7的vec7，vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE = 1 + thread_group_idx 3 * 2




为什么key要被加载到寄存器，而query却是被加载到shared memory？

首先看shared memory和register的scope：shared memory的scope是整个thread block，也就是说thread block内的每一条线程都可以拿到shared mem里面的东西，在这里就是q_vecs；但是寄存器的scope是thread，也就是说在key加载的时候，NUM_TOKENS_PER_THREAD个vecs仅被那条负责加载的线程看到。

再来看我们定义的问题处理的规模：每个thread block处理一个query token和整个context key之间的计算。也就是说，所有key token是和同一个query token做点积运算的，不同的key token被分配给了同一个thread block的不同的thread group，这些thread group要拿到同一个query token的值进行点积计算得出注意力分数，所以query必须对整个thread block内的不同thread group可见。假设我把query加载到寄存器里面，那就只有负责加载的线程能获取它自己负责的vecs，那些跟这条加载线程在不同的thread group的线程加载了某个key token，当这个key想要query来做点积的时候，根本就拿不到那条加载线程负责的query vecs，计算也就无法进行。




load key

并行设计：三层循环。注意，这里K向量的数据是直接加载到每个线程的寄存器的。

最外层循环，就是用一个warp处理一个block, 对一个SM来说，通常都是4个Wraps， 可以同时并行处理四个block。

2. 中间层循环，就是一个thread group处理一个token。因为每个block有多个token，所以一次iteration处理一个token。

如果warp_size > block size，那么一个thread group有多个thread，处理一个token；

如果warp_size < block size，那么一个thread group有一个thread，处理多个token。

3. 内层循环，就是一次循环加载一个K向量的一部分, size为 x 。注意，这里对VEC_SIZE做了一个变换，最后用的是x作为加载的size，不是VEC_SIZE。




key的内存布局：

K Cache的布局为[num_blocks, num_kv_heads, head_size/x, block_size, x]，这是为了优化写入shared memory的操作。

在Q和K矩阵的同一行元素被读入寄存器并进行点乘运算后，结果需要被存入shared memory。如果一个warp中所有线程都计算Q、K同一行数据，会导致写入shared memory的同一个位置，这将造成warp内不同线程顺序地写入。因此，为了优化，warp的线程最好计算Q和K的不同行数据。

因此，在设计K Cache布局时，我们将block_size放在比head_size更低的维度。由于warp size大于block_size，我们需要将head_size拆分为head_size/x和x两个维度，借x到最低维度，以确保每个线程读入的数据量和计算量都足够大。最后，每个线程组派一个线程去写入shared memory（也就是qk的结果logits在shared memory—），这样一个warp有blk_size个线程并行写入shared memory，从而增加了shared memory的访问带宽。这种设计策略是为了实现高效的并行计算和内存访问，以提高整体的计算性能。




如果warp_size > block size，那么一个thread group有多个thread，处理一个token；

t0，t1... 为thread 编号，相同颜色的方块表示同一个thread group，每次iteration处理一个token

如果warp_size < block size，那么一个thread group有一个thread，处理多个token。

for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
    const int physical_block_offset =
        (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    K_vec k_vecs[NUM_VECS_PER_THREAD];

k cache的layout从里到外分别是：每x个元素连续存储，然后按照block size递增排列，最后按照head size/x的维度进行排列。每一行表示的一个token，由一个thread group进行处理；每一列表示x的大小的向量。

要找到当前token所对应的k向量的位置需要越过下面3个的数据：

(1)physical_block_number * kv_block_stride 是前面所有的block中元素的数量；

(2)进入当前block后， kv_head_idx * kv_head_stride 是前面所有head中元素的数量；

(3)进入当前head，token的k cache是这样存储的：block_size*x，对照当前的block偏移physical_block_offset，乘以x的元素个数，就是当前seq在当前head下的token的k cache的首地址。

这样得到k_ptr：

const cache_t* k_ptr =
    k_cache + physical_block_number * kv_block_stride +
    kv_head_idx * kv_head_stride + physical_block_offset * x;

同时，我们还会以VEC_SIZE的向量大小来load数据，所以对每个thread的VEC number进行循环，定位到当前load的vec的idx。要找到当前处理的vector的首字节地址，那么换算出来就是先将VEC_SIZE转换成元素个数，vec_idx*VEC_SIZE；offset1确定了当前的vec_idx的首地址是哪个小矩形，也就是位于哪一列，每一列是block_size个；offset2表示当前的vec_idx的首地址位于小矩形中0-7的哪个index。

从而得到：

k_vecs[j] = *reinterpret_cast<const K_vec*>(
    k_ptr + offset1 * BLOCK_SIZE * x + offset2);
QK
// Compute dot product.
// This includes a reduction across the threads in the same thread group.
float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(
                        q_vecs[thread_group_offset], k_vecs);
Softmax

get max value: 先是在单个warp内部，获取点积qk最大值，然后获取所有warps的点积qk最大值

如果lane==0，也就是当前线程是本warp中的第一个线程的话，就将本warp中最大的qk值放到共享内存red_smem中，并且再次进行线程同步。

到目前为止red_smem的前4个元素分别存储了4个warp中的最大qk值。

for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
}
if (lane == 0) {
red_smem[warp_idx] = qk_max;
}
__syncthreads();

首先本线程如果是当前warp中的前4个线程的话，就分别存放4个warp中的最大值，这就将不同的warp中的最大qk值集中到了各个warp中的前4个线程中，每个warp携带所有warp中的最大值，通过共享内存red_smem实现。

// TODO(woosuk): Refactor this part.
// Get the max qk value for the sequence.
qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
}
// Broadcast the max qk value to all threads.
qk_max = VLLM_SHFL_SYNC(qk_max, 0);

为了防止溢出，统一减去最大值之后，再求和

// Get the sum of the exp values.
float exp_sum = 0.f;
for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
float val = __expf(logits[i] - qk_max);
logits[i] = val;
exp_sum += val;
}
exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

计算最终的safe softmax结果。

// Compute softmax.
const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
logits[i] *= inv_sum;
}
__syncthreads();
Load V

V的形状是：[num_blocks, num_kv_heads, head_size, block_size]

为什么V Cache的layout是 [num_blocks, num_kv_heads, head_size, block_size]，和K Cache layout不一样？ 这是因为V要去做点乘的对象在shared memory，只需要读，不涉及并行写的问题。

最外层循环，一个warp处理一个block, 对一个SM来说，通常都是4个Wraps， 可以同时并行处理四个block。
一个thread处理V_VEC_SIZE个token，load V_VEC_SIZE个元素，也就是每个token load一个元素。循环NUM_ROWS_PER_THRAED次来处理，也就是说，warp要根据head size的大小循环多次处理当前的block

和key的layout不同有两点：

load data的单元不是thread group，而是一个thread；
memory layout不同，每一列表示一个token，每一行是block size个token同一个head position。

每个thread 一次load V_VEC_SIZE个token的data，一个warp 处理V_VEC_SIZE * warp_size个data。如图所示，thread0 会循环两次load浅蓝色位置的data。

在内循环，对于v_vec，要和之前计算出来的qk结果logits_vec直接相乘并累加得到acc；然后一个thread 会得到num_iteration=2个acc值，这两个acc来自于不同的head position。




LV
首先，在warp内做reduce，每个thread可以拿到对应head position处的block内所有token的acc和。
  // Perform reduction within each warp.
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
#pragma unroll
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
      acc += VLLM_SHFL_XOR_SYNC(acc, mask);
    }
    accs[i] = acc;
  }

2. 然后，在warp之间进行reduce，让每个thread得到对应head position处sequence 所有token的acc和。

所有的warp分成两部分：

upper warps，把对应head position的数据都写到shared memory out_smem
  // Perform reduction across warps.
  float* out_smem = reinterpret_cast<float*>(shared_mem);
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    int mid = i / 2;
    // Upper warps write to shared memory.
    if (warp_idx >= mid && warp_idx < i) {
      float* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          dst[row_idx] = accs[i];
        }
      }
    }
    __syncthreads();
lower warps，因为不同的warp 对应了不同的block，也就是不同的token的结果，所以需要把sequence上的所有token都累加起来，也就是说要做reduce，更新结果
    // Lower warps update the output.
    if (warp_idx < mid) {
      const float* src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          accs[i] += src[row_idx];
        }
      }
    }
    __syncthreads();




Output

每个thread上的register memory的结果写出到最后的global memory：

首先，拿到对应sequence，对应head的起始地址out_ptr，然后迭代每个不同的head position写出对应的acc结果

  // Write the final output.
  if (warp_idx == 0) {
    scalar_t* out_ptr =
        out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
        head_idx * max_num_partitions * HEAD_SIZE + partition_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        from_float(*(out_ptr + row_idx), accs[i]);
      }
    }
  }




Paged Attention vs. Flash Attention
Paged attention V1和flash attention有什么不同？

FA用了两层循环（Q循环和KV循环），每次写一个Tile的output tensor，而PA一直只有一层循环，每次写一行output tensor。因为每次都有整行的QK^T中间结果，没有online softmax。

2. Paged attention V2

相比于V1，在sequence序列维也做了并行处理，一个block处理一个序列的分块。所以这里也涉及到reduce_kernel的修改。整体的思想也类似于从flash attention1到flash attention2的变化。




PageAttention代码走读 - 知乎 (zhihu.com)

vLLM皇冠上的明珠：深入浅出理解PagedAttention CUDA实现 - 知乎 (zhihu.com)

paged attention之将key加载到寄存器 (qq.com)

PageAttention V1 核心CUDA源代码阅读 - 知乎 (zhihu.com)

杨鹏程：聊聊CUDA编程中线程划分和数据分块 之 PagedAttention（V1/V2）分析
