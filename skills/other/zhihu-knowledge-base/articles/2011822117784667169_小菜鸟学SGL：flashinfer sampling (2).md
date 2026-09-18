# 小菜鸟学SGL：flashinfer sampling (2)

**作者**: xieym我是little菜鸟-菜菜菜没有敌

**原文链接**: https://zhuanlan.zhihu.com/p/2011822117784667169

---

本文聚焦top-k normalization

背景

假定选择flashinfer作为sampling backend，如果没有设置min_p，在flashinfer调用的kernel是TopKTopPSamplingFromProbKernel。单纯的top-k sampling是没有走radix select逻辑的。前面的文章覆盖了部分逻辑，这里不赘述。下面代码加了点简单注释，感兴趣可以瞅瞅。

本文的代码基于如下commit：

flashinfer: commit #2635
template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE, bool DETERMINISTIC,
          typename DType, typename IdType>
__global__ void TopKTopPSamplingFromProbKernel(DType* probs, IdType* top_k_arr, float* top_p_arr,
                                               IdType* output, IdType* indices, IdType top_k_val,
                                               float top_p_val, uint32_t d, uint64_t* seed_arr,
                                               uint64_t seed_val, uint64_t* offset_arr,
                                               uint64_t offset_val) {
  const uint32_t batch_size = gridDim.x;
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  ...
  vec_t<float, VEC_SIZE> probs_vec;
  float aggregate;
  float q = 1;
  double low = 0, high = 1.f;
  int sampled_id;
  do {
    temp_storage.sampled_id = d;
    __syncthreads();
    float u = curand_uniform(&state) * q;
    aggregate = 0;
#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(0);
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }

      DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM,
                             DETERMINISTIC>(
          // low is the lower bound of pivot_prob (exclusive)
          // high is the upper boound of pivot_prob (inclusive)
          i, d, [&](float x) { return x > low; }, u, probs_vec, aggregate, &temp_storage);
      if (aggregate > u) {
        break;
      }
    }
    __syncthreads();
    sampled_id = temp_storage.sampled_id;
    ...
    double pivot_0 = probs[row_idx * d + sampled_id];
    double pivot_1 = (pivot_0 + high) / 2;

    ValueCount<float> aggregate_gt_pivot_0{0, 0}, aggregate_gt_pivot_1{0, 0};
    ValueCount<float> threadlocal_aggregate_gt_pivot_0{0, 0};
    ValueCount<float> threadlocal_aggregate_gt_pivot_1{0, 0};

    // thread-level reduction info: threadlocal_aggregate_gt_pivot_0, threadlocal_aggregate_gt_pivot_1
    // ValueCount.value: accumulated prob
    // ValueCount.count: number of contributors for ValueCount.value
    // block-level reduction info: aggregate_gt_pivot_0, aggregate_gt_pivot_1
    // after blockwise synchronization, aggregate_gt_pivot_0 is consistent for all threads within block
    // similar handing for aggregate_gt_pivot_1 as aggregate_gt_pivot_0
    ...
    if (aggregate_gt_pivot_0.count < k && aggregate_gt_pivot_0.value < p) {
      // case 1: pivot_0 accepted
      break;
    }
    if (aggregate_gt_pivot_1.count < k && aggregate_gt_pivot_1.value < p) {
      // case 2: pivot_0 rejected, pivot_1 accepted
      low = pivot_0;
      high = pivot_1;
      // aggregate_gt_pivot_0.value is upper bound of pivot_prob
      q = aggregate_gt_pivot_0.value;
    } else {
      // case 3: pivot_0 rejected, pivot_1 rejected
      low = pivot_1;
      // both aggregate_gt_pivot_0.value and aggregate_gt_pivot_1.value are upper bound of pivot_prob
      // but, aggregate_gt_pivot_1.value is the smaller one
      q = aggregate_gt_pivot_1.value;
    }
  } while (low < high);
  __syncthreads();
  if (tx == 0) {
    output[bx] = sampled_id;
  }
}
top-k norm

RadixTopKRenormProbMultiCTA参数

probs: shape[bs, vocab_size]
renorm_probs: shape[bs, vocab_size]
batch_size: bs
row_states_buffer: gmem buffer, size: 1m，可以提供300+个RadixRowState对象
// Global state for multi-CTA radix reduction (one per group)
struct RadixRowState {
  uint32_t histogram[3][256];  // Triple-buffered histograms for 1-barrier-per-round
  uint32_t remaining_k;        // Remaining k after current round
  uint32_t prefix;             // Accumulated prefix (high bits of k-th element)
  int arrival_counter;         // For inter-CTA synchronization
  int output_counter;          // For collecting top-k indices (RadixTopK)
  float sum_topk;              // For RenormProb: sum of top-k elements
};
radix select设计思路

为了提升probs每一行（seq的last token)的处理速度，使用多个block同时处理同一行的probs。在nvidia的硬件架构中，没有使用特殊的launch API的话，block被assign的sm的顺序，以及连续2个block被assign到sm映射的关系，甚至连续的2个block同时在sm上运行。以上这些行为都是没有定义的。

需要在不同的sm之间进行数据交换需要利用gmem，不同block的对于相同gmem地址访问通过atomic写保证原子性。

无论sampling还是normalization，我们都是要对vocabulary反复扫描。为了提升处理速度，在radix select采用smem空间作为缓存，从gmem读取的prob(浮点)进行1-1映射转换成整数(ordered data)并保存到smem里。radix select逻辑把浮点数转换的整数(ordered data)按字节方式映射到固定size的bucket（8 bits共256个bucket），通过统计计数来找到后缀和（suffix_sum）>= k的最小bucket。利用这个办法来寻找pivot_prob，具体流程后面有介绍。

bucket映射处理过程可能进行多次遍历，radix select算法将ordered data存放在smem优化访存部分性能。

smem数据结构

这个数据结构定义是从代码注释总结的。

// summary from comments
struct smem_storage {
    uint32_t histogram[256];
    uint32_t suffix[256];
    uint_32_t scalars[4];
    float sum_local;
    // aligned to 16 bytes
    OrderedType chunk_data[]; 
}
RadixTopKRenormProbMultiCTA
RadixTopKRenormProbKernel_MultiCTA

GPU的smem maximum size可通过cuda API获得，基于此可以计算block中可以最多可以处理多少个ordered data。目前看到的sampling kernel的blockdim都是BLOCK_THREADS个线程。意味着，一个block处理数据至少是BLOCK_THREADS*vec_size。

代码通过兼顾smem_size和sm个数来计算gridDim，大体逻辑如下：

block内处理多少个prob被记作chunk_size，考虑smem size，且至少BLOCK_THREADS*vec_size
vocabulary需要多少个block共同处理被记作ctas_per_group，表示vocab_size需要分多少个chunk
gridDim：理论上是ctas_per_group*bs，从代码实现看至多使用#sm个block来执行这个kernel。

当bs > (#sm / ctas_per_group)，处理每行(seq's last token)分到的block数量小于ctas_per_group，使用循环方式来弥补。换而言之，在做radix select时尽量提升每行处理的并行度，通过smem size来确定切分的chunk，再通过chunk来确定一行prob需要多少个block共同完成；然而为了保证硬件利用率，kernel执行的thread spawning阶段不会因为没有空闲sm产生等待，tail block执行时只有部分sm active造成硬件资源限制。因此，限制了blocks数量不能超过simultaneously可运行block数目（每个block最大化使用smem资源作为缓存）。在RadixTopKRenormProbKernel_MultiCTA实现中是有一个loop，有点persistent kernel的味道咯。

RadixTopKRenormProbKernel_MultiCTA
kernel launch参数
blockDim	BLOCK_THREADS
ctas_per_group	ceil_div(vocab_size, chunk_size)
num_groups	std::min(num_sms/ ctas_per_group, bs)
gridDim	num_groups * ctas_per_group

如果ctas_per_group是1，直接依赖smem中数据结构来同步，不使用row_states (gmem)。函数RadixTopKRenormProbKernel_MultiCTA模板参数中第三个参数是SINGLE_CTA，位true表示不使用row_states进行block间的数据同步。

RadixTopKRenormProbKernel_MultiCTA实现

代码太多了，没办法复制了[忧伤]。

k >= vocab_size

含义是不需要选k，直接做normalization（归一化，所有prob加和等于1）。逻辑简单，值得一提的是非SINGLE_CTA情况下，通过多个block共同处理seq's last token的logits投影到vocabulary上token的概率。此时需要借助gmem的row_states缓存，通过global memory的atomic操作来同步数据。

RadixTopKRenormProbKernel_MultiCTA

但是为了确保不同block中的所有线程读/写row_state缓存(sum_topk)视图一致。代码使用两组barrier标记block-level规约sum结果提交到gmem的row_state这个事件开始和结束的边界。在barrier执行完成后，所有线程看到视图是一致的。同步大致逻辑:

每个block的tx0负责往state->arrival_counter累加1，等到state->arrival_counter变为state->arrival_counter+ctas_per_group说明处理同一行的所有block的tx0线程都执行到这里了。加上后面的__syncthreads() (block内线程同步)原语，处理同一行的所有线程完成同步。
barrier_phase: 线程寄存器，用于从当前线程角度描述state->arrival_counter
每个group使用row_state是不同的，是row_states数组的一个元素
row_states数组是一个复用buff，分配时清0
top_k_renorm_probs
  // State pointer only used when not SINGLE_CTA
  RadixRowState* state = nullptr;
  if constexpr (!SINGLE_CTA) {
    state = &row_states[group_id];
  }

  int barrier_phase = 0;
RadixTopKRenormProbKernel_MultiCTA

row_state中的histogram以RR方式使用，通常清理发生在RadixSelectFindPivot清理；k >= vocab_size, 不会走RadixSelectFindPivot，这里显式清理。

RadixTopKRenormProbKernel_MultiCTA

radix select处理完成，state->arrival_counter会被清0

RadixTopKRenormProbKernel_MultiCTA
处理radix select的外层调用逻辑
template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, bool SINGLE_CTA, typename DType,
          typename IdType>
__global__ void __launch_bounds__(BLOCK_THREADS) RadixTopKRenormProbKernel_MultiCTA(
    DType* probs,          // [batch, vocab_size]
    DType* renormed_prob,  // [batch, vocab_size]
    IdType* top_k_arr,     // [batch] or nullptr
    uint32_t top_k_val, uint32_t vocab_size, uint32_t batch_size,
    RadixRowState* row_states,  // [num_groups] (nullptr if SINGLE_CTA)
    uint32_t chunk_size,        // elements per CTA
    uint32_t ctas_per_group)    // CTAs per row (1 if SINGLE_CTA)
{
  using Traits = RadixTopKTraits<DType>;
  using OrderedType = typename Traits::OrderedType;

  constexpr uint32_t RADIX = 256;

  const uint32_t global_cta_id = blockIdx.x;
  const uint32_t group_id = global_cta_id / ctas_per_group;
  const uint32_t cta_in_group = global_cta_id % ctas_per_group;
  const uint32_t tx = threadIdx.x;

  // Shared memory layout: [fixed storage] [ordered values cache]
  extern __shared__ uint8_t smem[];

  // Fixed shared memory (at the beginning)
  // histogram[256] + suffix[256] + scalars[4] + sum_local[1]
  constexpr size_t fixed_smem_size = sizeof(uint32_t) * (RADIX + RADIX + 4) + sizeof(float);
  uint32_t* local_histogram = reinterpret_cast<uint32_t*>(smem);
  uint32_t* suffix_sum = local_histogram + RADIX;
  uint32_t* shared_scalars = suffix_sum + RADIX;

  // Persistent loop over rows
  for (uint32_t iter = 0; iter < total_iterations; iter++) {
    uint32_t row_idx = group_id + iter * num_groups;

    if (row_idx >= batch_size) break;

    const uint32_t chunk_start = cta_in_group * chunk_size;
    const uint32_t chunk_end = min(chunk_start + chunk_size, vocab_size);
    const uint32_t actual_chunk_size = chunk_end - chunk_start;

    uint32_t k = top_k_arr == nullptr ? top_k_val : top_k_arr[row_idx];

    // For RenormProb, pivot is compared with probs (must be non-negative)
    DType pivot = DType(0);
    float normalizer = 1.0f;

    if (k >= vocab_size) {
      ...
      continue;
    }

    pivot = RadixSelectFindPivot<BLOCK_THREADS, VEC_SIZE, SINGLE_CTA, DType>(
        probs + row_idx * vocab_size, shared_ordered, local_histogram, suffix_sum, shared_scalars,
        state, chunk_start, actual_chunk_size, k, barrier_phase, ctas_per_group, cta_in_group, tx,
        iter);

    ...
  }
}
RadixSelectFindPivot

这个函数是一个wrapper, 先看下caller传入参数中跟smem有关的参数

probs + row_idx * vocab_size: row_idx-th's probs, row_idx starts from 0
shared_ordered: addr to store ordered data
local_histogram: addr to store accumulated counter for block-wise radix mapping
suffix_sum: addr to store suffix counter for single row of probs
shared_scalars: addr to store temporary counter for radix select
template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, bool SINGLE_CTA, typename DType>
__device__ __forceinline__ DType RadixSelectFindPivot(
    const DType* input, typename RadixTopKTraits<DType>::OrderedType* shared_ordered,
    uint32_t* local_histogram, uint32_t* suffix_sum, uint32_t* shared_scalars, RadixRowState* state,
    uint32_t chunk_start, uint32_t actual_chunk_size, uint32_t k, int& barrier_phase,
    uint32_t ctas_per_group, uint32_t cta_in_group, uint32_t tx, uint32_t iter = 0) {
  using Traits = RadixTopKTraits<DType>;
  using OrderedType = typename Traits::OrderedType;

  // Stage 1: Load and convert to ordered representation
  LoadToSharedOrdered<BLOCK_THREADS, VEC_SIZE, DType, Traits>(input, shared_ordered, chunk_start,
                                                              actual_chunk_size, tx);

  // Stage 2: Radix select to find pivot
  uint32_t local_gt_count = 0;  // Not used in this function
  OrderedType ordered_pivot = RadixSelectFromSharedMemory<BLOCK_THREADS, SINGLE_CTA, OrderedType>(
      shared_ordered, actual_chunk_size, k, local_histogram, suffix_sum, shared_scalars, state,
      barrier_phase, ctas_per_group, cta_in_group, tx, iter, local_gt_count);

  // Convert ordered representation back to DType pivot
  return Traits::FromOrdered(ordered_pivot);
}
RadixTopKTraits

RadixTopKTraits模板类实现浮点数转换成等长度的整数，映射满足双射并保持单调性。LoadToSharedOrdered通过向量load方式从gmem浮点数经过reg中转把数据搬运到smem，在保存smem之前通过Traits::ToOrdered转换把浮点数映射到整数，并且保持单调性。

小菜鸟个人观点，这保证radix select中通过8-bit一组的radix mapping找到的prefix就是pivot_prob。Traits::ToOrdered和Traits::FromOrdered，都是双射且满足单调性。

注: input_vec是寄存器变量
// Specialization for nv_bfloat16 (16-bit)
template <>
struct RadixTopKTraits<nv_bfloat16> {
  using OrderedType = uint16_t;

  template <uint32_t RADIX_BITS>
  static __host__ __device__ constexpr uint32_t num_rounds() {
    return sizeof(OrderedType) * 8 / RADIX_BITS;
  }

  __device__ __forceinline__ static OrderedType ToOrdered(nv_bfloat16 val) {
    uint16_t bits = __bfloat16_as_ushort(val);
    return (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits ^ 0x8000);
  }

  __device__ __forceinline__ static nv_bfloat16 FromOrdered(OrderedType ordered) {
    uint16_t bits = (ordered & 0x8000) ? static_cast<uint16_t>(ordered ^ 0x8000)
                                       : static_cast<uint16_t>(~ordered);
    return __ushort_as_bfloat16(bits);
  }

  __device__ __forceinline__ static nv_bfloat16 NegInf() {
    return __ushort_as_bfloat16(static_cast<uint16_t>(0xFF80));  // -inf in bf16
  }
};

template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, typename DType, typename Traits>
__device__ __forceinline__ void LoadToSharedOrdered(const DType* input,
                                                    typename Traits::OrderedType* shared_ordered,
                                                    uint32_t chunk_start,
                                                    uint32_t actual_chunk_size, uint32_t tx) {
  using OrderedType = typename Traits::OrderedType;
  vec_t<DType, VEC_SIZE> input_vec;
  const uint32_t aligned_size = (actual_chunk_size / VEC_SIZE) * VEC_SIZE;

#pragma unroll 2
  for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
    input_vec.cast_load(input + chunk_start + i);
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      shared_ordered[i + j] = Traits::ToOrdered(input_vec[j]);
    }
  }
  // Handle tail
  for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
    shared_ordered[i] = Traits::ToOrdered(input[chunk_start + i]);
  }
  __syncthreads();
}
RadixSelectFromSharedMemory

Radix select实现在RadixSelectFromSharedMemory。如果是多个block共同处理同一行prob，需要借助gmem的row_state进行block间的数据同步。

这个代码看起来有点绕，其实用cuda来写data structure的algorithm。

先看个晦涩部分：

shared_scalars[4]大部分场景是integer的counter；
在RadixTopKRenormProbKernel_MultiCTA中k >= vocab_size && SINGLE_CTA被用来当作prob的累加和
这种写法在一般kernel是很少见的。我们C/C++ programming，这是"大宝天天见"，暴露年龄[汗]

仔细看两个路径还是有点区别的：

RadixTopKRenormProbKernel_MultiCTA: 先写后读
RadixSelectFromSharedMemory: counter累加，需要先清零
RadixSelectFromSharedMemory
RadixTopKRenormProbKernel_MultiCTA
RadixSelectFromSharedMemory入口处的同步
RadixSelectFromSharedMemory

涉及代码太多，copy-paste也搞不过来呀。选重点看看哈！

RADIX(256), RADIX_BITS(8): 每8 bit一组进行映射，8 bit无符号数最大可表达256，即256个bucket。且每个bucket表达范围是正交的
ORDERED_BITS：ordered data数据类型的字节数，即浮点数数据类型的字节数
NUM_ROUNDS：以8 bit一组进行映射，需要执行多少轮才能遍历ordered data数据类型的所有字节

下面的代码有两重循环，设计是把vocabulary的prob(已经是ordered datatype了)映射到256个bucket里。再以bucket为单位计数汇总，通过汇总信息确定bucket及其后继的累加和大于等于k的最小子集. 如果这个最小子集的元素个数 > k, 那么bucket后继(假设这部分累积和是m)都是满足topk的,k-m部分需要把bucket子集再次应用映射去筛选。应用映射的循环就是外层循环;内层循环是遍历vocabulary。

在多个block共同处理同一行probs的场景下,每个block处理不同的chunk。多个block共同处理vocabulary。

虽然是data structure的algorithm，代码没有判断k/remaining_k_cache等于0提前退出。小菜鸟的理解是cuda代码算力海海的，尽量避免复杂的控制逻辑；我觉得即使判断remaining_k_cache == 0提前退出外层循环，不会引入thread divergence问题。

remaining_k_cache 在函数开始部分被每个block的tx0初始化为k。

内存循环中，存储在smem中的shared_ordered被block中的所有线程（BLOCK_THREADS）以RR方式并行处理，每个线程把自己负责的ordered data映射到相对应的bucket上，并通过smem中的local_histogram进行计数。前面提到的prefix如何处理呢？

RadixSelectFromSharedMemory

计算完当前block的ordered data的计数，执行__syncthreads同步block所有线程。如果是多个block合作处理同一行，需借助state->arrival_counter来累加各个block的计数，原理跟上面k>=vocab_size处理类似，借助两个barrier来标记smem计数向gmem计数累加的arrival和completion的边界。

gmem中的统计是以RR方式来轮转的，一共3个buff (maximum iter diff是3么)。

每次计数前需要把local_histogram清零；计数过程中把下一个histogram的buff清零。仔细看会发现都是多线程并行执行的，smem中histogram每个block维护一份，每个block都执行清零；next_hist则是block0来进行清零的。

RadixSelectFromSharedMemory
RadixSelectFromSharedMemory

current_hist收集到的多个block汇报的计数后，每个block会把它复制到自己的smem的suffix_sum中。suffix_sum就是用来计算bucket后继的计数累加和的。

RadixSelectFromSharedMemory
RadixSuffixSum

RadixSuffixSum类似于warp shuffle（纯寄存器)，利用smem在suffix_sum维护了统计计数（bucket后继的计数累加和)。

处理同一行的不同block看到视图是一致的，找到suffix_sum[buket] >=k && suffix_sum[bucket+1] < k && bucket < 255;

如果suffix_sum[bucket] > k, k - suffix_sum[bucket+1]部分将由下一个round+1外层循环进一步处理。

为了保证block内部同步，用了多个__syncthreads来同步block内的所有线程。

RadixSelectFromSharedMemory

smem数据：

suffix_sum
found_bucket
found_remaining_k
remaining_k_cache
prefix_cache

registerfile：

tx
count_ge
count_gt

这三个值都跟那个thrd id有关，block内不同线程看到的是不同的。

remaining_k
prefix
shift

这三个值在block内是一致的。

RadixSuffixSum跟warp shuffle类似，通过把线程分为2的m次幂（m是从RADIX-1到0）个子集进行规约。因为suffix_sum是从gmem的histogram复制的。映射一致，定义域一致，值域也是一致。RadixSuffixSum返回后，不同block的suffix_sum也是一致的。

RadixSuffixSum

愉快地结束sampling的代码探索咯:P
