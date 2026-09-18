# 小菜鸟学SGL：flashinfer sampling (1)

**作者**: xieym我是little菜鸟-菜菜菜没有敌

**原文链接**: https://zhuanlan.zhihu.com/p/2011499655553389678

---

今日主题: sampling常用函数DeviceSamplingFromProb

背景

当model.forward()执行结束生成logits，在forward_batch_generation会调用model_runner.sampler.forward().假设我们选择了flashinfer作为sampling backend会执行flashinfer的两个kernel：

top_k_renorm_prob，top_p_renorm_prob，min_p_sampling_from_probs
top_k_top_p_sampling_from_probs

samping部分都会调用DeviceSamplingFromProb在vocabulary里选择(sample)一个token。

sampler.foward

下面是sglang的sampler.forward()调用路径。

sglang commit：pr #19004
class Sampler(nn.Module):
    ...
    def forward(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        return_logprob: bool,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
        positions: torch.Tensor,
    ):
        ...
        logits = logits_output.next_token_logits
        ...

        if sampling_info.is_all_greedy:
            ...
        else:
            simple_sampling_case = (
                not sampling_info.need_top_p_sampling
                and not sampling_info.need_top_k_sampling
                and not sampling_info.need_min_p_sampling
            )
            ...

            if self.rl_on_policy_target is not None:
                ...
            if self.use_ascend_backend:
                ...
            elif (
                self.use_log_softmax_logprob
                and self.enable_deterministic
                and simple_sampling_case
            ):
                ...
            else:
                ...
                probs = logits

                batch_next_token_ids = self._sample_from_probs(
                    probs, sampling_info, positions, simple_sampling_case
                )
                ...

        ...
        return batch_next_token_ids

    def _sample_from_probs(
        self,
        probs: torch.Tensor,
        sampling_info: SamplingBatchInfo,
        positions: torch.Tensor,
        simple_sampling_case: bool,
    ) -> torch.Tensor:
        ...
        if simple_sampling_case:
            ...
        else:
            backend = get_global_server_args().sampling_backend
            if backend == "flashinfer":
                ...
                if sampling_info.need_min_p_sampling:
                    probs = top_k_renorm_prob(probs, sampling_info.top_ks)
                    probs = top_p_renorm_prob(probs, sampling_info.top_ps)
                    batch_next_token_ids = min_p_sampling_from_probs(
                        probs, sampling_info.min_ps
                    )
                else:
                    batch_next_token_ids = top_k_top_p_sampling_from_probs(
                        probs.contiguous(),
                        sampling_info.top_ks,
                        sampling_info.top_ps,
                        filter_apply_order="joint",
                        check_nan=self.use_nan_detection,
                    )
            elif backend == "pytorch":
                ...
            else:
                raise ValueError(f"Invalid sampling backend: {backend}")
        return batch_next_token_ids
sampling目标

先回顾一下sampling的目的吧。sampling作用是为了输出的多样性又要保持语义连贯，当last token的logits生成之后，我们会把它映射到词表vocabulary里。映射方式是把logits与vocab权重矩阵相乘，之后按row进行softmax计算得到概率，映射到词表每个token的概率。

最简单的sampling方式是从每行中选出概率最高的词。为了输出的多样性，可以采取top-k，top-p，min-p等方式选出一个token。

top-k: 只考虑概率最大的k个token
top-p: 只考虑概率总和为p以上的token
min-p: 指定factor，只考虑最大概率概率大于等于max{p}*factor的token

确定好候选的token集合（candidate tokens），在其中随机选择一个token。

为什么要多样性？小菜鸟是自学的，只能说说我的理解。

用户问题：今天天气如何？回答：小雨，晴朗，台风。这是确定问题，查看当天的天气预报，直接copy或者找到一个近义词即可。
用户问题：请写一篇500字以内的文章解释“人不知者不愠”。这是开放问题，如果是对话agent场景，我们不希望agent是个复读机，希望它根据之前聊天内容来判断对话者的文化程度或者喜好来做一个通俗易懂的解释。
极端的场景：文本生成的例子，输出中"的",“地","也","了", "阿", "哦", "呢"等词频繁出现；其实我们在日常说话时候为了思考也会常说一些意义不大修饰词，生成文本中如果包含很多实际意义不大的副词助词会影响文字的可读性。直觉希望去抑制一些词的反复生成或者换种表达方式代替。
万一，以后短剧，短视频的脚本都是AI生成，除了内容规范之外，最大问题会不会是雷同？
DeviceSamplingFromProb的调用范式

选择flashinfer作为sampling的backend，sample的工作主要包括两部分：

从probs (shape: [bs, vocab_size]), 对每行找出满足条件的(candidate probs)。这里满足条件的candidate probs可能是
top-k: 最大的k个prob组成的sub probs
top-p: 累加和为p的最小sub probs
min-p: 所有GE max{prob} * factor构成的sub probs
从上面找到的candidte probs按照一定分布选择一个token id

如果是min-p的场景，在调用DeviceSamplingFromProb之前会对从top-k，top-p的角度对candidate probs进行归一化（normalize）。后面会专门写一篇top-k的radix select文章，top-p就是以binary search方式逼近来寻找pivot prob（浮点数），由于篇幅限制就没法涉及了。如果有时间，可以单写一篇。

下面分析基于最新的flashinfer代码

flashinfer commit：pr #2635

执行sampling kernel的launch参数如下：

BLOCK_THREADS在sm8X以上系列芯片都是1024

gridDim	batch_size
blockDim	BLOCK_THREADS

kernel执行每行（seq的last token）的sampling，每个thread block负责一行，即每个seq的last token的sampling是在同一个block中执行的。

一般来说vocab_size比BLOCK_THREADS要大很多，需要通过多次迭代完成对vocabulary遍历。

假设通过random（满足概率分布）选择一个prob，计算GE prob的count和prob累加和就可以满足top-k和top-p的条件。min-p会复杂一些，需要找到max(prob)，后面小节会补充找到max(p)的逻辑。

到这里，我们已经基本上分析出sampling的范式

选择一个锚定的概率u
通过迭代方式计算从vocabulary第一个token开始到当前token，从所有满足条件prob的累加和GE锚定概率u的子集中找到最小token_id
top-k/top-p场景需要在DeviceSamplingFromProb执行完成后再次检查条件，确认对于新选出来的token_id检查是否满足sampling条件，如果满足，返回token_id即可。不满足通过binary search方式逼近，不属于本文内容。
选择锚定概率u
float u = curand_uniform(&state) * q;

# 排除prob < max{prob} * factor后，所有prob的累加和
# temp_storage.block_aggregate.value存储在smem中，用于block内部规约
min-p：float q = temp_storage.block_aggregate.value; 
top-k/top-p： float q = 1;
caller调用DeviceSamplingFromProb
SCAN_ALGORITHM: BLOCK_SCAN_WARP_SCANS
REDUCE_ALGORITHM: BLOCK_REDUCE_WARP_REDUCTIONS
d: vocab_size
temp_storage: smem storage to execute block-level reduction
const uint32_t vec_size = std::gcd(16 / sizeof(DType), vocab_size);
float aggregate = 0；

#pragma unroll 2
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    probs_vec.fill(0);
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      probs_vec.cast_load(probs + row_idx * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
    }

    DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM,
                           DETERMINISTIC>(
        i, d, [](float x) { return x > 0; }, u, probs_vec, aggregate, &temp_storage);
    if (float(aggregate) > u) {
      break;
    }
  }
DeviceSamplingFromProb实现

DeviceSamplingFromProb是sampling的核心函数，实现从vocabulary找到满足条件的token_id的功能

先看函数定义：

i: 遍历vocabulary的迭代器
d: vocab_size
pred: 谓词表示只看某些token（初步筛选条件）
u: 锚定的概率累加和
prob_vec: 加载的prob (寄存器变量)
aggregate: 之前循环计算好的prob累加和，要跟u做比较
temp_storage: smem存储，用于block level的reduction

理解DeviceSamplingFromProb关键在于搞明白跟外层循环的配合，重点在pred，u和aggregate这三个参数。

min_p: 遍历probs便可得出max{prob}，很容易计算pivot_prob，pred就是真正的筛选条件
top-k/top-p: 通过binary search方式来逼近pivot_prob的，pred只是初步筛选条件。如果有兴趣看TopKTopPSamplingFromProbKernel会发现，只有满足pivot_0条件才算找到token_id。因为pivot_0是probs中的概率；pivot_1是为了利用binary search逼近计算的mid（不是概率哦）。low是确定被筛选出局的概率上限，即pivot_prob的下限(不包含哦)；high是满足pivot_prob的上限(包含)。binary search就是通过逐次迭代收敛pivot_prob位置。这些都是小菜鸟瞎说的，万万不能当真呀。
TopKTopPSamplingFromProbKernel

下面就是boring的走读代码咯，分为四个部分

规约满足pred的prob的累加和

这部分最简单，通过向量load prob到线程寄存器中，然后计算prob的累加和。

计算线程内部的prob累加和
通过block level原语，计算block内的prob累加和
通过smem在block中每个线程同步block内一致的prob累加和
template <uint32_t VEC_SIZE, uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, bool DETERMINISTIC, typename Predicate>
__device__ __forceinline__ void DeviceSamplingFromProb(
    uint32_t i, uint32_t d, Predicate pred, float u, vec_t<float, VEC_SIZE> prob_vec,
    float& aggregate,
    SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>* temp_storage) {
  const uint32_t tx = threadIdx.x;
  float prob_greater_than_threshold[VEC_SIZE];
  float inclusive_cdf[VEC_SIZE];
  bool greater_than_u[VEC_SIZE], valid[VEC_SIZE];
#pragma unroll
  for (uint32_t j = 0; j < VEC_SIZE; ++j) {
    prob_greater_than_threshold[j] = pred(prob_vec[j]) ? prob_vec[j] : 0;
    valid[j] = pred(prob_vec[j]) && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d;
  }
  float aggregate_local =
      BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage->block_prim.reduce)
          .template Sum<VEC_SIZE>(prob_greater_than_threshold);
  if (tx == 0) {
    temp_storage->block_aggregate.value = aggregate_local;
  }
  __syncthreads();
  aggregate_local = temp_storage->block_aggregate.value;
  ...
}
满足aggregate GE u的最小token_id

条件"aggregate + aggregate_local > u"表示什么?

aggregate: 之前迭代(i是迭代器)计算好的累加和
aggregate_local: 此次迭代计算的累加和
DeviceSamplingFromProb执行结束，会把aggregate_local累加到aggregate返回给caller

"aggregate + aggregate_local > u"表示本次迭代之中存在满足累加和 GE u的token_id。前文有提过sampling目标是找到满足条件的最小的token_id。这也是DeviceSamplingFromProb最复杂的规约逻辑。

在迭代器(i)同一个iteration的迭代中，probs是保存在BLOCK_THREADS个线程的寄存器中的。可以参考上一个代码片段，在同一次迭代中prob_vec是每个线程依次进行向量load读到线程的寄存器中。

找到最小的token_id意味着：

当前iteration：找到最小的tx
不同的iteration：找到vocabulary的最小索引id

通过block level的规约原语，计算从线程0到当前线程tx所有满足pred的prob累加和inclusive_cdf；inclusive_cdf大于0表示[0,tx)存在满足pred的token_id，意味着我们要寻找的token_id就在这个范围里。

template <uint32_t VEC_SIZE, uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, bool DETERMINISTIC, typename Predicate>
__device__ __forceinline__ void DeviceSamplingFromProb(
    uint32_t i, uint32_t d, Predicate pred, float u, vec_t<float, VEC_SIZE> prob_vec,
    float& aggregate,
    SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>* temp_storage) {
  ...
  if (aggregate + aggregate_local > u) {
    if constexpr (DETERMINISTIC) {
      DeterministicInclusiveSum<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>(
          prob_greater_than_threshold, inclusive_cdf, temp_storage);
    } else {
      BlockScan<float, BLOCK_THREADS, SCAN_ALGORITHM>(temp_storage->block_prim.scan)
          .template InclusiveSum<VEC_SIZE>(prob_greater_than_threshold, inclusive_cdf);

      __syncthreads();
    }
    ...
  }
}

但是，我们需要的token_id不是包含的范围, 还需要进一步计算。

greater_than_u的含义

inclusive_cdf：满足条件的token_id在[0,tx)
aggregate > u：满足条件的token_id就在tx
valid：vocabulary合法token

再次利用block level的规约原语，找到跳变的上沿。因为tx's greater_than_u是(tx+1)'s greater_than_u的子集。tx's greater_than_u_diff为true表示，跳变的上沿就在当前的线程内部。

有一点需要注意，DeviceSamplingFromProb和其调用者都是线程上下文的，同一个block中的线程会并行执行。vocabulary中满足上述条件的最小token_id，需要借助memory atomic原语实现。这里probs的每行都是在同一个block中执行的，只需借助shared memory的atomic语义。

template <uint32_t VEC_SIZE, uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, bool DETERMINISTIC, typename Predicate>
__device__ __forceinline__ void DeviceSamplingFromProb(
    uint32_t i, uint32_t d, Predicate pred, float u, vec_t<float, VEC_SIZE> prob_vec,
    float& aggregate,
    SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>* temp_storage) {
  ...
  if (aggregate + aggregate_local > u) {
     ...
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      greater_than_u[j] = (inclusive_cdf[j] + aggregate > u) && valid[j];
    }

    bool greater_than_u_diff[VEC_SIZE];
#ifdef FLASHINFER_CUB_SUBTRACTLEFT_DEFINED
    BlockAdjacentDifference<bool, BLOCK_THREADS>(temp_storage->block_prim.adj_diff)
        .SubtractLeft<VEC_SIZE>(greater_than_u, greater_than_u_diff, BoolDiffOp());
#else
    BlockAdjacentDifference<bool, BLOCK_THREADS>(temp_storage->block_prim.adj_diff)
        .template FlagHeads<VEC_SIZE>(greater_than_u_diff, greater_than_u, BoolDiffOp(), 0);
#endif
    __syncthreads();
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      if (greater_than_u_diff[j]) {
        atomicMin(&(temp_storage->sampled_id), (i * BLOCK_THREADS + tx) * VEC_SIZE + j);
      }
    }
    __syncthreads();
  }
  ...
}
退而求其次的选择

前文有提到u是通过random生成的，如果本次迭代没有满足"aggregate + aggregate_local > u"，aggregate_local还是会被累加到caller的aggregate中；不幸的是，经过数次迭代(i迭代器)都没有找到合适token_id，就会用最后一次迭代中找到的满足pred的最大的token_id。

此时，u又是什么呢？看注释是u接近1情况。为什么找不到合适的token_id呢？

pred谓词排除后，vocabulary所有满足pred的token prob累加和 < u
每次迭代只看min(BLOCK_THREADS*VEC_SIZE, rem token_id in vocabulary)个token，而且浮点数不连续，累加过程的误差导致。这是我瞎说的，只是把能想到原因都罗列出来了。
template <uint32_t VEC_SIZE, uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, bool DETERMINISTIC, typename Predicate>
__device__ __forceinline__ void DeviceSamplingFromProb(
    uint32_t i, uint32_t d, Predicate pred, float u, vec_t<float, VEC_SIZE> prob_vec,
    float& aggregate,
    SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>* temp_storage) {
  ...
  // update the last valid index
  int valid_index[VEC_SIZE];
#pragma unroll
  for (uint32_t j = 0; j < VEC_SIZE; ++j) {
    if (valid[j]) {
      valid_index[j] = (i * BLOCK_THREADS + tx) * VEC_SIZE + j;
    } else {
      valid_index[j] = -1;
    }
  }
  int max_valid_index =
      BlockReduce<int, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage->block_prim.reduce_int)
          .Reduce(valid_index, MaxReduceOp{});
  if (tx == 0 && max_valid_index != -1) {
    temp_storage->last_valid_id = max_valid_index;
  }
  __syncthreads();
  aggregate += aggregate_local;
}
计算max{prob}

以BLOCK_THREADS * VEC_SIZE为单位遍历probs，然后通过block level的原语计算(规约)最大值。

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE, bool DETERMINISTIC,
          typename DType, typename IdType>
__global__ void MinPSamplingFromProbKernel(DType* probs, float* min_p_arr, IdType* output,
                                           IdType* indices, float min_p_val, uint32_t d,
                                           uint64_t* seed_arr, uint64_t seed_val,
                                           uint64_t* offset_arr, uint64_t offset_val) {
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  ...
  const uint32_t row_idx = indices == nullptr ? bx : indices[bx];

  extern __shared__ __align__(
      alignof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>))
      uint8_t smem_sampling[];
  auto& temp_storage =
      reinterpret_cast<SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>&>(
          smem_sampling);

  float max_val = GetMaxValue<VEC_SIZE, BLOCK_THREADS, REDUCE_ALGORITHM,
                              SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>>(
      probs, row_idx, d, temp_storage);
  float pivot = max_val * p;
  ...
}

template <uint32_t VEC_SIZE, uint32_t BLOCK_THREADS, BlockReduceAlgorithm REDUCE_ALGORITHM,
          typename TempStorage>
__device__ __forceinline__ float GetMaxValue(float* in_data, uint32_t row_idx, uint32_t d,
                                             TempStorage& temp_storage) {
  const uint32_t tx = threadIdx.x;
  vec_t<float, VEC_SIZE> in_data_vec;

  // Thread-local max accumulation (deferred reduction)
  float thread_max = 0.0f;
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    in_data_vec.fill(0);
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      in_data_vec.cast_load(in_data + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      thread_max = max(thread_max, static_cast<float>(in_data_vec[j]));
    }
  }

  // Single block reduction after loop completes
  float max_val =
      BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
          .Reduce(thread_max, MaxReduceOp{});
  if (tx == 0) {
    temp_storage.max_val = max_val;
  }
  __syncthreads();
  return temp_storage.max_val;
}
