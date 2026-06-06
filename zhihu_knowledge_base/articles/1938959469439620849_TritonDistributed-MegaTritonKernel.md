# TritonDistributed-MegaTritonKernel

**作者**: xg20

**原文链接**: https://zhuanlan.zhihu.com/p/1938959469439620849

---

项目代码

https://github.com/ByteDance-Seed/Triton-distributed

TL;DR;

我们在Triton-distributed中新增了MegaTritonKernel的特性，将模型 forward 计算融合为一个 Triton Kernel，在 H800 和 H20 GPU 上的decode latency 相比现有baseline有一定的加速。


背景

对于基于LLM的应用，延迟对用户体验有较大的影响。特别是在模型think过程生成长文本输出时，延迟问题会变得尤为显著。
在小batch的decode这种memory bound的场景，尽管H20等GPU具备高带宽，但实际运行过程中GPU资源无法充分利用：多数kernel的计算时间小于10us，而kernel的launch开销通常需要5us以上，同时框架本身也会引入开销，这些因素导致gpu的利用率不足，且难以通过单算子优化降低单token的latency。NVGPU上虽然引入了CUDA Graph和PDL技术优化kernel launch的overhead，但这些技术本质上没有消除kernel的边界，无法实现极致的延迟优化。
为了降低decode的延迟，MegaKernel(mirage[1]/hazy reasearch megakernel[2])将整个模型forward融合成一个大kernel，彻底消除kernel边界。但基于CUDA进行实现，在不同厂商的卡上移植成本较高，在不同型号的GPU上，还需进行性能调优；同时需要编写大量CUDA代码，对用户而言开发调试负担较重。
TritonDistributed在计算和通信上都有较好的编程性和性能可移植性。基于TritonDistributed进行实现和优化可以得到通用的技术方案。


构建MegaTritonKernel

不同LLM结构相似但算子层面存在细微差异。为了让MegaTritonKernel具备较好的灵活性和易用性，我们采用task + interpreter的架构。具体而言，对于给定的模型，将其中每个算子拆分为多个 task，每个 task 代表最小的可执行任务，并建立task之间的依赖关系。由scheduler将任务分配到每个SM上workqueue中。运行时，每个SM则充当 interpreter，根据workqueue中的 task 信息按顺序执行相应的计算任务。在用户层面，仍通过算子级别的接口用于构建模型，由框架负责task拆分和MegaTritonkernel的生成。需要解决以下问题：

Op如何拆分成task
Op之间的data dependency如何处理
每个SM如何获取所需要执行的task
mega triton kernel的生成
Task Builder：Op到task的拆分和转换

Op拆分成task
算子对应的cuda/triton实现中，通常都需要将其切分成多个block，每个block负责一部分的计算，并最终调度到某个SM上执行。因此我们将一个block的计算作为一个task，用户可以根据算子的计算逻辑以及相应的最高性能实现，自定义相应的task拆分方式。
一个算子拆分得到的task是基本同构的，例如matmul按照output进行切分，不同task负责不同的tile的计算，task之间的区别仅有负责计算的tile位置不同。因此，对于一个task，除了需要保留原算子的输入输出信息以外，仅需额外增加一个tile_id即可描述该task对应的计算。

class TaskBase:
    tile_id: int
    io_tensors: List[Tensor]
    ...

以 matmul 算子为例，拆分后的 task 如下：

# matmul Op-> tasks
(tile_id=0, io_tensors = [(a_ptr, M, K), (b_ptr, N, K), (c_ptr, M, N)])
(tile_id=1, io_tensors = ...)
(tile_id=2, io_tensors = ...)
.....

Triton算子到task kernel的转换
对于算子已有的triton 实现，不同block之间的执行逻辑仅通过program_id进行区分。因此可以复用，将其简单封装即可得到task对应的triton实现。并且tile_id和program_id通常存在一一映射的关系（persistent kernel即多个tile对应一个program_id)。
以下为matmul的task kernel，其逻辑和matmul的triton kernel基本一致：

@triton.jit
def matmul_task_compute(tile_id, a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                             NUM_STAGES):
    # linear: a (M, K) x b (N, K) -> c (M, N)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)

    pid_m = tile_id // num_pid_n
    pid_n = tile_id % num_pid_n
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N
    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_am = tl.where(offs_am < M, offs_am, 0)
    offs_bn = tl.where(offs_bn < N, offs_bn, 0)
    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for ki in tl.range(0, k_tiles, num_stages=NUM_STAGES):
        offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * K + offs_k[None, :])
        b_ptrs = b_ptr + (offs_bn[:, None] * K + offs_k[None, :])

        a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b.T, accumulator)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + N * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c = accumulator.to(c_ptr.dtype.element_ty)
    tl.store(c_ptrs, c, mask=c_mask)
Scoreboard：task间数据依赖管理

task是最小的可执行单元，算子级别的依赖需转换为task级别的依赖。与原先算子级别的依赖相比，task级别的依赖支持更细粒度的优化，每个task只需要等待前置算子的部分task完成，即可开始运算（目前MegaTritonKernel中task间的data dependency比较冗余，需进一步优化）。
具体实现上，MegaTritonKernel通过global memory中的scoreboard tensor管理task间的dependency，每个task都有一个对应的signal，scoreboard支持两种操作：

release_task：task执行完成后将对应的signal置1
wait_deps：通过自旋锁(spinlock)等待当前task所依赖的task完成（即等待对应 signal 置 1）。
Scheduler：task到SM的分配

算子拆分成task并构建好task之间的依赖关系之后，需要将task分配到不同的SM上。这里可以采取不同的策略，例如round-robin或者基于cost的方式（当前MegaTritonKernel中使用round-robin策略，调度策略未来会进一步扩展优化）。通过特定的调度策略为每个SM生成workqueue后，由于task间的依赖由scoreboard处理，每个SM仅需要充当interpreter，按workqueue顺序执行task。

Codegen

Task编码
生成workqueue之后， 需将每个task进行编码，转换为kernel可识别的形式，当前实现中，task按照如下顺序进行编码，每个字段用固定长度的int32表示。task编码和kernel中解码约定好形式，kernel执行时每个SM即可获取对应task的信息以执行计算。

task_type | layer_id | task_id | tile_id_or_start | dependency | io_tensors | extra_params

生成MegaTritonKernel
Codegen根据workqueue中的task类型，动态生成最终的MEGA_TRITON_KERNEL。这个Kernel本质上是一个巨大的switch-case，根据解码出的task信息，分发到对应的Task Kernel实现。

@triton.jit
def MEGA_TRITON_KERNEL(
    work_queues, # 
    num_tasks_per_wq, #[num_sms,]
    scoreboard_ptr,
    ...
    NUM_SMS: tl.constexpr,
    num_warps: tl.constexpr
):
    scoreboard = Scoreboard(scoreboard_ptr, ...)
    num_tasks = tl.load(num_tasks_per_wq + sm_id)

    for i in range(num_tasks):
        task_base_info = load_task(work_queues + i)
         
        #0. 等待前置task 完成
        scoreboard.wait_deps(task_base_info)
        
        #1. 根据task type执行特定的task
        run_task(task_base_info)
        
        #2. task完成后，设置相应的signal
        scoreboard.release_task(task_base_info)


其他优化
算子融合：基于Triton易开发融合算子（如RMSNorm和RoPE 、Norm和Add等），用户可根据具体的模型结构针对优化，编写相应的triton kernel并转换成task kernel，注册到MegaTritonKernel中即可。
低延迟通信：通过TritonDistributed可以实现低延迟的通信算子（例如low latency all reduce），进一步提高e2e的性能。
参数调优：对于每个算子的task，通过triton的autotune获取不同problem size下的最佳参数，可直接集成到MegaTritonKernel中。



性能总览

在H20和H800 GPU上，测试MegaTritonKernel在decoding场景(bs=1, ctx=512)的性能，并进行对比：

torch eager: torch + nccl + flashinfer
mirage: mirage persistent kernel
torch + cudagraph: 在torch基础上开启cudagraph
triton_dist_AR + cudagraph：增加了TritonDistributed实现的低延迟通信等优化，并开启cudagraph
mega_triton_kernel

上图为qwen3-32b tp8的decode性能结果。在H800和H20上，MegaTritonKernel的延迟分别为7.41ms/8.34ms，相比于TritonDistributed + cudagraph、torch + cudagraph，分别取得了1.23x/1.43x和1.45x/1.66x的加速。mirage在H20上未跑通，在H800上可取得1.83x的加速。


在qwen3-8b tp8，MegaTritonKernel在H800/H20上对比mirage、torch+cudagraph、TritonDistributed + cudagraph分别取得1.66x/2.52、1.64/1.74和1.39/1.42x加速。每秒产生的token可达300个。


尽管当前的baseline已经使用了高性能的算子和cudagraph，MegaTritonKernel仍能取得加速。mirage当前的优化主要针对于A100，在hopper卡上可能需要重新调优优化。得益于TritonDistributed和Triton在计算和通信上的通用性和可移植性，MegaTritonKernel在不同GPU上均能保持较高的性能。


参考资料


[1]Mirage: A Multi-Level Superoptimizer for Tensor Programs
[2]Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B
