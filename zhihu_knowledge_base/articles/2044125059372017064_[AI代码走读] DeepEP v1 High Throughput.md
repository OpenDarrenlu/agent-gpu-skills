# [AI代码走读] DeepEP v1 High Throughput

**作者**: ZhangZVibe Coding忠实拥趸手撸醋打蒜籽

**原文链接**: https://zhuanlan.zhihu.com/p/2044125059372017064

---

本文由gpt5.4生成

这篇文章只讲 DeepEP V1 的 high-throughput 路径，也就是 normal kernels。

不讨论 low-latency mode，不讨论 V2。目标很明确：让读者真正搞懂下面三件事。

V1 high-throughput 在解决什么问题。
get_dispatch_layout -> dispatch -> combine 这条主链路怎么跑。
每个核心函数的输入、输出、以及这些输入输出为什么存在。

本文所有源码链接都固定到这个提交：

https://github.com/deepseek-ai/DeepEP/tree/d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb

如果只记一句话：

DeepEP V1 high-throughput 本质上是在做一次高性能 MoE all-to-all。先把 token 按 expert 路由到对应 rank 做计算，再把结果按原 token 聚合回来。
1. 先建立整体心智模型

先别钻进实现细节，先把 high-throughput 路径看成这条流水线：

输入 token x
  -> 根据 topk_idx 计算路由布局
  -> dispatch: 把 token 发往目标 rank
  -> 每个 rank 上的本地 expert 做计算
  -> combine: 把 expert 输出聚合回原 token
  -> 得到最终结果

在 V1 里，这条流水线对应三个核心 API：

Buffer.get_dispatch_layout(...)
Buffer.dispatch(...)
Buffer.combine(...)

Python 包装层定义在：

https://github.com/deepseek-ai/DeepEP/blob/d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb/deep_ep/buffers/legacy.py
2. 从官方用法先看一遍全流程

官方文档里的 V1 high-throughput 示例在：

https://github.com/deepseek-ai/DeepEP/blob/d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb/docs/legacy.md#example-use-in-model-training-or-inference-prefilling

核心调用大概是这样：

num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, _ = \
    _buffer.get_dispatch_layout(topk_idx, num_experts)

recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event = \
    _buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
    )

# 本地 expert 计算

combined_x, _, event = _buffer.combine(expert_out, handle)

这三步正是本文的主线：

先算布局
再 dispatch
最后 combine

最容易卡住的问题通常也是三个：

为什么 layout 要单独算一次
为什么 dispatch 要返回 handle
为什么 combine 必须吃这个 handle
3. Buffer 初始化时到底准备了什么

Python 构造函数在：

https://github.com/deepseek-ai/DeepEP/blob/d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb/deep_ep/buffers/legacy.py#L33

关键代码：

self.runtime = _C.Buffer(self.rank, self.group_size, num_nvl_bytes, num_rdma_bytes, low_latency_mode,
                         explicitly_destroy, enable_shrink, allow_mnnvl)

local_device_id = self.runtime.get_local_device_id()
device_ids = all_gather_object(local_device_id)

local_ipc_handle = self.runtime.get_local_ipc_handle()
ipc_handles = all_gather_object(local_ipc_handle)

root_unique_id = None
if self.runtime.get_num_rdma_ranks() > 1 or low_latency_mode:
    root_unique_id = self.runtime.get_local_nvshmem_unique_id()
    nvshmem_unique_ids = all_gather_object(root_unique_id)
    root_unique_id = nvshmem_unique_ids[...]

self.runtime.sync(device_ids, ipc_handles, root_unique_id)

这段代码的职责很简单：

Python 层收集分布式元信息
native runtime 真正搭建通信环境

high-throughput 模式里，Buffer 会准备两类空间：

num_nvl_bytes
给 intranode NVLink 路径用




num_rdma_bytes
给 internode RDMA 路径用




native 构造函数在：

https://github.com/deepseek-ai/DeepEP/blob/d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb/csrc/legacy/buffer.hpp#L84

其中有几行很关键：

shared_memory_allocator.malloc(&buffer_ptrs[nvl_rank],
                               num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes + barrier_signal_ptr_bytes);

CUDA_RUNTIME_CHECK(cudaMalloc(&workspace, LEGACY_NUM_WORKSPACE_BYTES));

CUDA_RUNTIME_CHECK(cudaMallocHost(&moe_recv_counter, sizeof(int64_t), cudaHostAllocMapped));
CUDA_RUNTIME_CHECK(cudaHostGetDevicePointer(&moe_recv_counter_mapped, const_cast<int*>(moe_recv_counter), 0));

这几行揭示了 V1 的设计：

GPU kernel 跑在 comm_stream
GPU 会把接收规模等元信息写到 host-pinned counter
CPU 读取这些 counter，再决定该分配多大的接收 tensor

这就是后面 CPU wait 的来源。

重要发现
V1 normal mode 是典型的 CPU 控制面 + GPU 数据面：

GPU 负责算 layout、生成接收规模元信息、执行真实通信
CPU 负责 launch kernel、busy-wait 元信息、分配紧凑输出 tensor




3.1 use_fabric=True/False 时，本地 NVLink buffer 怎么分配

原文没把这个点讲清楚，这里单独补上。

这部分逻辑在共享内存分配器里，不在 legacy/buffer.hpp 主逻辑里：

https://github.com/deepseek-ai/DeepEP/blob/d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb/csrc/utils/shared_memory.hpp

它影响的是 本地 NVLink 共享 buffer 的分配与跨进程共享方式，不影响 rdma_buffer_ptr 那条 NVSHMEM/RDMA 路径。

use_fabric=False

走经典 CUDA IPC 路径。

if (use_fabric) {
    ...
} else {
    CUDA_RUNTIME_CHECK(cudaMalloc(ptr, size));
}

导出和导入句柄时：

CUDA_RUNTIME_CHECK(cudaIpcGetMemHandle(&mem_handle->inner.cuda_ipc_mem_handle, ptr));
CUDA_RUNTIME_CHECK(cudaIpcOpenMemHandle(ptr, mem_handle->inner.cuda_ipc_mem_handle, cudaIpcMemLazyEnablePeerAccess));

也就是：

分配：cudaMalloc
共享：cudaIpcGetMemHandle / cudaIpcOpenMemHandle
模型：老的 CUDA IPC 模型
use_fabric=True

走新的 cuMem driver API 路径。

prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

CUDA_DRIVER_CHECK(lazy_cuMemCreate(&handle, size, &prop, 0));
CUDA_DRIVER_CHECK(lazy_cuMemAddressReserve(reinterpret_cast<CUdeviceptr*>(ptr), size, alignment, 0, 0));
CUDA_DRIVER_CHECK(lazy_cuMemMap(reinterpret_cast<CUdeviceptr>(*ptr), size, 0, handle, 0));
cu_mem_set_access_all(*ptr, size);

导出和导入句柄时：

CUDA_DRIVER_CHECK(lazy_cuMemExportToShareableHandle(&mem_handle->inner.cu_mem_fabric_handle, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
CUDA_DRIVER_CHECK(lazy_cuMemImportFromShareableHandle(&handle, &mem_handle->inner.cu_mem_fabric_handle, CU_MEM_HANDLE_TYPE_FABRIC));

也就是：

分配：cuMemCreate
预留虚拟地址：cuMemAddressReserve
映射：cuMemMap
设置访问权限：cuMemSetAccess
共享：fabric shareable handle
这两条路径的本质区别
use_fabric=False
cudaMalloc + CUDA IPC
先拿到一块显存，再把它的 IPC handle 分发给其他进程




use_fabric=True
cuMemCreate + reserve/map + fabric handle
先创建 memory object，再决定映射到哪个地址、谁能访问、如何导出




对 V1 high-throughput 来说，两者承担的是同一个角色：

都是在搭本地 NVLink 共享 buffer

但底层内存管理模型不同：

一个是老的 runtime/IPC 路径
一个是新的 driver/vmm/fabric 路径
重要发现
use_fabric=True/False 改变的是 本地 NVLink 共享 buffer 的内存分配/共享模型，不是 high-throughput 的算法主流程：

False: cudaMalloc + CUDA IPC
True: cuMemCreate + reserve/map + fabric handle




4. get_dispatch_layout：输入、输出、作用都是什么

Python 包装层定义在：

https://github.com/deepseek-ai/DeepEP/blob/d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb/deep_ep/buffers/legacy.py#L293

代码：

def get_dispatch_layout(self, topk_idx, num_experts, previous_event=None, async_finish=False,
                        allocate_on_comm_stream=False):
    num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, event = \
        self.runtime.get_dispatch_layout(topk_idx, num_experts, getattr(previous_event, 'event', None),
                                         async_finish, allocate_on_comm_stream)
    return num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, EventOverlap(event)
输入分别是干什么的
topk_idx
shape 是 [num_tokens, num_topk]
每个 token 选择了哪些 expert
这是整个路由计算的核心输入




num_experts
全局 expert 总数
用来把 topk_idx 里的 expert id 映射到 rank 和 local expert




previous_event
可选依赖事件
表示 layout kernel 开始前，需要先等待之前的某个 CUDA 事件完成




async_finish
如果为 True，当前 compute stream 不会等 layout 完成，而是返回一个 event 给调用方自己管理




allocate_on_comm_stream
控制输出 tensor 是否在通信 stream 所属的时序上分配和记录




输出分别是干什么的
num_tokens_per_rank
shape 是 [num_ranks]
表示当前 rank 的输入 token 一共要发给每个 rank 多少份




num_tokens_per_rdma_rank
shape 是 [num_rdma_ranks]
只在 internode 下有意义
表示跨 RDMA domain 后每个 RDMA rank 的流量规模




num_tokens_per_expert
shape 是 [num_experts]
表示每个 expert 最终要接收多少 token




is_token_in_rank
shape 是 [num_tokens, num_ranks]
布尔矩阵
第 i 个 token 是否需要发到第 r 个 rank




event
当 async_finish=True 时有效
供后续 stream 同步使用




把这组输出放在一起看，会更清楚：

topk_idx:                 [num_tokens, num_topk]
num_tokens_per_rank:      [num_ranks]
num_tokens_per_rdma_rank: [num_rdma_ranks]      # 仅 internode
num_tokens_per_expert:    [num_experts]
is_token_in_rank:         [num_tokens, num_ranks]

这里最重要的是理解两类 shape：

[num_tokens, ...]
还是站在“原始输入 token”视角




[num_ranks] / [num_experts]
已经是在统计“发往哪里、每个目的地有多少”




为什么这个函数必须单独存在

因为 dispatch 真正搬数据前，必须先知道：

token 要发给谁
每个目的地大概有多少数据
每个 local expert 最终会收到多少 token

native 实现在：

https://github.com/deepseek-ai/DeepEP/blob/d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb/csrc/legacy/buffer.hpp#L337
auto num_tokens_per_rank = torch::empty({num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
auto num_tokens_per_expert = torch::empty({num_experts}, dtype(torch::kInt32).device(torch::kCUDA));
auto is_token_in_rank = torch::empty({num_tokens, num_ranks}, dtype(torch::kBool).device(torch::kCUDA));
if (is_internode_available())
    num_tokens_per_rdma_rank = torch::empty({num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA));

layout::get_dispatch_layout(...);

这一步不搬运 x 本身，只计算后续要怎么搬。

重要发现
layout 不是“统计信息之外的额外信息”。
更准确地说：

topk_idx 只是原始路由选择
layout 是把它翻译成“通信可执行格式”
num_tokens_per_rank、num_tokens_per_expert、is_token_in_rank 都是这一步的直接产物




5. dispatch：输入、输出、作用都是什么

Python 定义在：

https://github.com/deepseek-ai/DeepEP/blob/d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb/deep_ep/buffers/legacy.py#L322

这是 V1 high-throughput 的核心入口。

dispatch 的输入分别是干什么的
x
要被路由的 token 数据
普通 BF16 形式是 [num_tokens, hidden]
FP8 形式是 (data, scales)
data shape 是 [num_tokens, hidden]
scales shape 通常是 [num_tokens, hidden // 128]




handle
可选
如果传入，表示走 cached mode，复用之前的布局元数据




num_tokens_per_rank
每个 rank 会收到多少 token
通常来自 get_dispatch_layout




num_tokens_per_rdma_rank
internode 模式下的 RDMA rank 级别流量统计




is_token_in_rank
token 到 rank 的布尔映射矩阵




num_tokens_per_expert
每个 expert 会收到多少 token




topk_idx
每个 token 选中的 expert id




topk_weights
每个 token 对应的 expert 权重




expert_alignment
要求每个本地 expert 接收 token 数按这个粒度对齐




num_worst_tokens
只限 intranode
如果调用方愿意按最坏情况预分配，就可以避免 CPU 同步等待精确接收规模




config
调优参数，决定使用多少 SM、每个 channel 的 chunk 大小等




previous_event
dispatch 启动前需要等待的依赖事件




async_finish
如果为 True，dispatch 结束后不强制同步 compute stream，而是返回 event




allocate_on_comm_stream
是否把分配出来的输出 tensor 归到通信 stream 的生命周期里




把输入 shape 放在一起看：

x (BF16):                 [num_tokens, hidden]
x (FP8 data):             [num_tokens, hidden]
x (FP8 scales):           [num_tokens, hidden // 128]   # 常见情况
num_tokens_per_rank:      [num_ranks]
num_tokens_per_rdma_rank: [num_rdma_ranks]              # 仅 internode
is_token_in_rank:         [num_tokens, num_ranks]
num_tokens_per_expert:    [num_experts]
topk_idx:                 [num_tokens, num_topk]
topk_weights:             [num_tokens, num_topk]

这里有一个关键观察：

x、topk_idx、topk_weights 都还在 token 视角
num_tokens_per_rank、num_tokens_per_expert 已经是 layout 视角

dispatch 的作用，就是把“token 视角的数据”和“layout 视角的规划”合起来，产出真正按目标 rank 排列的接收结果。

dispatch 的输出分别是干什么的
recv_x
当前 rank 实际收到的 token
是一个紧凑 tensor，只包含真实收到的数据
BF16 时 shape 是 [num_recv_tokens, hidden]
FP8 时返回 (recv_x, recv_x_scales)：
recv_x shape 是 [num_recv_tokens, hidden]
recv_x_scales shape 通常是 [num_recv_tokens, hidden // 128]







recv_topk_idx
收到 token 对应的 local expert 索引信息
如果本次没传 topk_idx/topk_weights，可能为空
shape 是 [num_recv_tokens, num_topk]




recv_topk_weights
收到 token 对应的权重信息
shape 是 [num_recv_tokens, num_topk]




num_recv_tokens_per_expert_list
Python list
当前 rank 上每个 local expert 实际收到了多少 token
这通常直接喂给后续 GEMM 调度
长度是 num_local_experts




handle
这是最重要的返回值之一
它保存了 combine 阶段需要的回程元数据，也能供下一次 dispatch 复用




event
当 async_finish=True 时有效




把 dispatch 的输出 shape 放在一起：

recv_x (BF16):                    [num_recv_tokens, hidden]
recv_x (FP8 data):                [num_recv_tokens, hidden]
recv_x_scales (FP8):              [num_recv_tokens, hidden // 128]   # 常见情况
recv_topk_idx:                    [num_recv_tokens, num_topk]
recv_topk_weights:                [num_recv_tokens, num_topk]
num_recv_tokens_per_expert_list:  len = num_local_experts

这里的 num_recv_tokens 是最关键的中间量：

它不是输入的 num_tokens
它是“当前 rank 最终实际收到的 token 条目数”
因为一个 token 可能发往多个 expert/rank，所以 num_recv_tokens 一般和输入 token 数不同
重要发现
dispatch 返回的是 紧凑接收结果，不是按最坏情况预留的大 buffer。
所以 normal mode 才需要先得到精确接收规模，再分配 recv_x: [num_recv_tokens, hidden]。
dispatch 返回的 handle 里每个张量的 shape

这部分原文只讲了语义，这里把 shape 补全。

intranode handle
handle = (
    rank_prefix_matrix,
    channel_prefix_matrix,
    recv_channel_prefix_matrix,
    recv_src_idx,
    is_token_in_rank,
    send_head,
)

各字段 shape：

rank_prefix_matrix
[num_ranks, num_ranks]




channel_prefix_matrix
[num_ranks, num_channels]




recv_channel_prefix_matrix
[num_ranks, num_channels]




recv_src_idx
[num_recv_tokens]




is_token_in_rank
[num_tokens, num_ranks]




send_head
[num_tokens, num_ranks]




各字段在后续流程里的作用：

rank_prefix_matrix
记录按 rank 组织后的前缀和布局
combine 用它知道每个 rank 返回的数据段在整体回程里的边界




channel_prefix_matrix
记录发送侧各个 communication channel 的切分边界
dispatch cached mode 会直接复用它
combine 也要依赖它按相同 channel 布局回收数据




recv_channel_prefix_matrix
记录接收侧各个 channel 的数据前缀布局
主要用于 dispatch 结果的接收组织和后续调试/复用语义
在 Python combine(...) 入口里不会直接解包使用




recv_src_idx
记录每个收到的 token 条目对应哪个源 token
combine 需要靠它把 expert 输出加回原 token 位置




is_token_in_rank
记录原始 token 到 rank 的布尔映射
cached dispatch 会直接复用它，避免重新计算 layout




send_head
记录回程发送队列的 head 元数据
combine 用它知道每个原 token 对应的回传队列入口




internode handle

在 Python 层是：

handle = (
    is_token_in_rank,
    rdma_channel_prefix_matrix,
    gbl_channel_prefix_matrix,
    recv_rdma_channel_prefix_matrix,
    recv_rdma_rank_prefix_sum,
    recv_gbl_channel_prefix_matrix,
    recv_gbl_rank_prefix_sum,
    recv_src_meta,
    send_rdma_head,
    send_nvl_head,
)

各字段 shape：

is_token_in_rank
[num_tokens, num_ranks]




rdma_channel_prefix_matrix
[num_rdma_ranks, num_channels]




gbl_channel_prefix_matrix
[num_ranks, num_channels]




recv_rdma_channel_prefix_matrix
[num_rdma_ranks, num_channels]




recv_rdma_rank_prefix_sum
[num_rdma_ranks]




recv_gbl_channel_prefix_matrix
[num_ranks, num_channels]




recv_gbl_rank_prefix_sum
[num_ranks]




recv_src_meta
[num_recv_tokens, src_meta_bytes]
其中 src_meta_bytes = internode::get_source_meta_bytes()




send_rdma_head
[num_recv_tokens, num_rdma_ranks]




send_nvl_head
第一维等于 num_rdma_recv_tokens
第二维是本地 NVLink peer 维度，具体常量是 LEGACY_NUM_MAX_NVL_PEERS




各字段在后续流程里的作用：

is_token_in_rank
和 intranode 类似，记录原始 token 是否需要发往某个全局 rank
cached dispatch 会直接复用这份映射




rdma_channel_prefix_matrix
记录 RDMA 维度上各 channel 的发送前缀布局
cached dispatch 和 combine 都会依赖它组织跨节点传输




gbl_channel_prefix_matrix
记录全局 rank 维度上的 channel 切分布局
用来把 RDMA 和本地 NVLink 两层流量组织到统一的回程结构里




recv_rdma_channel_prefix_matrix
记录接收侧 RDMA channel 的布局
主要用于 dispatch 接收侧的数据组织和 cached 复用




recv_rdma_rank_prefix_sum
记录每个 RDMA rank 的接收前缀和
cached dispatch 用它快速恢复接收布局




recv_gbl_channel_prefix_matrix
记录接收侧全局 rank 维度的 channel 布局
用于恢复全局视角下的接收组织




recv_gbl_rank_prefix_sum
记录每个全局 rank 的接收前缀和
cached dispatch 会直接复用




recv_src_meta
记录每个收到条目的 source metadata
combine 的 internode 回程主要依赖它恢复原 token 来源和回传路径




send_rdma_head
记录回程 RDMA 队列的 head 元数据
combine 用它组织跨节点返回流量




send_nvl_head
记录回程 NVLink 队列的 head 元数据
combine 用它完成节点内回传与归并




dispatch 在 Python 层先做了什么分流
config = self.get_dispatch_config(self.group_size) if config is None else config

if self.runtime.get_num_rdma_ranks() > 1:
    return self.internode_dispatch(...)

x, x_scales = x if isinstance(x, tuple) else (x, None)
if handle is not None:
    ...
else:
    ...

它先分两层：

第一层：intranode 还是 internode
第二层：cached mode 还是 non-cached mode
intranode dispatch 真正在 native 层做了什么

native 实现在：

https://github.com/deepseek-ai/DeepEP/blob/d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb/csrc/legacy/buffer.hpp#L417

最值得抓住的不是全部参数，而是 4 个阶段。

阶段 1：确定 channel 数
EP_HOST_ASSERT(config.num_sms % 2 == 0);
int num_channels = config.num_sms / 2;

V1 把通信并行度组织成 channel，num_sms 直接决定并行通道数。

阶段 2：非 cached 模式先发送“接收规模元信息”
*moe_recv_counter = -1;
for (int i = 0; i < num_local_experts; ++i)
    moe_recv_expert_counter[i] = -1;

intranode::notify_dispatch(...);

这一步不是传真正的数据，而是先通知：

每个 rank 会收到多少 token
每个 local expert 会收到多少 token
prefix 信息如何组织

这里最容易混淆的是：

get_dispatch_layout(...) 并不会设置 moe_recv_counter
moe_recv_counter / moe_recv_expert_counter 是在 dispatch(...) 内部的 notify_dispatch(...) 阶段写出来的

也就是说，正确时序是：

get_dispatch_layout(...) 先产出发送侧 layout 信息
dispatch(...) 把 counter 清成 -1
notify_dispatch(...) 在 GPU 上把接收规模元信息写到这些 counter
CPU busy-wait 直到这些 counter 变成有效值
CPU 再按精确 shape 分配 recv_x 等输出 tensor
dispatch 的 recv size 信息到底是什么

对 normal mode 来说，最关键的接收规模信息有三类：

moe_recv_counter
当前 rank 最终总共会收到多少 token 条目
这就是后面 recv_x 第一维的来源




moe_recv_expert_counter
当前 rank 上每个 local expert 分别会收到多少 token
后面会变成 Python 返回的 num_recv_tokens_per_expert_list




internode 下还有 moe_recv_rdma_counter
当前 rank 在 RDMA 维度上会收到多少 token 条目




这些 counter 的职责分工是：

内存由 Buffer runtime 持有
CPU 先把它们置成 -1
GPU 执行 notify_dispatch(...) 时写入真实值
CPU 在 host 侧轮询这些值

可以把它们理解成：

dispatch 的“精确接收规模回执”。
为什么 dispatch 一定要先拿到 recv size

因为 V1 normal mode 返回的是紧凑输出，不是按最坏情况预分配的大 buffer。

例如：

auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());

这里的 num_recv_tokens 在 notify_dispatch(...) 完成前是未知的。

所以对 dispatch 来说，必须先有 recv size，才能创建：

recv_x: [num_recv_tokens, hidden]
recv_topk_idx: [num_recv_tokens, num_topk]
recv_topk_weights: [num_recv_tokens, num_topk]
recv_src_idx: [num_recv_tokens]

这也是为什么：

没有 moe_send_counter
但一定有 moe_recv_counter

因为发送规模在 layout 阶段已经知道了，而 dispatch 真正要分配的是 接收侧输出 tensor。

重要发现
notify_dispatch 不是在传真实 token 数据。
它先产生并传播的是：

接收规模元信息
prefix / channel 布局元信息

真正的数据搬运发生在后面的 dispatch kernel。

阶段 3：CPU busy-wait 等精确接收规模
while (true) {
    num_recv_tokens = static_cast<int>(*moe_recv_counter);
    bool ready = (num_recv_tokens >= 0);
    for (int i = 0; i < num_local_experts and ready; ++i)
        ready &= moe_recv_expert_counter[i] >= 0;
    if (ready)
        break;
}

这解释了 normal mode 为什么会有 CPU wait。

更精确地说，这里 CPU 在等的是：

num_recv_tokens = *moe_recv_counter
moe_recv_expert_counter[i]
internode 下还有 num_rdma_recv_tokens = *moe_recv_rdma_counter

只有这些值都 ready 了，CPU 才知道 dispatch 的输出 tensor 该分配成多大。

重要发现
CPU busy-wait 等的不是“数据本体收完了没有”，而是：

moe_recv_counter
moe_recv_expert_counter

这些接收规模元信息是否 ready。
CPU 拿到这些值之后，才能分配精确 shape 的输出 tensor。

阶段 4：按精确大小分配输出，再真正 dispatch
auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
auto recv_src_idx = torch::empty({num_recv_tokens}, dtype(torch::kInt32).device(torch::kCUDA));
auto recv_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
auto send_head = torch::empty({num_tokens, num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));

intranode::dispatch(...);

这几行也验证了上面讲的 output shape：

recv_x: [num_recv_tokens, hidden]
recv_src_idx: [num_recv_tokens]
recv_channel_prefix_matrix: [num_ranks, num_channels]
send_head: [num_tokens, num_ranks]
handle 为什么重要

intranode 第一次 dispatch 返回的 handle 是：

handle = (
    rank_prefix_matrix,
    channel_prefix_matrix,
    recv_channel_prefix_matrix,
    recv_src_idx,
    is_token_in_rank,
    send_head,
)

它不是随便存点缓存，而是在保存两类信息：

下一次相同路由模式下可复用的 dispatch 元数据
combine 阶段回程聚合必须用到的元数据
重要发现
handle 不是普通缓存，而是 V1 的核心设计点。
它同时承担两件事：

cached dispatch 的复用入口
combine 的回程地图




6. combine：输入、输出、作用都是什么

Python 定义在：

https://github.com/deepseek-ai/DeepEP/blob/d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb/deep_ep/buffers/legacy.py#L408
combine 的输入分别是干什么的
x
当前 rank 上各个 local expert 算完后的输出
shape 通常是 [num_dispatched_tokens, hidden]




handle
必填
由前面的 dispatch 返回
负责告诉 combine：这些结果原来属于哪些 token，要发回哪里




topk_weights
如果需要同时把 top-k 权重也归并回来，就传这个




bias
可选最终 bias
可以是 1 个 tensor，也可以是 2 个 tensor 的 tuple




config
combine 自己的调优参数




previous_event
combine 启动前需要等待的事件




async_finish
是否异步返回 event




allocate_on_comm_stream
是否把输出 tensor 绑定到 comm stream 生命周期




这里把 combine 输入 shape 再收紧一下。

对于 intranode normal path：

x:             [num_recv_tokens, hidden]
topk_weights:  [num_recv_tokens, num_topk]   # 如果传入

为什么这里是 num_recv_tokens 而不是 num_tokens：

因为 combine 的输入不是原始 token
而是当前 rank 的 local experts 在 dispatch 之后收到并计算过的那批 token

也就是说，combine 吃的正是前面 dispatch 返回的那批紧凑结果。

重要发现
dispatch 和 combine 不共享同一个输出 tensor。
更准确地说：

dispatch 分配自己的输出 tensor
expert compute 消费 dispatch 输出
combine 再分配自己的最终输出 tensor

跨阶段真正复用的是 handle 元数据，不是同一块结果 buffer。

combine 的输出分别是干什么的
recv_x
也就是 combine 后的最终输出
按原 token 顺序聚合后的结果
shape 是 [num_combined_tokens, hidden]




recv_topk_weights
如果传了 topk_weights，这里返回聚合后的权重结果
shape 是 [num_combined_tokens, num_topk]




event
异步模式下返回的同步事件




这里的 num_combined_tokens 在典型 forward combine 场景下，通常就等于原始输入 token 数 num_tokens。

换句话说：

dispatch 之后 tensor 规模变成 [num_recv_tokens, hidden]
combine 之后通常回到 [num_tokens, hidden]

这也是为什么可以把 combine 理解成“把 expert 空间里的结果重新折叠回原 token 空间”。

combine 在 Python 层做了什么
config = self.get_combine_config(self.group_size) if config is None else config

if self.runtime.get_num_rdma_ranks() > 1:
    return self.internode_combine(...)

rank_prefix_matrix, _, channel_prefix_matrix, src_idx, is_recv_token_in_rank, send_head = handle
recv_x, recv_topk_weights, event = self.runtime.intranode_combine(
    x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix,
    channel_prefix_matrix, send_head, config, ...)
intranode handle 字段和 combine(...) 参数是一一怎么对上的

这是 V1 最值得讲透的一处。先看 Python 层实际解包：

rank_prefix_matrix, _, channel_prefix_matrix, src_idx, is_recv_token_in_rank, send_head = handle

recv_x, recv_topk_weights, event = self.runtime.intranode_combine(
    x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix,
    channel_prefix_matrix, send_head, config, ...)

这里可以直接列成对照表：

handle[0] rank_prefix_matrix        -> intranode_combine(..., rank_prefix_matrix, ...)
handle[1] channel_prefix_matrix     -> Python combine 里没有直接用
handle[2] recv_channel_prefix_matrix-> intranode_combine(..., channel_prefix_matrix, ...)
handle[3] recv_src_idx              -> intranode_combine(..., src_idx, ...)
handle[4] is_token_in_rank          -> Python combine 里解包但本路径未继续传下去
handle[5] send_head                 -> intranode_combine(..., send_head, ...)

注意这里有一个容易误读的点：

Python 里的解包写成了
rank_prefix_matrix, _, channel_prefix_matrix, src_idx, is_recv_token_in_rank, send_head = handle




也就是说：
handle[1] 被丢掉了
handle[2] 被当作 channel_prefix_matrix 传给了 native combine




这和 dispatch 返回 handle 时的名字并不完全一致，所以只看字段名很容易绕晕。读这段代码时，应该以 “combine 最终实际消费了 handle 的哪些位置” 为准，而不是只看 Python 变量名。

把这 6 个槽位逐个讲清楚：

handle[0] = rank_prefix_matrix
shape: [num_ranks, num_ranks]
在 combine 中对应：rank_prefix_matrix
作用：描述按 rank 组织后的前缀边界
用途：native combine 依赖它判断不同 rank 回传的数据段如何拼回整体输出
handle[1] = channel_prefix_matrix
shape: [num_ranks, num_channels]
在 Python combine(...) 这条 intranode 路径里：没有直接传下去
更准确地说：它是 dispatch 发送侧的 channel 布局元数据
用途：
cached dispatch 会复用
但当前 Python intranode combine(...) 实际上传给 native 的不是这一份，而是 handle[2]




handle[2] = recv_channel_prefix_matrix
shape: [num_ranks, num_channels]
在 combine 中被解包成 Python 变量 channel_prefix_matrix
在 combine 中对应：channel_prefix_matrix
作用：描述接收侧 channel 维度上的实际前缀布局
用途：native combine 依赖它知道回程归并时各 channel 的数据边界
handle[3] = recv_src_idx
shape: [num_recv_tokens]
在 combine 中被解包成 src_idx
在 combine 中对应：src_idx
作用：记录每个 dispatch 后收到的条目来自哪个源 token
用途：这是 combine 把 expert 输出加回原 token 位置的关键索引
handle[4] = is_token_in_rank
shape: [num_tokens, num_ranks]
在 combine 中被解包成 is_recv_token_in_rank
但当前 intranode Python combine(...) 没有继续把它传给 native
用途：
cached dispatch 会复用它
从语义上它仍然代表原 token 到 rank 的映射关系




handle[5] = send_head
shape: [num_tokens, num_ranks]
在 combine 中对应：send_head
作用：记录回程发送队列的 head 元数据
用途：native combine 依赖它知道每个原 token 的回传入口以及队列组织方式

如果你想用一句最短的话记这张表，可以记成：

combine 真正最依赖的是 4 个东西：
- src_idx
- rank_prefix_matrix
- recv_channel_prefix_matrix
- send_head

也就是 native 调用里的这几个实参：

self.runtime.intranode_combine(
    x,
    topk_weights,
    bias_0,
    bias_1,
    src_idx,
    rank_prefix_matrix,
    channel_prefix_matrix,
    send_head,
    config,
    ...,
)

其中：

src_idx 负责“这份结果原来属于谁”
rank_prefix_matrix 负责“不同 rank 的回程边界在哪里”
recv_channel_prefix_matrix 负责“不同 channel 的回程边界在哪里”
send_head 负责“每个原 token 的回程队列入口在哪里”

这四者合在一起，才构成了完整的回程地图。

重要发现
看 intranode combine(...) 时，不能只盯 Python 变量名，必须按 handle 槽位 -> native 实参 去理解。
因为真正决定行为的是“最终传给 self.runtime.intranode_combine(...) 的是哪几个张量”。

最关键的一点是：

combine 并不会重新推导回程路径，而是直接从 handle 里拆出 dispatch 阶段已经准备好的元数据。
intranode combine 真正在 native 层做了什么

native 实现在：

https://github.com/deepseek-ai/DeepEP/blob/d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb/csrc/legacy/buffer.hpp#L719

关键代码：

intranode::cached_notify_combine(buffer_ptrs_gpu,
                                 send_head.data_ptr<int>(),
                                 num_channels,
                                 num_recv_tokens,
                                 ...);

auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());

intranode::combine(...,
                   src_idx.data_ptr<int>(),
                   rank_prefix_matrix.data_ptr<int>(),
                   channel_prefix_matrix.data_ptr<int>(),
                   send_head.data_ptr<int>(),
                   ...);

这段代码说明，在 intranode native combine 里：

输入 x 的第一维是当前 rank 手里待归并的 token 数
输出 recv_x 的第一维是 send_head.size(0)
而 send_head 正是在 dispatch 阶段构造出来、代表原 token 维度的一份回程元数据

所以从 shape 视角理解 combine，最简单的记忆方式是：

dispatch: [num_tokens, hidden] -> [num_recv_tokens, hidden]
combine:  [num_recv_tokens, hidden] -> [num_combined_tokens, hidden]

其中在常见 forward 路径里：

num_combined_tokens == num_tokens

这里最重要的是明白：

src_idx
告诉 combine 每份 expert 输出对应哪个源 token




rank_prefix_matrix
告诉 combine 回程时 rank 维度上的组织方式




channel_prefix_matrix
告诉 combine 各 channel 的切分方式




send_head
是 dispatch 时顺手构造出的发送队列元数据




所以可以把 handle 理解成一句话：

dispatch 阶段生成的“回程地图”。
7. 从测试代码看真实使用方式

最值得读的测试是：

https://github.com/deepseek-ai/DeepEP/blob/d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb/tests/legacy/test_intranode.py

里面有三段特别适合对照理解。

第一次 dispatch
dispatch_args = {
    'x': current_x,
    'num_tokens_per_rank': num_tokens_per_rank,
    'is_token_in_rank': is_token_in_rank,
    'num_tokens_per_expert': num_tokens_per_expert,
    'config': config,
    'async_finish': async_mode
}
recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle, event = buffer.dispatch(**dispatch_args)

这就是 non-cached mode。

复用 handle 的 cached dispatch
dispatch_args = {'x': current_x, 'handle': handle, 'config': config, 'async_finish': async_mode}
recv_x, _, _, _, _, event = buffer.dispatch(**dispatch_args)

这说明 handle 可以省掉部分布局重算。

combine
combine_args = {'x': recv_x, 'handle': handle, 'config': config, 'async_finish': async_mode}
combined_x, combined_topk_weights, event = buffer.combine(**combine_args)

这说明 combine 的主要依赖就是：

expert 输出 x
dispatch 留下来的 handle
8. internode high-throughput 和 intranode 的关系

Python 层的分流在：

https://github.com/deepseek-ai/DeepEP/blob/d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb/deep_ep/buffers/legacy.py#L377
if self.runtime.get_num_rdma_ranks() > 1:
    return self.internode_dispatch(...)

native 对应实现：

dispatch: https://github.com/deepseek-ai/DeepEP/blob/d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb/csrc/legacy/buffer.hpp#L875
combine: https://github.com/deepseek-ai/DeepEP/blob/d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb/csrc/legacy/buffer.hpp#L1248

它和 intranode 的大框架其实一样：

校验输入
非 cached 模式先发送元信息
CPU 等待接收规模元信息 ready
分配接收 tensor
真正执行数据搬运

差异是 internode 多了一层 RDMA 复杂度：

多了 num_tokens_per_rdma_rank
多了 RDMA 级别的 prefix 和 source metadata
handle 也更复杂

所以更准确地说：

internode 不是另一套编程模型，而是在同一个 high-throughput 框架上，多了一层 RDMA 元信息和 buffer 管理复杂度。
重要发现
intranode 和 internode 的主干并没有变：

先布局
先拿接收规模元信息
再分配紧凑输出
最后真实搬运数据

internode 只是比 intranode 多了一层 RDMA 维度的元数据组织。

9. 现在回头看，V1 high-throughput 最核心的设计点是什么
9.1 先算布局，再搬数据

get_dispatch_layout 不是可有可无的预处理，而是 dispatch 的路由规划阶段。

9.2 dispatch 返回 handle

handle 同时承担两件事：

复用 dispatch 元数据
保存 combine 回程所需的元数据
9.3 normal mode 用的是动态精确接收

好处：

输出 tensor 紧凑
不按最坏情况浪费空间

代价：

CPU 要等待接收规模元信息 ready
9.4 同一个 API，底层自动分成 intranode 和 internode

对用户来说，接口是统一的：

dispatch
combine

对底层来说，会根据是否存在 RDMA rank 自动选择：

intranode 实现
internode 实现
10. 一张图收尾
用户代码
  |
  | 1. Buffer(group, num_nvl_bytes, num_rdma_bytes)
  v
Python Buffer wrapper
  |
  | 2. get_dispatch_layout(topk_idx, num_experts)
  v
native layout::get_dispatch_layout
  |
  | 3. dispatch(x, layout...)
  |    - intranode_dispatch 或 internode_dispatch
  |    - 先通知接收规模元信息
  |    - CPU 等待 recv count
  |    - 分配 recv_x
  |    - 真正发 token
  v
recv_x + handle
  |
  | 4. 本地 expert 计算
  v
expert_out
  |
  | 5. combine(expert_out, handle)
  |    - intranode_combine 或 internode_combine
  |    - 使用 handle 中的回程元数据
  v
combined_x

如果你准备继续往下读源码，建议顺序是：

docs/legacy.md
deep_ep/buffers/legacy.py
tests/legacy/test_intranode.py
csrc/legacy/buffer.hpp
再往下进入 csrc/kernels/legacy/*

这样最不容易一开始就淹没在 kernel 细节里。
