# 如何用 Triton-TLE 写出比 SGLang 更快的 MoE Align Block Size

**作者**: SunnyCase​DL Compiler

**原文链接**: https://zhuanlan.zhihu.com/p/2014316594554226118

---

0. TL;DR

MoE Align Block Size 的核心任务：将 token 按 expert 分桶，并将每个 expert 的 token 段长度补齐（pad）到 block_size 的整数倍，同时输出：

sorted_token_ids：按 expert 分组后的 token 索引（padding 部分用 sentinel 填充）
expert_ids：以 block 为单位标记每个 block 所属的 expert
num_tokens_post_pad：padding 后的总 token 数

SGLang 的 CUDA 实现：在常规路径中将“计数 + scan + 填 expert_ids”放在一个 threadblock 中完成（另一个 block 仅用于填充），然后再启动第二个 kernel 进行 scatter。对于大量 token，单个 block 会成为性能瓶颈。

FlagGems 的 Triton 版本：采用典型的多阶段流程：histogram -> prefix-scan -> align+cumsum -> scatter。在有 TLE 支持时，可将 histogram 放到 shared memory 中执行，减少全局内存的原子操作冲突。

FlagTree 的 TLE 分布式 + cooperative grid：可将 Triton atomic 的多阶段融合成单 kernel（本文称为 triton_atomic_fused / atomic_fused）：通过 distributed_barrier 实现 grid 同步，并将中间大缓冲（如 tokens_cnts）从全局内存移走或压缩，进一步提升性能。

基于相同 TLE 原语的 cluster/DSMEM 版本（tle_cluster_fused）：将关键中间状态放在 cluster 内部，借助 remote 访问和 cluster 域同步，在单个 kernel 内完成统计、对齐与 scatter。该版本在小规模输入上通常更具优势。

性能实测结果：在 Qwen3 Next 的 real dump（num_tokens=163840, num_experts=512）上，atomic_fused 相比 sglang_cuda 提速约 4.06x（RTX 5060 Ti）与 3.94x（H800）；在小 token（num_tokens=256）场景，tle_cluster_fused 相比 sglang_cuda 提速约 1.42x（RTX 5060 Ti）与 1.20x（H800）。

1. 背景

MoE Align Block Size 主要完成两件事：根据 topk_ids 将 token 分组，并将每个 expert 对应的 token 段对齐到 block_size 的整数倍，方便后续的 GEMM/GroupGEMM 处理。

该算子的开销容易被放大：当 num_tokens 和 num_experts 同时增大、且 expert 分布呈长尾时，计数、对齐前缀和、scatter、padding 中任何一个环节出现并行度不足或内存访问不理想，都可能成为主要瓶颈。

2. SGLang CUDA

从 CUDA 生态的两条代表性实现路线入手：先看 SGLang 的主线 kernel 结构与其单 block 瓶颈，再看 cooperative grid 如何把多阶段串进一次 launch。

参考代码（SGLang kernel 源码）：
https://raw.githubusercontent.com/sgl-project/sglang/refs/heads/main/sgl-kernel/csrc/moe/moe_align_kernel.cu

优化过程解析（BBuf）：
sgl-kernel MoE Align Block Size Kernel 优化过程解析

2.1 结构

第一次 kernel 启动：moe_align_block_size_kernel<<<2, 1024, shared_mem>>>

blockIdx.x==0 负责计数、对齐前缀和、填写 expert_ids，并写入 cumsum_buffer
blockIdx.x==1 负责填充 sorted_token_ids 中的 sentinel（可选，但通常需要确定性输出）

第二次 kernel 启动：count_and_sort_expert_tokens_kernel<<<many blocks>>>

rank_post_pad = atomicAdd(&cumsum_buffer[topk_id+1], 1)
sorted_token_ids[rank_post_pad] = token_id

关键约定：

dummy slot 0：scatter 中使用 topk_id+1，因此 host 侧的 num_experts 与 cumsum_buffer 需要预留 dummy slot；padding 方式也会影响输出的确定性和校验口径。
2.2 核心瓶颈：前缀和对齐阶段受限于单 block

虽然 scatter kernel 可以扩展到多个 block，但计数、对齐前缀和、填写 expert_ids 这些步骤仍集中在单个 block 内，且需要遍历所有 token。随着 num_tokens 增大，这部分工作很容易成为主要耗时。

2.3 关键点总结

BBuf 的解析文章可以归纳为三条优化思路：

向量化 padding：将 sorted_token_ids 的 sentinel 填充整合到 kernel 内部，并尽量采用合并/向量化写入，避免额外的 fill launch 和低效带宽占用。
并行化前缀和：将 expert 维度的对齐前缀和从串行扫描改为并行 scan（例如 Blelloch scan），让更多线程参与计算。
减少 __syncthreads()：通过 warp-scan 等手段减少 block 级同步点，降低同步开销。

此外，针对小规模输入还有一种融合路径：将计数、scan、scatter 全部放入一个 block 内完成，利用局部计数降低全局原子操作的竞争。

3. yiakwy CUDA

SGLang CUDA 暴露的一个关键问题是：前缀与对齐阶段被单个 block 限制并行度时，大 token 很快会撞上瓶颈。另一条常见路线是 cooperative launch：把多阶段串在一次 launch 里，用 grid.sync() 分段推进。

实现要点：cooperative launch + grid.sync()，单次 launch 串多阶段。

参考文章（yiakwy-xpu-team）：
https://huggingface.tw/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang

参考实现（CUDA 源码）：
https://raw.githubusercontent.com/yiakwy-xpu-ml-framework-team/AMD-sglang-benchmark-fork/d9831e330bd312fc00557187834d0f5b12ea5c70/sgl-kernel/src/sgl-kernel/csrc/moe_align_kernel.cu

3.1 核心流程
每个 block 执行 shared histogram：按 token 切分，每个 block 在 shared memory 中统计各自负责的 token，得到 per-expert 计数。
物化 per-block counts 并归约：将每个 block 的计数写入 tokens_cnts_buffer，然后进行跨 block 归约，得到每个 expert 的全局计数。
对齐与全局前缀和：对全局计数按 block_size 对齐并执行 scan，得到每个 expert 的起始偏移（写入 cumsum_buffer），同时填写 expert_ids。
scatter 写 sorted_token_ids：再次遍历 tokens，使用 atomicAdd(&cumsum[expert], 1) 获取写入位置，并写出 token id（此阶段 cumsum 同时作为计数器使用）。
3.2 约束与 benchmark 口径
代码中硬编码了 MAX_NUM_EXPERTS 256，并假设 num_experts <= 256（许多 shared memory 布局和并行分工依赖此限制）。
输入 topk_ids 在实现中被视为二维张量 [num_tokens, K]（K 为 top-k 路由数）；对于 Top-1 场景，可视为 K=1。
cooperative kernel 对 grid 规模有硬上限：当 num_blocks 过大时，启动时会报 too many blocks in cooperative launch（不同 GPU 的上限不同）。
本文的 benchmark 仅在 num_experts <= 256 时尝试运行该实现；若 cooperative 启动失败，则记为 na。
4. FlagGems

FlagGems 是一个面向训练与推理场景的算子与 kernel 集合，常见实现路径是 “PyTorch wrapper + Triton kernel”。本文选取其中的 moe_align_block_size 作为 baseline：结构清晰、stage 边界明确，便于后续把瓶颈逐个收敛掉。

参考源码链接（FlagGems）：
https://raw.githubusercontent.com/flagos-ai/FlagGems/refs/heads/master/src/flag_gems/fused/moe_align_block_size.py

4.1 Triton 四阶段流程

FlagGems 采用典型的 4 阶段拆分：

Stage1：Histogram（计数）
每个 program 统计一段 token，将结果写入 tokens_cnts_ptr[(pid+1), :]

2. Stage2：对 tokens_cnts 按列扫描

对每个 expert 做列方向的 cumsum，得到每个 program 的 prefix_before

3. Stage3：对齐与 expert 维度 cumsum

从最后一行获取总计数，进行对齐和 expert 维度的 cumsum，得到起始偏移量

4. Stage4：scatter

填写 expert_ids，并以 tokens_cnts 作为计数器，scatter 写入 sorted_token_ids




4.2 中间状态的开销

tokens_cnts 是一个形状为 (num_experts+1, num_experts) 的全局矩阵。以 num_experts=512 为例，其大小约为 513*512*4B ≈ 1.0 MiB，且在 Stage1/2/4 中被多次读写；加上多次 kernel 启动和输出初始化所需的带宽，整体开销不容忽视。

4.3 主要瓶颈

瓶颈通常集中在两个方面：

在没有 TLE 的情况下，Stage1 的全局内存原子操作冲突明显。
即使引入了 shared histogram，tokens_cnts 的带宽消耗以及多次 kernel launch 仍是主要成本。
5. Triton Atomic

FlagGems baseline 暴露了一个直接的问题：tokens_cnts 这类中间态既占带宽又引入额外 launch。这里先用 atomic 返回值把列 scan 折叠掉，作为融合前的过渡版本。

Triton Atomic 的核心思路是用 atomic 返回值构造 prefix_before，省掉按列 scan。

5.1 融合 Stage2

对每个 program 的 local_counts[e]，执行：

prefix_before[e] = atomicAdd(global_cumsum[e], local_counts[e])

即可得到该 program 在 expert e 下的起始偏移，这与原 Stage2 的列扫描结果等价。代价是需要进行 num_programs * num_experts 次原子写操作，并且后续仍需一次同步来衔接“rank0 对齐+scan”与 scatter 阶段。

6. TLE

atomic 化虽然减少了一个扫描阶段，但 stage 之间的边界依然存在。要想将这些边界整合到同一次 kernel 启动中，并将关键中间状态尽量放入 shared memory，就需要一个能够表达“域内同步 + shared 指针视图”的扩展层。

TLE（Triton Language Extensions）是 FlagTree 在 Triton 语言层之上的一组扩展，用于处理传统 Triton 难以覆盖的共享内存与分布式同步语义。它提供了“显式 shared 指针视图”和“可组合的同步域”两类能力，从而可以将多个 stage 合并到一次 launch 中，并将中间状态从全局内存迁移到更贴近计算的位置。

本文仅用到 TLE 的两类能力：

shared-memory 编程：tle.alloc + tle.local_ptr
域内同步与分布式视图：
tled.device_mesh + tled.distributed_barrier（mesh 覆盖 blocks 时对应 grid sync；覆盖 cluster shards 时对应 cluster sync）
tled.shard_id（在 cluster mesh 内获取 shard/rank）
tled.remote（构造 remote DSMEM/shared 的视图，用于 cluster 内跨 shard 读写）




TLE 目前位于 FlagTree 的 triton_v3.5.x 分支：
https://github.com/flagos-ai/FlagTree/tree/triton_v3.5.x

7. Atomic Fused

有了这些原语，就可以把 Triton atomic 的多 stage 收敛到单 kernel：用 cooperative grid 覆盖足够多的 blocks，用 distributed_barrier 在同一次 launch 内串阶段。

实现见 tutorials/tle/02-moe_align_block_size.py 的 moe_align_block_size_triton_atomic_fused_coop。

7.1 流程
Stage0：初始化输出，将 cumsum_ptr 清零，执行 distributed_barrier。
Stage1：每个 program 在 shared memory 中做 histogram，然后通过 prefix_before = atomic_add(cumsum_ptr, local_counts) 获得前缀偏移，并将 prefix_before 写回 shared 计数器，再次执行 distributed_barrier。
Stage2：仅 pid==0 的 program 负责对齐总计数并做 exclusive-scan，得到 expert_starts，写入 num_tokens_post_pad，然后执行 distributed_barrier。
Stage3：填写 expert_ids。
Stage4：第二次扫描 token，利用 shared 计数器（初始值为 prefix_before）得到 rank_in_prog，并写入 sorted_token_ids[expert_starts + rank_in_prog]。

TLE 的关键代码集中在两处：grid 同步与 shared/local_ptr 的原子操作。

mesh = tled.device_mesh({"block": [("block_x", NUM_BLOCKS)]})
if tl.program_id(0) == 0:
    tl.store(cumsum_ptr + tl.arange(0, BLOCK_EXPERT), 0)
tled.distributed_barrier(mesh)
local_counts = tle.alloc([BLOCK_EXPERT], dtype=tl.int32, scope=tle.smem)
e = tl.arange(0, BLOCK_EXPERT)
ptrs = tle.local_ptr(local_counts, (e,))
tl.store(ptrs, 0)
local_counts_vals = tl.load(ptrs)
prefix_before = tl.atomic_add(cumsum_ptr + e, local_counts_vals)
7.2 关键收益
移除了 tokens_cnts 矩阵，节省了其带宽与多次读写。
利用 atomic 前缀构造 prefix_before，省去了列扫描阶段。
在单次 launch 内通过 distributed_barrier 串接多个阶段。
7.3 与 yiakwy 的对比
两者都依赖 cooperative launch，在同一 kernel 内通过多次同步串接阶段。
yiakwy 的实现物化了 tokens_cnts_buffer 并做跨 block 归约；scatter 阶段以全局内存计数器为主。
atomic_fused 只保留全局 cumsum_ptr（既作计数又作起始偏移）和 shared 计数器，避免了 tokens_cnts 的物化；同步由 mesh+barrier 表达。
7.4 参数选择
NUM_BLOCKS <= ceil_div(num_tokens, BLOCK_TOKENS)，并以 SM_count * cap_mult 作为上限。
若 cooperative 启动失败，则回退减半，直至可启动。
8. Cluster Fused

atomic_fused 将并行度交由 cooperative grid 管理，更适合大 token 的吞吐场景。TLE 还提供了 cluster/DSMEM 路线：将执行域缩小到 SM90 的 cluster，把关键中间状态保留在 DSMEM 中，通过 remote 访问和 cluster 域同步完成聚合与 scatter。

moe_align_block_size_tle_cluster_fused 正是利用 SM90 cluster/DSMEM，将关键中间状态留在 cluster 内部，借助 remote 访问与 cluster 域同步，在单个 kernel 内完成统计、对齐与 scatter。

8.1 核心流程
每个 shard 在 shared memory 中做 histogram，然后将计数通过向量化的 atomic_add 累加到 rank0 的 DSMEM cumsum 中，得到本 shard 的 prefix_before。
rank0 对齐总计数并进行 scan，得到 expert_start_offsets 与 num_tokens_post_pad。
各 shard 第二次扫描 token，执行 scatter 并填写 expert_ids。
8.2 TLE 原语

该路线依赖三个关键点：cluster 维度的 mesh、cluster 域同步、以及 DSMEM remote 访问。代码形态大致如下：

mesh = tled.device_mesh({"block_cluster": [("cluster_x", 8)]})
cluster_rank = tled.shard_id(mesh, "cluster_x")
tled.distributed_barrier(mesh)

# 访问 rank0 的 DSMEM 视图（remote shared）
rank0_view = tled.remote(cumsum_local, 0, scope=mesh)
rank0_ptrs = tle.local_ptr(rank0_view, (expert_offsets,))
prefix_before = tl.atomic_add(rank0_ptrs, local_counts_vals)
8.3 约束与取舍
硬件/后端：需要 SM90 及以上架构，且后端支持 cluster remote lowering。
规模：当前实现要求 num_experts <= 1024（受 shared memory 向量长度和 alloc 限制）。
并行度：cluster 内的 shard 数量固定，整体并行度上限明显低于 cooperative grid 路线；因此该版本在小规模输入上更容易占优，但在大 token 场景下容易因并行度受限而放大瓶颈。
8.4 与 Atomic Fused 的对比
atomic_fused 依赖 cooperative grid，在 block 域内同步；tle_cluster_fused 以 cluster 为执行域，在 DSMEM 上聚合与同步。
两者都使用 atomic 返回值构造 prefix_before，主要差异在于同步域和并行度上限。
9. Benchmark

为了让数据可复现且可解释，这里先定义 benchmark 口径与数据集。

9.1 方法

将所有实现统一到同一测试脚本中，使用 triton.testing.do_bench 测量 p50 耗时，确保输入和输出语义一致。

对比对象：

triton：baseline
triton_atomic：atomic 前缀化版本
triton_atomic_fused：TLE cooperative 单 kernel（ours）
tle_cluster_fused：TLE cluster/DSMEM 单 kernel（ours）
sglang_cuda：SGLang CUDA 实现
yiakwy_cuda：yiakwy cooperative CUDA 实现（仅在 num_experts<=256 时运行）
9.2 数据集
synthetic：按 Zipf 分布采样 expert（更贴近真实 MoE 的长尾分布）
real：来自 Qwen3 Next 一次实际推理路由的 topk_ids dump（Top-1，num_tokens=163840）
10. 结果

首先给出结果表格与图示，然后回到参数空间，解释 NUM_BLOCKS 与 num_experts 的关系，以及它如何影响大 token 场景下的性能表现。

说明：以下数据为 p50 单次耗时（毫秒），数值越小越好。
RTX 5060 Ti 数据来自本地测试（Torch 2.8.0 + CUDA 12.8）。H800 数据来自同口径的日志输出。

10.1 RTX 5060 Ti
num_tokens	triton	triton_atomic	tle_atomic_fused [ours]	tle_cluster_fused [ours]	sglang_cuda
256	0.0348	0.0302	0.0323	0.0097	0.0138
512	0.0369	0.0301	0.0240	0.0117	0.0138
1024	0.0369	0.0313	0.0179	0.0117	0.0139
2048	0.0368	0.0313	0.0158	0.0131	0.0138
4096	0.0369	0.0301	0.0138	0.0143	0.0148
8192	0.0369	0.0313	0.0138	0.0164	0.0179
16384	0.0369	0.0301	0.0158	0.0205	0.0240
32768	0.0389	0.0322	0.0179	0.0301	0.0312
65536	0.0430	0.0374	0.0225	0.0486	0.0507
163840	0.0609	0.0512	0.0384	0.1036	0.1001
小 token：tle_cluster_fused 更占优。
大 token：triton_atomic_fused 更稳定；tle_cluster_fused 更容易受并行度上限影响。
10.2 RTX 5060 Ti Real

num_tokens=163840, num_experts=512, block_size=16

provider	ms
triton	0.0512
triton_atomic	0.0384
triton_atomic_fused [ours]	0.0261
tle_cluster_fused [ours]	0.0537
sglang_cuda	0.1060

Speedup：sglang_cuda / triton_atomic_fused = 0.1060 / 0.0261 ≈ 4.06x。

10.3 RTX 5060 Ti 256 Experts

yiakwy_cuda 依赖 cooperative launch；在 RTX 5060 Ti 上，较大的 token 规模会触发启动失败（too many blocks in cooperative launch），因此显示为 na。

num_tokens	sglang_cuda	yiakwy_cuda
8192	0.0179	0.0251
16384	0.0240	0.0292
32768	0.0313	0.0415
65536	0.0509	na
163840	0.1029	na
10.4 H800
num_tokens	triton	triton_atomic	triton_atomic_fused [ours]	tle_cluster_fused [ours]	sglang_cuda
256	0.0260	0.0408	0.0445	0.0133	0.0160
512	0.0262	0.0399	0.0315	0.0140	0.0162
1024	0.0274	0.0401	0.0239	0.0158	0.0163
2048	0.0509	0.0422	0.0226	0.0169	0.0173
4096	0.0265	0.0412	0.0200	0.0177	0.0187
8192	0.0476	0.0416	0.0192	0.0211	0.0230
16384	0.0548	0.0441	0.0219	0.0256	0.0286
32768	0.0443	0.0441	0.0221	0.0358	0.0401
65536	0.0361	0.0481	0.0273	0.0561	0.0645
163840	0.0509	0.0626	0.0451	0.1177	0.1323
10.5 H800 Real

num_tokens=163840, num_experts=512, block_size=16

provider	ms
triton	0.0397
triton_atomic	0.0497
triton_atomic_fused [ours]	0.0358
tle_cluster_fused [ours]	0.0604
sglang_cuda	0.1412

大 token 下，SGLang CUDA 的单 block 前缀阶段更容易成为瓶颈；triton_atomic_fused 将计数并行化，并把 scan 压到 O(num_experts)。

Speedup：sglang_cuda / triton_atomic_fused = 0.1412 / 0.0358 ≈ 3.94x。

注：SGLang 的 dummy slot 与 padding 填充存在语义约定，复现时需要对齐 host wrapper 的参数与输出口径。

11. Blocks vs Experts

一个常见的调参问题是：NUM_BLOCKS 是否可以、且是否应该超过 num_experts？

在 triton_atomic_fused 中，NUM_BLOCKS 的硬上限为：

token_programs = ceil_div(num_tokens, BLOCK_TOKENS)
NUM_BLOCKS <= token_programs（多出的 block 没有 token 可处理）

因此只有当 token_programs > num_experts 时，尝试设置 NUM_BLOCKS > num_experts 才可能带来收益；否则上限由 token_programs 决定。

12. 总结

本文的结论可以归纳为三点：

SGLang CUDA 的实现思路清晰，但在大 token 场景下受限于单 block 阶段的并行度。
FlagGems 的 Triton 多阶段实现是一个优秀的 baseline，具备良好的可读性和可维护性。
在支持 TLE 的平台上，借助 device_mesh + distributed_barrier 可以将 Triton atomic 的多阶段融合成单 kernel（atomic_fused），在大 token、num_experts=512 的真实数据上，性能显著超越 SGLang CUDA。
