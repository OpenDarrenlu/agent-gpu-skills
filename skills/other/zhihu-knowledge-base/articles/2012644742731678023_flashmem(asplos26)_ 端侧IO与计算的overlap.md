# flashmem(asplos26): 端侧IO与计算的overlap

**作者**: Arsmart​上海交通大学 计算机科学技术博士在读

**原文链接**: https://zhuanlan.zhihu.com/p/2012644742731678023

---

核心主线：基于 I/O 与计算重叠（I/O-Compute Overlap）的流式显存管理。通过 Kernel 间的全局约束调度与 Kernel 内的软件流水线设计，掩盖 Disk -> UM -> TM 的多级数据传输延迟。

1. 动机：粗粒度 I/O 带来的显存墙与延迟墙

在移动端 SoC 上，GPU 计算需要跨越复杂的内存层级：权重需经历 磁盘 (Disk) -> 统一内存 (UM) -> 纹理内存 (TM) -> 流多处理器 (SM) 的数据通路。

现有的推理框架（如 ExecuTorch, TVM, LiteRT）大多采用串行、粗粒度的全量预加载（Static Preloading）：在执行任何 Kernel 之前，将上述 I/O 与格式转换（UM to TM）一次性做完。这种缺乏 Overlap 的设计导致了两个致命问题：

显存峰值爆炸：无法应对现代大模型（如 2.7B 的 GPT-Neo），也无法支持多模型（Multi-DNN）并发流转。
初始化延迟极高：沉重的数据搬运完全暴露在关键路径（Critical Path）上，未能利用 GPU 算数逻辑单元（ALU）的执行时间来掩盖 I/O 延迟。

为此，FlashMem 提出了一套内存流式框架，其核心技术贡献可以完全归结为在 Kernel 间 与 Kernel 内 两个维度的精细化 Overlap 优化。

2. Kernel 间优化 (Inter-Kernel)：全局视野下的精细化重叠调度

在 Kernel 间做 Overlap，最大的挑战是“阻抗匹配”：不同的 Kernel 算术强度（Arithmetic Intensity）不同，对并发 I/O 的容忍度也不同。如果在访存密集的 Kernel 旁强行并发高带宽 I/O，会引发严重的内存总线竞争（Contention），适得其反。

FlashMem 通过以下三步实现了精准的 Kernel 间调度：

2.1 算子负载容量感知 (Load Capacity Profiling)

FlashMem 将“算子能够掩盖多少额外 I/O 且不拖慢自身计算”的能力定义为负载容量（Load Capacity, 
𝐶
𝑙
）。 系统利用 XGBoost 回归模型对算子进行离线 Profiling：

元素级算子（如 ReLU, Add）：对 I/O 容忍度极高，适合在其执行期间大批量并发搬运后续层的权重。
层级算子（如 LayerNorm）：需要频繁的同步与归约，对 I/O 争抢极度敏感，其 
𝐶
𝑙
 几乎为 0，不能用来做 Overlap。
2.2 基于 CP-SAT 的静态调度求解 (LC-OPG Solver)

在获得了所有 Kernel 的 C_l 后，FlashMem 将 I/O 调度建模为一个约束编程可满足性（CP-SAT）问题。 求解器（Solver）会在全局计算图中，决定“第 N 层的权重，应该切分成多少个 Chunk，分别安插在前面第 i, j, k 层的计算间隙中进行搬运（UM to TM）”。 这种精细的 Chunk 级切分与排布，确保了 I/O 流量平滑且严格受控于峰值内存预算（M_{peak}）。

2.3 为 Overlap 让路的自适应算子解耦 (Adaptive Operator Un-fusion)

这是一个非常反直觉但极具系统洞察的设计。传统的编译器（如 TVM）喜欢做极致的算子融合（Operator Fusion）来减少 Kernel 启动开销。但融合会导致多个执行阶段合并，极大地缩减了可用于 Overlap 的时间窗口与有效负载容量（C_{fused} \approx \min(C_1,...,C_k)）。 当 CP-SAT 求解器发现当前融合策略导致无解时，FlashMem 会主动将大型融合算子解耦（Un-fuse，例如将 MatMul+Add+GeLU 拆开），释放出高 C_l 的算子窗口，以换取更加从容的 I/O 重叠空间。

3. Kernel 内优化 (Intra-Kernel)：消除分支发散的微架构级流水线

仅仅在 Kernel 间做粗粒度的调度还不够，真正执行 I/O（从 UM 到 TM 的数据拉取）与计算时，必须在 GPU 线程级（Thread/Warp-level）实现微观的 Overlap。

3.1 Naive 调度的失败（Warp 发散问题）

如果在同一个 Kernel 内部简单地分配任务，例如 if (thread_id < compute_size) do_compute(); else do_load();，这在 CPU 上可行，但在 GPU 上会触发灾难性的 Warp 级分支发散（Branch Divergence），极大破坏 SIMT（单指令多线程）的执行效率。

3.2 软件流水线重构 (Software Pipelining / Kernel Rewriting)

FlashMem 设计了无分支的流水线内核模板（Pipelined Kernel Template）。在汇编/指令调度的逻辑上，重构了执行循环：

统一行为：让所有的线程在同一个循环内执行完全一致的分阶段动作。
Prefetch 掩盖：在每一次迭代开始时，所有线程先发出指令，预取（Prefetch）下一个数据块（Tile）的权重，然后立即对当前数据块进行乘加（MAC）计算。

这种精妙的流水线编排（类似于 CPU 的 Tomasulo 算法思想），在没有引入任何分支发散的前提下，让当前的算术执行时间完美覆盖了下一块权重的 TM 访存延迟。

4. 实验验证：精细化 Overlap 带来的降维打击

FlashMem 通过剥离 I/O 到后台，实现了对基线框架（MNN, TVM, ExecuTorch 及前沿的 SmartMem）的绝对优势：

证明精细化调度的必要性：作者对比了朴素的重叠策略（如“无脑预取下一层” Always-Next Loading）。由于 Naive 策略无视了 Kernel 间的算力与 I/O 阻抗匹配，导致 GPU 时常处于饥饿（Wait I/O）状态，执行速度比 FlashMem 慢最高 4.3 倍。
端到端加速与内存削减：得益于完美的 Overlap，FlashMem 实现了 1.7x 至 75.0x 的端到端加速（相比 SmartMem 平均 8.6x），同时内存消耗降低了 2.0x 至 8.4x。由于内存占用极低，在 1.5GB 的软性限制下，多模型并发（Multi-DNN）不会再产生剧烈的 OOM 尖峰。
总结与延伸探讨

无论是分布式系统中的通信与计算重叠（Comm-Compute Overlap），还是单机架构中的 I/O 与计算重叠（I/O-Compute Overlap），其底层的体系结构哲学是高度一致的：计算是昂贵的，等待是奢侈的。必须通过静态调度规划与底层的流水线重构，将数据移动完全隐藏在数据处理的阴影之中。

与上海交大 IPADS 实验室的 PowerInfer 相比，PowerInfer 是利用大模型的激活稀疏性（数学特性）来规避一部分算力和 I/O 需求；而 FlashMem 则是不改变任何计算量，纯粹依靠微架构感知的调度器（LC-OPG）与软件流水线（Pipelined Kernel），通过极致的时间差把 I/O 掩盖掉。这对于所有致力于系统底层 Overlap 优化的研究而言，都提供了一个极其严谨且优雅的方法论范本。
