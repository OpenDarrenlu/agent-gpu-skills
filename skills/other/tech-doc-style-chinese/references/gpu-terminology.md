# GPU、推理与训练领域术语表

仅在技术报告涉及 GPU kernel、推理框架、训练框架或分布式通信术语时读取本文件。本表是默认约定；目标项目已有术语表时，以目标项目为准。规则只作用于可见正文，不修改代码、命令、metric 名、配置项和官方字面量。

## 使用原则

- 同一概念在同一文档中只用一个首选写法。首次出现可以给出中英对照或全称，例如「共享内存（shared memory，下文简称 SMEM）」，之后统一使用简称。
- 领域内通用的英文术语保留英文，不硬译。硬译词读者需要反向翻译才能理解，例如把 warp 写成「线程束」、把 Tensor Core 写成「张量核心」。
- 有成熟中文译名、且中文读者更熟悉的通用概念使用中文，例如寄存器、流水线、访存、带宽、归约。
- 不造中英混搭的新词，例如「warp 束」「tile 块」。
- 英文术语作为普通名词时使用小写，例如 kernel、warp、tile；专有名词和缩写按官方写法。
- 指令、API、类名、函数名、配置项写在行内代码中并保持原样，例如 `tcgen05.mma`、`cp.async.bulk.tensor`、`--mem-fraction-static`。

## 保留英文的术语

| 术语 | 首次出现建议写法 | 说明 |
| --- | --- | --- |
| kernel | kernel | 一次 launch 在 GPU 上执行的函数；不写「内核」「核函数」 |
| warp | warp | 32 个线程组成的调度单位 |
| warp group | warp group（4 个连续 warp，共 128 线程，起始 warp 编号为 4 的倍数） | Hopper WGMMA 的发射单位 |
| CTA | CTA（cooperative thread array，即 thread block） | 同一文档不与「线程块」「block」混用 |
| cluster | thread block cluster（下文简称 cluster） | Hopper 起支持 |
| tile | tile | 名词保留英文；动作「分块」可以用中文 |
| SM | SM（streaming multiprocessor） | 不写「流多处理器」 |
| Tensor Core | Tensor Core | 不写「张量核心」 |
| TMA | TMA（Tensor Memory Accelerator） | 指令写 `cp.async.bulk.tensor` |
| TMEM | TMEM（Tensor Memory） | 数据中心 Blackwell（sm_100 系列）`tcgen05` MMA 使用的片上存储，存放累加器等操作数 |
| mbarrier | mbarrier | 基于共享内存的异步屏障 |
| CLC | CLC（Cluster Launch Control） | Blackwell 动态调度机制 |
| prologue / mainloop / epilogue | mainloop、epilogue | 不写「主循环」「尾处理」以免与普通循环混淆 |
| swizzle | swizzle | 共享内存地址重排，不写「交织」 |
| layout | layout | CuTe 语境下指 shape 与 stride 的映射 |
| occupancy | occupancy | 不写「占用率」，以免与利用率混淆 |
| bank conflict | bank conflict | 共享内存 bank 冲突 |
| stall | stall，具体类型保留原名 | 例如 long scoreboard、barrier |
| persistent kernel | persistent kernel | 不写「常驻 kernel」 |
| warp specialization | warp specialization | 角色写作 producer/consumer 或「生产者/消费者」，同一文档统一 |
| split-K / split-KV / Stream-K | 原样 | 切分策略名称 |
| wave quantization | wave quantization | CTA 总数不是「SM 数 × 每个 SM 可驻留的 CTA 数」的整数倍，导致最后一个 wave 不满；不与尾效混用 |
| roofline | roofline 模型 | 不翻译 |
| prefill / decode | prefill、decode | 不写「预填充」「解码阶段」 |
| KV cache | KV cache | 大小写见下文 |
| continuous batching / chunked prefill | 原样 | 调度机制名称 |
| speculative decoding | speculative decoding | 同一文档可统一用「投机解码」，二选一 |
| PD 分离 | PD 分离（prefill-decode disaggregation） | 中文社区通用说法 |
| rollout | rollout | RL 中的采样生成阶段 |
| log-prob | log-prob | 指 token 级对数概率 |
| MoE / expert / router | MoE、expert、router | 门控可写「路由」 |
| dispatch / combine | dispatch、combine | EP 通信的两个阶段 |
| AllReduce 等集合通信 | AllReduce、AllGather、ReduceScatter、AlltoAll | 按 NCCL 写法 |

## 使用中文的通用概念

| 中文 | 英文 | 说明 |
| --- | --- | --- |
| 寄存器 | register | 寄存器溢出写作「寄存器溢出（spill）」 |
| 共享内存 | shared memory | 首次出现后可简称 SMEM |
| 全局内存 | global memory | 与显存指同一物理存储时，同一文档选一种 |
| 显存 | device memory | 指 GPU 侧存储；主机侧写「主机内存」 |
| 流水线 | pipeline | 不写「管线」 |
| 空泡 | bubble | 同一文档不与「气泡」混用 |
| 尾效 | tail effect | kernel 末尾部分 SM 已空闲、其他 SM 仍在工作；来源包括 wave quantization 和负载不均，报告中分开说明 |
| 访存 | memory access | 访存合并写作「访存合并（coalescing）」 |
| 带宽 | bandwidth | 存储或互联的数据传输速率 |
| 吞吐 | throughput | 单位时间完成的工作量 |
| 延迟 | latency | 不与「时延」混用 |
| 归约 | reduction | 不写「规约」 |
| 累加 | accumulation | 累加精度、累加顺序 |
| 算术强度 | arithmetic intensity | 不与「计算强度」「计算访存比」混用 |
| 算子融合 | operator fusion | 融合后的实现称为 fused kernel |
| 量化 / 反量化 | quantization / dequantization | 不写「解量化」 |
| 缩放因子 | scale factor | 粒度写 per-tensor、per-channel、per-token、block-wise |
| 负载均衡 | load balancing | |
| 数据并行等 | DP、TP、PP、EP、CP、SP | 首次出现写全称，见下文 |

## 并行策略与指标缩写

首次出现写「中文全称（缩写）」，之后使用缩写：

- 数据并行（DP）、张量并行（TP）、流水线并行（PP）、专家并行（EP）、上下文并行（CP）、序列并行（SP）
- ZeRO、FSDP、FSDP2 保留原名
- 首 token 延迟（TTFT）、每个输出 token 的耗时（TPOT）、token 间延迟（ITL）
- 模型算力利用率（MFU），写明计算量公式和峰值口径

## 易混淆概念

| 概念 | 区别 | 写法要求 |
| --- | --- | --- |
| 算子与 kernel | 算子是框架层的 op，可能由一个或多个 kernel 实现；kernel 是一次 launch 执行的函数 | 讨论 launch、资源和 profile 时用 kernel |
| occupancy 与 SM 利用率 | occupancy 是 active warp 占最大 warp 数的比例；SM 利用率反映 SM 在时间上的忙碌程度 | 分开报告，不用一个推出另一个 |
| SM 利用率与 Tensor Core 利用率 | SM 忙碌不代表 Tensor Core 忙碌 | GEMM、attention 报告两者都给 |
| 带宽与吞吐 | 带宽描述数据传输速率；吞吐描述完成的工作量 | 「达成带宽 2.9 TB/s」「吞吐 1.2 万 tokens/s」 |
| 延迟与耗时 | 延迟指从发起到完成的时间；耗时指 kernel 或阶段的执行时间 | kernel 计时写「耗时」 |
| 并行与并发 | 并行指同时执行；并发指多个任务在同一时间段内推进 | 多 stream 重叠写「并发」 |
| 确定性 | 同一系统、同一输入多次运行结果逐位相同（run-to-run） | 不等同于训推一致 |
| bitwise 一致 | 两个不同系统（如训练引擎与推理引擎）对同一输入的输出逐位相同 | 写明比较的是哪两个量 |
| batch invariance | 单个样本的结果不随 batch 大小或同 batch 其他样本变化 | 训推一致的常见前提 |
| 精度、准确度与误差 | 精度指数据类型或位宽；准确度指模型指标；误差指与参考结果的差 | 「BF16 精度下最大绝对误差为……」 |
| 标称峰值与可达峰值 | 前者来自官方资料，后者来自 microbenchmark | 利用率写明分母 |
| FLOP、FLOPs 与 FLOPS | 计算量与每秒计算量 | 计算量写 FLOP，速率写 FLOP/s 或 TFLOPS |
| 显存与 HBM | 显存是功能概念；HBM 是存储技术 | 容量和带宽按语境选择 |
| step、iteration 与 epoch | 框架之间定义不同 | 首次出现说明本文口径 |
| global batch、mini batch 与 micro batch | RL 框架中还有 rollout batch、PPO mini batch | 写明单位是样本、序列还是 token |

## 官方写法与大小写

以下写法用于可见正文。代码、命令、包名和路径保持原样，例如 `import triton`、`pip install sglang`。

- 工具链：CUDA、PTX、SASS、cuBLAS、cuBLASLt、cuDNN、Nsight Compute、Nsight Systems（命令写 `ncu`、`nsys`）、`nvcc`、`ptxas`
- kernel 框架与库：CUTLASS、CuTe、CuTe DSL、Triton、Gluon、TileLang、FlashAttention（简称 FA2、FA3 首次出现给出全称）、FlashInfer、FlashMLA、DeepGEMM、DeepEP
- 框架：PyTorch、`torch.compile`、SGLang、vLLM、TensorRT-LLM、Megatron-LM、Megatron-Core、DeepSpeed、verl、Ray
- 通信与互联：NCCL、NVSHMEM、NVLink、NVSwitch、InfiniBand、RoCE、GPUDirect RDMA
- 硬件与架构：Ampere、Hopper、Blackwell、A100、H100、H200、B200、GB200；架构代号写在代码中，例如 `sm_90a`、`sm_100a`
- AMD：ROCm、HIP、MI300X
- 数据类型：FP32、TF32、BF16、FP16、FP8（E4M3、E5M2）、FP4、NVFP4、MXFP8、MXFP4、INT8、INT4、UE8M0
- 模型组件：GEMM、GEMV、MoE、MLA、MHA、GQA、MQA、RoPE、RMSNorm、LayerNorm、SwiGLU、KV cache

常见不推荐写法：

| 不推荐 | 推荐 |
| --- | --- |
| `cuda`、`Cuda` | `CUDA` |
| `cutlass`、`Cutlass` | `CUTLASS` |
| `pytorch`、`Pytorch` | `PyTorch` |
| `sglang`、`Sglang` | `SGLang` |
| `vllm`、`VLLM` | `vLLM` |
| `Flash Attention`、`flashattention` | `FlashAttention` |
| `deepgemm`、`DeepGemm` | `DeepGEMM` |
| `nvlink`、`NVlink` | `NVLink` |
| `kv cache`、`KV Cache`、`kvcache` | `KV cache` |
| `tensor core`、`TensorCore` | `Tensor Core` |
| `fp8`、`bf16`、`Fp16` | `FP8`、`BF16`、`FP16` |
| `moe`、`MOE` | `MoE` |
| `TFlops`、`Tflops` | `TFLOPS` |

## 领域错词与语境词

可以作为确定错误处理：

- `瓶劲` -> `瓶颈`
- `阻赛` -> `阻塞`
- `兼融` -> `兼容`
- `冗于` -> `冗余`
- `操做` -> `操作`

必须结合语境判断：

- `规约`：reduction 语境写 `归约`；`规约` 指规范、协议。
- `通讯`：技术文档通常写 `通信`，例如「集合通信」。
- `内核`：GPU 语境写 `kernel`；`内核` 容易与操作系统内核混淆。
- `内存泄露`、`显存泄露`：通常写 `泄漏`；`泄露` 用于信息泄露。
- `重迭`：通常写 `重叠`。
- `线程束`、`张量核心`、`线程块`：本表约定使用 warp、Tensor Core、CTA。
- `时延`、`气泡`、`计算访存比`、`解量化`：与首选词「延迟」「空泡」「算术强度」「反量化」二选一，同一文档统一。
