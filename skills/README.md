# Skill Layout

本地维护的 skill 按使用边界分组；安装时仍使用 skill 自己的稳定名称，不会把目录层级暴露给 Cursor、Claude Code、Codex、Gemini CLI 或 Kimi Code CLI。

## Kernel

写 kernel、底层 CUDA/​PTX/​Triton/CUTLASS、kernel 级 profiling、性能建模与调度：

| Skill | 用途 |
|---|---|
| `cuda-skill` | CUDA C++、PTX、CUDA API、基础 Nsight/调试 |
| `triton-skill` | Triton/Gluon kernel 与编译器 |
| `cutlass-skill` | CUTLASS、CuTe、CuTeDSL、GEMM |
| `deepgemm-skill` | DeepGEMM、grouped/MoE GEMM |
| `ncu-report-skill` | Nsight Compute 报告、source/SASS 与瓶颈诊断 |
| `ncu-persistent-kernel-diagnosis` | Persistent kernel 的 NCU 证据分析 |
| `persistent-kernel-scheduling` | Static/Dynamic/CLC/Stream-K 调度 |
| `persistent-kernel-utilization` | SM/TC 利用率、pipeline bubble、warp specialization |
| `nv-gpu-kernel-performance-modeling` | Roofline、流水线、延迟与跨架构建模 |
| `iket-profiling` | CuTe DSL kernel 内 per-warp 时间线 |
| `colfax-research-skill` | CUTLASS/CuTe/FlashAttention 研究资料 |
| `KernelWiki` | Blackwell/Hopper kernel 知识库与 PR 交叉引用，来自 `kernel-design-agents` |
| `tilelang-optimization-loop` | TileLang 基线、候选、验证、测量和晋级流程 |
| `writing-tilelang-kernels` | TileLang PrimFunc、JIT 和 kernel 模板 |
| `debugging-tilelang-programs` | TileLang 编译、lowering、运行时和数值问题 |
| `optimizing-tilelang-programs` | TileLang tile、layout、pipeline 和 autotuning |
| `profiling-tilelang-programs` | TileLang benchmark、IKET、NSYS/NCU profiling |
| `torch-profiling-tilelang-programs` | PyTorch/torch.compile baseline 和 allocation profiling |
| `testing-fwd-bwd-kernels` | TileLang autograd、前后向和梯度验证 |
| `writing-tilelang-design-docs` | TileLang kernel 设计文档和可视化说明 |
| `designing-with-tilefoundry` | TileFoundry HIR、拓扑、placement 和静态分析 |

## Framework

框架、运行时、端到端 serving、RL 后训练与 framework-level profiling：

| Skill | 用途 |
|---|---|
| `sglang-skill` | SGLang runtime、KV cache、attention backend、scheduler |
| `verl-skill` | verl/HybridFlow RL 后训练：PPO/GRPO、model engine、rollout 权重同步、reward、agent loop、训推一致 |
| `llm-serving-router` | LLM serving、吞吐/延迟、容量规划与部署 |
| `gpu-framework-router` | 框架问题与 kernel 问题之间的分流 |
| `nsys-profile-analysis` | VeloQ 的 whole-application timeline 分析 |
| `llm-serving-auto-benchmark` | serving workload benchmark，来自 Cursor skills |
| `perf-workload-profiling` | workload-level profiling，来自 Cursor skills |
| `llm-torch-profiler-analysis` | PyTorch profiler/runtime 分析，来自 Cursor skills |
| `perf-nsight-systems` / `perf-nsight-compute-analysis` | 框架侧 Nsight 分析，来自 Cursor skills |

后四类外部 skill 由 `repos/cursor-skills` 或 VeloQ 提供；未初始化对应来源时，
安装器会跳过它们，不影响本地 skill。

## Other

通信、AMD/ROCm、研究资料和非 GPU 专项：

| Skill | 用途 |
|---|---|
| `gpu-communication-libraries` | NCCL、NVSHMEM、RDMA、集合通信 |
| `nccl-skill` | NCCL API、源码、插件、hang 与调优 |
| `amd-gpu-docs` | AMD/ROCm/HIP 文档检索 |
| `amd-instinct-isa` | CDNA1-CDNA5 ISA 查询 |
| `amd-instinct-cdna4-isa` | CDNA4 ISA 兼容入口 |
| `giantpandallm-knowledge` | GiantPandaLLM 技术文章知识库 |
| `zhihu-knowledge-base` | 知乎技术文章本地知识库 |
| `salary-negotiation-skill` | 薪资谈判与 offer 评估 |

## Routers

跨领域入口单独放在 `skills/routers/`：

`gpu-development-catchall` → `gpu-kernel-authoring-router` /
`gpu-performance-router` / `gpu-framework-router` /
`llm-serving-router` → 具体 skill。

当用户没有点名 skill 时，先命中 `gpu-development-catchall`；当问题同时涉及
框架和 kernel 时，先用 `gpu-framework-router` 判断边界，再把实测 hot kernel
交给 `gpu-performance-router`。TileLang 问题优先进入 TileLang 专项 skill，
再按 profiling 证据转入 `ncu-report-skill`、`nsys-profile-analysis` 或
`gpu-performance-router`。
