# Skill Layout

本地维护的 skill 按使用边界分组；安装时仍使用 skill 自己的稳定名称，不会把目录层级暴露给 Cursor、Claude Code、Codex、Gemini CLI 或 Kimi Code CLI。

## Kernel

写 kernel、底层 CUDA/​PTX/​Triton/CUTLASS、kernel 级 profiling、性能建模与调度：

| Skill | 用途 |
|---|---|
| `cuda-skill` | CUDA C++、PTX、CUDA API、基础 Nsight/调试 |
| `gpu-kernel-reverse` | cubin/fatbin/SASS 语义与布局恢复、CUDA/CUTLASS 重建、正确性和性能对齐 |
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
| `train-inference-consistency` | bitwise 训推一致（true on-policy RL）：vLLM batch invariance、C0–C10 gate、BI kernel、反向/STE 与量化契约；来自 vLLM 仓库 `.agents/skills` |
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
| `ponytail` / `ponytail-review` / `ponytail-audit` | 最小完整实现、改动审查、仓库审计；来自 `repos/ponytail`，另含 debt/gain/help |
| `zotero-cli` | Zotero 文献搜索、PDF 阅读、元数据与笔记管理 |
| `sepia`、`sepia-write`、`sepia-review`、`sepia-refactor`、`sepia-recreate`、`sepia-hemingway` | 去 AI 味写作（小说与专业文本）；来自 `repos/sepia`（MIT）。操作入口需与 `sepia` 总路由同级安装 |
| `wx-cli` | 本地微信数据只读查询；来自 `repos/wx-cli`（fork 的 SKILL.md）。需要 `npm install -g @jackwener/wx-cli@0.3.0`，且只在用户明确授权时使用 |
| `reverse-engineer-anything` | REA 逆向分析 skill；来自 `rea-agents@6.7.0` npm 包（`runtime-skills`）。需要 `npm install -g rea-agents@6.7.0` |
| `mattpocock` 工程 skill（`tdd`、`grill-me`、`code-review`、`diagnosing-bugs`、`to-tickets` 等） | 来自 `repos/mattpocock-skills` 1.3.1（MIT）；与 cursor-skills 同名的改写版优先 |
| `salary-negotiation-skill` | 薪资谈判与 offer 评估 |
| `tech-doc-style-chinese` | 中文技术文档与技术原理报告（算子/推理/训练框架）：结构模板、证据等级、术语与排版检查；Fenng/Tech-Doc-Style-Chinese 的本地定制版 |
| `claude-delegate` | 把任务委托给 Claude Code CLI；原样复制自 amElnagdy/delegate-skills（MIT） |
| `codex-background` | `claude-delegate` 在 Codex 中所需的后台 MCP server，只为 Codex 安装；安装后需运行 `scripts/bootstrap.mjs` 注册并重启 Codex，见[根目录 README](../README.md#claude-委托) |

## Research

论文研究两个来源放在一起，详细对照（阶段、来源、许可）见 [`research/README.md`](research/README.md)：

- `research/ccfa-skills/` → `repos/ccfa-skills`：17 个 `ccf-*` runtime skill 与 `ccf-latex-templates` 支撑目录（MIT）
- `research/supervisor-skills/` → `repos/supervisor-skills`：12 个 skill，固定到 `a20d76e`（根许可 CC BY-NC-SA 4.0，逐 skill 许可不同）

这两个目录只是导航链接，安装时 skill 仍按各自的名称安装，不复制到本仓库。

## Routers

跨领域入口单独放在 `skills/routers/`：

`gpu-development-catchall` → `gpu-kernel-authoring-router` /
`gpu-performance-router` / `gpu-framework-router` /
`llm-serving-router` → 具体 skill。

当用户没有点名 skill 时，GPU 问题先命中 `gpu-development-catchall`；当问题同时涉及
框架和 kernel 时，先用 `gpu-framework-router` 判断边界，再把实测 hot kernel
交给 `gpu-performance-router`。TileLang 问题优先进入 TileLang 专项 skill，
再按 profiling 证据转入 `ncu-report-skill`、`nsys-profile-analysis` 或
`gpu-performance-router`。

跨域或非 GPU 的任务（论文阅读与审稿、Zotero、中文技术写作、通用代码审查、TDD、架构、
微信聊天记录等）使用 `agent-skill-router`。它只在本地运行 `scripts/route_skill.py`，
先看显式点名与制品后缀，再选一个主 skill；目录与可用性由
`references/inventory.tsv` 给出，需要时重新生成：

```bash
python3 skills/routers/agent-skill-router/scripts/build_inventory.py
```

`agent-skill-router` 的 `kind` 字段区分三类：`instruction`（只读 SKILL.md）、
`runtime-cli`（需要宿主安装命令，部分需要本机授权）、`user-level`（用户级 symlink，
不由本仓库管理）。

外部来源（NVIDIA、AMD、Cursor、CCFA、Supervisor-Skills、TileLang、Ponytail、mattpocock、sepia、wx-cli、rea）
的清单与获取方式见 [`sources/external-sources.conf`](../sources/external-sources.conf) 和
[`INSTALL.md`](../INSTALL.md#外部来源)；审查与覆盖情况见
[`docs/skills-usage-review.md`](../docs/skills-usage-review.md)。
