# agent-gpu-skills

GPU 开发 Agent Skill 集合，适用于 Cursor / Claude Code / Codex / Gemini CLI。

| Skill | 层级 | 使用场景 |
|:------|:-----|:---------|
| **cuda-skill** | 底层 (PTX/CUDA C++) | 查 PTX 指令、CUDA API、Programming Guide，nsys/ncu 分析 |
| **cutlass-skill** | 中间层 (CUTLASS/CuTeDSL) | 写 CUTLASS/CuTe kernel，查 CuTeDSL 示例 |
| **triton-skill** | 高层 (Python DSL) | 写 Triton/Gluon 内核，查教程和示例 |
| **sglang-skill** | 应用层 (LLM Serving) | SGLang 推理引擎开发，KV cache、Attention backend |
| **colfax-research-skill** | 参考资料 (技术文章) | 查 Colfax Research 文章：CUTLASS/CuTe、FlashAttention-2/3/4、Hopper/Blackwell 优化 |
| **gpu-communication-libraries** | 参考资料 (通信库) | 查 NCCL/NVSHMEM/DeepEP/UCX/GIN 等 GPU 通信库特性、API、选型指南 |
| **ncu-persistent-kernel-diagnosis** | 参考资料 (性能诊断) | 用 NCU 诊断 Persistent Kernel 的 SM/TC 空泡、流水线 stall、负载不均 |
| **persistent-kernel-scheduling** | 参考资料 (调度策略) | Persistent Kernel 调度策略：Static/Dynamic/CLC/Stream-K/尾效决策树 |
| **persistent-kernel-utilization** | 参考资料 (极致优化) | Persistent Kernel 利用率优化：warp spec、multistage、TMA multicast、setmaxnreg |
| **nsys-profile-analysis** | 性能分析 (timeline) | 用 VeloQ 查 `.nsys-rep`：GPU 空闲、kernel 启动因果、CPU↔GPU 关联、NVTX 归因、并发 |
| **ncu-profile-analysis** | 性能分析 (kernel) | 用 VeloQ 查 `.ncu-rep`：occupancy、warp stall、访存吞吐、指令构成、source/SASS 关联 |

> `nsys-profile-analysis` / `ncu-profile-analysis` 由 [VeloQ](https://github.com/lucifer1004/veloq) 提供 —— 一个把 nsys/ncu profile 转成稳定 JSON 契约的 agent-friendly CLI（`veloq`）。本仓库不 vendored 其内容，安装时委托其官方安装器（细节见 [VeloQ](#veloq-profile-查询-cli) 一节）。

## 安装

```bash
git clone https://github.com/slowlyC/agent-gpu-skills.git
cd agent-gpu-skills

# 1. 获取外部源码 repo (sparse checkout, ~130MB)
bash update-repos.sh

# 2. 安装 skill (默认 Cursor，用 --agent claude/codex/gemini 安装到其他工具)
#    同时会安装 VeloQ（veloq 二进制 + 两个 profiling skill）；加 --no-veloq 可跳过
bash install.sh
```

脚本会创建目录并复制 SKILL.md（Cursor 不识别软链接的 SKILL.md），其余文件软链接回项目目录，自动同步更新。详细安装说明参考 [INSTALL.md](INSTALL.md)。

### 其他工具

SKILL.md 格式兼容 Claude Code、Codex、Gemini CLI 等支持 Agent Skills 的工具。安装脚本提供了 `--agent` 参数将文件复制到对应路径:

| 工具 | 安装路径 | 命令 |
|:-----|:---------|:-----|
| Cursor | `~/.cursor/skills/` | `bash install.sh` (默认) |
| Claude Code | `~/.claude/skills/` | `bash install.sh --agent claude` |
| Codex | `~/.codex/skills/` | `bash install.sh --agent codex` |
| Gemini CLI | `~/.gemini/skills/` | `bash install.sh --agent gemini` |

注: 本项目只在 Cursor 下完整验证过。其他工具的 skill 发现和搜索机制可能有差异，如遇问题可以让对应工具的 AI 协助排查。

### 路径约定说明

不同工具对配置目录和 skill 目录的命名不完全一样，容易混淆。

| 工具 | 配置目录 | 常见 skill 目录 | 说明 |
|:-----|:---------|:----------------|:-----|
| Cursor | `~/.cursor/` | `~/.cursor/skills/` | Cursor 自己的目录约定 |
| Claude Code | `~/.claude/` | `~/.claude/skills/` | Claude Code 自己的目录约定 |
| Codex | `~/.codex/` | `~/.codex/skills/` | 本仓库统一使用 `.codex` 目录 |
| Gemini CLI | `~/.gemini/` | `~/.gemini/skills/` | Gemini CLI 自己的目录约定 |

对 Codex 来说要特别区分两件事：

- `~/.codex/config.toml` 是 Codex 的用户级配置文件
- `~/.codex/skills/` 是本仓库约定的 Codex skill 安装目录

因此本仓库的 `install.sh --agent codex` 会直接安装到 `~/.codex/skills/`。

## 目录结构

```
agent-gpu-skills/
├── README.md
├── INSTALL.md                       # 详细安装指南
├── install.sh                       # 安装脚本 (支持 --agent cursor|claude|codex|gemini，含 VeloQ)
├── install-veloq.sh                 # VeloQ 最小封装 (veloq 二进制 + 两个 profiling skill)
├── update-repos.sh                  # 克隆/更新外部 repo (triton, cutlass, sglang, veloq)
├── scrape_docs.py                   # 文档爬虫 (uv script)
├── cuda_skill/
│   ├── SKILL.md
│   └── references/                  # CUDA 文档库 (~700 files)
├── triton_skill/
│   ├── SKILL.md
│   ├── quick-reference.md
│   └── repos/triton/                # sparse checkout (~8MB, .gitignore)
├── cutlass_skill/
│   ├── SKILL.md
│   └── repos/cutlass/               # sparse checkout (~62MB, .gitignore)
├── sglang_skill/
│   ├── SKILL.md
│   └── repos/sglang/                # sparse checkout (~44MB, .gitignore)
├── colfax-research-skill/
│   ├── SKILL.md
│   ├── scripts/update_kb.py         # 增量抓取 Colfax 文章
│   └── colfax_knowledge_base/       # 文章 markdown + PDF (随仓库提交)
├── gpu-communication-libraries/
│   └── SKILL.md                     # GPU 通信库全景指南（纯文本 skill，无外部 repo）
├── ncu-persistent-kernel-diagnosis/
│   └── SKILL.md                     # NCU 诊断 Persistent Kernel SM/TC 空泡与流水线 stall
├── persistent-kernel-scheduling/
│   └── SKILL.md                     # Persistent Kernel 调度策略：Static/Dynamic/CLC/Stream-K
└── persistent-kernel-utilization/
    └── SKILL.md                     # Persistent Kernel 利用率极致优化指南
```

`repos/` 目录通过 `.gitignore` 排除，用 `bash update-repos.sh` 重建。

## cuda-skill

NVIDIA CUDA 全套文档转换为可搜索的 Markdown:

| 文档 | 文件数 | 大小 | 来源 |
|:-----|:-------|:-----|:-----|
| PTX ISA 9.1 完整规范 | 405 | 2.3MB | [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) |
| PTX 精简参考 | 13 | 149KB | [triton/.claude/knowledge](https://github.com/facebookexperimental/triton) |
| CUDA Runtime API 13.1 | 107 | 0.9MB | [NVIDIA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/) |
| CUDA Driver API 13.1 | 128 | 0.8MB | [NVIDIA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/) |
| CUDA Programming Guide v13.1 | 39 | 1.6MB | [NVIDIA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) |
| CUDA C++ Best Practices Guide | 73 | 585KB | [NVIDIA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) |
| Nsight Compute 文档 | 9 | 741KB | [NVIDIA Nsight Compute](https://docs.nvidia.com/nsight-compute/) |
| Nsight Systems 文档 | 5 | 833KB | [NVIDIA Nsight Systems](https://docs.nvidia.com/nsight-systems/) |
| 工具指南 (nsys/ncu/debug) | 6 | - | 手写参考 |

文档通过 `scrape_docs.py` 管理，用 `uv run scrape_docs.py all --force` 更新。

## cutlass-skill

引用 CUTLASS 源码:

| 内容 | 路径 |
|:-----|:-----|
| CuTeDSL Python 教程 | `cutlass/python/CuTeDSL/` |
| CuTe Python bindings | `cutlass/python/pycute/` |
| CUTLASS C++ 示例 | `cutlass/examples/` |
| CuTe 头文件 | `cutlass/include/cute/` |

## triton-skill

直接引用 Triton 源码中的教程、示例和内核实现:

| 内容 | 路径 | 说明 |
|:-----|:-----|:-----|
| Triton 教程 (01-11) | `triton/python/tutorials/` | 从 vector add 到 block-scaled matmul |
| Gluon 教程 (01-12) | `triton/python/tutorials/gluon/` | layouts, TMA, wgmma, tcgen05, warp spec |
| Gluon 示例 | `triton/python/examples/gluon/` | Flash Attention (Blackwell) |
| 生产级内核 | `triton/python/triton_kernels/` | matmul, reduce, top-k, SwiGLU, MXFP |
| 语言定义 | `triton/python/triton/language/` | tl.* 操作语义 |

## sglang-skill

引用 SGLang 源码:

| 内容 | 路径 |
|:-----|:-----|
| SRT 推理引擎 | `sglang/python/sglang/srt/` |
| JIT 内核 | `sglang/python/sglang/jit_kernel/` |
| SGL-Kernel (CUDA) | `sglang/sgl-kernel/` |

## colfax-research-skill

[Colfax Research](https://research.colfax-intl.com) 技术文章本地知识库（29 篇，2023-2026），随仓库提交无需额外抓取:

| 内容 | 路径 |
|:-----|:-----|
| 文章索引 (标题/分类/摘要) | `colfax_knowledge_base/metadata.json` |
| 文章正文 markdown | `colfax_knowledge_base/articles/` |
| 原文 PDF | `colfax_knowledge_base/pdfs/` |
| 增量更新脚本 | `scripts/update_kb.py` |

增量抓取新文章（依赖 `requests`、`beautifulsoup4`）:

```bash
cd colfax-research-skill
python3 scripts/update_kb.py
```

## VeloQ (profile 查询 CLI)

[VeloQ](https://github.com/lucifer1004/veloq) 是一个 agent-friendly 的 GPU profile 查询 CLI：**纯 CLI 入、JSON 契约出**，一次调用一个结果，无需打开 Nsight GUI。它读 `nsys export -t parquetdir` 的产物和 `.ncu-rep` 报告，输出带版本号的稳定 envelope（`data.rows[]`，每行带可跨 trace diff 的 `key`），专为 coding agent / shell 脚本推理 GPU profile 设计。

本仓库**不 vendored** VeloQ 内容，只做最小封装 [install-veloq.sh](install-veloq.sh)，按 `本地 ../VeloQ 源码 → veloq self-update → 官方 curl 安装器` 的优先级取得二进制与 skill，全程非致命。

| 安装方式 | 命令 |
|:---------|:-----|
| 随本仓库一起装（推荐） | `bash install.sh --agent claude`（加 `--no-veloq` 跳过） |
| 只装/更新 veloq 二进制 | `bash update-repos.sh veloq` |
| 只装 skill 到某 agent | `bash install-veloq.sh --agent claude --no-binary` |
| 用本地 VeloQ 源码 | `VELOQ_SRC=/path/to/VeloQ bash install-veloq.sh --agent claude` |
| 升级（VeloQ 原生） | `veloq self-update` |

装好后两个 skill（`nsys-profile-analysis` / `ncu-profile-analysis`）会落到 `~/.<agent>/skills/`，agent 遇到 `.nsys-rep` / `.ncu-rep` 问题时自动触发；也可手动用 CLI：

```bash
# 这个文件能被谁读？属于哪种 profile？
veloq info trace.nsys-rep

# NSys（timeline）：总览 → 热点 kernel → GPU 空闲气泡
veloq summary trace.nsys-rep
veloq stats   trace.nsys-rep --limit 10
veloq gaps    trace.nsys-rep --scope device
veloq recipes                       # 列出内置工作流（gpu-idle-audit、nvtx-breakdown…）

# NCU（kernel 报告）：总览 → 列 launch → 钻取单个 launch 的全部 metric + rule
veloq ncu summary  report.ncu-rep
veloq ncu launches report.ncu-rep --kernel '*gemm*'
veloq ncu inspect  report.ncu-rep --row-id launch:0

# 默认 JSON（agent 契约），也能投影成 csv/table 给人看
veloq stats trace.nsys-rep --limit 10 --format table
```

更多用法见 `veloq <verb> --help`（从响应结构体的 JsonSchema 投影，不会与实际 wire 格式漂移）和两个 skill 的 `SKILL.md`。

## 致谢

cuda-skill 的文档爬取方案受 [technillogue/ptx-isa-markdown](https://github.com/technillogue/ptx-isa-markdown) 启发。

## 许可

CUDA 文档内容 (c) NVIDIA Corporation. Triton、CUTLASS、SGLang 源码遵循各自原始许可.
