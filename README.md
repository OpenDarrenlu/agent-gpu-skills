# agent-gpu-skills

GPU 开发、性能分析、框架运行时和科研工作流的 Agent Skill 集合，兼容
Cursor、Claude Code、Codex、Gemini CLI 与 Kimi Code CLI。

## 快速入口

不知道该用哪个 skill 时，从 `gpu-development-catchall` 开始。

| 任务 | 入口 |
|---|---|
| 写 CUDA/PTX/Triton/CUTLASS/DeepGEMM kernel | `gpu-kernel-authoring-router` |
| kernel 慢、NCU/NSYS、occupancy、stall、SM/TC 利用率 | `gpu-performance-router` |
| TileLang kernel 编写、调试、优化、profiling | `writing-tilelang-kernels` / `profiling-tilelang-programs` |
| SGLang/vLLM、PyTorch、端到端延迟、框架 profiling | `gpu-framework-router` |
| LLM serving、KV cache、吞吐/延迟、容量规划 | `llm-serving-router` |
| Blackwell/Hopper kernel 资料与上游 PR | `KernelWiki` |

完整分类索引见 [`skills/README.md`](skills/README.md)，路由规则见
[`AGENTS.md`](AGENTS.md)。

## 目录

```text
skills/
├── kernel/     # kernel 实现、kernel profiling、性能建模
├── framework/  # 框架、runtime、serving、framework profiling
├── other/      # 通信、AMD/ROCm、知识库和其他专项
└── routers/    # catch-all 与领域路由
repos/          # NVIDIA/AMD/Cursor/CCFA/TileLang 外部 skill submodule
kernel-design-agents/
                # Kernel Design Agents + KernelWiki
```

## 安装

```bash
git clone --recursive git@github.com:OpenDarrenlu/agent-gpu-skills.git
cd agent-gpu-skills

# Codex
bash bootstrap.sh --agent codex

# 其他 agent
bash bootstrap.sh --agent cursor
bash bootstrap.sh --agent claude
bash bootstrap.sh --agent gemini
bash bootstrap.sh --agent kimi
```

只安装本地 skills 并跳过大型外部 catalog：

```bash
bash bootstrap.sh --agent codex \
  --no-nvidia-skills --no-amd-skills --no-cursor-skills --no-ccfa-skills --no-veloq
```

验证安装器而不修改用户目录：

```bash
bash install.sh --dest /tmp/agent-gpu-skills-test --no-veloq \
  --no-nvidia-skills --no-amd-skills --no-cursor-skills --no-ccfa-skills
```

## 更新

```bash
bash update-repos.sh
bash update-repos.sh triton
bash update-repos.sh cutlass
bash update-repos.sh deepgemm
bash update-repos.sh sglang
bash update-repos.sh nccl
bash update-repos.sh kernel-design-agents
bash update-repos.sh tilelang-cuda-skills
bash update-repos.sh nvidia-skills
bash update-repos.sh amd-skills
bash update-repos.sh cursor-skills
bash update-repos.sh ccfa-skills
```

`repos/nvidia-skills`、`repos/amd-skills`、`repos/cursor-skills`、
`repos/ccfa-skills`、`repos/tilelang-cuda-skills` 和 `kernel-design-agents`
是 submodule。更新后需要提交主仓库中的 submodule 指针变化。

## TileLang Skills

[`sablin39/tilelang-cuda-skills`](https://github.com/sablin39/tilelang-cuda-skills)
已作为 `repos/tilelang-cuda-skills` submodule 接入。安装器默认安装其中的
TileLang 工作流：

- `tilelang-optimization-loop`
- `writing-tilelang-kernels`
- `debugging-tilelang-programs`
- `optimizing-tilelang-programs`
- `profiling-tilelang-programs`
- `torch-profiling-tilelang-programs`
- `testing-fwd-bwd-kernels`
- `writing-tilelang-design-docs`
- `designing-with-tilefoundry`

上游的 `cuda_skill` 不重复安装；CUDA 基础工作流统一使用本仓库更完整的
`cuda-skill`。如需跳过这组 skills，可使用：

```bash
bash install.sh --no-tilelang-skills
```

## 许可边界

AMD ISA PDF、AMD 文档缓存和按需下载的上游源码不会提交到仓库。AMD 文档和
ISA 只在用户明确接受官方文档条款后下载到 git-ignored 本地目录。
