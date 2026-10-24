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
| verl RL 后训练（PPO/GRPO、rollout、reward、agent loop、训推一致） | `verl-skill` |
| bitwise 训推一致（vLLM batch invariance、logprob 0 diff、BI kernel） | `train-inference-consistency` |
| LLM serving、KV cache、吞吐/延迟、容量规划 | `llm-serving-router` |
| Blackwell/Hopper kernel 资料与上游 PR | `KernelWiki` |
| 写/改/审中文技术原理报告、设计文档、性能报告、源码解析 | `tech-doc-style-chinese` |
| 把任务委托给 Claude Code（Codex 需额外注册 MCP，见[下文](#claude-委托)） | `claude-delegate` |

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
bash update-repos.sh verl
bash update-repos.sh vllm
bash update-repos.sh nccl
bash update-repos.sh kernel-design-agents
bash update-repos.sh tilelang-cuda-skills
bash update-repos.sh nvidia-skills
bash update-repos.sh amd-skills
bash update-repos.sh cursor-skills
bash update-repos.sh ccfa-skills
bash update-repos.sh tech-doc-style-chinese
bash update-repos.sh iket-profiling
bash update-repos.sh salary-negotiation
bash update-repos.sh ponytail
# 文档单独刷新，耗时较长；提交前检查索引与抓取完整性
bash update-repos.sh docs
```

`tech-doc-style-chinese` 的上游 [Fenng/Tech-Doc-Style-Chinese](https://github.com/Fenng/Tech-Doc-Style-Chinese)
clone 到 git-ignored 的 `repos/tech-doc-style-chinese/`，只用于比对；安装的是
`skills/other/tech-doc-style-chinese/` 中的本地定制版，同步步骤见其 `README.md`。

`repos/nvidia-skills`、`repos/amd-skills`、`repos/cursor-skills`、
`repos/ccfa-skills`、`repos/tilelang-cuda-skills`、`repos/ponytail` 和 `kernel-design-agents`
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

## Claude 委托

`claude-delegate` 把任务交给本机的 Claude Code CLI 执行；在 Codex 中它依赖
支撑 skill `codex-background` 提供的后台 MCP server。两者原样复制自
[`amElnagdy/delegate-skills`](https://github.com/amElnagdy/delegate-skills)
commit `3aa72d681a79e2a9d09bd9a316353003ae1cb4a7`，放在
`skills/other/claude-delegate/` 和 `skills/other/codex-background/`；许可为
MIT，见各目录下的 `LICENSE`。

前置条件：

- Node.js 18+；Linux 下 `codex-background` 还需要 `/usr/bin/cc` 和 procfs。
- Claude Code CLI：`claude` 在 `PATH` 上，并已完成登录认证。Linux 下执行
  Claude 的受限 shell 命令还需要 `bubblewrap` 和 `socat`。

安装器为所有 agent 安装 `claude-delegate`，`codex-background` 只在
`--agent codex` 时安装。注册 MCP server 会写入全局 Codex 配置，所以安装器
不会自动执行这一步。Codex 用户需在安装后手动运行，然后重启 Codex 使其生效：

```bash
bash install.sh --agent codex
node ~/.codex/skills/codex-background/scripts/bootstrap.mjs
```

使用 `--dest DIR` 时，脚本路径换成 `DIR/codex-background/scripts/bootstrap.mjs`。
默认混合模式下，已安装 skill 的 `scripts/` 是指向本仓库的软链接；移动或删除
本仓库后，需要重新安装并重新运行 bootstrap。

## 许可边界

AMD ISA PDF、AMD 文档缓存和按需下载的上游源码不会提交到仓库。AMD 文档和
ISA 只在用户明确接受官方文档条款后下载到 git-ignored 本地目录。

## 研究笔记与文献工具

[verl 课程笔记](notes/verl-course/README.md)收录按源码快照编写的离线 HTML 笔记。
文献检索与 Zotero 阅读使用 `skills/other/zotero-cli`，随本地 skills 一起安装；
使用前运行 `zotero-cli config` 检查本机连接。Agent 的安装副本和本机 MCP 配置不提交到仓库。

## Ponytail

[Ponytail](https://github.com/DietrichGebert/ponytail) 通过 `repos/ponytail` 子模块接入，
安装器安装其 6 个 skill：ponytail、review、audit、debt、gain、help。
完整 Codex 插件及生命周期 hooks 的安装见 [安装指南](INSTALL.md#ponytail)。
