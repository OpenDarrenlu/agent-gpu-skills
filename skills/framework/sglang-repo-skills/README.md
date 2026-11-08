# SGLang 仓库内建 skill（sgl-project/sglang `.agents/skills` 精选）

本目录 vendored 自 [sgl-project/sglang](https://github.com/sgl-project/sglang) 仓库自带的
agent skills（`.agents/skills/`，`.claude/skills` 是它的 symlink）。它们面向 SGLang 框架的
调试、profile、kernel 贡献与一致性测试，与本仓库已有的 `sglang-skill`（SGLang 框架知识）互补：
`sglang-skill` 讲框架原理，这里的是可执行的操作流程。

## 上游来源

- 仓库：`https://github.com/sgl-project/sglang.git`
- 路径：`.agents/skills/`
- 基线：commit `3bcde0c80eeb07b40d3334c4b3d7341602efd089`（main，2026-11-08）
- 许可：SGLang 为 Apache-2.0（见上游仓库根目录 LICENSE）。
- 上游完整备份：`skills/framework/sglang-skill/repos/sglang/` 的稀疏 clone（含 `.agents/skills`）；
  运行 `bash update-repos.sh sglang` 获取或更新。

## 本地修改

所有文件均**原样** vendored，未做修改（包括各 skill 的 `references/`、`scripts/` 与
`agents/openai.yaml`）。`SKILL.md` 中的相对链接（如 `scripts/…`、`references/…`）按原样工作。

## 收录的 skill

| skill | 作用 |
| --- | --- |
| `kl-consistency-test` | SGLang prefill vs decode logprob（KL）一致性测试：两个独立条件（算子 batch-invariant + 两路同一函数）、helper 选择、阈值标定与单算子定位。与 `zero-mismatch-rl` / `train-inference-consistency` 互补 |
| `debug-cuda-crash` | 用 kernel API 日志调试 SGLang CUDA 崩溃 |
| `debug-distributed-hang` | SGLang 分布式推理（TP/PP/DP/EP）hang 定位：py-spy/watchdog/coredump、per-rank 日志、二分 |
| `sglang-prod-incident-triage` | SGLang 线上服务问题 replay-first 排查流程 |
| `compute-mamba-ratio` | 混合注意力 + 线性注意力（Mamba/GDN/KDA）模型的 `--mamba-full-memory-ratio` 计算 |
| `generate-profile` | 生成 SGLang server 运行的端到端 profiling trace |
| `add-jit-kernel` | 给 `sglang.kernels` JIT 基础设施添加轻量 JIT CUDA kernel |
| `add-sgl-kernel` | 给 `sgl-kernel`（AOT CUDA/C++）添加重型 kernel（含测试与 benchmark） |
| `kernel-organization` | SGLang kernels RFC：API 拆分、registry 元数据、测试与 benchmark 组织 |
| `write-sglang-test` | SGLang CI/UT 测试编写（CustomTestCase、tier、fixture、模型选择） |

未收录的上游 skill（SGLang 仓库维护类：`babysit-pr-to-pass-ci`、`ci-*`、`cookbook-*`、
`env-var-conventions`、`large-class-style`、`mechanical-refactor-verify`、`processor-*-parity`、
`rerun-test`、`scripted-runtime-notes`、`sglang-bisect-ci-regression`、`sglang-cherrypick`、
`sglang-multi-graph-check`、`sglang-runtime-context`、`speculative-naming`、`clean-startup-log`）
与本 bundle 的 GPU 开发/性能/框架目标关系较弱，需要时可直接从上游 `.agents/skills/`
取用。`llm-torch-profiler-analysis` 上游也有，但本 bundle 已由 `cursor-skills` 提供同名 skill，
故不重复收录（本地先到先得，cursor 版优先）。

## 与其他 skill 的分工

| 问题 | 使用 |
| --- | --- |
| SGLang 框架原理、scheduler/attention backend/KV cache 行为 | `sglang-skill` |
| prefill/decode 0 diff 或 logprob 对齐（端到端 workflow） | `zero-mismatch-rl`、`train-inference-consistency` |
| SGLang 侧 KL 一致性测试的写法与调试 | `kl-consistency-test` |
| SGLang CUDA 崩溃 / 分布式 hang / 线上事故 / 显存比 | `debug-cuda-crash`、`debug-distributed-hang`、`sglang-prod-incident-triage`、`compute-mamba-ratio` |
| 给 SGLang 贡献 kernel 或测试 | `add-jit-kernel`、`add-sgl-kernel`、`kernel-organization`、`write-sglang-test` |
| kernel 性能分析（ncu/nsys） | `ncu-report-skill`、`nsys-profile-analysis` 等 |

## 同步上游

1. `bash update-repos.sh sglang`（sparse clone 含 `.agents/skills`）。
2. 比对：`diff -ru repos/sglang/.agents/skills skills/framework/sglang-repo-skills`（排除本 README）。
3. 上表列出的 skill 原样覆盖；新增/删除 skill 时同步更新本 README 与 `install.sh` 的
   `SKILL_NAMES`/`SKILL_DIRS`。
4. 重新生成路由清单：`python3 skills/routers/agent-skill-router/scripts/build_inventory.py`，
   并确认 `build_inventory.py --check` 通过。
