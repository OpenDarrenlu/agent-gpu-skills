# train-inference-consistency（来自 vLLM）

本目录是 vLLM 仓库内置 agent skill [`train-inference-consistency`](https://github.com/vllm-project/vllm/tree/2d8efee9a73366d6b4e753780af32443d2fef91a/.agents/skills/train-inference-consistency) 的本地副本。它的目标是让 RL 训练侧与 vLLM rollout 对同一 token 给出 bitwise 相同的 logprob（diff = 0），并在保持 0 diff 的前提下把性能拉回高性能基线附近。入口是 [SKILL.md](SKILL.md)，工作流从 [workflow/README.md](workflow/README.md) 的 C0–C10 gate 开始。

## 上游来源

- 仓库：`https://github.com/vllm-project/vllm.git`
- 路径：`.agents/skills/train-inference-consistency/`
- 基线：commit `2d8efee9a73366d6b4e753780af32443d2fef91a`（2026-10-17，`[Docs] Add train-inference consistency agent skill (#61551)`）
- 许可：Apache-2.0，见本目录 [LICENSE](LICENSE)（复制自 vLLM 仓库根目录）。
- 本地稀疏 clone：本目录下的 `repos/vllm/`（git-ignored），在仓库根目录运行 `bash update-repos.sh vllm` 获取或更新。

## 本地修改

| 文件 | 状态 | 说明 |
| --- | --- | --- |
| `SKILL.md` | 修改 | description 末尾增加中文触发词；batch invariance 文档链接从 vLLM 仓库内相对路径改为本目录的稀疏 clone（`repos/vllm/docs/features/batch_invariance.md`），并附上游固定链接；增加来源注释和指向本文件的说明 |
| `LICENSE` | 新增 | vLLM 的 Apache-2.0 许可原文 |
| `README.md` | 新增 | 本文件 |
| 其余全部文件 | 原样 | `agents/`、`guides/`、`practices/`、`workflow/` 与上游逐字节一致 |

## vLLM 源码地图

运行 `bash update-repos.sh vllm` 后，本目录的 `repos/vllm/` 中会有以下与 batch invariance 相关的 vLLM 源码（固定在同一 commit）：

| 路径 | 内容 |
| --- | --- |
| `docs/features/batch_invariance.md` | vLLM 侧开启 batch-invariant 推理的说明，SKILL.md 直接引用 |
| `vllm/model_executor/determinism/` | `batch_invariant.py`、`batch_invariant_configs.py` 等 batch invariant 实现 |
| `tests/v1/determinism/` | batch invariance 测试，覆盖 matmul、RMSNorm、CUTLASS、NVFP4、MX linear、LoRA shrink、VLM、在线服务等 |
| `examples/rl/` | RL 集成示例，含 `batch_invariance/reproducibility_offline.py` |
| `.agents/skills/` | vLLM 仓库全部内置 skill，本仓库只安装 `train-inference-consistency` |

`benchmarks/benchmark_batch_invariance.py` 没有纳入稀疏 clone，需要时查看上游。skill 中出现的 `csrc/batch_invariant/`、`vllm/primitive/` 等路径来自其引用的 PR（vLLM #54955、Megatron-LM #7050），不在该 commit 的 vLLM 主线中。

## 与本仓库其他 skill 的分工

| 问题 | 使用 |
| --- | --- |
| bitwise 训推一致的完整工程流程：C0–C10 gate、训练前向复用推理 kernel、反向/STE 校验、量化转换链、交付与验收 | 本 skill |
| verl 框架内的训推差异指标（`rollout_probs_diff`）、rollout correction、`full_determinism`、router replay 等配置 | `verl-skill` |
| SGLang 作为 rollout 引擎时的 scheduler、attention backend、KV cache 行为 | `sglang-skill` |
| C10 阶段的 BI kernel 优化（skill 规定此阶段不用 Triton，使用 CuTe DSL、CUDA/CUTLASS 或已有实现） | `gpu-kernel-authoring-router`、`cutlass-skill`、`cuda-skill`、`gpu-performance-router` |
| NCCL、DeepEP 等通信路径的归约顺序与确定性 | `gpu-communication-libraries`、`nccl-skill` |
| 用中文写训推一致分析报告 | `tech-doc-style-chinese`；交付报告的字段以本 skill 的 [workflow/report-template.md](workflow/report-template.md) 为准 |

## 同步上游

1. 运行 `VLLM_SKILLS_REF=main bash update-repos.sh vllm`，或把 `update-repos.sh` 中 `VLLM_SKILLS_REF` 的默认值改为新的 commit 后运行 `bash update-repos.sh vllm`。
2. 比对：`diff -ru --exclude=repos --exclude=README.md --exclude=LICENSE skills/framework/train-inference-consistency/repos/vllm/.agents/skills/train-inference-consistency skills/framework/train-inference-consistency`。
3. 标为「原样」的文件直接覆盖；`SKILL.md` 用上游新版本重新应用上表中的修改。
4. 更新本文件、`SKILL.md` 来源注释中的基线 commit，以及 `update-repos.sh` 中 `VLLM_SKILLS_REF` 的默认值。
