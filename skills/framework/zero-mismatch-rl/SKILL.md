---
name: zero-mismatch-rl
description: Build and deliver bitwise-identical train-inference consistency (zero mismatch, true on-policy RL) between the vLLM rollout engine and the trainer (diff = 0), not small mismatch. Use for the full C0–C10 gate workflow, mLite training-side support, forward/backward/quantization and STE contracts, weight update and online sync, checkpoint/resume, per-model case recipes and evidence (DeepSeek-V4-Flash, DeepSeek-V4.1-Flash, Nemotron, Kimi-K3.1), and paired performance/quality acceptance. 中文触发词：零不一致、真 on-policy、true on-policy、训推 0 diff、logprob 0 diff、bitwise 训推、rollout 与训练对不齐、C0–C10、mLite、BI=1、权重同步、在线同步、权重版本、θ 版本。
---

<!-- Source: Inferact/zero-mismatch-rl `skill/` + `cases/` @ 136aca87d8158adae94b5a272fa4a8a6d5cd6939 (Apache-2.0, see LICENSE). Vendored verbatim into this directory; the only local files are SKILL.md, README.md and README.upstream.md. All other files are upstream bytes; see README.md. -->

# zero-mismatch-rl: train-inference consistency workflow (C0–C10)

Turn train-inference differences into experiments with fixed inputs, fixed weights and
repeatability; locate the first divergence, fix or optimize, then return to the real training
loop. The target is **zero mismatch**: the RL rollout engine (vLLM) and the trainer compute
bitwise identical token probabilities for the same token (diff = 0), and only plain unclipped
token-level importance sampling is needed to correct weight lag.

Start here: **[skill/README.md](skill/README.md)** — the C0–C10 gate table, "ask the human
first" items, performance budgets, the two human stop points, baseline/consistency parallel
lines, and the delivery conventions. Acceptance details are in [skill/validation.md](skill/validation.md)
and [skill/numerical-contracts.md](skill/numerical-contracts.md); process rules in
[skill/practices/](skill/practices/); mechanism guides in [skill/guides/](skill/guides/);
record templates in [skill/templates/](skill/templates/).

Model cases with per-model recipe, failure records, acceptance evidence and delivered
results: [cases/](cases/) — [DeepSeek-V4-Flash](cases/deepseek-v4-flash/), [DeepSeek-V4.1-Flash](cases/deepseek-v4.1-flash/),
[Nemotron](cases/nemotron/), [Kimi-K3.1](cases/kimi-k3.1/). Do not apply historical switch
lists directly to a new task.

Upstream overview (why zero mismatch, support matrix, code branches, acknowledgments):
[README.upstream.md](README.upstream.md) and https://github.com/Inferact/zero-mismatch-rl.

Local notes for this repository (provenance, layout, sync procedure, division of labor with
`train-inference-consistency`): [README.md](README.md).

## Relationship to other skills

| Question | Use |
| --- | --- |
| Full zero-mismatch / true on-policy workflow: C0–C10 gates, per-model cases and evidence, weight update and online sync, checkpoint/resume, precision/QAT, delivery and paired acceptance | this skill |
| vLLM `.agents/skills` snapshot of the same workflow, with the pinned vLLM sparse checkout, batch-invariance implementation and tests, and its source map | `train-inference-consistency` |
| verl framework internals (PPO/GRPO metrics, rollout correction, router replay) | `verl-skill` |
| SGLang as the rollout engine (scheduler, attention backend, KV cache) | `sglang-skill` |
| C10 BI kernel optimization (no Triton; CuTe DSL / CUDA / CUTLASS) | `cutlass-skill`, `cuda-skill`, `gpu-kernel-authoring-router` |
| NCCL / DeepEP / all-to-all collectives determinism | `nccl-skill`, `gpu-communication-libraries` |
| Writing the analysis report in Chinese | `tech-doc-style-chinese`; report fields follow [skill/templates/](skill/templates/) |

## Install

Installed by `install.sh` (this repository's bundle): the file layout is a real `SKILL.md`
plus the `skill/` and `cases/` directories; the install links `skill/` and `cases/` into the
target agent's skill directory so relative links between them keep working. Verify with:

```bash
bash install.sh --agent claude   # ~/.claude/skills/zero-mismatch-rl/
bash install.sh --agent codex    # ~/.codex/skills/zero-mismatch-rl/
```
