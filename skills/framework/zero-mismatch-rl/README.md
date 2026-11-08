# zero-mismatch-rl（来自 Inferact/zero-mismatch-rl）

本目录是 [Inferact/zero-mismatch-rl](https://github.com/Inferact/zero-mismatch-rl) 仓库的本地副本：让 RL 训练侧与 vLLM rollout 对同一 token 给出 **bitwise 相同**的 logprob（diff = 0，零不一致，而不是小不一致），并在保持 0 diff 的前提下完成交付性能与质量验收。入口是 [SKILL.md](SKILL.md)，工作流从 [skill/README.md](skill/README.md) 的 C0–C10 gate 开始。

## 上游来源

- 仓库：`https://github.com/Inferact/zero-mismatch-rl.git`
- 基线：commit `136aca87d8158adae94b5a272fa4a8a6d5cd6939`（2026-11-08，main）
- 许可：Apache-2.0，见本目录 [LICENSE](LICENSE)（复制自上游仓库根目录）
- 上游原始 README：[README.upstream.md](README.upstream.md)（原样保存，含 support matrix、为什么零不一致、代码分支与致谢）

## 本地修改

| 文件 | 状态 | 说明 |
| --- | --- | --- |
| `SKILL.md` | 新增 | 本仓库的安装入口：frontmatter（name/description + 中文触发词）、workflow 导航、与其他 skill 的分工表 |
| `README.md` | 新增 | 本文件 |
| `README.upstream.md` | 新增（原样） | 上游根 README 的逐字节副本 |
| `LICENSE` | 新增（原样） | 上游 Apache-2.0 许可原文 |
| `skill/` | 原样 | 上游 `skill/`（README、validation、numerical-contracts、guides、practices、templates）逐字节一致，含其中的 `../cases/...` 相对链接 |
| `cases/` | 原样 | 上游 `cases/`（deepseek-v4-flash、deepseek-v4.1-flash、nemotron、kimi-k3.1 的 recipe、报告、history 与结果）逐字节一致 |

## 目录结构与链接关系

```text
skills/framework/zero-mismatch-rl/
├── SKILL.md              # 入口（本仓库新增）
├── README.md             # 本文件
├── README.upstream.md    # 上游根 README（原样）
├── LICENSE               # Apache-2.0（原样）
├── skill/                # 上游 skill/：C0–C10 工作流正文
│   ├── README.md         # 工作流入口（gate 表、human stop points）
│   ├── validation.md     # 验收细则（admission 表、双路契约、分层判据）
│   ├── numerical-contracts.md
│   ├── guides/…          # 各机制指南（weight-update、checkpoint-resume-memory、…）
│   ├── practices/…       # evidence-and-experiments、code-and-delivery、operations-and-handoff
│   └── templates/…       # consistency-experiment.yaml、consistency-report.md、audit-verdict.yaml、…
└── cases/                # 上游 cases/：按模型的配方、故障记录与证据
    ├── deepseek-v4-flash/  deepseek-v4.1-flash/  nemotron/  kimi-k3.1/
```

`skill/guides/**` 中的文档用 `../../../cases/...` 引用模型案例，因此 `skill/` 与 `cases/` 必须保持同级，安装时两者一起链接进目标 skill 目录（`install.sh` 已按目录项处理）。

## 与其他 skill 的分工

| 问题 | 使用 |
| --- | --- |
| 零不一致 / true on-policy 的完整工程流程：C0–C10 gate、按模型配方与故障记录、权重更新与在线同步、checkpoint/resume、精度与 QAT、交付与配对验收 | 本 skill |
| 同一工作流的 vLLM 快照版：vLLM `.agents/skills` 原文 + 固定 commit 的 vLLM 稀疏源码（batch invariance 实现与测试）、vLLM 侧源码地图 | `train-inference-consistency`（固定在 vLLM commit `2d8efee9`，不含模型案例） |
| verl 框架内的训推差异指标与配置 | `verl-skill` |
| SGLang 作为 rollout 引擎时的行为 | `sglang-skill` |
| C10 阶段 BI kernel 优化（不用 Triton，用 CuTe DSL / CUDA / CUTLASS） | `cutlass-skill`、`cuda-skill`、`gpu-kernel-authoring-router` |
| NCCL、DeepEP 等集合通信归约顺序与确定性 | `nccl-skill`、`gpu-communication-libraries` |
| 中文写分析报告 | `tech-doc-style-chinese`；字段以本 skill 的 `skill/templates/` 为准 |

## Code branches <a id="code-branches"></a>

上游按框架与模型维护代码分支（每个分支两个 commit：未修改的上游快照 + zero-mismatch 修改）。完整表格见 [README.upstream.md](README.upstream.md#code-branches) 与上游仓库 https://github.com/Inferact/zero-mismatch-rl 。本仓库只 vendored `main` 的 `skill/` 与 `cases/`，不含代码分支。

## 同步上游

1. 重新获取上游 main（克隆到 git-ignored 的 `repos/zero-mismatch-rl/`）：

   ```bash
   bash update-repos.sh zero-mismatch-rl
   ```

2. 比对并覆盖 `skill/` 与 `cases/`（本仓库这两个目录没有本地修改，可直接覆盖）：

   ```bash
   diff -ru --exclude=README.md --exclude=LICENSE --exclude=SKILL.md \
     repos/zero-mismatch-rl/skill skills/framework/zero-mismatch-rl/skill
   diff -ru --exclude=README.md --exclude=LICENSE --exclude=SKILL.md \
     repos/zero-mismatch-rl/cases skills/framework/zero-mismatch-rl/cases
   cp -R repos/zero-mismatch-rl/skill/. skills/framework/zero-mismatch-rl/skill/
   cp -R repos/zero-mismatch-rl/cases/. skills/framework/zero-mismatch-rl/cases/
   ```

3. 标为「原样」的文件直接覆盖；`README.upstream.md` 从上游根 README 覆盖；`SKILL.md` 保持本地版本，仅在工作流入口变化时更新导航链接。
4. 更新本文件的基线 commit 与 `update-repos.sh` 中 `zero-mismatch-rl` 的 ref（如需固定版本）。
5. 重新生成路由清单：`python3 skills/routers/agent-skill-router/scripts/build_inventory.py`，并确认 `build_inventory.py --check` 通过。
