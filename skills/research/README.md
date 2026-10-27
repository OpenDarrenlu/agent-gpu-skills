# Research 论文研究 Skills

本目录把两个论文研究 skill 集合放在一起：CCFA-Skills（中文 CCF 论文工作流）与 Supervisor-Skills（HKUST(GZ) 的论文写作与评审方法）。本目录只含两个导航链接和本说明，没有复制任何外部内容。

| 来源 | 本目录链接 | 实际检出位置 | 固定版本 | 安装的 skill |
|---|---|---|---|---|
| CCFA-Skills（mikubaka88） | [`ccfa-skills/`](ccfa-skills/) | `repos/ccfa-skills`（submodule，MIT） | `e33ccfa` | 17 个 `ccf-*` runtime skill，外加 `ccf-latex-templates` 支撑目录 |
| Supervisor-Skills（HKUSTDial） | [`supervisor-skills/`](supervisor-skills/) | `repos/supervisor-skills`（git-ignored，见 `sources/external-sources.conf`） | `a20d76e`（2.1.0） | 12 个 skill |

两个来源都保持稳定的 checkout 路径。安装后的 skill 名与来源目录名一致，不带目录前缀。

## 怎么选

- 通用论文任务（检索、评审、投稿检查、rebuttal、写作）默认走 CCFA 的 `ccf-*`。`agent-skill-router` 按既有规则选择它们。
- Supervisor 的具名功能（如 `idea-evaluator`、`intro-drafter`、`pre-submission-reviewer`）在用户点名时直接使用；未点名时，只有它特有的短语才会路由过去（如“致命缺陷”“综述级”“reviewer mindset”“draw.io”）。
- 用户点名的 skill 永远优先，不会被替换成别的 skill。
- GPU 事实（如 CUDA、kernel 性能数字）仍由对应的 GPU skill 提供；本目录的 skill 不替代它们。

## 按阶段对照

下表列出全部 29 个 runtime skill：CCFA 的 17 个与 Supervisor 的 12 个。另有一个支撑目录 `ccf-latex-templates`，不计入 skill 数，列在“工作流与支撑”中。每个阶段的第一项是该阶段默认的 CCFA 入口。

### 1. Idea（选题与评估）

| Skill | 来源 | 作用 |
|---|---|---|
| `ccf-idea-optimizer` | CCFA | 把粗糙想法整理为问题、洞见、机制与证据计划 |
| `ccf-idea-reviewer` | CCFA | 评估想法的价值、新颖性与可行性，给出评分与排序 |
| `idea-evaluator` | Supervisor | 五维框架（Higher、Faster、Stronger、Cheaper、Broader）、生命周期与能力匹配、范式转移探测、致命缺陷检查 |
| `ccf-experiment-designer` | CCFA | 设计实验协议、数据集、基线、指标与消融 |

### 2. 文献（检索与调研）

| Skill | 来源 | 作用 |
|---|---|---|
| `ccf-literature-searcher` | CCFA | 检索并核验外部文献、先验工作与引用候选 |
| `ccf-literature-monitor` | CCFA | 跟踪 arXiv、OpenReview、会议与竞品动态，做新颖性监控 |
| `deep-research` | Supervisor | 综述级文献调研：冻结研究问题、多视角检索、逐条引用核验、MECE 综合 |
| `ccf-paper-to-exemplar` | CCFA | 把提供的论文 PDF 提炼为写作范例卡片，登记个人文库 |

### 3. 写作

| Skill | 来源 | 作用 |
|---|---|---|
| `ccf-paper-writer` | CCFA | 起草、修改、润色、压缩 CCF 论文正文（所有 CCFA 任务的前置预检见 `ccf-humanization`） |
| `paper-writer` | Supervisor | 基于作者材料起草论文段落到全文，事实主张可追溯，引用需通过验证阶梯 |
| `intro-drafter` | Supervisor | 按六段流程图起草引言 |
| `paper-polish` | Supervisor | 保留原意的语言润色、语气校准与去 AI 味，不做意义层面的改写 |
| `tech-paper-template` | Supervisor | 用思考模板表格搭建技术论文的完整逻辑骨架 |
| `benchmark-paper-template` | Supervisor | 按五支柱框架组织 benchmark 与评测论文 |
| `ccf-humanization` | CCFA | 所有 CCFA skill 的必经预检：路由、范围、证据与写作规则 |

### 4. 图表

| Skill | 来源 | 作用 |
|---|---|---|
| `ccf-visual-composer` | CCFA | 渲染与重设计 CCF 图、视觉表格与方法架构图 |
| `figure-designer` | Supervisor | 论文三张核心图（Motivated Example、Solution Overview、Experimental Results）的设计建议 |
| `drawio-reconstruction` | Supervisor | 把参考图重建为可编辑的 `.drawio` 文件并生成预览（见下方运行条件） |

### 5. 评审与投稿

| Skill | 来源 | 作用 |
|---|---|---|
| `ccf-paper-reviewer` | CCFA | 评审论文主张、证据与写作，不改写正文 |
| `pre-submission-reviewer` | Supervisor | 投稿前五维自检：宏观逻辑、写作细节、英文语法、LaTeX 排版、图的质量 |
| `ccf-integrity-auditor` | CCFA | 核验既有论文的主张、数字、术语与引用（BibTeX） |
| `ccf-submission-checker` | CCFA | 检查 CCF 会议规则与投稿包：模板、页数、匿名、PDF 构建、元数据 |
| `ccf-rebuttal-writer` | CCFA | 撰写 rebuttal、回复信与修订台账 |
| `rebuttal-guidance` | Supervisor | 给出 rebuttal 策略指导（审稿人心态匹配、策略维度、回复模板选择），不直接写回复全文 |

### 6. 工作流与支撑

| Skill | 来源 | 作用 |
|---|---|---|
| `ccf-common` | CCFA | CCFA 共享规则（路由、任务模式、产物契约、证据），其他 `ccf-*` 通过 `../ccf-common/` 引用 |
| `ccf-pipeline-orchestrator` | CCFA | 规划与协调研究阶段、门禁与 `ccfa.yaml` 状态 |
| `ccf-project-scaffolder` | CCFA | 初始化项目目录、LaTeX 模板与 `ccfa.yaml` |
| `vibe-research-workflow` | Supervisor | AI 辅助科研的三个子流程（Vibe Coding、Vibe Figure、Vibe Writing）与行为规则 |
| `ccf-skill-forger` | CCFA | 维护与审计 CCFA/Codex skill（面向维护者，普通论文任务不需要） |
| `ccf-latex-templates/` | CCFA（支撑目录） | 会议 LaTeX 模板（如 `AAAI/aaai2026_template.tex`）。模板模式读取安装根目录下的同级目录，安装器把它与 `ccf-*` 一起放在同一层 |

## 安装与依赖

```bash
# 获取两个来源（checkout 缺失时才拉取；Supervisor 固定到 a20d76e）
bash update-repos.sh ccfa-skills
bash install-external.sh --no-runtime

# 安装（默认混合模式：skill 目录内文件 symlink 到 checkout；--copy 则复制真实文件）
bash install.sh --agent codex
bash install.sh --agent codex --copy
```

- 两种模式都会安装 29 个 runtime skill（CCFA 17 个 + Supervisor 12 个），并安装 1 个支撑目录 `ccf-latex-templates`；`drawio-reconstruction` 的 `scripts/`、`examples/`、`requirements.txt` 与 `LICENSE` 也随之安装。
- 两个集合之间的引用都是相对路径：CCFA 用 `../ccf-common/...` 等同级路径，Supervisor 用各自目录内的 `references/`、`scripts/`。`--copy` 不依赖源码 symlink。
- `--no-ccfa-skills` 跳过全部 CCFA 内容（17 个 runtime skill 与 `ccf-latex-templates`）。`--no-integrations` 跳过包括 Supervisor 在内的全部外部来源安装步骤。没有单独的 Supervisor 开关，也不需要为了跳过它而修改 `sources/external-sources.conf` 中的 `mode`。
- 安装器不安装任何命令行运行时。`drawio-reconstruction` 的检查与批处理脚本只用标准库；`crop_assist.py` 需要 Pillow，SKILL.md 要求先经用户批准再用 `requirements.txt` 安装；`export_drawio.py` 需要本机的 draw.io 桌面应用，安装器不提供，也不检测其版本。使用前应先确认这些依赖存在。
- Supervisor-Skills 仓库的 `handbook/`（根目录教材）不随任何 skill 安装。`idea-evaluator` 与 `vibe-research-workflow` 只把它作为可选的深入阅读，缺失时不影响 skill 执行。

## 来源、版本与许可

- **Supervisor-Skills**：固定到 commit `a20d76ef3979b804d7dbf599d577826957e0a8a9`，原始 origin 为 `https://github.com/HKUSTDial/Supervisor-Skills.git`。
  - 根许可文件：[`LICENSE`](../../repos/supervisor-skills/LICENSE)。该文件声明 **CC BY-NC-SA 4.0**，版权行为 `Copyright (c) 2026 Yuyu Luo (methodology) and contributors`。
  - 逐 skill 的 SKILL.md frontmatter 含 `license` 字段，取值如下（与根许可文件并列记录，不作取舍）：`CC-BY-4.0`（figure-designer、idea-evaluator、vibe-research-workflow、tech-paper-template、benchmark-paper-template、pre-submission-reviewer）；`CC-BY-NC-SA-4.0`（deep-research、paper-writer、intro-drafter、paper-polish）。
  - `rebuttal-guidance` 的 frontmatter 没有 `license` 字段。
  - `drawio-reconstruction` 的 frontmatter 没有 `license` 字段，但目录内有单独的 [`LICENSE`](../../repos/supervisor-skills/skills/drawio-reconstruction/LICENSE)，文件为 **MIT**，版权行为 `Copyright (c) 2026 VCG-Bench Authors`。该文件随 skill 目录一起安装（混合模式为 symlink 到 checkout，`--copy` 为真实文件）。
- **CCFA-Skills**：[`LICENSE`](../../repos/ccfa-skills/LICENSE) 为 MIT License，Copyright (c) 2026 Chaoyue Li。
- 本仓库不选择其中任何一个许可，也不改写上游的许可文本或版权归属。

## 维护

- 更新 Supervisor 版本时，先在 `repos/supervisor-skills` 检视新 commit 的 `LICENSE`、各 skill 的 frontmatter 与相对引用，再更新 `sources/external-sources.conf` 中的 ref 与 provenance，然后运行 `bash update-repos.sh external` 与 `python3 skills/routers/agent-skill-router/scripts/build_inventory.py`。
- 本目录的两个链接指向 `../../repos/`，不要替换为复制的目录。
- 路由规则在 `skills/routers/agent-skill-router/references/routing-rules.json`，回归测试在 `tests/test_research_routes.py` 与 `tests/test_skills_router.py`，安装与来源一致性测试在 `tests/test_research_integration.py`。
