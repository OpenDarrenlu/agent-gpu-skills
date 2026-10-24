# Skill 使用审查与高效使用方案

本文基于 2026-10-24 的工作区。四个请求来源已固定 commit 并检视；安装器与路由器已实现并测试。
**本机安装已完成并独立验证**：清单内 552 个普通 skill 全部存在，默认 PATH 下 wx 0.3.0、REA 6.7.0 可运行；REA MCP 握手成功。
所有 skill 名都已对照 `skills/routers/agent-skill-router/references/inventory.tsv` 核对；§5.3 的结果来自本轮对路由脚本的实际重跑。

## 1. 结论

- 路由器的自动规则覆盖了主要 GPU、论文、写作、工程与四个来源的领域。规则带有领域门控（`requires_any`）：TileLang 等规则只有出现领域词时才计分，通用动作词（debug、pipeline、layout、baseline、设计文档、grad、优化）不能单独选中领域 skill。
- 显式点名（清单中任意名称，包括没有规则的 skill）与真实制品（`.ncu-rep` 等）可以绕过门控，优先级不变。
- 本轮修正的误路由：CUDA 文献检索（现在选中 `ccf-literature-searcher`）、仓库审计（现在选中 `ponytail-audit`）、DeepStream 视频 pipeline（现在选中 `deepstream-generate-pipeline`，不再进入 TileLang），以及 DOCA、BioNeMo、Earth2、cuOpt 的产品词锚点。
- 仍然存在的已测量限制见 §5.6：清单中没有 Agent Reach；通用服务设计文档现在没有证据（这是有意的，不是误路由）；没有规则的领域只能靠清单回退，且标注为低置信度。
- 宿主是否自动选中这些 skill，**没有测量**。下文所有“自动”结论都只指脚本的输出。
- 安装清单约 136 KB，552 个安装名。路由器减少的是选中后 skill 正文的加载，不减少宿主对描述的索引负担（§5.4）。

## 2. 四个请求来源

| 来源 | 检出 commit | 安装模式与运行时 | 许可 | 用途与边界 |
|---|---|---|---|---|
| sepia | `fb06be76abeae7dd606960a32f2b10c1d1798e0d` | `skills`，目录 `skills/`；无运行时 | MIT | 去 AI 味写作。总路由 `sepia`，操作入口 `sepia-write`、`sepia-review`、`sepia-refactor`、`sepia-recreate`、`sepia-hemingway` 按 `../sepia/SKILL.md` 读取，需同级安装 |
| wx-cli | `08af894594b4afd468e23e17dbd783f15403f13b`（Steady775 fork） | `skills`，根目录 SKILL.md（`name: wx-cli`）；运行时 `npm:@jackwener/wx-cli@0.3.0:wx` | Apache-2.0 | 本地微信数据只读 CLI。SKILL.md 来自 fork，二进制来自 npm 官方包（jackwener 上游）。真实聊天数据访问需要用户执行 SKILL.md 中的 `wx init`；安装器不执行 `wx init`、sudo、codesign、tccutil 或重启微信 |
| rea | `3ce311757dd8ac444c8297b2871464536f93816d`（源码 6.7.0） | `skills`，目录 `runtime-skills`；运行时 `npm:rea-agents@6.7.0:rea` | MIT | 逆向工程 CLI 与 skill `reverse-engineer-anything`。skill 取自 npm 包，经 `runtime-skills` 链接。Codex MCP 由 `install-rea-mcp.sh` 注册。原生 provider（Hopper、Ghidra、IDA）与 Frida 类安装脚本未验证，不在安装范围内 |
| mattpocock-skills | `5beb35b040e3de97e54fa9e386d7e42d15c295c2`（1.3.1） | `skills`，`engineering`、`productivity`、`misc`；无运行时；不装 `in-progress` | MIT | 工程流程 skill。与 cursor-skills 改写版同名的 6 个（`tdd`、`grill-with-docs`、`improve-codebase-architecture`、`setup-matt-pocock-skills`、`prototype`、`handoff`）由 cursor-skills 占用（清单中为 `shadowed`），保留你的推理触发词改写 |

## 3. 清单覆盖

`inventory.tsv`：576 行，552 个唯一安装名。

| catalog | 安装名 | 备注 |
|---|---|---|
| local（`install.sh` SKILL_DIRS） | 45 | 含 ponytail ×6、TileLang ×9、`agent-skill-router` |
| kernel-design-agents（KernelWiki） | 1 | |
| nvidia-skills | 406 | 含 BioNeMo、DOCA、DeepStream、Earth2Studio、cuOpt 等家族；`nvidia-skill-finder` 另有一份被 `shadowed` |
| cursor-skills | 39 | 含 `perf-nsight-systems`、`llm-serving-auto-benchmark`、`pdf`、`docx`、`pptx`、`review` 等 |
| ccfa-skills | 17 | 另有 `ccf-latex-templates` 支撑目录 |
| amd-skills | 11 | |
| sepia | 6 | |
| wx-cli | 1 | |
| mattpocock-skills | 25 | 另有 6 个 `shadowed` |
| rea | 1 | `reverse-engineer-anything`，来自 `runtime-skills` |

受保护名 `claude-delegate` 与 `codex-background` 由仓库内的 bundled 本地条目提供（`skills/other/`，取自上游 commit `3aa72d6…`），清单中标为 `protected`，不计入可安装的 552 个。真实安装时：已存在的目标原样保留；目标缺失且 `~/.agents/skills/<name>` 有效时只创建别名；`--dest` 与无用户级副本时复制 bundled 版本；任何 catalog 都不能占用这两个名称。`codex-background` 只为 `--agent codex` 安装。

## 4. 本轮路由修正

| 问题 | 修正 | 位置 |
|---|---|---|
| 通用动作词单独选中 TileLang 规则（pipeline、layout、debug、baseline、设计文档、grad） | 规则可声明 `requires_any`，文本中没有任何领域词时规则不计分；显式点名与强制制品绕过门控 | `scripts/route_skill.py` 的 `score_rule`；`routing-rules.json` 中 TileLang 规则、`tech-doc-style-chinese`、`gpu-framework-router`、`persistent-kernel-*`、`nv-gpu-kernel-performance-modeling`、`deepstream-*`、`earth2studio-*`、`cuopt-*` |
| CUDA 文献检索被 `persistent-kernel-scheduling` 的“调度”捕获 | 调度规则需要 `persistent`、`stream-k`、`clc`、`tail effect` 之一；文献检索规则的意图短语权重为 20 | `routing-rules.json` |
| 仓库审计未命中 | `ponytail-audit` 增加审计意图短语（审计、风险代码、慢路径、audit this codebase、find bloat 等），排除论文审稿词 | `routing-rules.json` |
| DeepStream 视频 pipeline 误入 TileLang | 生成 pipeline 的 skill 为 `deepstream-generate-pipeline`（其 SKILL 描述即“构建 DeepStream GStreamer pipeline”）；`deepstream-dev` 是 pyservicemaker Python API 开发，两者边界不同 | `routing-rules.json` |
| DOCA、BioNeMo、Earth2、cuOpt 的产品词落入低置信度回退 | 为具体产品词加精确锚点（如“doca dma”→`doca-dma`，“evo 2”→`bionemo-evo2-nim`，“earth2studio”→`earth2studio-deterministic-forecast`，“cuopt”/“milp”→`cuopt-numerical-optimization-api`） | `routing-rules.json` |
| 澄清问题在分数明显更弱的候选下也出现 | 只有当 runner-up 至少为主候选的 3/4 时才提问；同分仍提问 | `scripts/route_skill.py` 的 `route` |
| 清单回退被通用工程词命中（如 `tao-run-automl-deft-pipeline`） | 把 pipeline、baseline、layout、design、data、service 等列入回退的停用词 | `scripts/route_skill.py` 的 `STOPWORDS` |

## 5. 高效使用方案

### 5.1 原则

1. **单一入口。** 非 GPU 或跨域任务进入 `agent-skill-router`；纯 GPU 任务进入 `gpu-development-catchall` 或其窄路由。
2. **一个主 skill，至多两个伴随 skill。** 伴随 skill 需要至少两条独立证据；通用词不构成伴随。
3. **渐进加载。** 路由先给元数据（名称、不超过 140 字符的描述、三层可用性标签）；选中后只打开主 skill 的 `SKILL.md`，必要时再打开两个伴随 skill。不要把清单或规则文件整体读入上下文。
4. **显式点名覆盖一切。** 用户写 `$name`、`` `name` ``、`name skill` 或 `skill: name` 时，清单中任意名称都直接使用；未安装则报告 `absent`，不替换。
5. **确定性信号优先。** 顺序：显式点名 → 真实制品（`.ncu-rep`、`.nsys-rep`、`.pdf`、`.docx`、`.bib`、`.cu`）→ 带领域门控的规则短语 → 清单名称与描述的低置信度回退 → 一个澄清问题或无证据。

### 5.2 决策与命令

```bash
python3 skills/routers/agent-skill-router/scripts/route_skill.py '用 $ccf-literature-searcher 检索 CUDA 调度相关论文' --json
python3 skills/routers/agent-skill-router/scripts/route_skill.py '分析这个报告为什么慢' --artifact prof/run.ncu-rep
python3 skills/routers/agent-skill-router/scripts/route_skill.py '用 tdd skill 给这个函数补测试' --skills-root "$HOME/.codex/skills"
```

- 用单引号包住含 `$name` 的请求，避免 shell 展开。
- `--json` 用于程序解析；文本输出适合人读。
- `--skills-root` 只判断 `<目录>/<name>/SKILL.md` 是否存在。不传时显示 `skill_copy=not-checked`。本次只对临时目录验证过 `not-found` 的输出，没有对 `~/.codex/skills` 执行。
- 三层标签互不替代：`source`（清单是否可安装）、`skill_copy`（副本是否存在）、`runtime`（`runtime_bin` 是否在 PATH 中）。

### 5.3 代表性工作流（本轮重跑结果）

“路由来源”列区分：**规则**（自动，短语或门控规则命中）、**制品**（真实制品）、**显式**（用户点名）、**回退**（清单元数据低置信度回退）、**无证据**（没有候选，不给出结论）。

| 工作流 | 请求 | 路由结果（主 skill） | 路由来源 | 伴随 / 澄清 | 建议的 skill 组合（已核对） |
|---|---|---|---|---|---|
| GPU kernel 写作 | 用 triton 写一个 fused softmax kernel | `triton-skill` | 规则 | 无 | triton-skill；需要 CUTLASS 时加 cutlass-skill |
| 性能剖析 | 分析这个 ncu 报告为什么慢 `--artifact prof/run.ncu-rep` | `ncu-report-skill` | 制品 | 无 | ncu-report-skill；nsys 时间线用 perf-nsight-systems |
| 框架 serving | SGLang 的 KV cache 吞吐太低，帮我看 serving 设计 | `llm-serving-router` | 规则 | 无澄清（sglang-skill 得分 12，低于 3/4 阈值） | llm-serving-router；实现细节用 sglang-skill；benchmark 用 llm-serving-auto-benchmark |
| 文献检索（CUDA 主题） | 帮我做相关工作的文献检索，找 CUDA 调度的论文 | `ccf-literature-searcher` | 规则（意图权重 20） | 无伴随、无澄清 | ccf-literature-searcher；论文内容的 CUDA 事实由 cuda-skill 或 persistent-kernel-scheduling 提供，由用户明确要求时再加 |
| Zotero 文库 | 我的 Zotero 文库里找 attention 的文献 | `zotero-cli` | 规则（runtime 在本 shell 为 on-path） | 无 | zotero-cli；需要 `zotero-cli config` 检查连接 |
| 中文技术写作 | 润色这篇中文技术原理报告，讲 kernel 设计 | `tech-doc-style-chinese` | 规则 | 无 | tech-doc-style-chinese 负责结构与排版；事实由领域 skill 提供（§5.5） |
| sepia 去 AI 味 | 把这段文字去掉 AI 味，像人写的 | `sepia` | 规则 | 无 | sepia；操作入口 sepia-write、sepia-review 等需同级安装 |
| Matt 工程流程 | 用 tdd skill 给这个函数补测试 | `tdd` | 显式 | 无 | tdd、to-tickets、triage、grill-me、code-review、diagnosing-bugs；handoff 为 cursor 改写版 |
| 代码审查 | review 这个 PR 的代码改动 | `ponytail-review` | 规则（得分较低，8） | 无 | ponytail-review；PR 正文可加 mattpocock 的 `pr` |
| 仓库审计 | 审计一下这个仓库的风险代码和慢路径 | `ponytail-audit` | 规则 | 无 | ponytail-audit；只读报告，不修改仓库 |
| 微信数据 | 帮我导出我微信群昨天的聊天记录并总结 | `wx-cli` | 规则（本 shell 中 runtime=missing） | 无 | wx-cli；仅在用户明确请求时查询；`wx init` 由用户自行执行 |
| 逆向分析 | 逆向分析这个 Electron 应用的 asar 包 | `reverse-engineer-anything` | 规则（本 shell 中 runtime=missing） | 无 | reverse-engineer-anything；Codex MCP 由安装器注册 |
| 网页抓取（Agent Reach） | 用 agent reach 抓取这个网页的内容 | 无 | 无证据 | 无 | 清单中没有 Agent Reach；使用宿主工具，不要伪造 skill |
| 生物 NIM | 用 Evo 2 NIM 生成一段 DNA 序列 | `bionemo-evo2-nim` | 规则（产品词锚点） | 无 | bionemo-evo2-nim；其他 BioNeMo NIM 按名称显式点名 |
| DOCA 网卡/DPU | 在 BlueField DPU 上配置 DOCA DMA | `doca-dma` | 规则（产品词锚点） | 无 | doca-dma；其他 DOCA 技能按名称显式点名 |
| DeepStream 视频 | 用 deepstream 搭一个视频分析 pipeline | `deepstream-generate-pipeline` | 规则 | 无 | deepstream-generate-pipeline；pyservicemaker 程序开发用 deepstream-dev |
| Earth2 天气预报 | 用 earth2studio 跑一个天气预报 forecast | `earth2studio-deterministic-forecast` | 规则（产品词锚点） | 无 | earth2studio-deterministic-forecast；prognostic 模型包装用 earth2studio-create-prognostic |
| cuOpt 优化 | 用 cuOpt 求解一个 MILP 混合整数规划 | `cuopt-numerical-optimization-api` | 规则（产品词锚点） | 无 | cuopt-numerical-optimization-api；路由用 cuopt-routing-api-python 显式点名 |
| Triton 数值调试 | 帮我调试这个 Triton kernel 的数值误差 | `triton-skill` | 规则 | 无；不出现 TileLang skill | triton-skill |
| 通用服务设计文档 | 帮我设计一个订单服务的数据 pipeline，写设计文档，并做 layout 和 baseline 对比 | 无 | 无证据 | 无 | 直接回答或只问一个澄清问题；不要自动选 GPU 或中文写作 skill |
| CPU 线程调度 | 帮我设计一个 CPU 线程池的任务调度策略 | 无 | 无证据 | 无 | 直接回答；不选 persistent-kernel-scheduling |
| 显式 TileLang（无 TileLang 词） | 用 $writing-tilelang-kernels 帮我看看这个 sample | `writing-tilelang-kernels` | 显式 | 无伴随 | writing-tilelang-kernels |
| 显式 DeepStream | 用 $deepstream-dev 帮我写一个 pyservicemaker 程序 | `deepstream-dev` | 显式 | 无 | deepstream-dev |
| 真实制品 + 通用词 | 这个 kernel 为什么慢，帮我分析一下 pipeline 和 layout 的问题 `--artifact prof/run.ncu-rep` | `ncu-report-skill` | 制品 | 无 TileLang 伴随 | ncu-report-skill |
| 缺少领域词的 TileLang 动作 | 帮我 debug 一下这个 pipeline 的 layout | 无 | 无证据 | 无 | 询问是否为 TileLang；不要自动选 TileLang skill |

以上行与 `tests/test_skills_router.py` 的 `RoutingRegressionMatrixTest` 对应。测试断言的是结果（主 skill、不得出现的主 skill 或伴随 skill、是否提问），不是实现细节。

### 5.4 描述负担与未实现的部分

- `inventory.tsv` 将每条描述截为最多 140 字符，约 136 KB 是清单大小。宿主可能读取完整 frontmatter；实际索引量与上下文成本取决于宿主，不能用清单大小代替。**路由器没有实现宿主的选择性索引**，只减少选中后的 skill 正文加载。
- 未实现：按项目的选择性安装或索引。合理的后续方向是每个项目一个 profile，只列出该项目需要的 catalog（如 GPU 项目列 GPU 与通用工程，不列 BioNeMo、DOCA、Earth2）。profile 尚未设计或实现，本文不把它当作现状。
- 当前安装仍包含全部 catalog。若要减少负担，应通过显式的 profile 或安装参数，而不是在路由中隐藏。
- 本机另有 514 个名称同时出现在 `~/.agents/skills` 与 `~/.codex/skills`，两处 `gpu-development-catchall` 正文不同。宿主是否索引两份、同名时选择哪份尚未测量。建议下一步统一 skill 发现目录，再按 GPU 研究、论文写作、通用工程等项目 profile 启用 catalog；迁移时保留现有委派补丁链接，并验证实际选择结果。当前 router 已减少正文加载，目录统一和 profile 尚未实现。

### 5.5 中文技术写作的分工

按 `AGENTS.md`：领域 skill 提供事实与证据（如 `ncu-report-skill`、`triton-skill`、`cuda-skill`），`tech-doc-style-chinese` 提供结构、证据等级与排版。写作前应先读取被选中的领域 skill 的 `SKILL.md`，再给出技术细节。`tech-doc-style-chinese` 的规则现在要求技术领域词（GPU、kernel、推理等）或明确的文档类型（原理报告、中文技术文章等）；通用“设计文档”不再单独触发它。

### 5.6 仍存在的限制

| 限制 | 已测量的证据 | 处理 |
|---|---|---|
| 清单中没有 Agent Reach，也没有通用网页抓取 skill | “用 agent reach 抓取…”无证据 | 不在本 bundle 内；不要伪造 skill |
| 没有规则的领域只能靠清单回退，结果为低置信度 | 回退只在规则未命中时使用，且需要 ASCII 词或足够的中文双字重合 | 对重要领域补充规则；或显式点名 |
| 通用服务设计、CPU 调度等请求无证据 | 结果为“无证据”，这是有意的：不把它们误判为 GPU 或写作 skill | 由宿主直接回答或追问一个问题 |
| 同一领域的相近 skill 仍需用户选择 | 例如 llm-serving-router 与 sglang-skill；只有分数接近时才提问 | 按请求中的名称显式点名 |
| 规则是人工编写的 | 当前 83 条规则，测试矩阵覆盖的是已知误路由与相邻领域 | 新的误路由应先加入回归矩阵，再改规则 |

## 6. 剩余风险

| 序号 | 严重度 | 内容 | 处理 |
|---|---|---|---|
| R1 | 已解决 | REA 根 symlink 漏装与 npm bin 不在默认 PATH 的问题已修复：本机重装后 552 个安装名全部存在，wx/rea 默认 PATH 可用，REA MCP 握手通过 | `find -H`、稳定 launcher、安装名与清单一致性检查 |
| R2 | 中 | 宿主自动选择行为未测量；路由器只报告脚本输出 | 不作为结论 |
| R3 | 中 | wx-cli 的 skill 文本来自 fork，二进制来自上游 npm 包 | 在清单与 requires 中写明 |
| R4 | 中 | REA skill 来自 npm 包，没有 tarball 完整性校验 | 版本一致性已检查；完整性未检查 |
| R5 | 中 | 路由器 GPU 正文未声明对 cursor-skills 与 VeloQ 的依赖，关闭来源后会静默失效 | 延期；测试只保证当前引用的名称已安装 |
| R6 | 中 | 上游 `nvidia-skill-finder` 描述宽泛，与本地 CUDA 路由竞争 | 上游 submodule，未改；可用 `--no-nvidia-skills` |
| R7 | 低 | 安装器没有 manifest，`rm -rf` 对非本仓库的同名真实目录仍生效 | 延期 |
| R8 | 低 | 默认 agent 仍为 cursor；Codex 需显式 `--agent codex` | 有意保留 |

## 7. 验证

本次运行（最终代码）：
- `bash -n bootstrap.sh install.sh update-repos.sh install-veloq.sh install-external.sh install-rea-mcp.sh scripts/node-preflight.sh`：通过
- `python3 -m unittest discover -s tests -p 'test_*.py' -v`：130 项，全部通过（安装器集成 65 项，路由器 65 项，其中回归矩阵 21 项）。本轮新增：npm launcher 的参数转发、路径含空格与撇号、幂等与不递归、无关文件保留、`--dest` 与 `--no-integrations` 不写 launcher；symlink 源根（REA 形态）的安装、空源与悬空源判为缺失；安装名与清单中已安装名称的一致性门禁（真实 checkout，`--dest`）。
- `python3 skills/routers/agent-skill-router/scripts/build_inventory.py --check`：通过，576 行
- `git diff --check`：通过

Codex 独立重跑的原始路径 gate：`bash install.sh --agent codex --dest /private/tmp/agent-gpu-skills-integrations-gate --no-veloq --no-nvidia-skills --no-amd-skills --no-cursor-skills --no-ccfa-skills`，退出码 0，94 通过、0 失败；包含远端新增的委派 skill 与本机版本保护规则。

Codex 还在隔离的 npm 前缀中（Node v24.22.1）安装了 `@jackwener/wx-cli@0.3.0` 与 `rea-agents@6.7.0`，两者 `--version` 均通过；`rea analyze-javascript-application` 对合成的 JavaScript 夹具完成，退出码 0；`rea setup --client codex --skill=false --dry-run --json` 只计划 Codex 注册，没有应用任何动作。这些是 Codex 的独立结果。

未验证或未测量的内容：
- 宿主自动 skill 选择的准确率。
- 原生 REA provider、Frida 安装脚本（npm 11 对 frida 的可选安装脚本给出过策略警告，未放宽全局 npm 策略）。
- wx 对真实聊天数据的访问（需用户自行 `wx init`）。
- 空白机器的端到端安装。

## 8. 本机安装验证与复用

以下命令已在本机执行并验证；已有源码依赖的机器可复用：

```bash
bash bootstrap.sh --agent codex --no-update
```

若 PATH 中的 node 不满足 rea 的 engines，把兼容版本放在 PATH 前面，例如：

```bash
PATH="$HOME/.nvm/versions/node/v24.22.1/bin:$PATH" bash bootstrap.sh --agent codex --no-update
```

该命令会写入 `~/.codex/skills`、npm 全局前缀、`repos/rea/runtime-skills` 链接，以及受 git 跟踪的 `inventory.tsv`（需要提交）。REA 的 Codex MCP 注册通过 `rea setup --client codex --skill=false --yes --json` 完成，备份与保留其他设置由 REA 负责。安装器不解析或编辑 `~/.codex` 配置，不读取凭据，不执行 wx 的 `init` 与原生 provider 安装。

## 9. 未测量

- 宿主 agent 的自动 skill 选择行为。
- 空白机器的完整安装与宿主自动选中 skill 的准确率。
- 原生 REA provider、Frida、wx 的真实数据访问。

最终本机结果（2026-10-24）：安装器退出码 0、101 项校验通过；独立核对 552/552 个普通安装名存在，四个新来源的副本与源码字节一致。默认 PATH 下 wx 0.3.0、REA 6.7.0 可运行；按 Codex 注册命令完成 MCP initialize 与 tools/list，REA 6.7.0 返回 207 个工具。原用户级委派链接保留；首次注册时独立验证其他 Codex 配置不变，最终重装的 `rea setup` 为 `ready` 且无配置写入动作。没有访问微信聊天数据或启用原生分析引擎。

首次 clone 运行 `bash bootstrap.sh --agent codex`，以恢复其他 GPU skill 的 git-ignored 源码依赖；不要在缺少这些依赖时跳过更新。
