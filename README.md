# agent-gpu-skills

GPU 开发、性能分析、框架运行时、科研与写作、工程审查等 Agent Skill 集合，兼容
Cursor、Claude Code、Codex、Gemini CLI 与 Kimi Code CLI。

本仓库是一个 skill bundle，不是应用。它提供的是“路由契约 + 可执行的 skill 文件”。在一次任务中，
agent 读取本 README，按下面的协议选出一个 skill，读它的 `SKILL.md`，然后完成用户的真实任务。

## 给 Agent 的执行协议（@README 入口）

> **Agent 必读。** 如果用户在请求中提到 `@README`，或附上了本文件，并在其后追加自然语言任务，请按以下步骤执行。
> 不要只回复“已选择某某 skill”就停止。路由只是入口，任务必须完成。

**输入。** 把 `@README` 之后的全部文本视为用户的原始任务，逐字保留，不要改写后再路由。

**步骤 1：检查显式点名。**
如果用户点名了 skill（`$name`、`` `name` ``、`name skill`、`skill: name`，或“用 name skill”），优先使用它。
先在清单 `skills/routers/agent-skill-router/references/inventory.tsv` 中确认该名称存在（`kind=skill` 且 `status` 为
`installed`、`shadowed` 或 `protected`），再按下一节的三层可用性判断它能否使用。点名的 skill 不可用时，告诉用户原因，
**不要替换为其他 skill**。

**步骤 2：区分纯 GPU 请求与其他请求。**

| 请求类型 | 入口 | 说明 |
|---|---|---|
| 纯 GPU：写 CUDA/PTX/Triton/Gluon/CUTLASS/CuTe/GEMM/attention/reduction/fused kernel；kernel 慢、NCU/NSYS、occupancy、stall、SM/TC 利用率、roofline、persistent kernel 性能；框架/运行时、SGLang/vLLM、verl、PyTorch profiler、端到端 launch/scheduling；LLM serving、KV cache、attention backend、batching、TTFT/TBT、容量规划、部署 | 先读 `gpu-development-catchall`，再选窄路由 | 见下方窄路由表 |
| 非 GPU、混合领域或不确定：论文与文献、中文技术写作、去 AI 味写作、通用代码审查与设计、TDD、仓库审计、微信数据、逆向分析、NVIDIA 垂直 SDK（DeepStream、DOCA、BioNeMo、Earth2、cuOpt 等）、以及你无法判断领域的请求 | `agent-skill-router` | 运行下面的路由脚本 |

纯 GPU 请求的窄路由：

- 写或改 kernel（CUDA、PTX、Triton、Gluon、CUTLASS、CuTe、GEMM、attention、reduction、fused kernel）→ `gpu-kernel-authoring-router`
- kernel 慢、NCU/NSYS 证据、occupancy、warp stall、SM/TC 利用率、roofline、persistent kernel 性能 → `gpu-performance-router`
- 框架或运行时行为、SGLang/vLLM、verl、PyTorch profiler、端到端 launch/scheduling 分析 → `gpu-framework-router`
- LLM serving、KV cache、attention backend、batching、TTFT/TBT、容量规划、部署 → `llm-serving-router`

`gpu-development-catchall` 只用于 GPU 请求；它不处理论文、写作、代码审查或其他非 GPU 任务。

**步骤 3：非 GPU、混合或不确定请求，运行路由脚本。**

```bash
python3 skills/routers/agent-skill-router/scripts/route_skill.py '<用户的原始任务>'
```

- 任务文本用**单引号**包住，避免 `$name` 被 shell 展开。若任务本身含单引号，改用 `"..."` 并转义 `$` 与反引号。
- 如果用户附上了真实存在的文件，把它作为制品传入。制品比关键词更有力：

  ```bash
  python3 skills/routers/agent-skill-router/scripts/route_skill.py '读这篇论文并总结方法' --artifact papers/attention.pdf
  ```

  只传真实存在的路径，不要编造路径。
- 需要程序化解析结果时加 `--json`。
- 需要检查某个安装目录中是否已有副本时，加 `--skills-root <目录>`（可重复）。例如 `--skills-root ~/.codex/skills`。
  不传时输出 `skill_copy=not-checked`，表示没有检查。
- 如果你在已安装的 skill 目录里运行，把脚本路径换成 `<安装目录>/agent-skill-router/scripts/route_skill.py`。

脚本输出一个主 skill、至多两个伴随 skill、可能的澄清问题，以及各候选的三层可用性标签（见步骤 4）。

**步骤 4：检查三层可用性。** 路由结果里每个候选都分三层报告，三层互不替代：

| 层 | 字段 | 含义 | 不满足时怎么办 |
|---|---|---|---|
| 来源 | `source=installable` / `absent` / `user-level` | 清单认为它来自已检出的来源，可以安装；`absent` 表示清单中没有；`user-level` 表示用户级 skill，本仓库不管理 | `absent`：按 `requires` 说明缺口，不替换。`user-level`：使用用户已有的副本，不安装、不覆盖 |
| 副本 | `skill_copy=found` / `not-found` / `not-checked` | 指定的 skill 目录中是否已有该 skill 的 `SKILL.md`；只有传了 `--skills-root` 才会检查 | `not-found`：来源可安装，但你的 agent 目录里还没有，先按安装步骤安装。`not-checked`：没有检查，读取来源路径即可 |
| 运行时 | `runtime=on-path` / `missing` / `n/a` | 该 skill 需要的命令（如 `wx`、`rea`、`zotero-cli`）是否在 PATH 中 | `missing`：告诉用户 `requires` 中的安装命令；需要时才执行安装；能不依赖该命令完成的部分先完成 |

- 输出中的 `source` 字段是相对仓库根的路径，例如 `repos/cursor-skills/skills/pdf`。读取时打开 `<source>/SKILL.md`。
- 输出中带 `低置信度` 的候选来自清单名称与描述的词面回退，不是规则命中。使用前先确认它确实符合任务。
- 输出中的 `澄清` 行是唯一允许的澄清问题。如果没有 `澄清`，就直接继续。

**步骤 5：决定是否继续。**

- 路由置信度足够（主 skill 为规则命中或制品命中，且 `source` 可用）：继续。
- 证据不足（没有主 skill，或只有低置信度回退）：如果任务很简单，直接解决；否则**最多问一个必要的澄清问题**。
- 点名的 skill 不可用：说明缺口和 `requires`，并停在这一步，等待用户决定，或只完成不依赖它的部分。

**步骤 6：读取 skill，然后完成任务。**

1. 完整读取主 skill 的 `SKILL.md`（使用 `source` 路径）。在读完之前，不要给出细节建议或代码。
2. 伴随 skill 最多两个。只在需要时读取，而且要在主 skill 之后读。
3. 按主 skill 的说明执行用户的真实任务，直到得到结果。
4. 如果主 skill 指向下游 skill，继续读取它的 `SKILL.md` 并执行。
5. 写作类任务：领域 skill 负责事实与证据，`tech-doc-style-chinese` 负责结构与排版（见 `AGENTS.md`）。

**步骤 7：遵守授权边界。**

- 微信（`wx-cli`）：只读取用户本次请求明确涉及的本机数据。用户已直接要求的普通查询不需要再次确认。发送、修改、删除消息，或超出请求范围的读取，需要用户本次明确指令。不要执行 `sudo wx init`、codesign、`tccutil reset` 或重启微信；这些是用户自行授权的初始化步骤。
- Zotero 文库与论文资料属于用户自有数据：可以读取与检索，未经明确要求不要写入、删除或重新导入。
- REA：只做静态分析与用户请求的证据收集。不要安装 Hopper、Ghidra、IDA 或 Frida 相关工具，除非用户明确授权。
- 任何 skill 都不能授权你读取凭据或密钥；需要账号认证时，告诉用户应当执行的命令。

**如果本 README 没有被加载。** 不同宿主对 `@` 文件引用的支持不同，本 README 不能保证每个宿主都以同样的语法加载它。
此时请让用户使用下面的纯文本提示：

```text
请先阅读仓库根目录的 README.md，按其中“给 Agent 的执行协议（@README 入口）”一节执行。
用户任务：<在这里写用户的原始请求>
```

## 三层可用性与 skill 名称的含义

- **来源可安装**（`source=installable`）：清单中的 skill 来自已检出的 submodule、外部来源或本仓库的本地目录。它说明“安装器能提供它”，**不等于它已经在你的 agent 目录里**。
- **副本存在**（`skill_copy=found`）：只有传入 `--skills-root` 并且目录中确实存在 `SKILL.md`，才能确认。
- **运行时可用**（`runtime=on-path`）：只说明命令在 PATH 中，不说明它已经完成账号认证或数据授权。

清单、路由结果和本 README 都不证明“宿主会自动选中这个 skill”。自动选择的行为没有在每个宿主上测量。

## 快速开始

1. 克隆并安装（Codex 示例）：

   ```bash
   git clone --recursive git@github.com:OpenDarrenlu/agent-gpu-skills.git
   cd agent-gpu-skills
   bash bootstrap.sh --agent codex
   ```

   首次安装会更新所有仓库，耗时较长。之后补齐缺失项、不拉取已有仓库时，使用 `bash bootstrap.sh --agent codex --no-update`。
2. 确认路由清单可用：

   ```bash
   python3 skills/routers/agent-skill-router/scripts/build_inventory.py --check
   ```

3. 在 agent 中使用 `@README` 发起任务，例如：

   ```text
   @README 帮我检索关于GPU调度的相关论文，并整理成文献综述提纲
   ```

4. 如果宿主没有加载 README，使用上面的纯文本提示。

## 安装

`bootstrap.sh` 依次完成：初始化缺失的 submodule → 更新仓库（`--no-update` 时只补齐缺失项）→
获取 `sources/external-sources.conf` 中的外部来源并安装其运行时 CLI → 安装 skills → 为 Codex 注册 REA MCP →
重新生成路由清单。任何已请求的组件缺失或安装失败时以非零退出，并列出缺失项；不会报告“安装完成”。

```bash
# Codex
bash bootstrap.sh --agent codex

# 其他 agent
bash bootstrap.sh --agent cursor
bash bootstrap.sh --agent claude
bash bootstrap.sh --agent gemini
bash bootstrap.sh --agent kimi
```

只安装本地与专项 skills，并跳过大型外部 catalog：

```bash
bash bootstrap.sh --agent codex \
  --no-nvidia-skills --no-amd-skills --no-cursor-skills --no-ccfa-skills --no-veloq
```

跳过外部来源（sepia、wx-cli、rea、mattpocock/skills）及其运行时和 REA MCP 注册：

```bash
bash bootstrap.sh --agent codex --no-integrations
```

隔离验证（只写指定目录；不运行 VeloQ、不安装运行时 CLI、不 fetch、不写真实 HOME，不修改受 git 跟踪的路由清单）：

```bash
bash install.sh --agent codex --dest "$TMPDIR/agent-gpu-skills-test" --no-veloq \
  --no-nvidia-skills --no-amd-skills --no-cursor-skills --no-ccfa-skills
```

请求的外部来源已固定到 commit 并写入 [`sources/external-sources.conf`](sources/external-sources.conf)。
其中 wx-cli 与 rea 需要 `npm` 全局安装的运行时 CLI，`bootstrap.sh` 会执行这一步；安装器为它们在
`$HOME/.local/bin` 生成 managed launcher，使 `wx`、`rea` 不依赖 nvm 前缀在 PATH 中。
Node 版本、launcher 边界与 REA MCP 注册的细节见 [`INSTALL.md`](INSTALL.md#外部来源)。

### 首次机器的前提

- 必需：Git、Bash（系统 Bash 3.2 可用）、Python 3（路由清单与测试）。
- 使用 npm 运行时（wx-cli、rea）时：需要 Node/npm，并且 PATH 中的 node 满足 `rea-agents@6.7.0` 已发布包的 engines
  （`^22.19.0 || ^24.11.0 || >=26.0.0`）。不满足时安装器以 BLOCKED 退出，并给出指引；它不会修改全局 Node。
- 不保证在空白机器上直接可用：请先安装上述工具。

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
# 只更新外部来源（sources/external-sources.conf）与其运行时 CLI
bash update-repos.sh external
# 文档单独刷新，耗时较长；提交前检查索引与抓取完整性
bash update-repos.sh docs
```

路由清单与 skill 清单变化后，重新生成并校验（`bootstrap.sh` 在真实安装的最后一步会自动重新生成）：

```bash
python3 skills/routers/agent-skill-router/scripts/build_inventory.py
python3 skills/routers/agent-skill-router/scripts/build_inventory.py --check
```

## 验证

1. 语法与测试（仓库内的回归测试，不访问网络，不写真实 HOME）：

   ```bash
   bash -n bootstrap.sh install.sh update-repos.sh install-veloq.sh install-external.sh install-rea-mcp.sh scripts/node-preflight.sh
   python3 -m unittest discover -s tests -p 'test_*.py' -v
   ```

2. 清单一致性：

   ```bash
   python3 skills/routers/agent-skill-router/scripts/build_inventory.py --check
   ```

3. 路由样例（应分别选中对应 skill）：

   ```bash
   python3 skills/routers/agent-skill-router/scripts/route_skill.py '润色这篇中文技术原理报告，讲 kernel 设计'
   python3 skills/routers/agent-skill-router/scripts/route_skill.py '审计一下这个仓库的风险代码和慢路径'
   python3 skills/routers/agent-skill-router/scripts/route_skill.py '用 $tdd skill 给这个函数补测试'
   ```

4. 安装后，在默认 PATH 下检查 npm 运行时的稳定入口：

   ```bash
   wx --version
   rea --version
   ```

   这两个命令只检查 CLI 本身，不代表已完成 wx 的数据授权或 REA 的 MCP 注册。

测试只使用临时目录与本地 git 仓库；它们不访问网络，也不写入真实 HOME。
测试结果不能替代真实安装检查。

## 故障排查

| 现象 | 处理 |
|---|---|
| `@README` 没有被加载 | 使用上面的纯文本提示，让 agent 先读取 `README.md` 的执行协议一节。不要假设宿主一定支持 `@` 语法。 |
| 路由没有主 skill（`主 skill: 无`） | GPU 请求改用窄路由（见步骤 2）。非 GPU 请求补充真实制品路径、使用更具体的术语，或直接点名 skill（`$name`）。仍然没有证据时，请 agent 直接回答或只追问一个问题。 |
| 只有低置信度回退 | 不要直接执行。确认候选是否符合任务，或点名 skill。 |
| 来源可安装，但 `runtime=missing` | 按 `requires` 中的命令安装运行时。随后检查 `wx --version` 或 `rea --version`。如果命令仍不可见，检查 `~/.local/bin` 是否在 PATH 中；bootstrap 只会在自身进程中前置它，不修改全局 PATH。 |
| `skill_copy=not-found` | 来源存在，但 agent 目录中没有副本。重新运行 `bootstrap.sh`（或 `install.sh`）后再检查。 |
| `source=absent` 或点名的 skill 不在清单中 | 清单中没有它。检查拼写，或搜索清单：`grep -i '关键词' skills/routers/agent-skill-router/references/inventory.tsv`。 |
| 外部 catalog 已更新，路由结果不一致 | 运行 `build_inventory.py` 重新生成，再运行 `--check`。`--check` 失败说明清单与当前 checkout 不一致。 |
| `claude-delegate`、`codex-background` 相关 | 这两个是受保护名。安装器不会替换已有的用户级副本。macOS 上若已配置补丁版，保持现有 MCP 注册，不要按上游 bootstrap 重新注册（见下文）。 |

## 域地图

下表给出按领域的入口与代表 skill。完整清单见 [`skills/routers/agent-skill-router/references/inventory.tsv`](skills/routers/agent-skill-router/references/inventory.tsv)（约 550 个可安装名称，加上 2 个受保护名称）。

| 领域 | @README 入口 | 代表 skill | 来源 |
|---|---|---|---|
| GPU kernel 编写 | `gpu-kernel-authoring-router` | `triton-skill`、`cuda-skill`、`cutlass-skill`、`deepgemm-skill`、`writing-tilelang-kernels` | 本地（TileLang 来自 `repos/tilelang-cuda-skills`） |
| GPU 性能与 profiling | `gpu-performance-router` | `ncu-report-skill`（`.ncu-rep`）、`perf-nsight-systems`（`.nsys-rep`）、`persistent-kernel-scheduling`、`nv-gpu-kernel-performance-modeling` | 本地；`perf-nsight-systems` 来自 `repos/cursor-skills` |
| 框架与运行时 | `gpu-framework-router` | `sglang-skill`、`verl-skill`、`train-inference-consistency` | 本地 |
| LLM serving | `llm-serving-router` | `llm-serving-auto-benchmark`、`llm-serving-capacity-planner` | 本地路由；两个 benchmark 与容量 skill 来自 `repos/cursor-skills` |
| 通信与 AMD | `gpu-development-catchall` → 对应路由 | `nccl-skill`、`gpu-communication-libraries`、`amd-gpu-docs`、`amd-instinct-isa` | 本地 |
| 论文与文献 | `agent-skill-router` | `ccf-literature-searcher`、`ccf-paper-reviewer`、`ccf-rebuttal-writer`、`ccf-submission-checker`、`zotero-cli`；Supervisor 具名 skill（如 `idea-evaluator`、`deep-research`、`pre-submission-reviewer`） | `ccf-*` 来自 `repos/ccfa-skills`；Supervisor 来自 `repos/supervisor-skills`；`zotero-cli` 为本地。完整对照见 [`skills/research/README.md`](skills/research/README.md) |
| PDF 文件处理 | `agent-skill-router` | `pdf` | `repos/cursor-skills`（通用 PDF 处理，不是论文专用） |
| 中文技术写作 | `agent-skill-router` | `tech-doc-style-chinese`（结构与排版）+ 领域 skill（事实） | 本地定制版 |
| 去 AI 味写作 | `agent-skill-router` | `sepia`、`sepia-write`、`sepia-review`、`sepia-refactor`、`sepia-recreate`、`sepia-hemingway` | `repos/sepia`（MIT）；操作入口需同级安装 |
| 通用工程与审查 | `agent-skill-router` | `tdd`、`code-review`、`diagnosing-bugs`、`to-tickets`、`triage`、`grill-me`（mattpocock） | `repos/mattpocock-skills`；`tdd`、`handoff` 等与 cursor 同名的由 cursor 版本占用 |
| 最小实现与审计 | `agent-skill-router` | `ponytail-review`、`ponytail-audit` | `repos/ponytail` |
| 逆向分析 | `agent-skill-router` | `reverse-engineer-anything` | `rea-agents@6.7.0` npm 包（`runtime-skills`），需要 `rea` 命令 |
| 微信数据 | `agent-skill-router` | `wx-cli` | `Steady775/wx-cli` 的 skill 文本 + npm 包 `@jackwener/wx-cli@0.3.0`，需要 `wx` 命令与用户授权 |
| NVIDIA 垂直 SDK | `agent-skill-router` | `deepstream-generate-pipeline`、`deepstream-dev`、`doca-dma`、`bionemo-evo2-nim`、`earth2studio-deterministic-forecast`、`cuopt-numerical-optimization-api` | `repos/nvidia-skills` |
| 委托给 Claude Code | `agent-skill-router` 或直接点名 | `claude-delegate`（Codex 另需 `codex-background`） | 本地，见下文「Claude 委托」 |

Agent Reach 不在本仓库的清单中，本 bundle 也没有提供通用网页抓取 skill。对于“用 agent reach 抓取网页”这类请求，
路由会返回无证据；请使用宿主自身的工具，不要伪造 skill。

## 可复制的 @README 示例

以下示例可以直接粘贴到 agent 中。`@README` 之后的文本就是原始任务。

**CUDA 或 Triton kernel 编写**（GPU 请求 → catchall → kernel 路由）

```text
@README 用 Triton 写一个 fused softmax kernel，要求数值与 torch.softmax 对齐，并给出测试
```

```text
@README 把这个 CUDA kernel 改成支持 FP8 输入，保持 PTX 检查通过
```

**NCU / NSYS 性能分析（带真实制品）**

```text
@README 分析这个 ncu 报告为什么 kernel 慢，制品是 <你的 .ncu-rep 路径>
```

```text
@README 这个 nsys 时间线里 launch 间隔很大，先定位瓶颈再给出 profiling 下一步。文件：<你的 .nsys-rep 路径>
```

> `<…>` 是占位符，请替换为真实存在的文件路径。如果只有症状没有 profile，请让 agent 先给出最小 profiling 步骤。

**SGLang 或 LLM serving**

```text
@README SGLang 部署的 KV cache 吞吐太低，帮我看 serving 设计和 benchmark 方案
```

```text
@README 估算 Qwen 服务在 4 张 GPU 上的容量规划，目标是 TTFT 小于 500ms
```

**文献检索、Zotero 与审稿**

```text
@README 帮我检索关于GPU调度的相关论文，并整理成文献综述提纲
```

```text
@README 我的 Zotero 文库里找 attention 相关的文献，列出标题和年份
```

```text
@README 帮我 review 这篇投稿的论文草稿，只给出审稿意见，不改写
```

**中文技术文章写作**

```text
@README 润色这篇中文技术原理报告，讲 kernel 设计，保持技术事实不变。以下是正文：……
```

**TDD、代码审查与仓库审计**

```text
@README 用 TDD 给这个函数补测试，先写失败用例
```

```text
@README review 这个 PR 的代码改动，重点看正确性和测试覆盖
```

```text
@README 审计一下这个仓库的风险代码和慢路径，给出排序后的报告
```

**Sepia（去 AI 味写作）**

```text
@README 把这段文字去掉 AI 味，像人写的，只做最小改动。以下是原文：……
```

**REA 逆向分析**

```text
@README 帮我逆向分析这个 Electron 应用的 asar 包，给出功能证据。文件：<你的 .asar 路径>
```

> REA 需要 `rea` 命令可用。如果 `runtime=missing`，agent 会给出安装命令，并停在这一步等待你的决定。

**微信（wx-cli）**

```text
@README 帮我总结一下我微信群昨天的聊天记录，我已经授权你读取本机数据，只读取该群
```

本示例的边界：

- 读取的数据属于你本机的微信数据。使用前，你必须已经在本机完成 `wx init` 等初始化步骤；这些步骤需要你自己执行，安装器不会代劳。
- 本示例只查询；不会发送、修改或删除消息。
- 如果请求超出你指定的群或时间范围，agent 应当先询问，而不是扩大读取范围。

**NVIDIA 垂直 SDK 请求**

```text
@README 用 deepstream 搭一个视频分析 pipeline，输入是一路 RTSP 视频
```

```text
@README 在 BlueField DPU 上配置 DOCA DMA，给出最小示例
```

```text
@README 用 Evo 2 NIM 生成一段 DNA 序列，并说明输入参数
```

```text
@README 用 earth2studio 跑一个天气预报 forecast，使用默认数据源
```

```text
@README 用 cuOpt 求解一个 MILP 混合整数规划，给出建模与求解代码
```

**显式点名 skill（覆盖路由）**

```text
@README 用 $tdd skill 给这个函数补测试
```

```text
@README 用 $deepstream-dev 帮我写一个 pyservicemaker 程序
```

显式点名会优先于路由。它只能点名清单中存在的名称；不存在或不可用时，agent 必须报告，而不是替换。

## 受保护的用户级 skill 与外部 catalog

**受保护名 `claude-delegate`、`codex-background`。** 这两个 skill 由本仓库的 `skills/other/` 提供，取自上游
[`amElnagdy/delegate-skills`](https://github.com/amElnagdy/delegate-skills) commit `3aa72d681a79e2a9d09bd9a316353003ae1cb4a7`（MIT，见各目录的 `LICENSE`）。
安装器把它们视为受保护名：

- 已存在的目标（包括指向本仓库之外的 symlink）原样保留，不替换。
- 目标缺失且 `~/.agents/skills/<name>/SKILL.md` 有效时，只创建指向它的别名，不改动原链接。
- `--dest` 隔离安装不读取真实 HOME；目标缺失时复制仓库内的版本。
- 任何外部 catalog 都不能占用这两个名称。`codex-background` 只为 `--agent codex` 安装。

**外部 catalog 与来源。** 外部来源位于 `repos/`，由 `sources/external-sources.conf` 固定版本：

| 来源 | 固定版本 | 说明 |
|---|---|---|
| NVIDIA / AMD / Cursor / CCFA / TileLang / Ponytail | submodule（见 `.gitmodules`） | 通过 `update-repos.sh` 更新 |
| Supervisor-Skills | `a20d76e`（CC BY-NC-SA 4.0；`drawio-reconstruction` 另为 MIT） | 论文研究 skill；`repos/supervisor-skills`，由 `sources/external-sources.conf` 固定，见 [`INSTALL.md`](INSTALL.md#supervisor-skills-论文研究-skills) |
| sepia | `fb06be76…`（MIT） | 去 AI 味写作；需要同级安装 |
| wx-cli | `08af8945…`（Apache-2.0，Steady775 fork） | skill 文本来自 fork；`wx` 二进制来自 npm 包 `@jackwener/wx-cli@0.3.0` |
| rea | `3ce31175…`（MIT，源码 6.7.0） | skill 来自 npm 包 `rea-agents@6.7.0`，不是源码构建 |
| mattpocock/skills | `5beb35b0…`（MIT，1.3.1） | 与 cursor-skills 同名的 6 个由 cursor 版本占用 |

## macOS 与已有用户级副本

- 上游 `codex-background` 的 bootstrap 仅支持 Linux 与 Windows。macOS 上若已配置补丁版用户级副本，保持现有 MCP 注册即可，不要按上文命令替换。
- 安装器在 macOS 上只打印这一提示，不自动注册，也不改写 Codex 的 MCP 配置。
- REA 的 Codex MCP 注册由 `install-rea-mcp.sh` 执行，它使用 REA 自身的备份与保留逻辑；`--no-rea-mcp` 可跳过。

## Claude 委托

`claude-delegate` 把任务交给本机的 Claude Code CLI 执行；在 Codex 中它依赖支撑 skill `codex-background`
提供的后台 MCP server。

前置条件：

- Node.js 18+；Linux 下 `codex-background` 还需要 `/usr/bin/cc` 和 procfs。
- Claude Code CLI：`claude` 在 `PATH` 上，并已完成登录认证。Linux 下执行 Claude 的受限 shell 命令还需要 `bubblewrap` 和 `socat`。

安装器为所有 agent 安装 `claude-delegate`，`codex-background` 只在 `--agent codex` 时安装。注册 MCP server 会写入全局 Codex 配置，
所以安装器不会自动执行这一步。Linux 或 Windows 上的 Codex 用户在安装后可手动运行，然后重启 Codex：

```bash
bash install.sh --agent codex
node ~/.codex/skills/codex-background/scripts/bootstrap.mjs
```

使用 `--dest DIR` 时，脚本路径换成 `DIR/codex-background/scripts/bootstrap.mjs`。默认混合模式下，已安装 skill 的 `scripts/`
是指向本仓库的软链接；移动或删除本仓库后，需要重新安装并重新运行 bootstrap。

## TileLang Skills

[`sablin39/tilelang-cuda-skills`](https://github.com/sablin39/tilelang-cuda-skills) 已作为 `repos/tilelang-cuda-skills`
submodule 接入。安装器默认安装其中的 TileLang 工作流：

- `tilelang-optimization-loop`
- `writing-tilelang-kernels`
- `debugging-tilelang-programs`
- `optimizing-tilelang-programs`
- `profiling-tilelang-programs`
- `torch-profiling-tilelang-programs`
- `testing-fwd-bwd-kernels`
- `writing-tilelang-design-docs`
- `designing-with-tilefoundry`

上游的 `cuda_skill` 不重复安装；CUDA 基础工作流统一使用本仓库更完整的 `cuda-skill`。如需跳过这组 skills：

```bash
bash install.sh --no-tilelang-skills
```

TileLang 相关的通用动作词（debug、pipeline、layout、baseline 等）单独出现时不会选中 TileLang skill；请在请求中写明 TileLang。

## 许可边界

AMD ISA PDF、AMD 文档缓存和按需下载的上游源码不会提交到仓库。AMD 文档和 ISA 只在用户明确接受官方文档条款后下载到 git-ignored 本地目录。

## 研究笔记与文献工具

[verl 课程笔记](notes/verl-course/README.md) 收录按源码快照编写的离线 HTML 笔记。
文献检索与 Zotero 阅读使用 `skills/other/zotero-cli`，随本地 skills 一起安装；使用前运行 `zotero-cli config` 检查本机连接。
Agent 的安装副本和本机 MCP 配置不提交到仓库。

## 外部来源（sources/）

[`sources/external-sources.conf`](sources/external-sources.conf) 记录不 vendored 的外部来源：checkout 位于 git-ignored 的
`repos/<name>/`，skill 由 `install.sh` 安装，运行时 CLI 由 `install-external.sh` 安装。来源模式为 `skills`（递归发现 `SKILL.md`）或 `none`；
`pending` 只用于尚未检视的来源，真实安装会把它视为失败。当前四个请求来源的 pin、用途与安装方式见 [`INSTALL.md`](INSTALL.md)。

## Ponytail

[Ponytail](https://github.com/DietrichGebert/ponytail) 通过 `repos/ponytail` 子模块接入，安装器安装其 6 个 skill：
ponytail、review、audit、debt、gain、help。完整 Codex 插件及生命周期 hooks 的安装见 [安装指南](INSTALL.md#ponytail)。

## 卸载

仓库没有内置的卸载命令。手动卸载时：

1. 从 agent 的 skill 目录（例如 `~/.codex/skills`）删除清单中 `status=installed` 的同名目录或 symlink。**不要删除** `claude-delegate` 与 `codex-background` 的用户级副本，也不要替换它们。
2. 删除带有 `agent-gpu-skills-managed` 标记的 launcher：`grep -l agent-gpu-skills-managed ~/.local/bin/*` 找到后逐个确认再删除。
3. 需要时卸载 npm 运行时：`npm uninstall -g @jackwener/wx-cli`、`npm uninstall -g rea-agents`。
4. REA 的 MCP 注册请按 REA 的文档处理；本 README 没有验证其卸载命令。
5. 删除仓库本身或 `repos/` 中的 checkout 与卸载 skill 是两件事；后者不会删除你的数据。

## 维护

- 修改 `install.sh` 的 skill 数组、外部来源清单或 checkout 后：重新生成并校验路由清单。
- 路由规则在 `skills/routers/agent-skill-router/references/routing-rules.json`，回归矩阵在 `tests/test_skills_router.py`。出现新的误路由时，先在回归矩阵里加入样例，再修改规则。
- 本 README 的协议与示例由 `tests/test_readme_contract.py` 检查：协议段落存在、示例中的 skill 名称存在于清单，以及 README 中提到的脚本路径与选项确实存在。非 GPU 示例还会经过真实路由脚本，确认选中的正是示例声明的 skill。
- 修改 README 后运行完整测试：`python3 -m unittest discover -s tests -p 'test_*.py' -v`。
- 路由回归与设计决策的记录见 [`docs/skills-usage-review.md`](docs/skills-usage-review.md)。
- 路由协议的完整说明（三层可用性、输出字段、澄清规则）见 [`skills/routers/agent-skill-router/SKILL.md`](skills/routers/agent-skill-router/SKILL.md)。
