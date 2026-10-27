# 安装指南

## 首次安装

推荐 clone 时直接拉取 submodules：

```bash
git clone --recursive git@github.com:OpenDarrenlu/agent-gpu-skills.git
cd agent-gpu-skills
```

如果已经普通 clone 了，也可以补拉：

```bash
git submodule update --init --recursive
```

推荐一条命令完成外部 repo 获取、submodule 初始化和 skill 安装：

```bash
# Codex
bash bootstrap.sh --agent codex

# Cursor / Claude Code / Gemini CLI / Kimi Code CLI
bash bootstrap.sh --agent cursor
bash bootstrap.sh --agent claude
bash bootstrap.sh --agent gemini
bash bootstrap.sh --agent kimi
```

`bootstrap.sh` 会依次运行：

1. `git submodule update --init --recursive`（总是执行，即使 `--no-update`）
2. `bash update-repos.sh`（`--no-update` 时跳过，改为 `install-external.sh` 只 clone 缺失项，不拉取已有 checkout）
3. 外部来源与运行时 CLI：`bash install-external.sh`（见下文「外部来源」）
4. `bash install.sh --agent <agent>`
5. REA MCP（仅 codex，且清单声明了 rea 运行时）：`bash install-rea-mcp.sh --agent codex`
6. 路由清单重新生成：`python3 skills/routers/agent-skill-router/scripts/build_inventory.py`（`--dest` 下不修改受 git 跟踪的清单）

真实 Codex 安装的推荐命令：

```bash
bash bootstrap.sh --agent codex
```

可继续传入 `install.sh` 的选项：

```bash
bash bootstrap.sh --agent codex --no-nvidia-skills --no-amd-skills --no-ccfa-skills --no-veloq
bash bootstrap.sh --agent cursor --copy
```

`bootstrap.sh` 专用选项：

| 选项 | 行为 |
|---|---|
| `--no-update` | 跳过 `update-repos.sh`；仍初始化缺失的 submodule，并补齐缺失的外部来源 |
| `--no-integrations` | 跳过 `sources/external-sources.conf` 中的外部来源（获取、运行时、skill） |
| `--no-rea-mcp` | 跳过 REA 的 Codex MCP 注册（`--no-integrations` 同样跳过） |
| `--dest DIR` | 隔离安装：只写 `DIR`。不初始化 submodule、不跑 `update-repos.sh`、不获取外部来源、不安装运行时、不注册 MCP、不修改受 git 跟踪的路由清单，不写真实 HOME |

## 外部来源

`sources/external-sources.conf` 是唯一的外部来源清单（tracked）。checkout 放在 git-ignored 的 `repos/<name>/`，不 vendored 进本仓库。

| 来源 | 远端 | 检出 commit | 安装模式 | 运行时（需宿主执行） |
|---|---|---|---|---|
| sepia | `Nanako0129/sepia` | `fb06be76…` | skills（`skills/`，含 sepia 与五个操作入口） | 无 |
| wx-cli | `Steady775/wx-cli`（fork） | `08af8945…` | skills（根 SKILL.md） | `npm install -g @jackwener/wx-cli@0.3.0`（官方 npm 包，`wx`） |
| rea | `morluto/rea` | `3ce31175…` | skills（npm 包的 `runtime-skills`） | `npm install -g rea-agents@6.7.0`（`rea`） |
| mattpocock-skills | `mattpocock/skills` | `5beb35b0…` | skills（engineering、productivity、misc） | 无 |

完整来源、许可与版本说明见 [`sources/external-sources.conf`](sources/external-sources.conf) 与 [`docs/skills-usage-review.md`](docs/skills-usage-review.md) §2。

`mode` 与 `runtime` 的含义：

- `skills`：递归查找 `skills_dir`（可用逗号分隔多个目录，`.` 为仓库根）下的 `SKILL.md`；skill 名取 frontmatter `name`。`rea` 的 `runtime-skills` 由 `install-external.sh` 在 npm 安装后链接到全局安装包的 `skills/`。
- `none`：只获取源码，不安装运行时。
- `npm:<包名@版本>:<可执行名>`：`npm install -g <包名@版本>`，版本必须固定；安装后检查 `command -v <可执行名>` 且 `<可执行名> --version` 成功。
- `uv:<相对目录>:<可执行名>`：`uv tool install --force <checkout>/<相对目录>`。

运行时安装写入用户的全局位置（npm 全局前缀或 uv tool 目录），并需要 registry 访问。登录、授权、账号绑定和 WeChat 的 `wx init` 不由安装器执行。REA 的 Codex MCP 注册随 bootstrap 执行，见下文。

### 单条命令（Codex）

```bash
bash bootstrap.sh --agent codex --no-update
```

运行前请确保 PATH 中的 `node`/`npm` 满足 rea 的 engines（见下文「首次机器的前提」）。如果当前 Node 版本不满足，安装器会在执行 npm 之前以 BLOCKED 退出。

一次运行依次完成：补齐缺失的 submodule 与外部来源 → 安装 wx-cli、rea 的 npm CLI（版本固定，`npm install -g`）并检查 `--version` → 链接 rea 的 `runtime-skills` → 安装 skill → 为 Codex 注册 REA MCP → 重新生成路由清单（`skills/routers/agent-skill-router/references/inventory.tsv`）。任一步失败都返回非零，并在末尾说明，不会报告“完成”。

REA MCP 注册：`install-rea-mcp.sh` 只在清单声明了 `rea-agents` 运行时时执行，且只针对 `--agent codex`，执行 `rea setup --client codex --skill=false --yes --json`。备份、保留其他 MCP 设置与用户级 symlink 由 REA 自身完成；安装器不解析也不编辑 `~/.codex` 的配置。不使用 `--all-detected`、`--install-hopper`。其他 agent 只获得 CLI 与 skill，不会被报告为已注册 MCP。`--no-rea-mcp` 或 `--no-integrations` 跳过注册；`--dest` 下也不执行。

### npm 运行时的稳定入口（managed launcher）

- npm 全局 bin 位于 npm 前缀下（`npm prefix -g` 的 `bin/`，nvm 环境中例如 `…/v24.22.1/bin`），它不一定在日常 shell 的 PATH 中，所以 `which wx`、`which rea` 可能找不到。
- 安装器为 npm 运行时在 `$HOME/.local/bin/<bin>` 生成 managed launcher。启动器把安装时选中的 node 所在目录前置到 PATH，然后 `exec` 真实的全局 bin，参数逐个保留（包括含空格与撇号的路径）。它不修改 node/npm 本身、全局 PATH、shell 配置或 npm 前缀。
- launcher 带有标记 `agent-gpu-skills-managed`，重复安装会幂等地更新它。已存在的无关文件或 symlink（例如你自己放的 `~/.local/bin/wx`）不会被替换；安装会报告 BLOCKED，请移走或重命名后重试。
- bootstrap 只在自身进程内把 `~/.local/bin` 前置到 PATH，用于安装验证与 REA 注册；不修改你的全局 PATH。如果 `~/.local/bin` 不在你的 PATH 中，bootstrap 会打印提示，请自行把它加入 shell 配置。
- `--dest` 与 `--no-integrations` 不写入任何 launcher。

### 已验证的安装结果

2026-10-24 在本机 macOS 上执行 `bash bootstrap.sh --agent codex --no-update`（Node v24.22.1）：退出码 0，安装器校验 101 项通过、0 失败。另做独立核对：

- 清单中 552 个普通安装名全部存在，四个新增来源的 `SKILL.md` 与来源字节一致；两个受保护的委派 skill 复用已有 macOS 补丁版，原用户级链接未改动。
- 默认 PATH 下 `wx --version` 为 0.3.0，`rea --version` 为 6.7.0，均通过 `~/.local/bin` 的入口运行。
- REA MCP 已注册；按注册命令完成独立 stdio 握手并取得 207 个工具。安装不代表原生分析引擎或 Frida 已配置。
- 130 项自动化测试通过；隔离安装与清单逐项核对，REA 的 symlink 源根会被枚举，空源不会被报告为成功。

首次 clone 使用 `bash bootstrap.sh --agent codex`，以恢复其他 GPU skill 的 git-ignored 源码依赖；已有依赖的机器可以用 `--no-update` 跳过刷新。

### 首次机器的前提

- 必需：Git、Bash（系统 Bash 3.2 可用）、Python 3（路由清单与测试）。
- npm 运行时（wx-cli、rea）：需要 Node/npm。安装器按已发布的 `rea-agents@6.7.0` npm 包的 engines（`^22.19.0 || ^24.11.0 || >=26.0.0`）检查 PATH 中的 node，不满足时以 BLOCKED 退出并给出指引；不会自动修改全局 Node。源码 checkout 的 `package.json` 中同一版本号的 engines 较新（`^22.23.4 || ^24.22.1 || >=26.12.1`），但运行时是从 npm 安装的，因此不以源码 engines 为准。
- 满足 engines 的 Node 放在 PATH 前面即可，例如（本机示例，路径因机器而异）：

  ```bash
  PATH="$HOME/.nvm/versions/node/v24.22.1/bin:$PATH" bash bootstrap.sh --agent codex --no-update
  ```

  安装器不修改全局 Node，也不切换版本。
- 不保证在空白机器上直接可用：需要用户自行安装上述工具。

### 不由安装器执行的步骤（需要用户单独授权）

- wx-cli 读取真实聊天数据：需要用户执行 SKILL.md 中的 `wx init`（可能需要 sudo、内存扫描、codesign、`tccutil reset` 或重启微信）。安装器只做 npm 安装与 `wx --version`。
- REA 原生 provider（Hopper、Ghidra、IDA）与 Frida 相关安装脚本：不在本安装范围内，未经验证。npm 11 对 frida 可选安装脚本给出过策略警告，核心 `rea --version` 仍然通过；不要为此放宽全局 npm 策略。

同名 skill 按先到先得：本地 skill → KernelWiki → NVIDIA → cursor-skills → CCFA → AMD → 外部来源。被占用或受保护的名称不会被覆盖，计入对应 catalog 的“跳过同名或受保护”数量；用户级 symlink（如 `claude-delegate`、`codex-background`）以及任何指向本仓库之外的顶层 symlink 都不会被替换。

如果需要手动分步执行：

```bash
# 1. 获取源码 repo（DeepGEMM 含递归依赖，其余为 sparse checkout）+ NVIDIA/AMD/Cursor/CCFA skills + veloq 二进制
bash update-repos.sh

# 2. 安装 skill (默认 Cursor，用 --agent claude/codex/gemini 安装到其他工具)
#    同时安装 VeloQ（veloq 二进制 + nsys/ncu-profile-analysis skill）
#    同时安装 TileLang skills、NVIDIA skills、AMD skills、Cursor skills、CCFA Skills、GPU router skills
bash install.sh

# 不想要 VeloQ 时:
bash install.sh --no-veloq

# 不安装 AMD 官方 skills 时（本地多代 ISA skills 仍会安装）:
bash install.sh --no-amd-skills

# 不安装 CCFA 论文研究 skills 时:
bash install.sh --no-ccfa-skills

# 不安装 TileLang/CUDA 工作流 skills 时:
bash install.sh --no-tilelang-skills
```

## AMD 文档本地知识库

安装脚本会安装本仓库的 `amd-gpu-docs` skill；它与 `repos/amd-skills/` 中的 AMD 官方 workflow skills 分工不同：前者负责广泛的官方公开文档检索，后者负责具体工作流。文档缓存只在本机生成，并由 Git 忽略。

```bash
# ROCm + Instinct 核心目录
python3 skills/other/amd-gpu-docs/scripts/sync_amd_docs.py --profile core

# 加上 ROCm Blog、Ryzen AI、Quark、GPUOpen 当前技术目录
python3 skills/other/amd-gpu-docs/scripts/sync_amd_docs.py --profile all

# 查询本地文档和 amd/skills
python3 skills/other/amd-gpu-docs/scripts/query_amd_docs.py "hipGraph"
python3 skills/other/amd-gpu-docs/scripts/query_amd_docs.py "gfx950" --limit 20
```

同步器是增量式的：已存在的页面默认复用，`--refresh` 才会重新下载；`--workers 4` 等参数可降低站点压力。AMD 文档站偶尔会返回 Cloudflare `429` challenge，脚本会对 429/5xx/超时退避重试并保留已成功页面，稍后可重新运行续抓。`all` 表示当前公开技术目录，不包含私有支持门户、无限历史版本或营销页面。

PDF 默认不抓取。只有在逐份确认 AMD 的 Specification Agreement/文档许可后，才运行：

```bash
python3 skills/other/amd-gpu-docs/scripts/sync_amd_docs.py --profile all \
  --include-pdfs --accept-document-terms
```

PDF 以及 HTML/Markdown 缓存均位于 `skills/other/amd-gpu-docs/cache/`，该目录不会进入 Git；查询 PDF 需要额外安装 `pdfplumber`，然后加 `--include-pdfs`。

## 安装目标

```bash
bash install.sh                    # Cursor (默认，已验证)
bash install.sh --agent claude     # Claude Code
bash install.sh --agent codex      # Codex
bash install.sh --agent gemini     # Gemini CLI
bash install.sh --agent kimi       # Kimi Code CLI
```

| 工具 | Skill 安装路径 | 验证状态 | 官方文档 |
|:-----|:---------------|:---------|:---------|
| Cursor | `~/.cursor/skills/` | 已验证 | [Cursor Skills](https://cursor.com/docs/context/skills) |
| Claude Code | `~/.claude/skills/` | 未验证 | [Claude Code Skills](https://docs.anthropic.com/en/docs/claude-code/skills) |
| Codex | `~/.codex/skills/` | 未验证 | [Codex Skills](https://developers.openai.com/codex/skills) |
| Gemini CLI | `~/.gemini/skills/` | 未验证 | [Gemini CLI Skills](https://geminicli.com/docs/cli/skills/) |
| Kimi Code CLI | `~/.agents/skills/` | 未验证 | — |

注: SKILL.md 格式是跨工具通用的，但 skill 发现机制和搜索工具的行为可能因工具而异。Cursor 与 Kimi Code CLI 已在本仓库验证安装；Claude Code / Codex / Gemini CLI 如遇问题，建议让对应 AI 协助排查。

## 路径说明

本仓库对 Codex 统一使用：

- 配置文件：`~/.codex/config.toml`
- skill 目录：`~/.codex/skills/`

因此 `bash install.sh --agent codex` 会直接安装到 `~/.codex/skills/`。

对 Kimi Code CLI 统一使用：

- skill 目录：`~/.agents/skills/`

因此 `bash install.sh --agent kimi` 会直接安装到 `~/.agents/skills/`。

## 安装方式

默认使用**混合模式**安装:

- skill 目录: 真实目录（多数工具不识别软链接目录）
- `SKILL.md`: 复制真实文件（多数工具不识别软链接的 SKILL.md）
- 其余文件: 软链接到项目目录（repo、references 等）

使用 `--copy` 进行全量复制（适用于无法软链接的场景）。

需要隔离安装或在 CI 中验证时，可指定自定义目录：

```bash
bash install.sh --agent codex --dest /absolute/path/to/test-skills --no-veloq
```

`--dest` 是隔离安装：只写入指定目录下的 skill；不运行 VeloQ，不运行 `install-external.sh`，也不安装运行时 CLI。外部来源缺失时只提示，不返回失败。

## CCFA 论文研究 Skills

`install.sh` 默认从 `repos/ccfa-skills/` 安装 17 个 `ccf-*` runtime skill，并安装 `ccf-latex-templates` 支撑目录，保持上游完整安装语义。`install.sh` 会在 submodule 尚未初始化时自动初始化；也可以单独更新和安装：

```bash
bash update-repos.sh ccfa-skills
bash install.sh --agent codex
```

如不需要论文 idea、文献、实验、科研绘图、写作、评审、投稿和 rebuttal 工作流，可传入 `--no-ccfa-skills`。

## Supervisor-Skills 论文研究 Skills

`sources/external-sources.conf` 中的 `supervisor-skills` 固定到 commit `a20d76ef3979b804d7dbf599d577826957e0a8a9`（版本 2.1.0），由默认的外部来源安装步骤安装 `skills/` 下的 12 个 skill。它没有单独的 `--no-` 开关；`--no-integrations` 会一并跳过它。两种安装模式都会保留 `drawio-reconstruction` 的 `scripts/`、`examples/`、`requirements.txt` 与 `LICENSE`。

```bash
bash install-external.sh --no-runtime   # checkout 缺失时拉取并固定到 ref
bash install.sh --agent codex           # 混合模式；加 --copy 复制真实文件
```

许可：根 `LICENSE` 为 CC BY-NC-SA 4.0（Yuyu Luo 等），逐 skill frontmatter 的许可不同，`drawio-reconstruction` 另有 MIT 的 `LICENSE`。运行条件：安装器不装命令行运行时；`crop_assist.py` 需要 Pillow（需用户批准后安装 `requirements.txt`），`export_drawio.py` 需要本机 draw.io 桌面应用。两个来源的对照见 [`skills/research/README.md`](skills/research/README.md)。

## TileLang/CUDA Skills

本仓库通过 `repos/tilelang-cuda-skills/` 接入
[`sablin39/tilelang-cuda-skills`](https://github.com/sablin39/tilelang-cuda-skills)。
它提供 TileLang kernel 编写、调试、优化、profiling、PyTorch baseline 分析、
前后向测试、设计文档和 TileFoundry 工作流。`install.sh` 默认安装这些
TileLang skills，并保留上游提交的 submodule pin 以便复现。

上游仓库中的 `cuda_skill` 与本仓库的 `cuda-skill` 职责重叠，因此不额外
安装第二份 CUDA skill。TileLang skills 仍可调用本地 `cuda-skill`、NCU
和 NSYS 相关 skills。

如需单独更新：

```bash
bash update-repos.sh tilelang-cuda-skills
bash install.sh --agent codex
```

## AMD 官方 skills 与 Instinct ISA

`install.sh` 默认从 `repos/amd-skills/skills/` 安装 AMD 官方稳定 skills；可用 `--no-amd-skills` 跳过。`bootstrap.sh` 会自动初始化该 submodule，也可以单独运行：

```bash
bash update-repos.sh amd-skills
bash install.sh --agent codex
```

`amd-instinct-cdna4-isa` 是本仓库额外提供的 ISA 检索 skill。由于 AMD 官方 PDF 的 Specification Agreement 限制复制和再分发，PDF 不随 Git 仓库或 `install.sh` 分发。接受 AMD 条款后，在源仓库或安装后的 skill 目录运行：

```bash
bash skills/other/amd-instinct-cdna4-isa/download-cdna4-isa.sh
bash skills/other/amd-instinct-cdna4-isa/query-cdna4-isa.sh "v_mfma"
```

检索脚本优先使用 `pdftotext`（macOS: `brew install poppler`；Debian/Ubuntu: `apt install poppler-utils`），没有时可回退到 Python `pdfplumber`。提取文本不会提交到仓库；`pdftotext` 路径只使用临时文件并在脚本退出时删除。

多代 AMD Instinct ISA 使用 `amd-instinct-isa`。它维护 CDNA1、CDNA2、CDNA3、CDNA4 和 CDNA5 的官方入口，并允许在新 PDF 尚未加入目录时用 `--url` 登记。用户确认文档条款后可执行：

```bash
python3 skills/other/amd-instinct-isa/scripts/amd_instinct_isa.py list
python3 skills/other/amd-instinct-isa/scripts/amd_instinct_isa.py download cdna5-isa \
  --accept-document-terms
python3 skills/other/amd-instinct-isa/scripts/amd_instinct_isa.py query "WMMA" \
  --document cdna5-isa
```

下载新一代 ISA 或 white paper：

```bash
python3 skills/other/amd-instinct-isa/scripts/amd_instinct_isa.py download \
  --url https://www.amd.com/path/to/new-document.pdf \
  --id cdna6-isa --title "AMD Instinct CDNA6 ISA" \
  --accept-document-terms
```

下载的 PDF 只保存在 `skills/other/amd-instinct-isa/references/`，不会提交到 Git。`amd-instinct-cdna4-isa` 作为旧命令入口继续保留。

## VeloQ（profile 查询 CLI）

`install.sh` 默认一并安装 [VeloQ](https://github.com/lucifer1004/veloq)：`veloq` 二进制 + 两个 profiling skill（`nsys-profile-analysis` / `ncu-profile-analysis`）。VeloQ 不 vendored 进本仓库，由 [install-veloq.sh](install-veloq.sh) 获取，按以下优先级：

1. 本地 VeloQ 源码 checkout（`$VELOQ_SRC`，或自动探测 `../VeloQ`、`~/workspace/VeloQ`）：skill 复制 `SKILL.md` + 软链接 `references`，二进制优先用 `target/release/veloq`；
2. 已在 PATH 的 `veloq`：用其自带 `veloq self-update`；
3. VeloQ 官方 `curl` 安装脚本；
4. `cargo binstall veloq` / `cargo install veloq` fallback。

任一步失败都不致命（只 warn），不影响本仓库其他 skill 的安装。

```bash
bash install.sh --agent claude              # 含 VeloQ
bash install.sh --agent claude --no-veloq   # 跳过 VeloQ
bash install-veloq.sh --agent claude --no-binary  # 只装 skill
bash install-veloq.sh --no-skills                  # 只装/更新二进制

bash install.sh --agent kimi                # Kimi Code CLI（含 VeloQ）
bash install-veloq.sh --agent kimi          # 只更新 Kimi 的 VeloQ skill/二进制
```

二进制默认落在 `~/.local/bin/veloq` 或 `~/.cargo/bin/veloq`（确保它在 `PATH` 中）。`.nsys-rep` 首次查询需要 `nsys >= 2024.6` 在 `PATH` 上（`nsys export -t parquetdir`）。

如果服务器访问 GitHub raw/release 返回 403，最快的完整 skill 安装方式是先跳过 VeloQ：

```bash
bash bootstrap.sh --agent codex --no-veloq
```

之后有 Rust 工具链时再补装 VeloQ：

```bash
cargo binstall veloq   # 或 cargo install veloq
bash install-veloq.sh --agent codex
```

## 更新

```bash
# 更新所有源码 repo
bash update-repos.sh

# 只更新某个 repo
bash update-repos.sh triton
bash update-repos.sh cutlass
bash update-repos.sh deepgemm
bash update-repos.sh sglang
bash update-repos.sh verl
bash update-repos.sh nccl
bash update-repos.sh nvidia-skills
bash update-repos.sh amd-skills
bash update-repos.sh cursor-skills
bash update-repos.sh ccfa-skills

# 只更新 veloq 二进制（skill 用 veloq self-update 或重跑 install.sh）
bash update-repos.sh veloq
veloq self-update                  # VeloQ 原生升级：二进制 + skill

# 更新 CUDA 文档库
python3 scrape_docs.py all --force

# 更新 Colfax Research 文章知识库（增量抓取）
python3 colfax-research-skill/scripts/update_kb.py

# 同步 SKILL.md（修改源文件后重新安装）
bash install.sh                    # 或 --agent claude 等
```

## 验证

```bash
bash install.sh   # 安装时自动运行验证
```

验证结果决定退出码：任一已请求的组件（本地 skill、submodule 内的文档与源码、外部来源）缺失时返回非零，末尾打印“安装未完成”。
`--dest` 隔离安装中外部来源的缺失只显示为“可选缺失”，不计为失败。

仓库自检（不访问网络、不写真实 HOME）：

```bash
bash -n bootstrap.sh install.sh update-repos.sh install-veloq.sh install-external.sh
python3 -m unittest discover -s tests -p 'test_install_integrations.py' -v
python3 -m unittest discover -s tests -p 'test_skills_router.py' -v
```

## Skill 发现规则

| 层级 | 要求 | 可用软链接 |
|:-----|:-----|:----------|
| skill 目录 | 必须是真实目录 | 否 |
| SKILL.md | 必须是真实文件 | 否 |
| 其他内容 | 无限制 | 是 |

注: 以上规则在 Cursor 上验证通过。其他工具的 SKILL.md 发现和软链接行为可能不同，遇到问题可以让对应工具的 AI 协助排查。

## Ponytail

Ponytail 上游保留在 `repos/ponytail` 子模块。`install.sh` 将它的 6 个 skill 安装到所选 agent，
不自动修改 agent 的插件或 hooks 配置。更新源码运行：

```bash
bash update-repos.sh ponytail
```

Codex 的完整插件按上游方式单独安装（包含生命周期 hooks）：

```bash
codex plugin marketplace add DietrichGebert/ponytail
codex plugin add ponytail@ponytail
```

随后在 Codex 的 `/hooks` 中审核并信任插件 hooks，重启桌面应用或开始新会话。
插件与独立 skill 都提供相同指令；安装完整插件后可使用插件命名空间，例如
`$ponytail:ponytail-review`。仅通过 `install.sh` 安装 skill 时不包含自动激活 hooks。
