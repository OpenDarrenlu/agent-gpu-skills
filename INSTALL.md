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

1. `git submodule update --init --recursive`
2. `bash update-repos.sh`
3. `bash install.sh --agent <agent>`

可继续传入 `install.sh` 的选项：

```bash
bash bootstrap.sh --agent codex --no-nvidia-skills --no-amd-skills --no-veloq
bash bootstrap.sh --agent cursor --copy
```

如果需要手动分步执行：

```bash
# 1. 获取源码 repo（DeepGEMM 含递归依赖，其余为 sparse checkout）+ NVIDIA/AMD/Cursor skills + veloq 二进制
bash update-repos.sh

# 2. 安装 skill (默认 Cursor，用 --agent claude/codex/gemini 安装到其他工具)
#    同时安装 VeloQ（veloq 二进制 + nsys/ncu-profile-analysis skill）
#    同时安装 NVIDIA skills、AMD skills、Cursor skills、GPU router skills
bash install.sh

# 不想要 VeloQ 时:
bash install.sh --no-veloq

# 不安装 AMD 官方 skills 时（本地多代 ISA skills 仍会安装）:
bash install.sh --no-amd-skills
```

## AMD 文档本地知识库

安装脚本会安装本仓库的 `amd-gpu-docs` skill；它与 `repos/amd-skills/` 中的 AMD 官方 workflow skills 分工不同：前者负责广泛的官方公开文档检索，后者负责具体工作流。文档缓存只在本机生成，并由 Git 忽略。

```bash
# ROCm + Instinct 核心目录
python3 amd-gpu-docs/scripts/sync_amd_docs.py --profile core

# 加上 ROCm Blog、Ryzen AI、Quark、GPUOpen 当前技术目录
python3 amd-gpu-docs/scripts/sync_amd_docs.py --profile all

# 查询本地文档和 amd/skills
python3 amd-gpu-docs/scripts/query_amd_docs.py "hipGraph"
python3 amd-gpu-docs/scripts/query_amd_docs.py "gfx950" --limit 20
```

同步器是增量式的：已存在的页面默认复用，`--refresh` 才会重新下载；`--workers 4` 等参数可降低站点压力。AMD 文档站偶尔会返回 Cloudflare `429` challenge，脚本会对 429/5xx/超时退避重试并保留已成功页面，稍后可重新运行续抓。`all` 表示当前公开技术目录，不包含私有支持门户、无限历史版本或营销页面。

PDF 默认不抓取。只有在逐份确认 AMD 的 Specification Agreement/文档许可后，才运行：

```bash
python3 amd-gpu-docs/scripts/sync_amd_docs.py --profile all \
  --include-pdfs --accept-document-terms
```

PDF 以及 HTML/Markdown 缓存均位于 `amd-gpu-docs/cache/`，该目录不会进入 Git；查询 PDF 需要额外安装 `pdfplumber`，然后加 `--include-pdfs`。

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

## AMD 官方 skills 与 Instinct ISA

`install.sh` 默认从 `repos/amd-skills/skills/` 安装 AMD 官方稳定 skills；可用 `--no-amd-skills` 跳过。`bootstrap.sh` 会自动初始化该 submodule，也可以单独运行：

```bash
bash update-repos.sh amd-skills
bash install.sh --agent codex
```

`amd-instinct-cdna4-isa` 是本仓库额外提供的 ISA 检索 skill。由于 AMD 官方 PDF 的 Specification Agreement 限制复制和再分发，PDF 不随 Git 仓库或 `install.sh` 分发。接受 AMD 条款后，在源仓库或安装后的 skill 目录运行：

```bash
bash amd-instinct-cdna4-isa/download-cdna4-isa.sh
bash amd-instinct-cdna4-isa/query-cdna4-isa.sh "v_mfma"
```

检索脚本优先使用 `pdftotext`（macOS: `brew install poppler`；Debian/Ubuntu: `apt install poppler-utils`），没有时可回退到 Python `pdfplumber`。提取文本不会提交到仓库；`pdftotext` 路径只使用临时文件并在脚本退出时删除。

多代 AMD Instinct ISA 使用 `amd-instinct-isa`。它维护 CDNA1、CDNA2、CDNA3、CDNA4 和 CDNA5 的官方入口，并允许在新 PDF 尚未加入目录时用 `--url` 登记。用户确认文档条款后可执行：

```bash
python3 amd-instinct-isa/scripts/amd_instinct_isa.py list
python3 amd-instinct-isa/scripts/amd_instinct_isa.py download cdna5-isa \
  --accept-document-terms
python3 amd-instinct-isa/scripts/amd_instinct_isa.py query "WMMA" \
  --document cdna5-isa
```

下载新一代 ISA 或 white paper：

```bash
python3 amd-instinct-isa/scripts/amd_instinct_isa.py download \
  --url https://www.amd.com/path/to/new-document.pdf \
  --id cdna6-isa --title "AMD Instinct CDNA6 ISA" \
  --accept-document-terms
```

下载的 PDF 只保存在 `amd-instinct-isa/references/`，不会提交到 Git。`amd-instinct-cdna4-isa` 作为旧命令入口继续保留。

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
bash update-repos.sh nvidia-skills
bash update-repos.sh amd-skills
bash update-repos.sh cursor-skills

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

## Skill 发现规则

| 层级 | 要求 | 可用软链接 |
|:-----|:-----|:----------|
| skill 目录 | 必须是真实目录 | 否 |
| SKILL.md | 必须是真实文件 | 否 |
| 其他内容 | 无限制 | 是 |

注: 以上规则在 Cursor 上验证通过。其他工具的 SKILL.md 发现和软链接行为可能不同，遇到问题可以让对应工具的 AI 协助排查。
