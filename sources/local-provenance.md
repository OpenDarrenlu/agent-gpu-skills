# 本地 skill 来源证据（tracked）

记录 2026-11-12 审计中已核验的最小事实，供 `scripts/build_sources.py` 的 `LOCAL_ORIGINS` 引用。
只记录非敏感的来源声明；不含本机配置、凭据或个人数据。新 checkout 中本文件必然存在，不依赖 git-ignored 的本机文件。

## claude-delegate（protected）

- 证据：本机工作区 `skills-lock.json`（git-ignored，machine-local）的条目。
- 声明的来源：`amElnagdy/delegate-skills`（GitHub）。
- 声明的 skillPath：`skills/claude-delegate/SKILL.md`。
- ref：未声明（unknown）。
- 本地修改：保留 macOS 补丁，不由上游覆盖；用户级保留名，安装器不覆盖。
- 版权：`skills/other/claude-delegate/LICENSE` 为 MIT，版权人 Ahmed Mohammed (amElnagdy)。

## codex-background（protected）

- 本次审计未确立上游 URL（unknown）。LICENSE 与 claude-delegate 相同版权人，但这不足以确立来源。
- 本地修改：保留；用户级保留名，安装器不覆盖。

## zotero-cli（external-declared）

- 证据：已安装包元数据（仓库外）。包名 `zotero-mcp-server`，版本 `0.15.0`，`METADATA` 的 `Project-URL: Homepage` 为 `https://github.com/54yyyu/zotero-mcp`。
- 核验方式：编排器读取该元数据并记录；本次未读取包内文件。
- 本地 skill 与包内模板的一致性：未核对（baseline equality unknown）。
- skill 由 `zotero-mcp install-skill` 生成（AGENTS.md 托管块）。
