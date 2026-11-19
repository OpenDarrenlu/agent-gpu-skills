# 全局 Agent 配置

`AGENTS.md` 是 Claude Code 与 Codex 共用的全局开发习惯配置，其中第 12 节是 Cursor multitask 的模型分工，与 `/root/code/.cursor/rules/brain-and-hands-multitask.mdc` 保持同一规则。它被软链接为：

- `~/.claude/CLAUDE.md` → `global/AGENTS.md`
- `~/.codex/AGENTS.md` → `global/AGENTS.md`

因此只需要维护 `AGENTS.md` 一份。安装或刷新链接：

```bash
bash global/install.sh
```

脚本会备份已有的普通文件（`*.bak.<timestamp>`），然后创建软链接并校验两边内容一致。
若目标已是软链接但指向别处，脚本会拒绝覆盖；确认后可加 `--force`。

更新配置后无需重新安装：软链接会直接读取本文件。只有换机器或重建 HOME 时才需要再次运行安装脚本。
