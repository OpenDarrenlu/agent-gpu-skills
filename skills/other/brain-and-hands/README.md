# brain-and-hands（本地修改版 0.3.2）

支持 Codex、Claude 和 Cursor 的技能，同时保留 Claude 插件入口。主会话 Astra、Opus
或 Fable 负责决策、代码分析、测试结果分析和修复；Sol、最新 Sonnet，或当前会话可用的
其他手模型（DeepSeek V4 Flash 1017、Grok 4.7 等）执行普通测试；Terra、最新 Haiku
只查事实。非常简单的测试从任一宿主创建独立 Claude session，使用 a7n Noir。
Cursor 的 multitask 子代理在任务已写明、逻辑不算特别复杂时用 hand 模型并作为 hand，这档可以比普通 hand 稍高。Fable/Opus 子代理只负责设计、含糊诊断和最终核对，并可以继续派这样的 hand。

- 上游：https://github.com/prograsec/brain-and-hands，基线 `2b30b09798a5bb8fd91f1402905f1fb4a524b05e`（MIT，见 LICENSE）。2026-11-18 `git ls-remote HEAD` 核验相同。
- 原文：`README.upstream.md`；规范入口：`SKILL.md`。
- 旧版实验：仓库根 `docs/brain-and-hands-experiments/README.md`，仅代表 2026-11-17 的 Gris/Claude 夹具，不代表 Noir 或 Codex。

## 安装

```bash
bash install-brain-and-hands.sh --agent codex   # 只安装这一技能
bash install-brain-and-hands.sh --agent claude  # 安装技能和 Claude 插件
bash install-brain-and-hands.sh --agent cursor  # 安装到 ~/.cursor/skills
bash install-brain-and-hands.sh --agent codex --dest /absolute/devlog/skill-check
```

`install.sh` 的本地清单也包含此技能；独立安装器无需枚举整个仓库。
`--dest` 仅写目标技能，不调用 Claude、不注册插件。Claude 插件注册继续使用官方
`claude plugin` 命令，不手工改全局配置；卸载用相同 `--agent` 加 `--uninstall`。
技能副本省略 .claude-plugin，避免 Claude 同时按 skills-dir 重复加载插件；插件从仓库源目录注册。
安装后的 Codex、Claude 与 Cursor 新会话都能发现该技能；当前会话可直接读取本文件旁的 SKILL.md。
三个宿主共用同一份文件，Cursor 另由 `.cursor/rules/brain-and-hands-multitask.mdc` 指到本技能。

## 实际路由

`scripts/route_task.py` 从指定的模型目录选择同家族最新可用版本。2026-11-18 本机为
`gpt-6-astra` / `gpt-6.1-sol` / `gpt-6.1-terra`。Claude 使用 provider 的 opus、fable、
sonnet、haiku 别名，其中 fable 与 opus 都是大脑。Cursor 使用 `--host cursor` 和当前会话
的可用模型列表：默认手是列表里最新的 Sonnet 或 Haiku；列表里没有该家族时不自动换模型，
调用方显式指定 DeepSeek、Grok、Gemini、Muse 或其他非大脑 slug。显式版本和环境覆盖优先。
缓存可用不等于原生子代理工具可用，见 `references/models.md`。

`scripts/run_simple_test.py` 只执行调用者给定的测试 argv，记录真实退出码，再启动无工具的
独立 Noir session 总结证据。Noir 不生成 shell、不改测试、不分析根因。Noir 的 gateway ID
默认 `anthropic/a7n-noir[1m]`，可用 `BAH_NOIR_MODEL` 指定；失败不会偷偷换 Gris。
用法、隔离和退出状态见 `references/noir-session.md`。

## 相对上游的修改与维护

新增根 SKILL.md、agents/openai.yaml、模型解析器、Noir runner、三个按需参考文档、
tester/debugger 角色；delegate 改为共用入口；grunt 改成 Noir 测试兼容角色；原有角色增加
限定搜索和禁止继续委派的规则；SessionStart 仅提醒且不启动模型。插件元数据版本 0.3.0。
README.upstream.md、LICENSE、图标、hooks/hooks.json 保留上游内容。

同步上游时在已知小目录中比较上述基线与新版本，逐文件合并本地修改，不整目录覆盖。
更新基线记录后，只刷新此技能的 inventory；来源生成器可复用已生成清单避免遍历其他项目：

```bash
python3 skills/routers/agent-skill-router/scripts/build_inventory.py --local-skill brain-and-hands
python3 scripts/build_sources.py --local-skill brain-and-hands
python3 /absolute/skill-creator/scripts/quick_validate.py skills/other/brain-and-hands
python3 -m unittest discover -s tests -p test_brain_and_hands.py -v
```

两个生成器的 `--local-skill --check` 仅证明该技能登记和来源表与
缓存清单及已声明来源一致，不声称全面核验其他 checkout。具体变更记录在 sources/update-log.md。
