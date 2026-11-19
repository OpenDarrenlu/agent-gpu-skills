# 独立安装 agent-desktop

`install-agent-desktop.sh` 把上游 [vercel-labs/agent-desktop](https://github.com/vercel-labs/agent-desktop) 的 `agent-desktop` skill 与匹配的 npm 运行时安装到用户级位置。

这是一个独立脚本，**不属于统一安装流程**：`install.sh`、`bootstrap.sh`、`install-external.sh` 都不调用它，`sources/external-sources.conf` 也不登记它。只有显式运行 `install-agent-desktop.sh` 才会安装。

## 固定版本与来源

| 项目 | 固定值 | 说明 |
|------|--------|------|
| npm 运行时 | `agent-desktop@1.3.1` | npm 官方 registry；以 `--prefix` 安装到私有目录，不用 `-g` |
| skill 源码提交 | `b0b7277bea09b770ef9d1fb45aad667bc432cd57` | 上游 tag `v1.3.1` 的 release 提交 |
| 许可 | Apache-2.0 | 上游根目录 `LICENSE`，安装时随 skill 一并复制到 skill 目录 |

- **skill 取自 release 提交，而不是 main。** 上游 main（`8ffecc1d`）的 `skills/agent-desktop/SKILL.md` 与 `references/commands-system.md` 比 release 多出 sandbox 相关改动，而 npm 1.3.1 对应的是 release 提交。脚本通过 `git cat-file blob <提交>:<路径>` 读取文件，不读取 checkout 的工作区。
- skill 文件集合固定为 5 个文件：`SKILL.md` 与 `references/` 下的 4 个文件（`commands-interaction.md`、`commands-observation.md`、`commands-system.md`、`workflows.md`）。上游树与此不一致时脚本失败，需要人工复核后更新脚本。
- 本脚本是这个独立安装的唯一 pin 来源。`SOURCES.md` 由 `scripts/build_sources.py` 从 `install.sh` 生成，不包含本 skill。
- **npm 元数据**：已在官方 npm registry 核对（2026-11-15）：`gitHead` 为 `b0b7277bea09b770ef9d1fb45aad667bc432cd57`，`engines` 为 `node >=18`。下文的 `npm view` 命令可随时复核。

## 验证范围

- 2026-11-15 在 macOS arm64 上做过一次冒烟测试：真实独立安装通过；在临时 AppKit 夹具上，snapshot 找到按钮，语义点击返回成功，第二次 snapshot 确认标题已变化。
- 这是针对该机器与该夹具的一次特定冒烟测试，**不代表**跨 macOS 版本、Intel Mac、Linux、Windows 或任意应用的广泛验证。

## 前置条件

- macOS。脚本检查 `darwin-arm64` / `darwin-x64` 原生包；上游也列出 Linux 与 Windows，但本脚本没有在这些平台上验证。
- `git`，以及 PATH 中的 `node`（>= 18）和 `npm`。
- 能访问 `github.com`（获取固定提交）和 npm registry。
- `repos/agent-desktop/`（已被 git 忽略）：
  - 缺失时，脚本创建它并只获取固定提交（`git fetch --depth 1 origin <提交>`）。
  - 已存在时，必须是独立 git checkout，`origin` 为 `https://github.com/vercel-labs/agent-desktop.git`，且没有未提交或未跟踪的改动；否则脚本失败且不修改它。
- **macOS 辅助功能（Accessibility）**：上游 agent-desktop 需要运行它的应用（终端、IDE、Codex 等）拥有辅助功能权限，才能观察其他应用的界面树。**本脚本不请求任何权限，也不运行 agent-desktop 的子命令。** 授权需要用户自己在 系统设置 > 隐私与安全性 > 辅助功能 中完成；上游 `macos` 排障文档还提到部分操作需要屏幕录制权限。

## 命令

隔离安装（写 skill 目录；checkout 缺失时还会创建 `repos/agent-desktop`；不安装 npm 运行时；脚本本身不向 HOME 写入，但 git 等子进程可能读取用户配置（如 `~/.gitconfig`），因此不保证它们完全不读取 HOME）：

```bash
bash install-agent-desktop.sh --dest /path/to/skills
```

真实安装（默认 Codex；`--agent` 与 `install.sh` 的映射相同）：

```bash
bash install-agent-desktop.sh --agent codex
```

其他 agent 的 skill 目录与 `install.sh` 一致：`cursor`（`~/.cursor/skills`）、`claude`（`~/.claude/skills`）、`gemini`（`~/.gemini/skills`）、`kimi`（`~/.agents/skills`）。

## 真实安装写入的路径

脚本直接写入下表的路径，不修改 shell 配置或 PATH。npm 在安装时还会写入它的下载缓存（`~/.npm`）。

| 路径 | 内容 |
|------|------|
| `~/.codex/skills/agent-desktop/` | `SKILL.md`、`references/` 的 4 个文件、`LICENSE`、标记文件 `.agent-desktop-standalone-installer` |
| `~/.local/share/agent-gpu-skills/agent-desktop/1.3.1/` | npm 私有前缀（`npm install --prefix`，含 `node_modules/agent-desktop`）和标记文件 |
| `~/.local/bin/agent-desktop` | 带标记的 launcher：把安装时的 node 目录前置到 PATH，然后 `exec` 私有前缀中的命令 |
| `repos/agent-desktop/` | 上游 checkout（仅缺失时创建） |
| `~/.npm/` | npm 下载缓存（由 npm 管理，本脚本不改变它） |

## npm 参数与运行时成本

安装命令是：

```
npm install --prefix <私有前缀> --ignore-scripts --no-save --no-audit --no-fund agent-desktop@1.3.1
```

- `--ignore-scripts` 不运行上游 `postinstall`。npm 11 会拒绝项目级安装中的 `--allow-scripts`（报错 `EALLOWSCRIPTS`），因此脚本不使用它。
- 上游 `postinstall` 负责链接原生二进制。跳过它之后，`agent-desktop` 通过上游的 Node launcher 运行，上游提示每次调用约多 125 ms。
- 上游 launcher 在首次运行时会向 stderr 打印一次该提示，并在包的 `bin/` 中写入标记文件 `.launcher-notice-1.3.1`。
- 如需改用原生入口，上游提示的做法是对该包执行 `npm install-scripts approve agent-desktop`，再 `npm rebuild agent-desktop`。这一步没有在本脚本中执行，也没有在本环境验证。

## 冲突与重新运行

- 所有前置检查（checkout、上游文件集合、Node/npm、目标归属）在写入前完成。检查失败时返回 1，且不写入任何目标路径。
- 目标已存在时：
  - 带本脚本标记的 skill 目录属于本脚本，重新运行时会被**整体删除并重建**，其中的任何文件（包括你自己放进去的）都会丢失。不要在这个目录中放自己的文件。
  - 带标记的运行时目录不会被删除：脚本只在其中重新执行 `npm install --prefix`，由 npm 更新 `node_modules`。
  - 其他情况（用户目录、用户文件、symlink、无标记的 launcher）一律视为用户所有：返回 1，打印 `BLOCKED`，保持原样。同一 skill 目录下的其他 skill 目录不受影响。
- 重新运行是安全的：脚本拥有的 skill 目录和运行时被替换为同一版本，launcher 也会被重写。
- 脚本不会自动更新：要升级版本，需先复核上游 release，再修改脚本中的 `NPM_VERSION` 与 `UPSTREAM_SHA`。

## 路由器发现

统一清单（`inventory.tsv`）不列出这个独立 skill，因此它不会出现在统一安装结果中。`agent-skill-router` 通过规则表中的 `agent-desktop` 条目发现它，并在 `--skills-root` 指向的目录中检查副本，在 PATH 中检查运行时。

```bash
python3 skills/routers/agent-skill-router/scripts/route_skill.py 'agent-desktop skill 用于操作桌面应用' --skills-root "$HOME/.codex/skills" --json
```

输出中三项判断分开给出：

- `source` 为 `standalone`：独立安装，不是 `installable`。
- `status`：`standalone-installed`（检查过且找到副本）、`standalone-unchecked`（没有传 `--skills-root`，只说明未检查，不等于未安装）、`standalone-absent`（检查过的目录里没有副本）。
- `runtime`：`agent-desktop` 是否在 PATH 中（`on-path` 或 `missing`）。

不传 `--skills-root` 时，副本状态为 `not-checked`，路由器不会断言它缺失。

## 只读检查

以下命令不请求任何权限，也不操作其他应用。`agent-desktop status` 在上游 skill 中被描述为只报告平台与权限状态。首次通过 Node launcher 启动时，上游可能在包的 `bin/` 中创建一次性提示标记 `.launcher-notice-1.3.1`；这是唯一可能产生的写入。

```bash
ls -l ~/.local/bin/agent-desktop ~/.codex/skills/agent-desktop/SKILL.md
node -p "require(process.env.HOME + '/.local/share/agent-gpu-skills/agent-desktop/1.3.1/node_modules/agent-desktop/package.json').version"   # 应输出 1.3.1
npm view agent-desktop@1.3.1 gitHead engines                          # 只读查询 registry；应含 gitHead b0b7277... 与 node >=18
~/.local/bin/agent-desktop status                                     # 只读；报告平台与权限
~/.local/bin/agent-desktop skills get platform                        # 读取运行时自带的平台 skill（macOS）
```

运行 `status` 之前，请确认当前应用已经在系统设置中获得辅助功能授权；否则它会报告 `denied`。授权本身不由脚本完成。

## 卸载（手动）

本脚本没有卸载命令。先查看内容，再删除本脚本生成的项目：

```bash
ls -la ~/.codex/skills/agent-desktop ~/.local/share/agent-gpu-skills/agent-desktop ~/.local/bin/agent-desktop
rm -rf ~/.codex/skills/agent-desktop ~/.local/share/agent-gpu-skills/agent-desktop
rm -f ~/.local/bin/agent-desktop   # 只在它带有 agent-desktop-standalone-installer 标记时删除
```

## 已知限制

- **只安装 `agent-desktop` skill。** 平台 skill（如 `agent-desktop-macos`）不会被复制；上游要求通过 `agent-desktop skills get platform` 从运行时读取，因此运行时必须可用。
- **隔离安装不包含运行时。** `--dest` 与 `install.sh --dest` 一样只安装 skill。
- **Node 版本在安装时固定。** launcher 记录的是安装时 PATH 中的 node 目录；之后更换 Node 需要重新运行脚本。
- 验证只覆盖上文「验证范围」中的 2026-11-15 macOS arm64 冒烟测试与临时 AppKit 夹具；离线测试只使用临时 HOME、临时 `--dest` 与 npm 桩程序。
- 其他 macOS 版本、Intel Mac、Linux、Windows，以及除临时夹具外的其他应用，尚未验证。

## 许可与归属

- 上游：`vercel-labs/agent-desktop`，Apache-2.0。
- 安装的 skill 目录包含上游 `LICENSE`，标记文件记录了包名、提交和许可。
- 本仓库没有 vendoring 上游 skill 内容；脚本在安装时从固定提交读取，不修改内容。

## 测试

```bash
python3 -m unittest discover -s tests -p 'test_agent_desktop_install.py' -v
```

测试只使用临时目录，npm 与上游远端都是桩或本地仓库；只读取仓库中的脚本与本地 `repos/agent-desktop`（缺少固定提交时跳过真实 checkout 的字节比较）。
