# 全局 Agent 开发配置（Claude Code / Codex 共用）

本文件是 Claude Code（`~/.claude/CLAUDE.md`）与 Codex（`~/.codex/AGENTS.md`）的唯一配置来源。
两个入口都软链接到本文件；修改后只需维护这一份。加载本配置的主会话、子代理、远程执行与 MCP 均须遵守。

## 0. 默认工作方式

- 直接完成用户任务，不停在“已选择 skill”或“已制定计划”。需要计划时给出可执行步骤，然后实施。
- 非平凡任务先做 skill 路由：有明确匹配时直接调用最具体的已安装 skill；不确定时用
  `agent-skill-router` 按任务证据筛选，不把全量 skill 清单贴进上下文。新增常用外部 skill 后补进 router inventory。
- 一切结论以实际文件、diff、命令退出码、日志或运行指标为证据；不接受“应该可以”式的完成声明。
- 先正确性、后性能：没有 profile、A/B 或 bitwise 证据，不宣称优化有效，也不预设加速比例。
- 保持简洁：代码、注释、测试、commit、MR 和汇报都只保留必要内容；能一次校验的不要重复兜底，
  不用 `getattr` 默认值掩盖真实字段，不为展示进度制造文档。
- 不复杂、以及已经写明且逻辑不算特别复杂的任务交给 hand 模型；这档比普通 hand 稍高，仍由 hand 执行。设计、含糊诊断和最终核对留在 Astra、Opus 或 Fable 主会话。分工见第 12 节。
- 长任务拆成可独立验收的阶段；连续两次同因失败时停下，依据证据换方案或报告阻碍。
- 命令、任务或子代理明显变慢、异常等待或资源争用时，主动做轻量、有界的排查
  （相关进程、CPU、内存、I/O、GPU、网络或服务状态）；正常运行的长任务不因耗时而中止。
- 工具不可用时，在现有授权范围内尝试恢复并说明具体阻碍；不恢复任何禁用的 Claude 委派机制。
- 只执行用户明确要求的外向操作：push、rebase、force push、job 提交/停止、共享 checkout 修改、
  删除或覆盖文件都需要本次授权。`!244` 永不关闭、修改、rebase、cherry-pick 或再次提议处理。

## 1. 禁止硬盘扫描

- 禁止任何硬盘、分区或文件系统的大范围扫描，包括递归遍历、批量索引、统计和搜索；
  不得通过脚本、后台进程或其他工具绕过此限制。
- 禁止以根目录、用户主目录、挂载点、包含多个项目的工作区或其他大规模路径为范围运行
  `find`、`rg`、`grep -R`、`fd`、`tree`、`du`、`ls -R` 及同类递归/批量遍历工具。
  本机 `/`、`/root`、`/root/code`、`/mnt` 均属于此类路径。
- 只访问明确的目标文件，或在已知且足够小的目标目录内做必要的限定搜索；目录规模不明确时，
  先通过已知路径或非递归、定向查看缩小范围。
- 本规则同样适用于子代理和远程执行；任务说明及验收命令必须包含并遵守这些限制。

## 2. Shell、文件与临时产物

- 使用非登录 shell 和绝对路径，避免本机登录配置把工作目录改回 `/root/code`。
- 临时使用且不需要提交的脚本、实验工具和一次性报告放入 `/root/code/devlog`
  （本机约定；其他主机使用各自的 devlog），不要放进目标项目提交。
- 不提交生成物、大文件、profile 原始输出、镜像缓存或模型资产。
- 运行下载或解压得到的代码时使用 `python3 -I`，脚本与数据分目录，路径作为参数传入。
- 不原地修改模型/checkpoint 文件，包括写穿 symlink；确需修改时先复制到新路径，
  并断言目标不是链接。

## 3. Git commit 与 MR

- 创建 commit 时默认作者邮箱为 `tingzhan.lu@anuttacon.com`，保留原有作者姓名；提交前检查实际身份，
  提交后用 `git log -1 --format='%an <%ae>'` 验证。仓库配置或环境变量覆盖时，用命令级
  `GIT_AUTHOR_EMAIL=tingzhan.lu@anuttacon.com` 等方式确保正确。
- commit message 简洁；不添加 Claude/Anthropic 的 `Co-Authored-By`、`Generated with` 等署名行。
- 一个 commit 只做一件逻辑完整的事；相近修复可以 squash；独立特性拆成独立 branch/MR 分别验证；
  未 ready 时标 Draft；不夹带无关改动、临时脚本或实验配置。
- MR 标题必须全英文并采用 Conventional Commits，清晰说明变动边界：
  `[module] describe what changed`（例如 `[moe] support token-level balancing`）。
- MR 描述使用中文，按“变更背景与目标 / 具体变更 / 影响面评估 / 验证实验 / 作者自检”组织；
  删除与最终版本无关的方案试探、临时测试和冗余描述。模板见
  `/root/code/devlog/moe_parity/docs/MR_REVIEW_REQUIREMENTS.md`。
- squash 只能由相关作者执行并保留原作者；相似 commit 合并且涉及不同作者时，按用户要求以
  `tingzhan.lu` 为 author、另一位为 co-author。
- rebase 或 force push 前先查看远端目标分支是否有 squash commit 或新增内容，不要强制 rebase。
  需要改写历史时先备份 ref，使用 `--force-with-lease`，lease SHA 用 `git ls-remote` 解析；
  共享 checkout 在 push 前重查 HEAD，绝不用本地版本覆盖他人 commit。
- 避免 `git stash`（共享 checkout 中它是 repo-wide 操作，pop 可能捞到他人条目）；
  对照实验使用独立 worktree。
- MR 合入前 CI 必须全绿；涉及底层并行框架、算子、通信、显存或精度的改动按团队规范取得足够 approve。

## 4. 凭据与密钥

- 任何 API key、token、密码都不得写入 job spec、launcher、脚本、配置或本文件；
  平台密钥一律通过 secret 注入。
- `WANDB_API_KEY` 使用平台 secret（`secrets: ["9ab7c815-e3f8-4c51-8ca1-f1545a9c94d8"]`，
  共享 `rl-secrets`），不要复制到 job spec、launcher 或任何文件中。
- 不打印 secret 环境变量，也不从他人 job spec 复制 key。用户给出 token 时，提醒放入 secret/env，
  不落盘到全局配置。

## 5. 本机 GPU 使用

- 单元测试、GPU kernel/module 正确性测试和微基准可以在 dev box 本地 GPU 上运行；
  serving 和长时间 profiling 放到 MLP job 中。
- 需要独占 GPU 而资源被占用时，只停止阻塞所需 GPU 的进程。停止前记录其命令行、工作目录、
  环境变量、GPU 编号和日志路径；GPU 工作完成后按原样重启并确认。
- 不擅自终止其他会话、用户的任务或系统服务；回复中说明停止了什么以及是否已恢复。

## 6. MLP job

- 每个真实 job（training、RL、serving、model validation）都必须开启引擎监控和 wandb。
  A7N verl 栈使用 `++rollout.server.smg_engine_metrics=true`、
  `++trainer.async_snapshot.telemetry.enable=true`、
  `++rollout.client_config.dump_rollout_trace=true`、
  `++rollout.client_config.trace_session_span=true`、`TRAINER_LOGGER=[console,wandb]`、
  `WANDB_MODE=online`，并设置 `PROJECT_NAME` / `EXP_NAME`。
- 已提交但未开启监控/wandb 的 job 要重提；只有 0-GPU probe/reader job 豁免。
- 提交前查看各队列排队深度和输入可用性。用户要求竞速时，可同时向有希望快速调度的队列排队；
  一旦某个 job 开始运行，就停掉其余 job，避免重复占用。
- launcher 脚本一旦被 job 引用就必须不可变：每次提交使用单独拷贝并 `chmod 444`。
- 平台不解释 command 中换行时，将脚本 base64 内联进 command；driver stdout 通过对应 submitter
  pod/container 查看。
- job spec 不写入凭据；队列等环境差异用变量或 dotenv 传递，不把本机路径假设为目标集群路径。

## 7. 提交 job 前的模型资产核验

- 创建、重提或重启加载模型的 job 前，核验实际使用的 actor、rollout、draft、tokenizer/processor、
  初始化 checkpoint 和 distributed checkpoint 路径或模型引用。主 checkpoint 已恢复不代表其 HF 导出
  或 tokenizer 已就绪。
- 必须在目标集群的实际存储命名空间中核验。本机或另一集群能访问同名路径不等于目标 job 可访问；
  必要时使用目标环境中的有界只读检查或 0-GPU probe。
- 目录存在不算通过。按 loader 要求检查 config、metadata/index、权重分片和 tokenizer/processor 文件
  是否存在且可读，检查 symlink target；使用显式 manifest/index 及其引用路径，禁止扫盘或递归遍历
  大目录，也不要全量加载模型权重。
- 使用 tokenizer 的 job 必须用与实际入口相同的库和版本在 CPU 上加载 tokenizer，执行一次最小 encode，
  核对词表和 token ID 范围兼容。不得换用其他模型的 tokenizer、随机权重或静默下载/默认回退掩盖缺失。
- 发现资产为空、缺失、不可读、链接失效，或处于沉降/归档、恢复/同步/转换中时，停止提交并报告
  精确路径和缺失文件。恢复后重新验证实际可读状态；长排队、跨集群迁移或 archive/restore 后，
  在实际启动前轻量复查。
- 将目标集群、实际路径/引用、必要文件/清单、检查命令和结果记录到 devlog，并在聊天中简洁汇报。

## 8. TIC / parity 项目硬规则

- `mbridge` 永远为 `false`；`mbridge=true` 的运行不能作为参考基线或目标配置。
- parity 最小配置：vLLM 设置 `A7N_BATCH_INVARIANT=1` 且
  `parity_split_kv_chunk_size = MAGI_CHUNK_SIZE`；Megatron 使用封闭的显式 key 清单
  （见 `/root/code/devlog/M6lowbitRL/docs/PARITY_minimal_config.md` 与
  `/root/code/devlog/moe_parity/docs/CHECKLIST_tic_parity_config.md`），
  不要按 `aniac_*` 命名规律自行补键。
- parity 相关改动必须由单独 flag/env 控制；默认关闭时原路径行为必须保持不变。
  多个 feature 共用 flag 时，调整 commit 顺序或拆分，保证开关语义清晰。
- 验证按模块/引擎 A/B、再端到端的顺序进行；长 sweep 每完成一个 tier 立即汇报，最后汇总到 devlog 文档。
- 高性能 vLLM serving 默认不开 PDL；确需开启时必须说明原因并提供实测证据。

## 9. 公网搜索与 Claude/Codex 分工

- 公网搜索、公开网站与 GitHub 查询由 Codex 直接执行；Claude 不主动使用 WebSearch/WebFetch，
  除非用户本次明确要求。
- Claude 负责本机实现、测试、job 操作和验收；Codex 直接负责其任务范围内的需求澄清、计划、
  实现、测试、文档和最终验收，不转交 Claude。Codex 不得调用 Claude CLI、claude-delegate、
  codex-background、delegate_run/delegate_wait/delegate_abort、Anthropic/Claude API，
  或任何通过脚本、子进程、子代理、SSH、MCP、插件、技能包装调用 Claude 的机制。
- 唯一例外：brain-and-hands 的非常简单测试路径，使用插件 `scripts/run_simple_test.py`
  创建全新的独立 Claude/a7n Noir session。该 session 不恢复旧上下文，不使用工具、插件、hooks、
  MCP 或进一步委派，也不做实现、设计、研究或复杂 debug。
- Cursor multitask 的 hand 子代理不是这条例外，也不走禁用的 Claude 委派；模型分工按第 12 节。
- 不安装、注册或恢复上述禁用的委派技能和 MCP，也不要求用户为此重载 MCP。

## 10. 进展汇报

- 用中文简洁汇报实际成果、验证结果、问题/风险和下一步；不为展示进度自动创建或打开
  实时日志窗口、终端、浏览器或日志查看进程。
- 失败时报告 job/命令状态、首个真实 traceback（去掉 Ray 等包装帧）以及初步分类
  （OOM、config rejection、missing path、image problem 等）；不编造修复或数值。
- 未完成、未执行或未验证的内容必须明确说明，不用成功措辞掩盖。

## 11. 适用范围与维护

- 本配置适用于加载它的 Claude/Codex 会话；远程执行同样适用。其他主机的持久配置需单独核实。
- 修改本文件后，确认 `~/.claude/CLAUDE.md` 与 `~/.codex/AGENTS.md` 均指向本文件。
- `/root/code/devlog`、队列名、集群名和镜像 tag 等是本机/项目约定；换主机时只调整这些路径，
  不改变通用工作规则。

## 12. 模型分工与 Cursor multitask

与 `/root/code/.cursor/rules/brain-and-hands-multitask.mdc` 保持同一规则。技能正文在
`/root/code/agent-gpu-skills/skills/other/brain-and-hands/SKILL.md`。加载本配置的主会话和子代理，
包括 Cursor multitask 子代理，都按此执行；子代理不继承父会话里被省略的内容。

- 大脑只做开放设计、含糊失败的根因判断，以及最终 diff 核对。这些留在当前的 Fable、Opus 或 Astra。
- 任务已经写明、逻辑不算特别复杂时，multitask 子代理用 hand 模型，并且自己就作为 hand。这档比普通 hand 稍高：跨几个文件、构建、GPU 运行，或只做局部机械判断，仍然算 hand。不要因为它比单条命令更大就改用 Fable 或 Opus。
- 派子代理时必须带上述技能的绝对路径、角色（brain 或 hand）和明确的 `model`。`model` 来自本会话可用模型列表，先用该技能的 `scripts/route_task.py --host cursor` 解析。
- hand 默认用列表里最新的 Sonnet。只读事实默认用最新的 Haiku。列表里没有该家族时，显式指定其他非大脑模型，例如 `deepseek-v41-flash-1017-high` 或 `grok-4.7-high-fast`，也可以是列表里的 Gemini、Muse 或其他非大脑 slug。不要省略 `model`，也不要用 `generalPurpose` 继承父模型来充当手。
- 跑在 hand 模型上就是 hand：完成交代的任务，不要再派子代理。Fable 或 Opus 只在设计、含糊诊断和最终核对时继续派 hand；交给它的任务若落在上面这档，就自己作为 hand 做完。
- 极简、命令已经写死的测试仍走该技能的独立 Noir 会话，不走 Cursor Task。
