# Skill 使用审查与裁剪记录

本文记录 2026-11-19 按当前 LLM infra 工作范围完成的 skill 裁剪，以及保留后的路由使用方式。
结论基于 `inventory.tsv`、`routing-rules.json`、`excluded-skills.txt` 与本仓库测试；宿主是否自动选中 skill 仍未测量。

## 1. 结论

- 已移除与当前 LLM infra 工作无关的垂直领域 skill 及其路由逻辑：医疗/生物、DOCA/DPU、TAO、Jetson、DeepStream、BioNeMo、Earth2、cuOpt、VSS、PAIDF、HoloHub、前端/Office、微信/逆向、文献管理等。
- 移除不修改外部 submodule 的内容：精确安装名写入 `skills/routers/agent-skill-router/references/excluded-skills.txt`，由 `build_inventory.py` 与 `install.sh` 共同消费；外部来源清单只保留 `mattpocock-skills`，其与 cursor-skills 同名的 skill 由 cursor-skills 占用。
- 当前可安装 skill 180 个，受保护名称 2 个（`claude-delegate`、`codex-background`），另有 5 个 mattpocock 同名 skill 由 cursor-skills 占用；路由规则 71 条。
- 路由仍坚持单一入口、一个主 skill、至多两个伴随 skill；显式点名与真实制品优先于规则。

## 2. 当前来源与安装范围

| catalog | installed | 边界 |
|---|---:|---|
| local | 60 | GPU kernel、framework、router、provenance 与本地工具；另有 2 个 protected 名称 |
| nvidia-skills | 49 | Dynamo serving、Megatron Bridge / NeMo-RL、TileGym cuTile、Warp |
| cursor-skills | 34 | 通用工程流程：TDD、review、audit、性能分析等 |
| amd-skills | 11 | ROCm / AMD GPU 文档与 kernel 工作流 |
| kernel-design-agents | 1 | KernelWiki |
| mattpocock-skills | 25 | 通用工程流程；与 cursor 同名的 5 个由 cursor 占用 |

完整来源、pin 与许可说明见 [`sources/external-sources.conf`](../sources/external-sources.conf) 和生成的 [`SOURCES.md`](../SOURCES.md)。

## 3. 排除机制

- `excluded-skills.txt` 是按安装名精确匹配的中心排除清单（当前 465 个唯一名称），覆盖 CCFA、cursor 中的前端/Office、mattpocock 重复项、NVIDIA 无关垂直领域，以及本地 `salary-negotiation-agent`、`zotero-cli` 等。
- `build_inventory.py` 读取该清单后过滤候选；被排除的 catalog skill 不生成 `inventory.tsv` 行，catalog 行标记为 `absent` 或直接不出现。
- `install.sh` 读取同一清单，既不安装被排除的名称，也不让它们占用受保护名称或本地 skill。
- 回归测试断言已删除领域不再路由到旧 skill，且排除名称不出现在 inventory 中。继续裁剪时，把精确安装名加入清单，重新生成 inventory 与 SOURCES.md，并补充对应回归样例。

## 4. 保留的路由使用方式

```bash
python3 skills/routers/agent-skill-router/scripts/route_skill.py '用 triton 写一个 fused softmax kernel'
python3 skills/routers/agent-skill-router/scripts/route_skill.py '分析这个 ncu 报告为什么慢' --artifact prof/run.ncu-rep
python3 skills/routers/agent-skill-router/scripts/route_skill.py '用 tdd skill 给这个函数补测试'
```

- 非 GPU、跨域或不确定的任务进入 `agent-skill-router`；纯 GPU 任务进入 `gpu-development-catchall` 或其窄路由。
- 一个主 skill，至多两个伴随 skill；伴随需要至少两条独立证据。
- 渐进加载：先看路由输出的元数据，再只打开主 skill 的 `SKILL.md`，必要时打开伴随 skill。
- 显式 `$name`、反引号名称或 `name skill` 直接选中清单中的名称；未安装时报告 `absent`，不替换。

## 5. 验证

裁剪后应通过：

```bash
bash -n bootstrap.sh install.sh update-repos.sh install-veloq.sh install-external.sh scripts/node-preflight.sh
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 skills/routers/agent-skill-router/scripts/build_inventory.py --check
python3 scripts/build_sources.py --check
git diff --check
```

实际运行结果记录在提交说明中。

## 6. 未测量与边界

- 宿主 agent 的自动 skill 选择行为没有测量；本文只描述仓库内路由脚本的输出。
- 路由器减少的是选中后的 skill 正文加载，不减少宿主对名称与描述的索引负担。
- 外部 submodule checkout 仍可能留在 `repos/` 供源码对照，但被排除的名称不会安装，也不参与路由。
- `claude-delegate` 与 `codex-background` 是用户级委派入口，由 protected 规则保留，不由 catalog 覆盖。
