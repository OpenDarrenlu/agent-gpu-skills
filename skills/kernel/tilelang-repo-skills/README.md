# TileLang 仓库内建 skill（tile-ai/tilelang `.agents/skills` 精选）

本目录 vendored 自 [tile-ai/tilelang](https://github.com/tile-ai/tilelang) 仓库自带的 agent
skills（`.agents/skills/`，`.claude/skills` 下的 9 个 symlink 指向同一内容）。它们覆盖
TileLang 编译器/后端/layout 语义层面的知识，与本仓库已有的 TileLang kernel authoring 套件
（`repos/tilelang-cuda-skills`s：`writing-tilelang-kernels`、`debugging-tilelang-programs`、
`optimizing-tilelang-programs`、`profiling-tilelang-programs`、`testing-fwd-bwd-kernels`、
`torch-profiling-tilelang-programs`、`tilelang-optimization-loop`、`designing-with-tilefoundry`、
`writing-tilelang-design-docs`）互补：套件讲"怎么写 kernel"，这里讲"TileLang 内部怎么工作"，
在编译报错、layout 问题、后端/Ascend 适配时使用。

## 上游来源

- 仓库：`https://github.com/tile-ai/tilelang.git`
- 路径：`.agents/skills/`
- 基线：commit `a8f3d3bcfe7fdd1a3887b05ff8f054e7d08980de`（main，2026-11-08）
- 许可：Apache-2.0（见上游仓库根目录 LICENSE）。
- 上游稀疏 clone：`repos/tilelang/`（git-ignored，只取 `.agents/skills`）；
  运行 `bash update-repos.sh tilelang` 获取或更新。

## 本地修改

所有文件均**原样** vendored，未做修改（包括 `references/`、`scripts/` 与 `agents/openai.yaml`）。

## 收录的 skill

| skill | 作用 |
| --- | --- |
| `tilelang-layout` | Layout/Fragment 类、layout 推断与代价模型、CuTe 布局代数与 TileLang→CuTe 转换、swizzle，以及 layout 相关 pass 的调试 |
| `tilelang-semantic` | TileLang 语义规则与校验：循环嵌套、buffer 边界/初始化/所有权、数据竞争、同步与 async 生命周期、TileOp 契约、软件流水线（含 `scan_loop_nesting.py` 扫描脚本） |
| `tilelang-backend` | 集成或 review TileLang target backend / backend variant / 共享执行后端：dialect、PassPipeline、codegen、CMake 打包 |
| `tilelang-ascend` | 面向华为 Ascend NPU 编写、移植、调优与调试 TileLang kernel：Ascend dialect、AIC/AIV 分区、SIMD/SIMT 操作、T.Stage 调度 |

未收录的上游 skill（`tilelang-build`、`tilelang-dev-guide`、`tilelang-simplification`、
`tilelang-submit-pr`、`tilelang-tvm-ir`）是 TileLang 编译器自身的构建/C++ 开发/PR 流程，与本
bundle 的 kernel authoring 目标关系较弱，需要时可直接从上游 `.agents/skills/` 取用。

## 与其他 skill 的分工

| 问题 | 使用 |
| --- | --- |
| 写 TileLang kernel（authoring、调试、优化、profiling、fwd/bwd、设计文档） | `repos/tilelang-cuda-skills/skills/` 下的 kernel authoring 套件 |
| layout/Fragment、CuTe 布局代数、layout 推断问题 | `tilelang-layout` |
| 语义校验、数据竞争、同步/async 生命周期报错 | `tilelang-semantic` |
| 适配新后端 / Ascend NPU | `tilelang-backend`、`tilelang-ascend` |
| TileLang 报错属于 kernel 写法问题时的分层定位 | `debugging-tilelang-programs` → 需要时再进 `tilelang-semantic` |

## 同步上游

1. `bash update-repos.sh tilelang`（稀疏 clone，只取 `.agents/skills`）。
2. 比对：`diff -ru repos/tilelang/.agents/skills skills/kernel/tilelang-repo-skills`（排除本 README）。
3. 上表列出的 skill 原样覆盖；新增/删除 skill 时同步更新本 README 与 `install.sh` 的
   `SKILL_NAMES`/`SKILL_DIRS`。
4. 重新生成路由清单：`python3 skills/routers/agent-skill-router/scripts/build_inventory.py`，
   并确认 `build_inventory.py --check` 通过。
