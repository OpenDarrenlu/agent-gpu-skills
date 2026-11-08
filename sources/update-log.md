# 更新记录

手工维护。每次检查上游后在此追加一行（日期、对象、上游 tip、结论、动作）。检查命令：

```bash
git ls-remote <repo> main                      # submodule / clone 来源
curl -s https://api.github.com/repos/<owner>/<repo>/commits/main   # 外部来源
```

| 日期 | 对象 | 本地 pin | 上游 tip | 结论 | 动作 |
|---|---|---|---|---|---|
| 2026-11-08 | sglang `.agents/skills`（sglang-repo-skills） | `3bcde0c8` | `cbe0fed9` | main 前进，但 10 个 vendored skill 内容与上游一致 | 不更新 |
| 2026-11-08 | tile-ai/tilelang `.agents/skills`（tilelang-repo-skills） | `a8f3d3bc` | `a8f3d3bc` | 一致 | — |
| 2026-11-08 | Inferact/zero-mismatch-rl | `136aca87` | `136aca87` | 一致 | — |
| 2026-11-08 | vllm `.agents/skills/train-inference-consistency` | `2d8efee9` | 已删除该路径 | 上游 main 已移除该 skill，本仓库 vendored 副本成为唯一来源 | 保持本地副本（不覆盖） |
| 2026-11-08 | Fenng/Tech-Doc-Style-Chinese | `b119d01c` | `b119d01c` | 一致 | — |
| 2026-11-08 | NVIDIA/skills（submodule） | `fce3cca4` | `fce3cca4` | 一致 | — |
| 2026-11-08 | amd/skills（submodule） | `9eba3192` | `9eba3192` | 一致 | — |
| 2026-11-08 | CCFA-Skills（submodule） | `e33ccfa6` | `91ca9e28` | 11 commits：ccf-common / ccf-paper-writer / ccf-paper-reviewer / ccf-literature-searcher 等更新，新增 `ccf-research-engineer` | pin 已更新到 `91ca9e28` |
| 2026-11-08 | ponytail（submodule） | `09c434ab` | `96608e07` | 6 commits：仅 plugin 元数据/i18n/hooks，无 skill 内容变化 | pin 已更新到 `96608e07` |
| 2026-11-08 | cursor-skills（submodule） | `400c165e` | `400c165e` | 一致 | — |
| 2026-11-08 | tilelang-cuda-skills（submodule） | `0bc0a92a` | `842d1dc7` | 23 commits：reference/script 更新，skill 内容无变化 | pin 已更新到 `842d1dc7` |
| 2026-11-08 | kernel-design-agents（submodule） | `2e49985e` | `2e49985e` | 一致 | — |
| 2026-11-08 | sepia（external） | `fb06be76` | `0eb11d6c` | v0.13.0 release（skill 修复） | ref 已更新到 `0eb11d6c` |
| 2026-11-08 | mattpocock-skills（external） | `5beb35b0` | `4ae2317d` | 7 commits：code-review / implement-spec / improve-codebase-architecture / research / setup-matt-pocock-skills / wayfinder 小改，新增 in-progress/software-factory（不安装） | ref 已更新到 `4ae2317d` |
| 2026-11-08 | Supervisor-Skills（external） | `a20d76ef` | `4bd3fef3` | 1 commit：仅 README 增补，12 个 skill 未变 | ref 已更新到 `4bd3fef3` |
| 2026-11-08 | rea（external） | `3ce31175` / npm `6.7.0` | tag `rea-agents-6.8.0` (`160d821a`) | 6.8.0 发布：reverse-engineer-anything skill 更新（npm 包提供） | ref 更新到 `160d821ac`，npm 运行时更新到 `rea-agents@6.8.0`，`node-preflight.sh` engines 同步为 `^22.23.4 || ^24.22.1 || >=26.12.1`，已实装并验证 `rea --version = 6.8.0` |
| 2026-11-08 | wx-cli（external） | `08af8945` / npm `0.3.0` | `08af8945` | 一致 | — |
