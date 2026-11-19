# Skill 来源登记（SOURCES.md）

本文件由 `python3 scripts/build_sources.py` 生成，请勿手工编辑正文。
skill 的发现、同名先到先得与状态来自 `skills/routers/agent-skill-router/scripts/build_inventory.py`；
本文件逐 skill 给出来源类别、上游链接、ref 与说明。

## 类别说明

| 类别 | 含义 | 同步方式 |
| --- | --- | --- |
| vendored | 从上游逐字复制的快照，可能有本地修改 | 有本地修改的文件 merge，其余直接覆盖 |
| catalog | submodule / clone 中的 skill 目录，只读取 | `bash update-repos.sh <name>` 更新 pin |
| external | `sources/external-sources.conf` 的外部来源 | 更新 conf 中的 ref 后 `bash update-repos.sh external` |
| nested | catalog 的嵌套 submodule | 以嵌套仓库的 origin 与 gitlink 为准 |
| external-declared | 第三方来源已在本地文件或包元数据中声明；基线与内容比对未验证 | 按声明的上游手工核对，比对后再决定是否 merge |
| protected | 用户级保留名，安装器不覆盖；来源可能已声明，保留本地补丁 | 不由本仓库同步或覆盖 |
| unknown | 本次审计未确立来源（不断言无声明） | 人工确认来源；不得默认为原创 |
| optional-runtime | 可选运行时 skill（VeloQ） | 由 install-veloq.sh 管理 |

上游链接：`<repo>/tree/<ref>/<路径>`，仓库地址不带 `.git`。无 ref 时只给仓库地址。

## 1. 汇总

| 状态 | 数量 | 含义 |
| --- | --- | --- |
| installed | 226 | 来源可安装（install.sh 选中并可装入，不表示已安装到宿主） |
| shadowed | 5 | 同名已被先到来源占用，不安装 |
| protected | 2 | 用户级保留名，不计为可安装 |
| missing-source | 0 | 本地缺文件 |
| optional-runtime | 2 | VeloQ，可选，不在 install.sh 数组中 |
| tracked-not-installed | 0 | 仓库内有 SKILL.md，未安装 |

注：installed 数量与 `agent-skill-router/references/inventory.tsv` 的 installed 行一致，只表示安装器会选中该来源，不验证宿主上的安装结果。

## 2. 逐来源概览

| 来源 | 类别 | 上游仓库 | ref | 安装 skill 数 |
| --- | --- | --- | --- | --- |
| `skills/framework/sglang-repo-skills` | vendored | https://github.com/sgl-project/sglang | `3bcde0c80eeb07b40d3334c4b3d7341602efd089` | 10 |
| `skills/kernel/tilelang-repo-skills` | vendored | https://github.com/tile-ai/tilelang | `a8f3d3bcfe7fdd1a3887b05ff8f054e7d08980de` | 4 |
| `skills/framework/zero-mismatch-rl` | vendored | https://github.com/Inferact/zero-mismatch-rl | `136aca87d8158adae94b5a272fa4a8a6d5cd6939` | 1 |
| `skills/framework/train-inference-consistency` | vendored | https://github.com/vllm-project/vllm | `2d8efee9a73366d6b4e753780af32443d2fef91a` | 1 |
| `skills/other/brain-and-hands` | vendored | https://github.com/prograsec/brain-and-hands | `2b30b09798a5bb8fd91f1402905f1fb4a524b05e` | 1 |
| `skills/other/tech-doc-style-chinese` | vendored | https://github.com/Fenng/Tech-Doc-Style-Chinese | `b119d01c9bb7cd132b3c62fafdff34227c4410d1` | 1 |
| `nvidia-skills` | catalog | https://github.com/NVIDIA/skills | `fce3cca4170be75e82c64920654d56ee2811b005` | 59 |
| `amd-skills` | catalog | https://github.com/amd/skills | `9eba3192f6793237eaeacfabd61fb0bd557e53cf` | 17 |
| `cursor-skills` | catalog | https://github.com/Saddss/cursor-skills | `400c165e7a6afac09e0f84aab85bcc20f37252f5` | 34 |
| `ccfa-skills` | catalog | https://github.com/mikubaka88/CCFA-Skills | `91ca9e2801698466f934b557a75ce46db8ba46b5` | 18 |
| `ponytail` | catalog | https://github.com/DietrichGebert/ponytail | `96608e078b2d5ea607c5c3b8b5a54d1c1fba6846` | 6 |
| `tilelang-cuda-skills` | catalog | https://github.com/sablin39/tilelang-cuda-skills | `842d1dc7d684861d2da241cb3e51137a47d5cad6` | 9 |
| `kernel-design-agents` | catalog | https://github.com/OpenDarrenlu/kernel-design-agents | `2e49985e277f65ad6c07d31b943cf78628e06e98` | 1 |
| `tech-doc-style-chinese (clone)` | catalog | https://github.com/Fenng/Tech-Doc-Style-Chinese | `` | 0 |
| `mattpocock-skills` | external | https://github.com/mattpocock/skills | `4ae2317ddb22b046e26455e4f9c9b3aaaf62212d` | 25 |
| `supervisor-skills` | external | https://github.com/HKUSTDial/Supervisor-Skills | `4bd3fef393e5ffc891c74309a5ef6c5ad3ad12c8` | 12 |
| `veloq` | optional-runtime | https://github.com/lucifer1004/veloq | 未固定（main） | 2 |

## 3. 逐 skill 来源

| 安装名 | 状态 | 占用者 | 类别 | 上游链接 | ref | 本地路径 | 说明 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `KernelWiki` | installed | - | nested | https://github.com/mit-han-lab/KernelWiki/tree/76d27b56f804e7e7295d4c570e1e5d7eef4b0a75 | 76d27b56f804 | `kernel-design-agents/skills/KernelWiki` | install.sh 安装的 KernelWiki 来自此嵌套目录；origin=https://github.com/mit-han-lab/KernelWiki；.gitmodules=https://github.com/mit-han-lab/KernelWiki |
| `add-jit-kernel` | installed | - | vendored | https://github.com/sgl-project/sglang/tree/3bcde0c80eeb07b40d3334c4b3d7341602efd089/.agents/skills/add-jit-kernel | 3bcde0c80eeb | `skills/framework/sglang-repo-skills/add-jit-kernel` | 无（全部逐字复制） |
| `add-sgl-kernel` | installed | - | vendored | https://github.com/sgl-project/sglang/tree/3bcde0c80eeb07b40d3334c4b3d7341602efd089/.agents/skills/add-sgl-kernel | 3bcde0c80eeb | `skills/framework/sglang-repo-skills/add-sgl-kernel` | 无（全部逐字复制） |
| `agent-skill-router` | installed | - | unknown | - | - | `skills/routers/agent-skill-router` | 本次审计未确立来源（未逐一读取该目录全部文件，不断言其无上游声明） |
| `amd-gpu-docs` | installed | - | unknown | - | - | `skills/other/amd-gpu-docs` | 本次审计未确立来源（未逐一读取该目录全部文件，不断言其无上游声明） |
| `amd-instinct-cdna4-isa` | installed | - | unknown | - | - | `skills/other/amd-instinct-cdna4-isa` | 本次审计未确立来源（未逐一读取该目录全部文件，不断言其无上游声明） |
| `amd-instinct-isa` | installed | - | unknown | - | - | `skills/other/amd-instinct-isa` | 本次审计未确立来源（未逐一读取该目录全部文件，不断言其无上游声明） |
| `ask-matt` | installed | - | external | https://github.com/mattpocock/skills/tree/4ae2317ddb22b046e26455e4f9c9b3aaaf62212d/skills/engineering/ask-matt | 4ae2317ddb22 | `repos/mattpocock-skills/skills/engineering/ask-matt` | - |
| `benchmark-paper-template` | installed | - | external | https://github.com/HKUSTDial/Supervisor-Skills/tree/4bd3fef393e5ffc891c74309a5ef6c5ad3ad12c8/skills/benchmark-paper-template | 4bd3fef393e5 | `repos/supervisor-skills/skills/benchmark-paper-template` | - |
| `brain-and-hands` | installed | - | vendored | https://github.com/prograsec/brain-and-hands/tree/2b30b09798a5bb8fd91f1402905f1fb4a524b05e | 2b30b09798a5 | `skills/other/brain-and-hands` | 本地修改版 0.3.2（见其 README.md）：新增 Codex/Claude/Cursor 共用 SKILL.md、模型解析器（含 --host cursor 与 Fable 大脑）、独立 Noir 测试 runner、参考文档和 tester/debugger；grunt 改成 Noir 测试角色；delegate 复用根入口；角色与 hook 修正规则；保留上游 LICENSE、图标、hooks.json 和 README.upstream.md |
| `brainstorming` | installed | - | catalog | https://github.com/Saddss/cursor-skills/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/skills/brainstorming | 400c165e7a6a | `repos/cursor-skills/skills/brainstorming` | - |
| `ccf-common` | installed | - | catalog | https://github.com/mikubaka88/CCFA-Skills/tree/91ca9e2801698466f934b557a75ce46db8ba46b5/ccf-common | 91ca9e280169 | `repos/ccfa-skills/ccf-common` | - |
| `ccf-experiment-designer` | installed | - | catalog | https://github.com/mikubaka88/CCFA-Skills/tree/91ca9e2801698466f934b557a75ce46db8ba46b5/ccf-experiment-designer | 91ca9e280169 | `repos/ccfa-skills/ccf-experiment-designer` | - |
| `ccf-humanization` | installed | - | catalog | https://github.com/mikubaka88/CCFA-Skills/tree/91ca9e2801698466f934b557a75ce46db8ba46b5/ccf-humanization | 91ca9e280169 | `repos/ccfa-skills/ccf-humanization` | - |
| `ccf-idea-optimizer` | installed | - | catalog | https://github.com/mikubaka88/CCFA-Skills/tree/91ca9e2801698466f934b557a75ce46db8ba46b5/ccf-idea-optimizer | 91ca9e280169 | `repos/ccfa-skills/ccf-idea-optimizer` | - |
| `ccf-idea-reviewer` | installed | - | catalog | https://github.com/mikubaka88/CCFA-Skills/tree/91ca9e2801698466f934b557a75ce46db8ba46b5/ccf-idea-reviewer | 91ca9e280169 | `repos/ccfa-skills/ccf-idea-reviewer` | - |
| `ccf-integrity-auditor` | installed | - | catalog | https://github.com/mikubaka88/CCFA-Skills/tree/91ca9e2801698466f934b557a75ce46db8ba46b5/ccf-integrity-auditor | 91ca9e280169 | `repos/ccfa-skills/ccf-integrity-auditor` | - |
| `ccf-literature-monitor` | installed | - | catalog | https://github.com/mikubaka88/CCFA-Skills/tree/91ca9e2801698466f934b557a75ce46db8ba46b5/ccf-literature-monitor | 91ca9e280169 | `repos/ccfa-skills/ccf-literature-monitor` | - |
| `ccf-literature-searcher` | installed | - | catalog | https://github.com/mikubaka88/CCFA-Skills/tree/91ca9e2801698466f934b557a75ce46db8ba46b5/ccf-literature-searcher | 91ca9e280169 | `repos/ccfa-skills/ccf-literature-searcher` | - |
| `ccf-paper-reviewer` | installed | - | catalog | https://github.com/mikubaka88/CCFA-Skills/tree/91ca9e2801698466f934b557a75ce46db8ba46b5/ccf-paper-reviewer | 91ca9e280169 | `repos/ccfa-skills/ccf-paper-reviewer` | - |
| `ccf-paper-to-exemplar` | installed | - | catalog | https://github.com/mikubaka88/CCFA-Skills/tree/91ca9e2801698466f934b557a75ce46db8ba46b5/ccf-paper-to-exemplar | 91ca9e280169 | `repos/ccfa-skills/ccf-paper-to-exemplar` | - |
| `ccf-paper-writer` | installed | - | catalog | https://github.com/mikubaka88/CCFA-Skills/tree/91ca9e2801698466f934b557a75ce46db8ba46b5/ccf-paper-writer | 91ca9e280169 | `repos/ccfa-skills/ccf-paper-writer` | - |
| `ccf-pipeline-orchestrator` | installed | - | catalog | https://github.com/mikubaka88/CCFA-Skills/tree/91ca9e2801698466f934b557a75ce46db8ba46b5/ccf-pipeline-orchestrator | 91ca9e280169 | `repos/ccfa-skills/ccf-pipeline-orchestrator` | - |
| `ccf-project-scaffolder` | installed | - | catalog | https://github.com/mikubaka88/CCFA-Skills/tree/91ca9e2801698466f934b557a75ce46db8ba46b5/ccf-project-scaffolder | 91ca9e280169 | `repos/ccfa-skills/ccf-project-scaffolder` | - |
| `ccf-rebuttal-writer` | installed | - | catalog | https://github.com/mikubaka88/CCFA-Skills/tree/91ca9e2801698466f934b557a75ce46db8ba46b5/ccf-rebuttal-writer | 91ca9e280169 | `repos/ccfa-skills/ccf-rebuttal-writer` | - |
| `ccf-research-engineer` | installed | - | catalog | https://github.com/mikubaka88/CCFA-Skills/tree/91ca9e2801698466f934b557a75ce46db8ba46b5/ccf-research-engineer | 91ca9e280169 | `repos/ccfa-skills/ccf-research-engineer` | - |
| `ccf-skill-forger` | installed | - | catalog | https://github.com/mikubaka88/CCFA-Skills/tree/91ca9e2801698466f934b557a75ce46db8ba46b5/ccf-skill-forger | 91ca9e280169 | `repos/ccfa-skills/ccf-skill-forger` | - |
| `ccf-submission-checker` | installed | - | catalog | https://github.com/mikubaka88/CCFA-Skills/tree/91ca9e2801698466f934b557a75ce46db8ba46b5/ccf-submission-checker | 91ca9e280169 | `repos/ccfa-skills/ccf-submission-checker` | - |
| `ccf-visual-composer` | installed | - | catalog | https://github.com/mikubaka88/CCFA-Skills/tree/91ca9e2801698466f934b557a75ce46db8ba46b5/ccf-visual-composer | 91ca9e280169 | `repos/ccfa-skills/ccf-visual-composer` | - |
| `claude-delegate` | protected | - | protected | https://github.com/amElnagdy/delegate-skills | - | `skills/other/claude-delegate` | lockfile 声明来源；ref 未知；保留本地 macOS 补丁，用户级保留名，安装器不覆盖。LICENSE 为 MIT，版权人 Ahmed Mohammed (amElnagdy) |
| `code-review` | installed | - | external | https://github.com/mattpocock/skills/tree/4ae2317ddb22b046e26455e4f9c9b3aaaf62212d/skills/engineering/code-review | 4ae2317ddb22 | `repos/mattpocock-skills/skills/engineering/code-review` | - |
| `codebase-design` | installed | - | external | https://github.com/mattpocock/skills/tree/4ae2317ddb22b046e26455e4f9c9b3aaaf62212d/skills/engineering/codebase-design | 4ae2317ddb22 | `repos/mattpocock-skills/skills/engineering/codebase-design` | - |
| `codebase-onboarding` | installed | - | catalog | https://github.com/Saddss/cursor-skills/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/skills/codebase-onboarding | 400c165e7a6a | `repos/cursor-skills/skills/codebase-onboarding` | - |
| `codex-background` | protected | - | protected | - | - | `skills/other/codex-background` | MIT，版权人 Ahmed Mohammed (amElnagdy)；本次审计未确立上游 URL（unknown）；保留本地补丁，用户级保留名，安装器不覆盖 |
| `colfax-research-skill` | installed | - | unknown | - | - | `skills/kernel/colfax-research-skill` | 本次审计未确立来源（未逐一读取该目录全部文件，不断言其无上游声明） |
| `compute-mamba-ratio` | installed | - | vendored | https://github.com/sgl-project/sglang/tree/3bcde0c80eeb07b40d3334c4b3d7341602efd089/.agents/skills/compute-mamba-ratio | 3bcde0c80eeb | `skills/framework/sglang-repo-skills/compute-mamba-ratio` | 无（全部逐字复制） |
| `compute-sanitizer-diagnose-cuda-bugs` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/compute-sanitizer-diagnose-cuda-bugs | fce3cca4170b | `repos/nvidia-skills/skills/compute-sanitizer-diagnose-cuda-bugs` | - |
| `cuda-skill` | installed | - | unknown | - | - | `skills/kernel/cuda-skill` | 本次审计未确立来源（未逐一读取该目录全部文件，不断言其无上游声明） |
| `cutlass-skill` | installed | - | unknown | - | - | `skills/kernel/cutlass-skill` | 本次审计未确立来源（未逐一读取该目录全部文件，不断言其无上游声明） |
| `debug-cuda-crash` | installed | - | vendored | https://github.com/sgl-project/sglang/tree/3bcde0c80eeb07b40d3334c4b3d7341602efd089/.agents/skills/debug-cuda-crash | 3bcde0c80eeb | `skills/framework/sglang-repo-skills/debug-cuda-crash` | 无（全部逐字复制） |
| `debug-distributed-hang` | installed | - | vendored | https://github.com/sgl-project/sglang/tree/3bcde0c80eeb07b40d3334c4b3d7341602efd089/.agents/skills/debug-distributed-hang | 3bcde0c80eeb | `skills/framework/sglang-repo-skills/debug-distributed-hang` | 无（全部逐字复制） |
| `debugging-tilelang-programs` | installed | - | catalog | https://github.com/sablin39/tilelang-cuda-skills/tree/842d1dc7d684861d2da241cb3e51137a47d5cad6/skills/tilelang/debugging-tilelang-programs | 842d1dc7d684 | `repos/tilelang-cuda-skills/skills/tilelang/debugging-tilelang-programs` | - |
| `deep-research` | installed | - | external | https://github.com/HKUSTDial/Supervisor-Skills/tree/4bd3fef393e5ffc891c74309a5ef6c5ad3ad12c8/skills/deep-research | 4bd3fef393e5 | `repos/supervisor-skills/skills/deep-research` | - |
| `deepgemm-skill` | installed | - | unknown | - | - | `skills/kernel/deepgemm-skill` | 本次审计未确立来源（未逐一读取该目录全部文件，不断言其无上游声明） |
| `designing-with-tilefoundry` | installed | - | catalog | https://github.com/sablin39/tilelang-cuda-skills/tree/842d1dc7d684861d2da241cb3e51137a47d5cad6/skills/designing-with-tilefoundry | 842d1dc7d684 | `repos/tilelang-cuda-skills/skills/designing-with-tilefoundry` | - |
| `diagnose` | installed | - | catalog | https://github.com/Saddss/cursor-skills/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/skills/diagnose | 400c165e7a6a | `repos/cursor-skills/skills/diagnose` | - |
| `diagnosing-bugs` | installed | - | external | https://github.com/mattpocock/skills/tree/4ae2317ddb22b046e26455e4f9c9b3aaaf62212d/skills/engineering/diagnosing-bugs | 4ae2317ddb22 | `repos/mattpocock-skills/skills/engineering/diagnosing-bugs` | - |
| `domain-modeling` | installed | - | external | https://github.com/mattpocock/skills/tree/4ae2317ddb22b046e26455e4f9c9b3aaaf62212d/skills/engineering/domain-modeling | 4ae2317ddb22 | `repos/mattpocock-skills/skills/engineering/domain-modeling` | - |
| `drawio-reconstruction` | installed | - | external | https://github.com/HKUSTDial/Supervisor-Skills/tree/4bd3fef393e5ffc891c74309a5ef6c5ad3ad12c8/skills/drawio-reconstruction | 4bd3fef393e5 | `repos/supervisor-skills/skills/drawio-reconstruction` | - |
| `dynamo-interconnect-check` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/dynamo-interconnect-check | fce3cca4170b | `repos/nvidia-skills/skills/dynamo-interconnect-check` | - |
| `dynamo-recipe-runner` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/dynamo-recipe-runner | fce3cca4170b | `repos/nvidia-skills/skills/dynamo-recipe-runner` | - |
| `dynamo-router-starter` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/dynamo-router-starter | fce3cca4170b | `repos/nvidia-skills/skills/dynamo-router-starter` | - |
| `dynamo-troubleshoot` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/dynamo-troubleshoot | fce3cca4170b | `repos/nvidia-skills/skills/dynamo-troubleshoot` | - |
| `executing-plans` | installed | - | catalog | https://github.com/Saddss/cursor-skills/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/skills/executing-plans | 400c165e7a6a | `repos/cursor-skills/skills/executing-plans` | - |
| `figure-designer` | installed | - | external | https://github.com/HKUSTDial/Supervisor-Skills/tree/4bd3fef393e5ffc891c74309a5ef6c5ad3ad12c8/skills/figure-designer | 4bd3fef393e5 | `repos/supervisor-skills/skills/figure-designer` | - |
| `flydsl-kernel-authoring` | installed | - | catalog | https://github.com/amd/skills/tree/9eba3192f6793237eaeacfabd61fb0bd557e53cf/skills/flydsl-kernel-authoring | 9eba3192f679 | `repos/amd-skills/skills/flydsl-kernel-authoring` | - |
| `flydsl-kernel-trace-analysis` | installed | - | catalog | https://github.com/amd/skills/tree/9eba3192f6793237eaeacfabd61fb0bd557e53cf/skills/flydsl-kernel-trace-analysis | 9eba3192f679 | `repos/amd-skills/skills/flydsl-kernel-trace-analysis` | - |
| `generate-profile` | installed | - | vendored | https://github.com/sgl-project/sglang/tree/3bcde0c80eeb07b40d3334c4b3d7341602efd089/.agents/skills/generate-profile | 3bcde0c80eeb | `skills/framework/sglang-repo-skills/generate-profile` | 无（全部逐字复制） |
| `giantpandallm-knowledge` | installed | - | unknown | - | - | `skills/other/giantpandallm-knowledge` | 本次审计未确立来源（未逐一读取该目录全部文件，不断言其无上游声明） |
| `git-guardrails-claude-code` | installed | - | external | https://github.com/mattpocock/skills/tree/4ae2317ddb22b046e26455e4f9c9b3aaaf62212d/skills/misc/git-guardrails-claude-code | 4ae2317ddb22 | `repos/mattpocock-skills/skills/misc/git-guardrails-claude-code` | - |
| `gpu-communication-libraries` | installed | - | unknown | - | - | `skills/other/gpu-communication-libraries` | 本次审计未确立来源（未逐一读取该目录全部文件，不断言其无上游声明） |
| `gpu-development-catchall` | installed | - | unknown | - | - | `skills/routers/gpu-development-catchall` | 本次审计未确立来源（未逐一读取该目录全部文件，不断言其无上游声明） |
| `gpu-framework-router` | installed | - | unknown | - | - | `skills/routers/gpu-framework-router` | 本次审计未确立来源（未逐一读取该目录全部文件，不断言其无上游声明） |
| `gpu-kernel-authoring-router` | installed | - | unknown | - | - | `skills/routers/gpu-kernel-authoring-router` | 本次审计未确立来源（未逐一读取该目录全部文件，不断言其无上游声明） |
| `gpu-kernel-reverse` | installed | - | unknown | - | - | `skills/kernel/gpu-kernel-reverse` | 本次审计未确立来源（未逐一读取该目录全部文件，不断言其无上游声明） |
| `gpu-performance-router` | installed | - | unknown | - | - | `skills/routers/gpu-performance-router` | 本次审计未确立来源（未逐一读取该目录全部文件，不断言其无上游声明） |
| `grill-me` | installed | - | external | https://github.com/mattpocock/skills/tree/4ae2317ddb22b046e26455e4f9c9b3aaaf62212d/skills/productivity/grill-me | 4ae2317ddb22 | `repos/mattpocock-skills/skills/productivity/grill-me` | - |
| `grill-with-docs` | installed | - | catalog | https://github.com/Saddss/cursor-skills/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/skills/grill-with-docs | 400c165e7a6a | `repos/cursor-skills/skills/grill-with-docs` | - |
| `grill-with-docs` | shadowed | cursor-skills | external | https://github.com/mattpocock/skills/tree/4ae2317ddb22b046e26455e4f9c9b3aaaf62212d/skills/engineering/grill-with-docs | 4ae2317ddb22 | `repos/mattpocock-skills/skills/engineering/grill-with-docs` | - |
| `grilling` | installed | - | external | https://github.com/mattpocock/skills/tree/4ae2317ddb22b046e26455e4f9c9b3aaaf62212d/skills/productivity/grilling | 4ae2317ddb22 | `repos/mattpocock-skills/skills/productivity/grilling` | - |
| `handoff` | installed | - | catalog | https://github.com/Saddss/cursor-skills/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/skills/handoff | 400c165e7a6a | `repos/cursor-skills/skills/handoff` | - |
| `handoff` | shadowed | cursor-skills | external | https://github.com/mattpocock/skills/tree/4ae2317ddb22b046e26455e4f9c9b3aaaf62212d/skills/productivity/handoff | 4ae2317ddb22 | `repos/mattpocock-skills/skills/productivity/handoff` | - |
| `high-performance-code-review` | installed | - | catalog | https://github.com/Saddss/cursor-skills/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/skills/high-performance-code-review | 400c165e7a6a | `repos/cursor-skills/skills/high-performance-code-review` | - |
| `hyperloom-workload-optimizer` | installed | - | catalog | https://github.com/amd/skills/tree/9eba3192f6793237eaeacfabd61fb0bd557e53cf/skills/hyperloom-workload-optimizer | 9eba3192f679 | `repos/amd-skills/skills/hyperloom-workload-optimizer` | - |
| `idea-evaluator` | installed | - | external | https://github.com/HKUSTDial/Supervisor-Skills/tree/4bd3fef393e5ffc891c74309a5ef6c5ad3ad12c8/skills/idea-evaluator | 4bd3fef393e5 | `repos/supervisor-skills/skills/idea-evaluator` | - |
| `iket-profiling` | installed | - | external-declared | https://github.com/humanfia/iket-profiling-skill | - | `skills/kernel/iket-profiling` | UPSTREAM.md 声明安装来源为 humanfia/iket-profiling-skill（无 pin）；内容是 NVIDIA CUTLASS 官方 IKET 文档的摘要，NVIDIA 文档是参考来源，不是本 skill 的 origin |
| `implement` | installed | - | external | https://github.com/mattpocock/skills/tree/4ae2317ddb22b046e26455e4f9c9b3aaaf62212d/skills/engineering/implement | 4ae2317ddb22 | `repos/mattpocock-skills/skills/engineering/implement` | - |
| `implement-spec` | installed | - | external | https://github.com/mattpocock/skills/tree/4ae2317ddb22b046e26455e4f9c9b3aaaf62212d/skills/engineering/implement-spec | 4ae2317ddb22 | `repos/mattpocock-skills/skills/engineering/implement-spec` | - |
| `improve-codebase-architecture` | installed | - | catalog | https://github.com/Saddss/cursor-skills/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/skills/improve-codebase-architecture | 400c165e7a6a | `repos/cursor-skills/skills/improve-codebase-architecture` | - |
| `improve-codebase-architecture` | shadowed | cursor-skills | external | https://github.com/mattpocock/skills/tree/4ae2317ddb22b046e26455e4f9c9b3aaaf62212d/skills/engineering/improve-codebase-architecture | 4ae2317ddb22 | `repos/mattpocock-skills/skills/engineering/improve-codebase-architecture` | - |
| `intro-drafter` | installed | - | external | https://github.com/HKUSTDial/Supervisor-Skills/tree/4bd3fef393e5ffc891c74309a5ef6c5ad3ad12c8/skills/intro-drafter | 4bd3fef393e5 | `repos/supervisor-skills/skills/intro-drafter` | - |
| `karpathy-guidelines` | installed | - | catalog | https://github.com/Saddss/cursor-skills/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/skills/karpathy-guidelines | 400c165e7a6a | `repos/cursor-skills/skills/karpathy-guidelines` | - |
| `kernel-cute-writing` | installed | - | catalog | https://github.com/Saddss/cursor-skills/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/skills/kernel-cute-writing | 400c165e7a6a | `repos/cursor-skills/skills/kernel-cute-writing` | - |
| `kernel-organization` | installed | - | vendored | https://github.com/sgl-project/sglang/tree/3bcde0c80eeb07b40d3334c4b3d7341602efd089/.agents/skills/kernel-organization | 3bcde0c80eeb | `skills/framework/sglang-repo-skills/kernel-organization` | 无（全部逐字复制） |
| `kernel-triton-writing` | installed | - | catalog | https://github.com/Saddss/cursor-skills/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/skills/kernel-triton-writing | 400c165e7a6a | `repos/cursor-skills/skills/kernel-triton-writing` | - |
| `kl-consistency-test` | installed | - | vendored | https://github.com/sgl-project/sglang/tree/3bcde0c80eeb07b40d3334c4b3d7341602efd089/.agents/skills/kl-consistency-test | 3bcde0c80eeb | `skills/framework/sglang-repo-skills/kl-consistency-test` | 无（全部逐字复制） |
| `launch-nemo-rl` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/launch-nemo-rl | fce3cca4170b | `repos/nvidia-skills/skills/launch-nemo-rl` | - |
| `lemonade-router-builder` | installed | - | catalog | https://github.com/amd/skills/tree/9eba3192f6793237eaeacfabd61fb0bd557e53cf/skills/lemonade-router-builder | 9eba3192f679 | `repos/amd-skills/skills/lemonade-router-builder` | - |
| `llm-pipeline-analysis` | installed | - | catalog | https://github.com/Saddss/cursor-skills/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/skills/llm-pipeline-analysis | 400c165e7a6a | `repos/cursor-skills/skills/llm-pipeline-analysis` | - |
| `llm-serving-auto-benchmark` | installed | - | catalog | https://github.com/Saddss/cursor-skills/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/skills/llm-serving-auto-benchmark | 400c165e7a6a | `repos/cursor-skills/skills/llm-serving-auto-benchmark` | - |
| `llm-serving-capacity-planner` | installed | - | catalog | https://github.com/Saddss/cursor-skills/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/skills/llm-serving-capacity-planner | 400c165e7a6a | `repos/cursor-skills/skills/llm-serving-capacity-planner` | - |
| `llm-serving-router` | installed | - | unknown | - | - | `skills/routers/llm-serving-router` | 本次审计未确立来源（未逐一读取该目录全部文件，不断言其无上游声明） |
| `llm-torch-profiler-analysis` | installed | - | catalog | https://github.com/Saddss/cursor-skills/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/skills/llm-torch-profiler-analysis | 400c165e7a6a | `repos/cursor-skills/skills/llm-torch-profiler-analysis` | - |
| `local-ai-app-integration` | installed | - | catalog | https://github.com/amd/skills/tree/9eba3192f6793237eaeacfabd61fb0bd557e53cf/skills/local-ai-app-integration | 9eba3192f679 | `repos/amd-skills/skills/local-ai-app-integration` | - |
| `local-ai-use` | installed | - | catalog | https://github.com/amd/skills/tree/9eba3192f6793237eaeacfabd61fb0bd557e53cf/skills/local-ai-use | 9eba3192f679 | `repos/amd-skills/skills/local-ai-use` | - |
| `magpie-kernel-evaluator` | installed | - | catalog | https://github.com/amd/skills/tree/9eba3192f6793237eaeacfabd61fb0bd557e53cf/skills/magpie-kernel-evaluator | 9eba3192f679 | `repos/amd-skills/skills/magpie-kernel-evaluator` | - |
| `mcore-create-issue` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/mcore-create-issue | fce3cca4170b | `repos/nvidia-skills/skills/mcore-create-issue` | - |
| `mcore-linting-and-formatting` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/mcore-linting-and-formatting | fce3cca4170b | `repos/nvidia-skills/skills/mcore-linting-and-formatting` | - |
| `mcore-run-on-slurm` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/mcore-run-on-slurm | fce3cca4170b | `repos/nvidia-skills/skills/mcore-run-on-slurm` | - |
| `mcore-split-pr` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/mcore-split-pr | fce3cca4170b | `repos/nvidia-skills/skills/mcore-split-pr` | - |
| `mcore-testing` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/mcore-testing | fce3cca4170b | `repos/nvidia-skills/skills/mcore-testing` | - |
| `migrate-to-shoehorn` | installed | - | external | https://github.com/mattpocock/skills/tree/4ae2317ddb22b046e26455e4f9c9b3aaaf62212d/skills/misc/migrate-to-shoehorn | 4ae2317ddb22 | `repos/mattpocock-skills/skills/misc/migrate-to-shoehorn` | - |
| `model-compute-simulation` | installed | - | catalog | https://github.com/Saddss/cursor-skills/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/skills/model-compute-simulation | 400c165e7a6a | `repos/cursor-skills/skills/model-compute-simulation` | - |
| `model-perf-binary-search` | installed | - | catalog | https://github.com/Saddss/cursor-skills/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/skills/model-perf-binary-search | 400c165e7a6a | `repos/cursor-skills/skills/model-perf-binary-search` | - |
| `nccl-skill` | installed | - | unknown | - | - | `skills/other/nccl-skill` | 本次审计未确立来源（未逐一读取该目录全部文件，不断言其无上游声明） |
| `ncu-persistent-kernel-diagnosis` | installed | - | unknown | - | - | `skills/kernel/ncu-persistent-kernel-diagnosis` | 本次审计未确立来源（未逐一读取该目录全部文件，不断言其无上游声明） |
| `ncu-profile-analysis` | optional-runtime | - | optional-runtime | https://github.com/lucifer1004/veloq | - | `install-veloq.sh` | install-veloq.sh 从 raw.githubusercontent.com/lucifer1004/veloq/main/scripts/install.sh 安装（main，无 pin）；--no-veloq 跳过 |
| `ncu-report-skill` | installed | - | external-declared | https://github.com/mit-han-lab/ncu-report-skill | - | `skills/kernel/ncu-report-skill` | README 称为 KDA 的独立 submodule；KDA .gitmodules 指向 mit-han-lab/ncu-report-skill；README 安装示例另指 DongyunZou/ncu-report-skill（两处来源待确认）；本地副本无 .git，ref 未知 |
| `nemo-automodel-distributed-training` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nemo-automodel-distributed-training | fce3cca4170b | `repos/nvidia-skills/skills/nemo-automodel-distributed-training` | - |
| `nemo-automodel-launcher-config` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nemo-automodel-launcher-config | fce3cca4170b | `repos/nvidia-skills/skills/nemo-automodel-launcher-config` | - |
| `nemo-automodel-model-onboarding` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nemo-automodel-model-onboarding | fce3cca4170b | `repos/nvidia-skills/skills/nemo-automodel-model-onboarding` | - |
| `nemo-automodel-recipe-development` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nemo-automodel-recipe-development | fce3cca4170b | `repos/nvidia-skills/skills/nemo-automodel-recipe-development` | - |
| `nemo-mbridge-mlm-bridge-training` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nemo-mbridge-mlm-bridge-training | fce3cca4170b | `repos/nvidia-skills/skills/nemo-mbridge-mlm-bridge-training` | - |
| `nemo-mbridge-multi-node-slurm` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nemo-mbridge-multi-node-slurm | fce3cca4170b | `repos/nvidia-skills/skills/nemo-mbridge-multi-node-slurm` | - |
| `nemo-mbridge-perf-activation-recompute` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nemo-mbridge-perf-activation-recompute | fce3cca4170b | `repos/nvidia-skills/skills/nemo-mbridge-perf-activation-recompute` | - |
| `nemo-mbridge-perf-cpu-offloading` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nemo-mbridge-perf-cpu-offloading | fce3cca4170b | `repos/nvidia-skills/skills/nemo-mbridge-perf-cpu-offloading` | - |
| `nemo-mbridge-perf-cuda-graphs` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nemo-mbridge-perf-cuda-graphs | fce3cca4170b | `repos/nvidia-skills/skills/nemo-mbridge-perf-cuda-graphs` | - |
| `nemo-mbridge-perf-expert-parallel-overlap` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nemo-mbridge-perf-expert-parallel-overlap | fce3cca4170b | `repos/nvidia-skills/skills/nemo-mbridge-perf-expert-parallel-overlap` | - |
| `nemo-mbridge-perf-hierarchical-context-parallel` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nemo-mbridge-perf-hierarchical-context-parallel | fce3cca4170b | `repos/nvidia-skills/skills/nemo-mbridge-perf-hierarchical-context-parallel` | - |
| `nemo-mbridge-perf-megatron-fsdp` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nemo-mbridge-perf-megatron-fsdp | fce3cca4170b | `repos/nvidia-skills/skills/nemo-mbridge-perf-megatron-fsdp` | - |
| `nemo-mbridge-perf-memory-tuning` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nemo-mbridge-perf-memory-tuning | fce3cca4170b | `repos/nvidia-skills/skills/nemo-mbridge-perf-memory-tuning` | - |
| `nemo-mbridge-perf-moe-comm-overlap` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nemo-mbridge-perf-moe-comm-overlap | fce3cca4170b | `repos/nvidia-skills/skills/nemo-mbridge-perf-moe-comm-overlap` | - |
| `nemo-mbridge-perf-moe-dispatcher-selection` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nemo-mbridge-perf-moe-dispatcher-selection | fce3cca4170b | `repos/nvidia-skills/skills/nemo-mbridge-perf-moe-dispatcher-selection` | - |
| `nemo-mbridge-perf-moe-hardware-configs` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nemo-mbridge-perf-moe-hardware-configs | fce3cca4170b | `repos/nvidia-skills/skills/nemo-mbridge-perf-moe-hardware-configs` | - |
| `nemo-mbridge-perf-moe-long-context` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nemo-mbridge-perf-moe-long-context | fce3cca4170b | `repos/nvidia-skills/skills/nemo-mbridge-perf-moe-long-context` | - |
| `nemo-mbridge-perf-moe-optimization-workflow` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nemo-mbridge-perf-moe-optimization-workflow | fce3cca4170b | `repos/nvidia-skills/skills/nemo-mbridge-perf-moe-optimization-workflow` | - |
| `nemo-mbridge-perf-moe-vlm-training` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nemo-mbridge-perf-moe-vlm-training | fce3cca4170b | `repos/nvidia-skills/skills/nemo-mbridge-perf-moe-vlm-training` | - |
| `nemo-mbridge-perf-parallelism-strategies` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nemo-mbridge-perf-parallelism-strategies | fce3cca4170b | `repos/nvidia-skills/skills/nemo-mbridge-perf-parallelism-strategies` | - |
| `nemo-mbridge-perf-sequence-packing` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nemo-mbridge-perf-sequence-packing | fce3cca4170b | `repos/nvidia-skills/skills/nemo-mbridge-perf-sequence-packing` | - |
| `nemo-mbridge-perf-tp-dp-comm-overlap` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nemo-mbridge-perf-tp-dp-comm-overlap | fce3cca4170b | `repos/nvidia-skills/skills/nemo-mbridge-perf-tp-dp-comm-overlap` | - |
| `nemo-mbridge-recipe-recommender` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nemo-mbridge-recipe-recommender | fce3cca4170b | `repos/nvidia-skills/skills/nemo-mbridge-recipe-recommender` | - |
| `nemo-mbridge-resiliency` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nemo-mbridge-resiliency | fce3cca4170b | `repos/nvidia-skills/skills/nemo-mbridge-resiliency` | - |
| `nemo-retriever-mcp` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nemo-retriever-mcp | fce3cca4170b | `repos/nvidia-skills/skills/nemo-retriever-mcp` | - |
| `nemo-rl-auto-research` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nemo-rl-auto-research | fce3cca4170b | `repos/nvidia-skills/skills/nemo-rl-auto-research` | - |
| `nemo-rl-brev-etiquette` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nemo-rl-brev-etiquette | fce3cca4170b | `repos/nvidia-skills/skills/nemo-rl-brev-etiquette` | - |
| `nemo-rl-docs` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nemo-rl-docs | fce3cca4170b | `repos/nvidia-skills/skills/nemo-rl-docs` | - |
| `nemo-rl-session-memory` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nemo-rl-session-memory | fce3cca4170b | `repos/nvidia-skills/skills/nemo-rl-session-memory` | - |
| `nsys-profile-analysis` | optional-runtime | - | optional-runtime | https://github.com/lucifer1004/veloq | - | `install-veloq.sh` | install-veloq.sh 从 raw.githubusercontent.com/lucifer1004/veloq/main/scripts/install.sh 安装（main，无 pin）；--no-veloq 跳过 |
| `nv-gpu-kernel-performance-modeling` | installed | - | unknown | - | - | `skills/kernel/nv-gpu-kernel-performance-modeling` | 本次审计未确立来源（未逐一读取该目录全部文件，不断言其无上游声明） |
| `nvshmem-collect-performance-data` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nvshmem-collect-performance-data | fce3cca4170b | `repos/nvidia-skills/skills/nvshmem-collect-performance-data` | - |
| `nvshmem-configure-nic-pe-mapping` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nvshmem-configure-nic-pe-mapping | fce3cca4170b | `repos/nvidia-skills/skills/nvshmem-configure-nic-pe-mapping` | - |
| `nvshmem-docs` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nvshmem-docs | fce3cca4170b | `repos/nvidia-skills/skills/nvshmem-docs` | - |
| `nvshmem-enable-tma` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nvshmem-enable-tma | fce3cca4170b | `repos/nvidia-skills/skills/nvshmem-enable-tma` | - |
| `nvshmem-get-started` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nvshmem-get-started | fce3cca4170b | `repos/nvidia-skills/skills/nvshmem-get-started` | - |
| `nvshmem-install` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nvshmem-install | fce3cca4170b | `repos/nvidia-skills/skills/nvshmem-install` | - |
| `nvshmem-select-remote-transport` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nvshmem-select-remote-transport | fce3cca4170b | `repos/nvidia-skills/skills/nvshmem-select-remote-transport` | - |
| `nvshmem-troubleshoot-and-report-bugs` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nvshmem-troubleshoot-and-report-bugs | fce3cca4170b | `repos/nvidia-skills/skills/nvshmem-troubleshoot-and-report-bugs` | - |
| `nvshmem-tune-performance` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/nvshmem-tune-performance | fce3cca4170b | `repos/nvidia-skills/skills/nvshmem-tune-performance` | - |
| `optimizing-tilelang-programs` | installed | - | catalog | https://github.com/sablin39/tilelang-cuda-skills/tree/842d1dc7d684861d2da241cb3e51137a47d5cad6/skills/tilelang/optimizing-tilelang-programs | 842d1dc7d684 | `repos/tilelang-cuda-skills/skills/tilelang/optimizing-tilelang-programs` | - |
| `paper-polish` | installed | - | external | https://github.com/HKUSTDial/Supervisor-Skills/tree/4bd3fef393e5ffc891c74309a5ef6c5ad3ad12c8/skills/paper-polish | 4bd3fef393e5 | `repos/supervisor-skills/skills/paper-polish` | - |
| `paper-writer` | installed | - | external | https://github.com/HKUSTDial/Supervisor-Skills/tree/4bd3fef393e5ffc891c74309a5ef6c5ad3ad12c8/skills/paper-writer | 4bd3fef393e5 | `repos/supervisor-skills/skills/paper-writer` | - |
| `parallel-exploring` | installed | - | catalog | https://github.com/Saddss/cursor-skills/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/skills/parallel-exploring | 400c165e7a6a | `repos/cursor-skills/skills/parallel-exploring` | - |
| `perf-analysis` | installed | - | catalog | https://github.com/Saddss/cursor-skills/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/skills/perf-analysis | 400c165e7a6a | `repos/cursor-skills/skills/perf-analysis` | - |
| `perf-host-analysis` | installed | - | catalog | https://github.com/Saddss/cursor-skills/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/skills/perf-host-analysis | 400c165e7a6a | `repos/cursor-skills/skills/perf-host-analysis` | - |
| `perf-host-optimization` | installed | - | catalog | https://github.com/Saddss/cursor-skills/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/skills/perf-host-optimization | 400c165e7a6a | `repos/cursor-skills/skills/perf-host-optimization` | - |
| `perf-nsight-compute-analysis` | installed | - | catalog | https://github.com/Saddss/cursor-skills/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/skills/perf-nsight-compute-analysis | 400c165e7a6a | `repos/cursor-skills/skills/perf-nsight-compute-analysis` | - |
| `perf-nsight-systems` | installed | - | catalog | https://github.com/Saddss/cursor-skills/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/skills/perf-nsight-systems | 400c165e7a6a | `repos/cursor-skills/skills/perf-nsight-systems` | - |
| `perf-optimization` | installed | - | catalog | https://github.com/Saddss/cursor-skills/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/skills/perf-optimization | 400c165e7a6a | `repos/cursor-skills/skills/perf-optimization` | - |
| `perf-workload-profiling` | installed | - | catalog | https://github.com/Saddss/cursor-skills/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/skills/perf-workload-profiling | 400c165e7a6a | `repos/cursor-skills/skills/perf-workload-profiling` | - |
| `persistent-kernel-scheduling` | installed | - | unknown | - | - | `skills/kernel/persistent-kernel-scheduling` | 本次审计未确立来源（未逐一读取该目录全部文件，不断言其无上游声明） |
| `persistent-kernel-utilization` | installed | - | unknown | - | - | `skills/kernel/persistent-kernel-utilization` | 本次审计未确立来源（未逐一读取该目录全部文件，不断言其无上游声明） |
| `ponytail` | installed | - | catalog | https://github.com/DietrichGebert/ponytail/tree/96608e078b2d5ea607c5c3b8b5a54d1c1fba6846/skills/ponytail | 96608e078b2d | `repos/ponytail/skills/ponytail` | - |
| `ponytail-audit` | installed | - | catalog | https://github.com/DietrichGebert/ponytail/tree/96608e078b2d5ea607c5c3b8b5a54d1c1fba6846/skills/ponytail-audit | 96608e078b2d | `repos/ponytail/skills/ponytail-audit` | - |
| `ponytail-debt` | installed | - | catalog | https://github.com/DietrichGebert/ponytail/tree/96608e078b2d5ea607c5c3b8b5a54d1c1fba6846/skills/ponytail-debt | 96608e078b2d | `repos/ponytail/skills/ponytail-debt` | - |
| `ponytail-gain` | installed | - | catalog | https://github.com/DietrichGebert/ponytail/tree/96608e078b2d5ea607c5c3b8b5a54d1c1fba6846/skills/ponytail-gain | 96608e078b2d | `repos/ponytail/skills/ponytail-gain` | - |
| `ponytail-help` | installed | - | catalog | https://github.com/DietrichGebert/ponytail/tree/96608e078b2d5ea607c5c3b8b5a54d1c1fba6846/skills/ponytail-help | 96608e078b2d | `repos/ponytail/skills/ponytail-help` | - |
| `ponytail-review` | installed | - | catalog | https://github.com/DietrichGebert/ponytail/tree/96608e078b2d5ea607c5c3b8b5a54d1c1fba6846/skills/ponytail-review | 96608e078b2d | `repos/ponytail/skills/ponytail-review` | - |
| `pp-separation-benchmark` | installed | - | catalog | https://github.com/Saddss/cursor-skills/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/skills/pp-separation-benchmark | 400c165e7a6a | `repos/cursor-skills/skills/pp-separation-benchmark` | - |
| `pr` | installed | - | external | https://github.com/mattpocock/skills/tree/4ae2317ddb22b046e26455e4f9c9b3aaaf62212d/skills/engineering/pr | 4ae2317ddb22 | `repos/mattpocock-skills/skills/engineering/pr` | - |
| `pre-submission-reviewer` | installed | - | external | https://github.com/HKUSTDial/Supervisor-Skills/tree/4bd3fef393e5ffc891c74309a5ef6c5ad3ad12c8/skills/pre-submission-reviewer | 4bd3fef393e5 | `repos/supervisor-skills/skills/pre-submission-reviewer` | - |
| `profiling-tilelang-programs` | installed | - | catalog | https://github.com/sablin39/tilelang-cuda-skills/tree/842d1dc7d684861d2da241cb3e51137a47d5cad6/skills/tilelang/profiling-tilelang-programs | 842d1dc7d684 | `repos/tilelang-cuda-skills/skills/tilelang/profiling-tilelang-programs` | - |
| `quark-install` | installed | - | catalog | https://github.com/amd/skills/tree/9eba3192f6793237eaeacfabd61fb0bd557e53cf/skills/quark-install | 9eba3192f679 | `repos/amd-skills/skills/quark-install` | - |
| `quark-torch-llm-ptq` | installed | - | catalog | https://github.com/amd/skills/tree/9eba3192f6793237eaeacfabd61fb0bd557e53cf/skills/quark-torch-llm-ptq | 9eba3192f679 | `repos/amd-skills/skills/quark-torch-llm-ptq` | - |
| `rebuttal-guidance` | installed | - | external | https://github.com/HKUSTDial/Supervisor-Skills/tree/4bd3fef393e5ffc891c74309a5ef6c5ad3ad12c8/skills/rebuttal-guidance | 4bd3fef393e5 | `repos/supervisor-skills/skills/rebuttal-guidance` | - |
| `research` | installed | - | external | https://github.com/mattpocock/skills/tree/4ae2317ddb22b046e26455e4f9c9b3aaaf62212d/skills/engineering/research | 4ae2317ddb22 | `repos/mattpocock-skills/skills/engineering/research` | - |
| `retro` | installed | - | external | https://github.com/mattpocock/skills/tree/4ae2317ddb22b046e26455e4f9c9b3aaaf62212d/skills/engineering/retro | 4ae2317ddb22 | `repos/mattpocock-skills/skills/engineering/retro` | - |
| `review` | installed | - | catalog | https://github.com/Saddss/cursor-skills/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/skills/review | 400c165e7a6a | `repos/cursor-skills/skills/review` | - |
| `rocm-doctor` | installed | - | catalog | https://github.com/amd/skills/tree/9eba3192f6793237eaeacfabd61fb0bd557e53cf/skills/rocm-doctor | 9eba3192f679 | `repos/amd-skills/skills/rocm-doctor` | - |
| `rocprofiler-pc-sampling` | installed | - | catalog | https://github.com/amd/skills/tree/9eba3192f6793237eaeacfabd61fb0bd557e53cf/skills/rocprofiler-pc-sampling | 9eba3192f679 | `repos/amd-skills/skills/rocprofiler-pc-sampling` | - |
| `rocprofiler-systems-choosing-a-profiling-preset` | installed | - | catalog | https://github.com/amd/skills/tree/9eba3192f6793237eaeacfabd61fb0bd557e53cf/skills/rocprofiler-systems-choosing-a-profiling-preset | 9eba3192f679 | `repos/amd-skills/skills/rocprofiler-systems-choosing-a-profiling-preset` | - |
| `rocprofiler-systems-discovering-profiling-options` | installed | - | catalog | https://github.com/amd/skills/tree/9eba3192f6793237eaeacfabd61fb0bd557e53cf/skills/rocprofiler-systems-discovering-profiling-options | 9eba3192f679 | `repos/amd-skills/skills/rocprofiler-systems-discovering-profiling-options` | - |
| `rocprofiler-systems-instrumenting-binaries` | installed | - | catalog | https://github.com/amd/skills/tree/9eba3192f6793237eaeacfabd61fb0bd557e53cf/skills/rocprofiler-systems-instrumenting-binaries | 9eba3192f679 | `repos/amd-skills/skills/rocprofiler-systems-instrumenting-binaries` | - |
| `scaffold-exercises` | installed | - | external | https://github.com/mattpocock/skills/tree/4ae2317ddb22b046e26455e4f9c9b3aaaf62212d/skills/misc/scaffold-exercises | 4ae2317ddb22 | `repos/mattpocock-skills/skills/misc/scaffold-exercises` | - |
| `search-first` | installed | - | catalog | https://github.com/Saddss/cursor-skills/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/skills/search-first | 400c165e7a6a | `repos/cursor-skills/skills/search-first` | - |
| `serving-llms-on-epyc` | installed | - | catalog | https://github.com/amd/skills/tree/9eba3192f6793237eaeacfabd61fb0bd557e53cf/skills/serving-llms-on-epyc | 9eba3192f679 | `repos/amd-skills/skills/serving-llms-on-epyc` | - |
| `serving-llms-on-instinct` | installed | - | catalog | https://github.com/amd/skills/tree/9eba3192f6793237eaeacfabd61fb0bd557e53cf/skills/serving-llms-on-instinct | 9eba3192f679 | `repos/amd-skills/skills/serving-llms-on-instinct` | - |
| `setup-matt-pocock-skills` | installed | - | catalog | https://github.com/Saddss/cursor-skills/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/skills/setup-matt-pocock-skills | 400c165e7a6a | `repos/cursor-skills/skills/setup-matt-pocock-skills` | - |
| `setup-matt-pocock-skills` | shadowed | cursor-skills | external | https://github.com/mattpocock/skills/tree/4ae2317ddb22b046e26455e4f9c9b3aaaf62212d/skills/engineering/setup-matt-pocock-skills | 4ae2317ddb22 | `repos/mattpocock-skills/skills/engineering/setup-matt-pocock-skills` | - |
| `setup-pre-commit` | installed | - | external | https://github.com/mattpocock/skills/tree/4ae2317ddb22b046e26455e4f9c9b3aaaf62212d/skills/misc/setup-pre-commit | 4ae2317ddb22 | `repos/mattpocock-skills/skills/misc/setup-pre-commit` | - |
| `sglang-prod-incident-triage` | installed | - | vendored | https://github.com/sgl-project/sglang/tree/3bcde0c80eeb07b40d3334c4b3d7341602efd089/.agents/skills/sglang-prod-incident-triage | 3bcde0c80eeb | `skills/framework/sglang-repo-skills/sglang-prod-incident-triage` | 无（全部逐字复制） |
| `sglang-skill` | installed | - | unknown | - | - | `skills/framework/sglang-skill` | 本次审计未确立来源（未逐一读取该目录全部文件，不断言其无上游声明） |
| `simplify-code` | installed | - | catalog | https://github.com/Saddss/cursor-skills/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/skills/simplify-code | 400c165e7a6a | `repos/cursor-skills/skills/simplify-code` | - |
| `tdd` | installed | - | catalog | https://github.com/Saddss/cursor-skills/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/skills/tdd | 400c165e7a6a | `repos/cursor-skills/skills/tdd` | - |
| `tdd` | shadowed | cursor-skills | external | https://github.com/mattpocock/skills/tree/4ae2317ddb22b046e26455e4f9c9b3aaaf62212d/skills/engineering/tdd | 4ae2317ddb22 | `repos/mattpocock-skills/skills/engineering/tdd` | - |
| `teach` | installed | - | external | https://github.com/mattpocock/skills/tree/4ae2317ddb22b046e26455e4f9c9b3aaaf62212d/skills/productivity/teach | 4ae2317ddb22 | `repos/mattpocock-skills/skills/productivity/teach` | - |
| `tech-doc-style-chinese` | installed | - | vendored | https://github.com/Fenng/Tech-Doc-Style-Chinese/tree/b119d01c9bb7cd132b3c62fafdff34227c4410d1 | b119d01c9bb7 | `skills/other/tech-doc-style-chinese` | 本地定制覆盖版（修改对照见其 README.md）；上游 clone 在 repos/tech-doc-style-chinese（git-ignored） |
| `tech-paper-template` | installed | - | external | https://github.com/HKUSTDial/Supervisor-Skills/tree/4bd3fef393e5ffc891c74309a5ef6c5ad3ad12c8/skills/tech-paper-template | 4bd3fef393e5 | `repos/supervisor-skills/skills/tech-paper-template` | - |
| `testing-fwd-bwd-kernels` | installed | - | catalog | https://github.com/sablin39/tilelang-cuda-skills/tree/842d1dc7d684861d2da241cb3e51137a47d5cad6/skills/tilelang/testing-fwd-bwd-kernels | 842d1dc7d684 | `repos/tilelang-cuda-skills/skills/tilelang/testing-fwd-bwd-kernels` | - |
| `tilegym-adding-cutile-kernel` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/tilegym-adding-cutile-kernel | fce3cca4170b | `repos/nvidia-skills/skills/tilegym-adding-cutile-kernel` | - |
| `tilegym-converting-cutile-to-julia` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/tilegym-converting-cutile-to-julia | fce3cca4170b | `repos/nvidia-skills/skills/tilegym-converting-cutile-to-julia` | - |
| `tilegym-converting-cutile-to-triton` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/tilegym-converting-cutile-to-triton | fce3cca4170b | `repos/nvidia-skills/skills/tilegym-converting-cutile-to-triton` | - |
| `tilegym-cutile-autotuning` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/tilegym-cutile-autotuning | fce3cca4170b | `repos/nvidia-skills/skills/tilegym-cutile-autotuning` | - |
| `tilegym-cutile-python` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/tilegym-cutile-python | fce3cca4170b | `repos/nvidia-skills/skills/tilegym-cutile-python` | - |
| `tilegym-improve-cutile-kernel-perf` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/tilegym-improve-cutile-kernel-perf | fce3cca4170b | `repos/nvidia-skills/skills/tilegym-improve-cutile-kernel-perf` | - |
| `tilegym-monkey-patch-kernels-to-transformers` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/tilegym-monkey-patch-kernels-to-transformers | fce3cca4170b | `repos/nvidia-skills/skills/tilegym-monkey-patch-kernels-to-transformers` | - |
| `tilelang-ascend` | installed | - | vendored | https://github.com/tile-ai/tilelang/tree/a8f3d3bcfe7fdd1a3887b05ff8f054e7d08980de/.agents/skills/tilelang-ascend | a8f3d3bcfe7f | `skills/kernel/tilelang-repo-skills/tilelang-ascend` | 无（全部逐字复制） |
| `tilelang-backend` | installed | - | vendored | https://github.com/tile-ai/tilelang/tree/a8f3d3bcfe7fdd1a3887b05ff8f054e7d08980de/.agents/skills/tilelang-backend | a8f3d3bcfe7f | `skills/kernel/tilelang-repo-skills/tilelang-backend` | 无（全部逐字复制） |
| `tilelang-layout` | installed | - | vendored | https://github.com/tile-ai/tilelang/tree/a8f3d3bcfe7fdd1a3887b05ff8f054e7d08980de/.agents/skills/tilelang-layout | a8f3d3bcfe7f | `skills/kernel/tilelang-repo-skills/tilelang-layout` | 无（全部逐字复制） |
| `tilelang-optimization-loop` | installed | - | catalog | https://github.com/sablin39/tilelang-cuda-skills/tree/842d1dc7d684861d2da241cb3e51137a47d5cad6/skills/tilelang-optimization-loop | 842d1dc7d684 | `repos/tilelang-cuda-skills/skills/tilelang-optimization-loop` | - |
| `tilelang-semantic` | installed | - | vendored | https://github.com/tile-ai/tilelang/tree/a8f3d3bcfe7fdd1a3887b05ff8f054e7d08980de/.agents/skills/tilelang-semantic | a8f3d3bcfe7f | `skills/kernel/tilelang-repo-skills/tilelang-semantic` | 无（全部逐字复制） |
| `to-issues` | installed | - | catalog | https://github.com/Saddss/cursor-skills/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/skills/to-issues | 400c165e7a6a | `repos/cursor-skills/skills/to-issues` | - |
| `to-questionnaire` | installed | - | external | https://github.com/mattpocock/skills/tree/4ae2317ddb22b046e26455e4f9c9b3aaaf62212d/skills/productivity/to-questionnaire | 4ae2317ddb22 | `repos/mattpocock-skills/skills/productivity/to-questionnaire` | - |
| `to-spec` | installed | - | external | https://github.com/mattpocock/skills/tree/4ae2317ddb22b046e26455e4f9c9b3aaaf62212d/skills/engineering/to-spec | 4ae2317ddb22 | `repos/mattpocock-skills/skills/engineering/to-spec` | - |
| `to-tickets` | installed | - | external | https://github.com/mattpocock/skills/tree/4ae2317ddb22b046e26455e4f9c9b3aaaf62212d/skills/engineering/to-tickets | 4ae2317ddb22 | `repos/mattpocock-skills/skills/engineering/to-tickets` | - |
| `torch-profiling-tilelang-programs` | installed | - | catalog | https://github.com/sablin39/tilelang-cuda-skills/tree/842d1dc7d684861d2da241cb3e51137a47d5cad6/skills/tilelang/torch-profiling-tilelang-programs | 842d1dc7d684 | `repos/tilelang-cuda-skills/skills/tilelang/torch-profiling-tilelang-programs` | - |
| `tracelens-analysis-orchestrator` | installed | - | catalog | https://github.com/amd/skills/tree/9eba3192f6793237eaeacfabd61fb0bd557e53cf/skills/tracelens-analysis-orchestrator | 9eba3192f679 | `repos/amd-skills/skills/tracelens-analysis-orchestrator` | - |
| `train-inference-consistency` | installed | - | vendored | https://github.com/vllm-project/vllm/tree/2d8efee9a73366d6b4e753780af32443d2fef91a/.agents/skills/train-inference-consistency | 2d8efee9a733 | `skills/framework/train-inference-consistency` | SKILL.md 增加中文触发词、vLLM 文档链接改为本目录稀疏 clone、增加 zero-mismatch-rl 交叉引用与来源注释；新增 README.md 与 LICENSE；其余为上游逐字复制 |
| `triage` | installed | - | external | https://github.com/mattpocock/skills/tree/4ae2317ddb22b046e26455e4f9c9b3aaaf62212d/skills/engineering/triage | 4ae2317ddb22 | `repos/mattpocock-skills/skills/engineering/triage` | - |
| `triton-skill` | installed | - | unknown | - | - | `skills/kernel/triton-skill` | 本次审计未确立来源（未逐一读取该目录全部文件，不断言其无上游声明） |
| `verification-before-completion` | installed | - | catalog | https://github.com/Saddss/cursor-skills/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/skills/verification-before-completion | 400c165e7a6a | `repos/cursor-skills/skills/verification-before-completion` | - |
| `verl-skill` | installed | - | unknown | - | - | `skills/framework/verl-skill` | 本次审计未确立来源（未逐一读取该目录全部文件，不断言其无上游声明） |
| `vibe-research-workflow` | installed | - | external | https://github.com/HKUSTDial/Supervisor-Skills/tree/4bd3fef393e5ffc891c74309a5ef6c5ad3ad12c8/skills/vibe-research-workflow | 4bd3fef393e5 | `repos/supervisor-skills/skills/vibe-research-workflow` | - |
| `wait-what` | installed | - | external | https://github.com/mattpocock/skills/tree/4ae2317ddb22b046e26455e4f9c9b3aaaf62212d/skills/productivity/wait-what | 4ae2317ddb22 | `repos/mattpocock-skills/skills/productivity/wait-what` | - |
| `warp-compile-time-optimizer` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/warp-compile-time-optimizer | fce3cca4170b | `repos/nvidia-skills/skills/warp-compile-time-optimizer` | - |
| `warp-debug-gradients` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/warp-debug-gradients | fce3cca4170b | `repos/nvidia-skills/skills/warp-debug-gradients` | - |
| `warp-eval` | installed | - | catalog | https://github.com/NVIDIA/skills/tree/fce3cca4170be75e82c64920654d56ee2811b005/skills/warp-eval | fce3cca4170b | `repos/nvidia-skills/skills/warp-eval` | - |
| `wayfinder` | installed | - | external | https://github.com/mattpocock/skills/tree/4ae2317ddb22b046e26455e4f9c9b3aaaf62212d/skills/engineering/wayfinder | 4ae2317ddb22 | `repos/mattpocock-skills/skills/engineering/wayfinder` | - |
| `wizard` | installed | - | external | https://github.com/mattpocock/skills/tree/4ae2317ddb22b046e26455e4f9c9b3aaaf62212d/skills/engineering/wizard | 4ae2317ddb22 | `repos/mattpocock-skills/skills/engineering/wizard` | - |
| `write-sglang-test` | installed | - | vendored | https://github.com/sgl-project/sglang/tree/3bcde0c80eeb07b40d3334c4b3d7341602efd089/.agents/skills/write-sglang-test | 3bcde0c80eeb | `skills/framework/sglang-repo-skills/write-sglang-test` | 无（全部逐字复制） |
| `writing-for-agents` | installed | - | external | https://github.com/mattpocock/skills/tree/4ae2317ddb22b046e26455e4f9c9b3aaaf62212d/skills/productivity/writing-for-agents | 4ae2317ddb22 | `repos/mattpocock-skills/skills/productivity/writing-for-agents` | - |
| `writing-plans` | installed | - | catalog | https://github.com/Saddss/cursor-skills/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/skills/writing-plans | 400c165e7a6a | `repos/cursor-skills/skills/writing-plans` | - |
| `writing-tilelang-design-docs` | installed | - | catalog | https://github.com/sablin39/tilelang-cuda-skills/tree/842d1dc7d684861d2da241cb3e51137a47d5cad6/skills/tilelang/writing-tilelang-design-docs | 842d1dc7d684 | `repos/tilelang-cuda-skills/skills/tilelang/writing-tilelang-design-docs` | - |
| `writing-tilelang-kernels` | installed | - | catalog | https://github.com/sablin39/tilelang-cuda-skills/tree/842d1dc7d684861d2da241cb3e51137a47d5cad6/skills/tilelang/writing-tilelang-kernels | 842d1dc7d684 | `repos/tilelang-cuda-skills/skills/tilelang/writing-tilelang-kernels` | - |
| `zero-mismatch-rl` | installed | - | vendored | https://github.com/Inferact/zero-mismatch-rl/tree/136aca87d8158adae94b5a272fa4a8a6d5cd6939/skill | 136aca87d815 | `skills/framework/zero-mismatch-rl` | SKILL.md 与 README.md 为本地新增入口与说明；skill/、cases/、LICENSE、README.upstream.md 逐字复制 |
| `zhihu-knowledge-base` | installed | - | unknown | - | - | `skills/other/zhihu-knowledge-base` | 本次审计未确立来源（未逐一读取该目录全部文件，不断言其无上游声明） |

## 4. 嵌套 checkout（nested）

| 嵌套路径 | 父 checkout | 实际 origin / 来源 | gitlink ref | 安装 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `kernel-design-agents/skills/KernelWiki` | `kernel-design-agents` | https://github.com/mit-han-lab/KernelWiki | `76d27b56f804e7e7295d4c570e1e5d7eef4b0a75` | KernelWiki | install.sh 安装的 KernelWiki 来自此嵌套目录 |
| `kernel-design-agents/skills/ncu-report-skill` | `kernel-design-agents` | https://github.com/mit-han-lab/ncu-report-skill | `d1887948c7d53690cfe6605f59c1329b8a1c6bb5` | 否 | 未安装；本仓库 skills/kernel/ncu-report-skill 为本地副本，来源见第 5 节 |
| `repos/tilelang-cuda-skills/external/TileFoundry` | `repos/tilelang-cuda-skills` | https://github.com/tile-ai/TileFoundry | `3a6a4c035e907ae8b0abe9c132e5098b1910b92d` | 否 | 未安装（tilelang-cuda-skills 的嵌套 submodule） |
| `repos/tilelang-cuda-skills/external/TileFoundry/third_party/cutlass` | `repos/tilelang-cuda-skills/external/TileFoundry` | https://github.com/NVIDIA/cutlass | `e406c186f510a15091cce01f782020ceb7ba8eb5` | 否 | 未安装（TileFoundry 的嵌套 submodule） |

## 5. 本地 skill 来源证据

本表只列有明确证据的条目。其余本地 skill 为 `unknown`，不默认原创。

| 安装名 | 类别 | 上游仓库 | ref | 证据文件 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `claude-delegate` | protected | https://github.com/amElnagdy/delegate-skills | - | `sources/local-provenance.md（claude-delegate 节）` | lockfile 声明来源；ref 未知；保留本地 macOS 补丁，用户级保留名，安装器不覆盖。LICENSE 为 MIT，版权人 Ahmed Mohammed (amElnagdy) |
| `codex-background` | protected | - | - | `sources/local-provenance.md（codex-background 节）` | MIT，版权人 Ahmed Mohammed (amElnagdy)；本次审计未确立上游 URL（unknown）；保留本地补丁，用户级保留名，安装器不覆盖 |
| `iket-profiling` | external-declared | https://github.com/humanfia/iket-profiling-skill | - | `skills/kernel/iket-profiling/UPSTREAM.md` | UPSTREAM.md 声明安装来源为 humanfia/iket-profiling-skill（无 pin）；内容是 NVIDIA CUTLASS 官方 IKET 文档的摘要，NVIDIA 文档是参考来源，不是本 skill 的 origin |
| `ncu-report-skill` | external-declared | https://github.com/mit-han-lab/ncu-report-skill | - | `skills/kernel/ncu-report-skill/README.md` | README 称为 KDA 的独立 submodule；KDA .gitmodules 指向 mit-han-lab/ncu-report-skill；README 安装示例另指 DongyunZou/ncu-report-skill（两处来源待确认）；本地副本无 .git，ref 未知 |

未验证来源（unknown，25 个）：`agent-skill-router`, `amd-gpu-docs`, `amd-instinct-cdna4-isa`, `amd-instinct-isa`, `colfax-research-skill`, `cuda-skill`, `cutlass-skill`, `deepgemm-skill`, `giantpandallm-knowledge`, `gpu-communication-libraries`, `gpu-development-catchall`, `gpu-framework-router`, `gpu-kernel-authoring-router`, `gpu-kernel-reverse`, `gpu-performance-router`, `llm-serving-router`, `nccl-skill`, `ncu-persistent-kernel-diagnosis`, `nv-gpu-kernel-performance-modeling`, `persistent-kernel-scheduling`, `persistent-kernel-utilization`, `sglang-skill`, `triton-skill`, `verl-skill`, `zhihu-knowledge-base`。

## 6. vendored 本地修改与 merge 要求

- **`skills/framework/sglang-repo-skills`**（sglang-repo-skills）：无（全部逐字复制）。上游 .claude/skills 是 .agents/skills 的 symlink；未收录的上游 skill 见其 README
- **`skills/kernel/tilelang-repo-skills`**（tilelang-repo-skills）：无（全部逐字复制）。未收录上游 tilelang-build / tilelang-dev-guide / tilelang-simplification / tilelang-submit-pr / tilelang-tvm-ir
- **`skills/framework/zero-mismatch-rl`**（zero-mismatch-rl）：SKILL.md 与 README.md 为本地新增入口与说明；skill/、cases/、LICENSE、README.upstream.md 逐字复制。skill/ 与 cases/ 必须保持同级（skill/guides 中以 ../../../cases/... 引用案例）
- **`skills/framework/train-inference-consistency`**（train-inference-consistency）：SKILL.md 增加中文触发词、vLLM 文档链接改为本目录稀疏 clone、增加 zero-mismatch-rl 交叉引用与来源注释；新增 README.md 与 LICENSE；其余为上游逐字复制。2026-11-08 审计时的上游 tip 不含 .agents/skills/train-inference-consistency；保留本地固定副本（不覆盖）；未来需网络恢复后重新检查上游是否恢复该路径
- **`skills/other/brain-and-hands`**（brain-and-hands）：本地修改版 0.3.2（见其 README.md）：新增 Codex/Claude/Cursor 共用 SKILL.md、模型解析器（含 --host cursor 与 Fable 大脑）、独立 Noir 测试 runner、参考文档和 tester/debugger；grunt 改成 Noir 测试角色；delegate 复用根入口；角色与 hook 修正规则；保留上游 LICENSE、图标、hooks.json 和 README.upstream.md。install.sh 安装普通 skill；install-brain-and-hands.sh 可单独安装（--agent codex|claude|cursor），--agent claude 另注册插件；同步上游逐文件 merge，不恢复禁用的旧 relay
- **`skills/other/tech-doc-style-chinese`**（tech-doc-style-chinese）：本地定制覆盖版（修改对照见其 README.md）；上游 clone 在 repos/tech-doc-style-chinese（git-ignored）。同步上游必须按其 README.md 的步骤 merge，不能直接覆盖本地修改的文件

2026-11-08 审计时上游 tip 不含路径的 vendored 副本（如 train-inference-consistency）保留本地固定副本，不得以上游缺失为由删除本地 skill。

## 7. 覆盖检查

- install.sh SKILL_NAMES 未被 installed/protected 覆盖：无
- 仓库内 SKILL.md 未安装（tracked-not-installed）：0 个（见第 3 节）

## 检查与更新流程

```bash
python3 scripts/build_sources.py            # 重新生成 SOURCES.md
python3 scripts/build_sources.py --check    # 校验 SOURCES.md 与当前 checkout 一致
bash update-repos.sh all                    # 更新全部 submodule / 外部 checkout
```

上游 tip 的检查结论记录在 `sources/update-log.md`（手工维护）。无网络时记为失败，不得记为无更新。

<!-- update-log -->
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
| 2026-11-12 | 全部上游 tip（上表全部来源 + npm `rea-agents@6.8.0`、`@jackwener/wx-cli@0.3.0`） | 见 SOURCES.md 第 2 节 | 未取得：本会话沙箱拒绝 github.com:443 与 registry.npmjs.org:443（CONNECT 403，user denied）。这是本次委派会话的沙箱策略，不代表网络全局不可用 | 结论未知。不能判定"无更新"，也不能判定"有更新"；上表 2026-11-08 记录仅为历史结论，未重新核对 | 未改任何 pin；在允许上述主机的会话中逐来源重查（`git ls-remote <repo> refs/heads/main`；npm `view`） |
| 2026-11-12 | 注册表修复（离线，无网络） | — | — | `scripts/build_sources.py` 复用 `build_inventory.py` 的状态（installed / shadowed / protected），不再自行统计候选；链接去掉 `.git`；删除不支持的 `--fetch`；加入 VeloQ 可选 skill；嵌套 KernelWiki 使用其 checkout origin（DongyunZou/KernelWiki）与父仓库 gitlink（`76d27b5`）；iket-profiling、salary-negotiation-skill、ncu-report-skill 改为 external-declared（有 UPSTREAM.md / README 声明）；claude-delegate 为 protected，来源 amElnagdy/delegate-skills（已声明，ref 未知，见 sources/local-provenance.md）；codex-background 为 protected，来源 unknown（本次未确立）；zotero-cli 为 external-declared，来源 https://github.com/54yyyu/zotero-mcp（编排器核验的已安装包 0.15.0 元数据，模板一致性未核对）；其余无声明的本地 skill 为 unknown，不默认原创 | 生成 SOURCES.md；新增 `tests/test_sources.py`；未改 skill 内容、pin 或 tracked 安装逻辑 |
| 2026-11-19 | skill 裁剪 | — | — | 按当前 LLM infra 工作范围移除医疗/生物、DOCA、TAO、Jetson、DeepStream、BioNeMo、Earth2、cuOpt、VSS、PAIDF、HoloHub、前端/Office、微信/逆向、文献管理等无关 skill；新增 `excluded-skills.txt` 精确名称清单，由 `build_inventory.py` 与 `install.sh` 消费；移除 CCFA submodule 与本地 salary/zotero 条目；保留 180 个 installed、5 个 shadowed 与 2 个 protected 名称 | 重新生成 `inventory.tsv` / `SOURCES.md`；更新 README、INSTALL、CLAUDE.md、测试与 `docs/skills-usage-review.md` |
| 2026-11-18 | brain-and-hands | `2b30b09798a5bb8fd91f1402905f1fb4a524b05e` | `git ls-remote https://github.com/prograsec/brain-and-hands.git HEAD` 返回同一 SHA（GitHub API 403，改用 Git 直接核验） | 上游基线不变；本地 0.3.0 支持 Codex/Claude，共用模型分工（Astra/Opus 分析、Sol/Sonnet 测试、Terra/Haiku 查事实），简单测试新建独立 Claude/Noir session；移除旧 Gris 实验结果的泛化与固定分片/提前结束规则；两个清单生成器支持限定单 skill 更新 | 28 个本地行为测试与 48 个既有路由行为/回归测试通过；技能/插件格式校验通过；三个真实 Noir session 分别验证指定单测通过、继承 Claude 父环境的单测失败退出码保留、Astra 主会话修复夹具后 3 项回归通过。产物在本机 devlog/brain-and-hands-20261118；不代表 GPU 工作负载或成本基准；未更新其他来源 pin，未恢复旧 relay |
| 2026-11-19 | CCFA-Skills / Supervisor-Skills（恢复） | `e33ccfa6` / 缺失 | `91ca9e28` / `4bd3fef3` | 此前裁剪误删了 CCFA submodule 与 Supervisor 外部来源；二者均属于论文研究工作流，恢复为可安装范围 | CCFA checkout 固定到 `91ca9e28`（18 个 runtime skill），Supervisor 固定到 `4bd3fef3`（12 个 skill）；从 `excluded-skills.txt` 移除对应名称，恢复安装逻辑、路由规则、README/INSTALL/AGENTS/CLAUDE 文档与回归测试；重新生成 `inventory.tsv` / `SOURCES.md` |
