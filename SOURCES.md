# Skill 来源登记（SOURCES.md）

本文件由 `python3 scripts/build_sources.py` 生成，请勿手工编辑正文。
它列出 `install.sh` 安装的全部 skill 的来源：本仓库路径、上游仓库与引用、本地修改情况。

## 来源分类

| 类别 | 含义 | 同步方式 |
| --- | --- | --- |
| local | 本仓库原创，无上游 | 直接修改本仓库 |
| vendored | 从上游逐字复制的快照，可能有本地修改 | 有本地修改的文件 merge，其余直接覆盖 |
| catalog | submodule 引入的 skill 目录树 | `bash update-repos.sh <name>` 更新 pin |
| external | `sources/external-sources.conf` 的外部来源 | 更新 conf 中的 ref 后 `bash update-repos.sh external` |

上游 URL 模板：`<repo>/tree/<pin>/<上游路径>`（vendored 与 catalog 逐 skill 给出，external 见 conf）。

## 1. 本仓库原创 skill（local，31 个）

来源：本仓库 https://github.com/OpenDarrenlu/agent-gpu-skills。

| skill | 本地路径 |
| --- | --- |
| `agent-skill-router` | `skills/routers/agent-skill-router` |
| `amd-gpu-docs` | `skills/other/amd-gpu-docs` |
| `amd-instinct-cdna4-isa` | `skills/other/amd-instinct-cdna4-isa` |
| `amd-instinct-isa` | `skills/other/amd-instinct-isa` |
| `claude-delegate` | `skills/other/claude-delegate` |
| `codex-background` | `skills/other/codex-background` |
| `colfax-research-skill` | `skills/kernel/colfax-research-skill` |
| `cuda-skill` | `skills/kernel/cuda-skill` |
| `cutlass-skill` | `skills/kernel/cutlass-skill` |
| `deepgemm-skill` | `skills/kernel/deepgemm-skill` |
| `giantpandallm-knowledge` | `skills/other/giantpandallm-knowledge` |
| `gpu-communication-libraries` | `skills/other/gpu-communication-libraries` |
| `gpu-development-catchall` | `skills/routers/gpu-development-catchall` |
| `gpu-framework-router` | `skills/routers/gpu-framework-router` |
| `gpu-kernel-authoring-router` | `skills/routers/gpu-kernel-authoring-router` |
| `gpu-kernel-reverse` | `skills/kernel/gpu-kernel-reverse` |
| `gpu-performance-router` | `skills/routers/gpu-performance-router` |
| `iket-profiling` | `skills/kernel/iket-profiling` |
| `llm-serving-router` | `skills/routers/llm-serving-router` |
| `nccl-skill` | `skills/other/nccl-skill` |
| `ncu-persistent-kernel-diagnosis` | `skills/kernel/ncu-persistent-kernel-diagnosis` |
| `ncu-report-skill` | `skills/kernel/ncu-report-skill` |
| `nv-gpu-kernel-performance-modeling` | `skills/kernel/nv-gpu-kernel-performance-modeling` |
| `persistent-kernel-scheduling` | `skills/kernel/persistent-kernel-scheduling` |
| `persistent-kernel-utilization` | `skills/kernel/persistent-kernel-utilization` |
| `salary-negotiation-skill` | `skills/other/salary-negotiation-skill` |
| `sglang-skill` | `skills/framework/sglang-skill` |
| `triton-skill` | `skills/kernel/triton-skill` |
| `verl-skill` | `skills/framework/verl-skill` |
| `zhihu-knowledge-base` | `skills/other/zhihu-knowledge-base` |
| `zotero-cli` | `skills/other/zotero-cli` |

## 2. vendored 固定上游快照（5 个目录）

| skill | 上游仓库 | 上游路径 | 基线 commit | 本地修改 |
| --- | --- | --- | --- | --- |
| `add-jit-kernel`, `add-sgl-kernel`, `compute-mamba-ratio`, `debug-cuda-crash`, `debug-distributed-hang`, `generate-profile`, `kernel-organization`, `kl-consistency-test`, `sglang-prod-incident-triage`, `write-sglang-test` | https://github.com/sgl-project/sglang.git | `.agents/skills` | `3bcde0c80eeb07b40d3334c4b3d7341602efd089` | 无（全部逐字复制） |
| `tilelang-ascend`, `tilelang-backend`, `tilelang-layout`, `tilelang-semantic` | https://github.com/tile-ai/tilelang.git | `.agents/skills` | `a8f3d3bcfe7fdd1a3887b05ff8f054e7d08980de` | 无（全部逐字复制） |
| `zero-mismatch-rl` | https://github.com/Inferact/zero-mismatch-rl.git | `skill/（另有 cases/ 模型案例）` | `136aca87d8158adae94b5a272fa4a8a6d5cd6939` | SKILL.md 与 README.md 为本地新增入口与说明；skill/、cases/、LICENSE、README.upstream.md 逐字复制 |
| `train-inference-consistency` | https://github.com/vllm-project/vllm.git | `.agents/skills/train-inference-consistency` | `2d8efee9a73366d6b4e753780af32443d2fef91a` | SKILL.md 增加中文触发词、vLLM 文档链接改为本目录稀疏 clone、增加 zero-mismatch-rl 交叉引用与来源注释；新增 README.md 与 LICENSE；其余为上游逐字复制 |
| `tech-doc-style-chinese` | https://github.com/Fenng/Tech-Doc-Style-Chinese.git | `.（仓库根）` | `b119d01c9bb7cd132b3c62fafdff34227c4410d1` | 本地定制覆盖版（修改对照见其 README.md）；上游 clone 在 repos/tech-doc-style-chinese（git-ignored） |

各 vendored 目录的说明：

- **`skills/framework/sglang-repo-skills`**（sglang-repo-skills）：上游 .claude/skills 是 .agents/skills 的 symlink；未收录的上游 skill 见其 README
- **`skills/kernel/tilelang-repo-skills`**（tilelang-repo-skills）：未收录上游 tilelang-build / tilelang-dev-guide / tilelang-simplification / tilelang-submit-pr / tilelang-tvm-ir
- **`skills/framework/zero-mismatch-rl`**（zero-mismatch-rl）：skill/ 与 cases/ 必须保持同级（skill/guides 中以 ../../../cases/... 引用案例）
- **`skills/framework/train-inference-consistency`**（train-inference-consistency）：2026-11-08 检查：上游 vLLM main 已删除 .agents/skills/train-inference-consistency，本 vendored 副本成为唯一来源，不再有上游更新可同步
- **`skills/other/tech-doc-style-chinese`**（tech-doc-style-chinese）：同步上游必须按其 README.md 的步骤 merge，不能直接覆盖本地修改的文件

## 3. catalog（submodule / clone，8 个来源）

| catalog | 上游仓库 | 本地 pin | 安装 skill 数 |
| --- | --- | --- | --- |
| `nvidia-skills` | https://github.com/NVIDIA/skills.git | `fce3cca4170be75e82c64920654d56ee2811b005` | 438 |
| `amd-skills` | https://github.com/amd/skills.git | `9eba3192f6793237eaeacfabd61fb0bd557e53cf` | 17 |
| `ccfa-skills` | https://github.com/mikubaka88/CCFA-Skills.git | `91ca9e2801698466f934b557a75ce46db8ba46b5` | 18 |
| `cursor-skills` | https://github.com/Saddss/cursor-skills.git | `400c165e7a6afac09e0f84aab85bcc20f37252f5` | 39 |
| `ponytail` | https://github.com/DietrichGebert/ponytail.git | `96608e078b2d5ea607c5c3b8b5a54d1c1fba6846` | 6 |
| `tilelang-cuda-skills` | https://github.com/sablin39/tilelang-cuda-skills.git | `842d1dc7d684861d2da241cb3e51137a47d5cad6` | 9 |
| `kernel-design-agents` | https://github.com/OpenDarrenlu/kernel-design-agents.git | `2e49985e277f65ad6c07d31b943cf78628e06e98` | 1 |
| `tech-doc-style-chinese (clone)` | https://github.com/Fenng/Tech-Doc-Style-Chinese.git | `b119d01c9bb7cd132b3c62fafdff34227c4410d1` | 0 |

install.sh 的 SKILL_DIRS 直接安装的 tilelang-cuda-skills skill：`debugging-tilelang-programs`, `designing-with-tilefoundry`, `optimizing-tilelang-programs`, `profiling-tilelang-programs`, `testing-fwd-bwd-kernels`, `tilelang-optimization-loop`, `torch-profiling-tilelang-programs`, `writing-tilelang-design-docs`, `writing-tilelang-kernels`。

### nvidia-skills

- 上游：https://github.com/NVIDIA/skills.git
- 本地 pin：`fce3cca4170be75e82c64920654d56ee2811b005`（submodule）
- 上游 URL：`https://github.com/NVIDIA/skills.git/tree/fce3cca4170be75e82c64920654d56ee2811b005/<路径>`
- install.sh 遍历 skills/ 与 plugins/nvidia-skills/skills/；同名先到先得（本地 skill 优先）

| 安装名 | 上游相对路径 |
| --- | --- |
| `accelerated-computing-cudf` | `skills/accelerated-computing-cudf` |
| `ambient-healthcare-agent-with-nemotron-voice-agent` | `skills/ambient-healthcare-agent-with-nemotron-voice-agent` |
| `amc-run-rtsp-calibration` | `skills/amc-run-rtsp-calibration` |
| `amc-run-sample-calibration` | `skills/amc-run-sample-calibration` |
| `amc-run-video-calibration` | `skills/amc-run-video-calibration` |
| `amc-setup-calibration-stack` | `skills/amc-setup-calibration-stack` |
| `bionemo-boltz2-nim` | `skills/bionemo-boltz2-nim` |
| `bionemo-codonfm-embed` | `skills/bionemo-codonfm-embed` |
| `bionemo-codonfm-finetune` | `skills/bionemo-codonfm-finetune` |
| `bionemo-codonfm-score` | `skills/bionemo-codonfm-score` |
| `bionemo-codonfm-setup` | `skills/bionemo-codonfm-setup` |
| `bionemo-complexa-evaluate-pdbs` | `skills/bionemo-complexa-evaluate-pdbs` |
| `bionemo-complexa-sweep` | `skills/bionemo-complexa-sweep` |
| `bionemo-complexa-target` | `skills/bionemo-complexa-target` |
| `bionemo-cuequivariance` | `skills/bionemo-cuequivariance` |
| `bionemo-cuequivariance-jax` | `skills/bionemo-cuequivariance-jax` |
| `bionemo-cuequivariance-torch` | `skills/bionemo-cuequivariance-torch` |
| `bionemo-diffdock-nim` | `skills/bionemo-diffdock-nim` |
| `bionemo-drug-discovery-pipeline` | `skills/bionemo-drug-discovery-pipeline` |
| `bionemo-evo2-nim` | `skills/bionemo-evo2-nim` |
| `bionemo-genmol-nim` | `skills/bionemo-genmol-nim` |
| `bionemo-genomics-workflow-acceleration` | `skills/bionemo-genomics-workflow-acceleration` |
| `bionemo-kermt-add-cmim-pretrain` | `skills/bionemo-kermt-add-cmim-pretrain` |
| `bionemo-kermt-continue-pretrain` | `skills/bionemo-kermt-continue-pretrain` |
| `bionemo-kermt-embed` | `skills/bionemo-kermt-embed` |
| `bionemo-kermt-finetune` | `skills/bionemo-kermt-finetune` |
| `bionemo-kermt-infer` | `skills/bionemo-kermt-infer` |
| `bionemo-kermt-monitor` | `skills/bionemo-kermt-monitor` |
| `bionemo-kermt-pretrain-scratch` | `skills/bionemo-kermt-pretrain-scratch` |
| `bionemo-kermt-setup` | `skills/bionemo-kermt-setup` |
| `bionemo-molmim-nim` | `skills/bionemo-molmim-nim` |
| `bionemo-msa-search-nim` | `skills/bionemo-msa-search-nim` |
| `bionemo-msa-structure-prediction-pipeline` | `skills/bionemo-msa-structure-prediction-pipeline` |
| `bionemo-nvmolkit-usage` | `skills/bionemo-nvmolkit-usage` |
| `bionemo-openfold2-nim` | `skills/bionemo-openfold2-nim` |
| `bionemo-openfold3-nim` | `skills/bionemo-openfold3-nim` |
| `bionemo-parabricks` | `skills/bionemo-parabricks` |
| `bionemo-proteinmpnn-nim` | `skills/bionemo-proteinmpnn-nim` |
| `bionemo-rfd4-proteina-nim` | `skills/bionemo-rfd4-proteina-nim` |
| `bionemo-rfdiffusion-nim` | `skills/bionemo-rfdiffusion-nim` |
| `cdf-anomalygen` | `skills/cdf-anomalygen` |
| `cdf-augmentation` | `skills/cdf-augmentation` |
| `cdf-auto-labeling` | `skills/cdf-auto-labeling` |
| `cdf-autonomous-vehicle-dataset-generation` | `skills/cdf-autonomous-vehicle-dataset-generation` |
| `cdf-curation-and-retrieval` | `skills/cdf-curation-and-retrieval` |
| `cdf-defect-image-generation` | `skills/cdf-defect-image-generation` |
| `cdf-event-video-generation` | `skills/cdf-event-video-generation` |
| `cdf-image-attribute-augmentation` | `skills/cdf-image-attribute-augmentation` |
| `cdf-orchestration-preflight` | `skills/cdf-orchestration-preflight` |
| `cdf-orchestration-setup` | `skills/cdf-orchestration-setup` |
| `cdf-orchestration-write-dag` | `skills/cdf-orchestration-write-dag` |
| `cdf-simulation` | `skills/cdf-simulation` |
| `cdf-video-data-augmentation` | `skills/cdf-video-data-augmentation` |
| `compute-sanitizer-diagnose-cuda-bugs` | `skills/compute-sanitizer-diagnose-cuda-bugs` |
| `cudaq-guide` | `skills/cudaq-guide` |
| `cudaq-importing` | `skills/cudaq-importing` |
| `cudf-query-engine-builder` | `skills/cudf-query-engine-builder` |
| `cuml` | `skills/cuml` |
| `cuopt-developer` | `skills/cuopt-developer` |
| `cuopt-install` | `skills/cuopt-install` |
| `cuopt-multi-objective-exploration` | `skills/cuopt-multi-objective-exploration` |
| `cuopt-numerical-optimization-api` | `skills/cuopt-numerical-optimization-api` |
| `cuopt-numerical-optimization-formulation` | `skills/cuopt-numerical-optimization-formulation` |
| `cuopt-routing-api-python` | `skills/cuopt-routing-api-python` |
| `cuopt-server-api-python` | `skills/cuopt-server-api-python` |
| `dali-dynamic-mode` | `skills/dali-dynamic-mode` |
| `data-designer` | `skills/data-designer` |
| `deepstream-dev` | `skills/deepstream-dev` |
| `deepstream-generate-pipeline` | `skills/deepstream-generate-pipeline` |
| `deepstream-import-vision-model` | `skills/deepstream-import-vision-model` |
| `deepstream-profile-pipeline` | `skills/deepstream-profile-pipeline` |
| `deepstream-run-mv3dt` | `skills/deepstream-run-mv3dt` |
| `deepstream-sop` | `skills/deepstream-sop` |
| `dgx-spark-skills` | `skills/dgx-spark-skills` |
| `dicom-metadata-extract` | `skills/dicom-metadata-extract` |
| `dicom-series-preflight` | `skills/dicom-series-preflight` |
| `dicom-series-to-volume` | `skills/dicom-series-to-volume` |
| `digital-health-clinical-asr-build` | `skills/digital-health-clinical-asr-build` |
| `digital-health-clinical-asr-eval` | `skills/digital-health-clinical-asr-eval` |
| `digital-health-clinical-asr-finetune` | `skills/digital-health-clinical-asr-finetune` |
| `digital-health-clinical-asr-setup` | `skills/digital-health-clinical-asr-setup` |
| `doca-aes-gcm` | `skills/doca-aes-gcm` |
| `doca-argp` | `skills/doca-argp` |
| `doca-argus` | `skills/doca-argus` |
| `doca-bare-metal-deployment` | `skills/doca-bare-metal-deployment` |
| `doca-bench` | `skills/doca-bench` |
| `doca-bf3-deployment` | `skills/doca-bf3-deployment` |
| `doca-bf4-deployment` | `skills/doca-bf4-deployment` |
| `doca-caps` | `skills/doca-caps` |
| `doca-comch` | `skills/doca-comch` |
| `doca-comm-channel-admin` | `skills/doca-comm-channel-admin` |
| `doca-common` | `skills/doca-common` |
| `doca-compress` | `skills/doca-compress` |
| `doca-container-deployment` | `skills/doca-container-deployment` |
| `doca-debug` | `skills/doca-debug` |
| `doca-devemu` | `skills/doca-devemu` |
| `doca-dma` | `skills/doca-dma` |
| `doca-dms` | `skills/doca-dms` |
| `doca-dpa` | `skills/doca-dpa` |
| `doca-dpdk-bridge` | `skills/doca-dpdk-bridge` |
| `doca-dts` | `skills/doca-dts` |
| `doca-erasure-coding` | `skills/doca-erasure-coding` |
| `doca-eth` | `skills/doca-eth` |
| `doca-firefly` | `skills/doca-firefly` |
| `doca-flow` | `skills/doca-flow` |
| `doca-flow-tune` | `skills/doca-flow-tune` |
| `doca-gpunetio` | `skills/doca-gpunetio` |
| `doca-hardware-safety` | `skills/doca-hardware-safety` |
| `doca-mgmt` | `skills/doca-mgmt` |
| `doca-ngauge` | `skills/doca-ngauge` |
| `doca-pcc` | `skills/doca-pcc` |
| `doca-pcc-counters` | `skills/doca-pcc-counters` |
| `doca-perftest` | `skills/doca-perftest` |
| `doca-pipeline-language-developer-tools` | `skills/doca-pipeline-language-developer-tools` |
| `doca-programming-guide` | `skills/doca-programming-guide` |
| `doca-public-knowledge-map` | `skills/doca-public-knowledge-map` |
| `doca-rdma` | `skills/doca-rdma` |
| `doca-rmax` | `skills/doca-rmax` |
| `doca-setup` | `skills/doca-setup` |
| `doca-sha` | `skills/doca-sha` |
| `doca-sha-offload-engine` | `skills/doca-sha-offload-engine` |
| `doca-socket-relay` | `skills/doca-socket-relay` |
| `doca-spcx-cc` | `skills/doca-spcx-cc` |
| `doca-sta` | `skills/doca-sta` |
| `doca-structured-tools-contract` | `skills/doca-structured-tools-contract` |
| `doca-telemetry` | `skills/doca-telemetry` |
| `doca-telemetry-exporter` | `skills/doca-telemetry-exporter` |
| `doca-telemetry-traceback-application` | `skills/doca-telemetry-traceback-application` |
| `doca-telemetry-utils` | `skills/doca-telemetry-utils` |
| `doca-upgrade` | `skills/doca-upgrade` |
| `doca-urom` | `skills/doca-urom` |
| `doca-urom-svc` | `skills/doca-urom-svc` |
| `doca-verbs` | `skills/doca-verbs` |
| `doca-version` | `skills/doca-version` |
| `dynamo-interconnect-check` | `skills/dynamo-interconnect-check` |
| `dynamo-recipe-runner` | `skills/dynamo-recipe-runner` |
| `dynamo-router-starter` | `skills/dynamo-router-starter` |
| `dynamo-troubleshoot` | `skills/dynamo-troubleshoot` |
| `earth2studio-create-datasource` | `skills/earth2studio-create-datasource` |
| `earth2studio-create-diagnostic` | `skills/earth2studio-create-diagnostic` |
| `earth2studio-create-prognostic` | `skills/earth2studio-create-prognostic` |
| `earth2studio-data-fetch` | `skills/earth2studio-data-fetch` |
| `earth2studio-deterministic-forecast` | `skills/earth2studio-deterministic-forecast` |
| `earth2studio-discover` | `skills/earth2studio-discover` |
| `earth2studio-install` | `skills/earth2studio-install` |
| `ethos-explore` | `skills/ethos-explore` |
| `ethos-orchestrate` | `skills/ethos-orchestrate` |
| `ethos-write` | `skills/ethos-write` |
| `foundationpose-pipeline` | `skills/foundationpose-pipeline` |
| `foundationpose-setup` | `skills/foundationpose-setup` |
| `g-assist-mcp-skill` | `skills/g-assist-mcp-skill` |
| `holoscan-app-debug-build-run` | `skills/holoscan-app-debug-build-run` |
| `holoscan-install-conda` | `skills/holoscan-install-conda` |
| `holoscan-install-container` | `skills/holoscan-install-container` |
| `holoscan-install-debian` | `skills/holoscan-install-debian` |
| `holoscan-install-source` | `skills/holoscan-install-source` |
| `holoscan-install-wheel` | `skills/holoscan-install-wheel` |
| `holoscan-module-lifecycle` | `skills/holoscan-module-lifecycle` |
| `holoscan-setup` | `skills/holoscan-setup` |
| `hsb-app` | `skills/hsb-app` |
| `hsb-flash` | `skills/hsb-flash` |
| `hsb-ip-create-top` | `skills/hsb-ip-create-top` |
| `hsb-ip-def` | `skills/hsb-ip-def` |
| `hsb-ip-packetizer` | `skills/hsb-ip-packetizer` |
| `hsb-setup` | `skills/hsb-setup` |
| `hsb-test` | `skills/hsb-test` |
| `i4h-lerobot-viz` | `skills/i4h-lerobot-viz` |
| `i4h-workflow` | `skills/i4h-workflow` |
| `i4h-workflow-create` | `skills/i4h-workflow-create` |
| `i4h-workflow-dataset-annotate` | `skills/i4h-workflow-dataset-annotate` |
| `i4h-workflow-dataset-convert` | `skills/i4h-workflow-dataset-convert` |
| `i4h-workflow-dataset-mimic` | `skills/i4h-workflow-dataset-mimic` |
| `i4h-workflow-dataset-replay` | `skills/i4h-workflow-dataset-replay` |
| `i4h-workflow-dataset-teleop` | `skills/i4h-workflow-dataset-teleop` |
| `i4h-workflow-e2e` | `skills/i4h-workflow-e2e` |
| `i4h-workflow-finetune` | `skills/i4h-workflow-finetune` |
| `i4h-workflow-scene-edit` | `skills/i4h-workflow-scene-edit` |
| `i4h-workflow-setup` | `skills/i4h-workflow-setup` |
| `i4h-workflow-train-rl` | `skills/i4h-workflow-train-rl` |
| `i4h-workflow-validate` | `skills/i4h-workflow-validate` |
| `isaac-cable-insertion-policy-deployment` | `skills/isaac-cable-insertion-policy-deployment` |
| `isaac-cable-insertion-policy-training` | `skills/isaac-cable-insertion-policy-training` |
| `isaac-humanoid-cloud-control` | `skills/isaac-humanoid-cloud-control` |
| `isaac-mission-control-showcase` | `skills/isaac-mission-control-showcase` |
| `jetson-build-source` | `skills/jetson-build-source` |
| `jetson-customize-camera` | `skills/jetson-customize-camera` |
| `jetson-customize-clocks` | `skills/jetson-customize-clocks` |
| `jetson-customize-fan` | `skills/jetson-customize-fan` |
| `jetson-customize-mgbe` | `skills/jetson-customize-mgbe` |
| `jetson-customize-nvpmodel` | `skills/jetson-customize-nvpmodel` |
| `jetson-customize-pcie` | `skills/jetson-customize-pcie` |
| `jetson-customize-pinmux` | `skills/jetson-customize-pinmux` |
| `jetson-customize-uphy` | `skills/jetson-customize-uphy` |
| `jetson-customize-usb` | `skills/jetson-customize-usb` |
| `jetson-derive-carrier` | `skills/jetson-derive-carrier` |
| `jetson-diagnostic` | `skills/jetson-diagnostic` |
| `jetson-download-bsp` | `skills/jetson-download-bsp` |
| `jetson-flash-image` | `skills/jetson-flash-image` |
| `jetson-generate-kb` | `skills/jetson-generate-kb` |
| `jetson-headless-mode` | `skills/jetson-headless-mode` |
| `jetson-inference-mem-tune` | `skills/jetson-inference-mem-tune` |
| `jetson-init-image` | `skills/jetson-init-image` |
| `jetson-init-source` | `skills/jetson-init-source` |
| `jetson-init-target` | `skills/jetson-init-target` |
| `jetson-link-docs` | `skills/jetson-link-docs` |
| `jetson-llm-benchmark` | `skills/jetson-llm-benchmark` |
| `jetson-llm-serve` | `skills/jetson-llm-serve` |
| `jetson-memory-audit` | `skills/jetson-memory-audit` |
| `jetson-optimize-memory` | `skills/jetson-optimize-memory` |
| `jetson-package` | `skills/jetson-package` |
| `jetson-print-bsp-info` | `skills/jetson-print-bsp-info` |
| `jetson-print-device-info` | `skills/jetson-print-device-info` |
| `jetson-promote-image` | `skills/jetson-promote-image` |
| `jetson-quick-start` | `skills/jetson-quick-start` |
| `jetson-set-target` | `skills/jetson-set-target` |
| `jetson-speculative-decoding` | `skills/jetson-speculative-decoding` |
| `jetson-validate-image` | `skills/jetson-validate-image` |
| `jetson-video-benchmark` | `skills/jetson-video-benchmark` |
| `jetson-video-capability` | `skills/jetson-video-capability` |
| `jetson-video-pipeline` | `skills/jetson-video-pipeline` |
| `jetson-video-recipe` | `skills/jetson-video-recipe` |
| `jetson-video-setup` | `skills/jetson-video-setup` |
| `launch-nemo-rl` | `skills/launch-nemo-rl` |
| `mcore-create-issue` | `skills/mcore-create-issue` |
| `mcore-linting-and-formatting` | `skills/mcore-linting-and-formatting` |
| `mcore-run-on-slurm` | `skills/mcore-run-on-slurm` |
| `mcore-split-pr` | `skills/mcore-split-pr` |
| `mcore-testing` | `skills/mcore-testing` |
| `medtech-model-evidence-export` | `skills/medtech-model-evidence-export` |
| `nemo-automodel-distributed-training` | `skills/nemo-automodel-distributed-training` |
| `nemo-automodel-launcher-config` | `skills/nemo-automodel-launcher-config` |
| `nemo-automodel-model-onboarding` | `skills/nemo-automodel-model-onboarding` |
| `nemo-automodel-recipe-development` | `skills/nemo-automodel-recipe-development` |
| `nemo-fabric-build-adapter` | `skills/nemo-fabric-build-adapter` |
| `nemo-fabric-integrate` | `skills/nemo-fabric-integrate` |
| `nemo-mbridge-mlm-bridge-training` | `skills/nemo-mbridge-mlm-bridge-training` |
| `nemo-mbridge-multi-node-slurm` | `skills/nemo-mbridge-multi-node-slurm` |
| `nemo-mbridge-perf-activation-recompute` | `skills/nemo-mbridge-perf-activation-recompute` |
| `nemo-mbridge-perf-cpu-offloading` | `skills/nemo-mbridge-perf-cpu-offloading` |
| `nemo-mbridge-perf-cuda-graphs` | `skills/nemo-mbridge-perf-cuda-graphs` |
| `nemo-mbridge-perf-expert-parallel-overlap` | `skills/nemo-mbridge-perf-expert-parallel-overlap` |
| `nemo-mbridge-perf-hierarchical-context-parallel` | `skills/nemo-mbridge-perf-hierarchical-context-parallel` |
| `nemo-mbridge-perf-megatron-fsdp` | `skills/nemo-mbridge-perf-megatron-fsdp` |
| `nemo-mbridge-perf-memory-tuning` | `skills/nemo-mbridge-perf-memory-tuning` |
| `nemo-mbridge-perf-moe-comm-overlap` | `skills/nemo-mbridge-perf-moe-comm-overlap` |
| `nemo-mbridge-perf-moe-dispatcher-selection` | `skills/nemo-mbridge-perf-moe-dispatcher-selection` |
| `nemo-mbridge-perf-moe-hardware-configs` | `skills/nemo-mbridge-perf-moe-hardware-configs` |
| `nemo-mbridge-perf-moe-long-context` | `skills/nemo-mbridge-perf-moe-long-context` |
| `nemo-mbridge-perf-moe-optimization-workflow` | `skills/nemo-mbridge-perf-moe-optimization-workflow` |
| `nemo-mbridge-perf-moe-vlm-training` | `skills/nemo-mbridge-perf-moe-vlm-training` |
| `nemo-mbridge-perf-parallelism-strategies` | `skills/nemo-mbridge-perf-parallelism-strategies` |
| `nemo-mbridge-perf-sequence-packing` | `skills/nemo-mbridge-perf-sequence-packing` |
| `nemo-mbridge-perf-tp-dp-comm-overlap` | `skills/nemo-mbridge-perf-tp-dp-comm-overlap` |
| `nemo-mbridge-recipe-recommender` | `skills/nemo-mbridge-recipe-recommender` |
| `nemo-mbridge-resiliency` | `skills/nemo-mbridge-resiliency` |
| `nemo-relay-debug-runtime-integration` | `skills/nemo-relay-debug-runtime-integration` |
| `nemo-relay-get-started` | `skills/nemo-relay-get-started` |
| `nemo-relay-install` | `skills/nemo-relay-install` |
| `nemo-relay-instrument-calls` | `skills/nemo-relay-instrument-calls` |
| `nemo-relay-instrument-context-isolation` | `skills/nemo-relay-instrument-context-isolation` |
| `nemo-relay-instrument-typed-wrappers` | `skills/nemo-relay-instrument-typed-wrappers` |
| `nemo-relay-migrate-from-flow` | `skills/nemo-relay-migrate-from-flow` |
| `nemo-relay-plugin-adaptive-tuning` | `skills/nemo-relay-plugin-adaptive-tuning` |
| `nemo-relay-plugin-build` | `skills/nemo-relay-plugin-build` |
| `nemo-relay-plugin-observability` | `skills/nemo-relay-plugin-observability` |
| `nemo-retriever` | `skills/nemo-retriever` |
| `nemo-retriever-mcp` | `skills/nemo-retriever-mcp` |
| `nemo-rl-auto-research` | `skills/nemo-rl-auto-research` |
| `nemo-rl-brev-etiquette` | `skills/nemo-rl-brev-etiquette` |
| `nemo-rl-docs` | `skills/nemo-rl-docs` |
| `nemo-rl-session-memory` | `skills/nemo-rl-session-memory` |
| `nemoclaw-user-guide` | `skills/nemoclaw-user-guide` |
| `nemotron-asr-finetune` | `skills/nemotron-asr-finetune` |
| `nemotron-customize` | `skills/nemotron-customize` |
| `nemotron-policy-generator` | `skills/nemotron-policy-generator` |
| `nemotron-retrieval-recipes` | `skills/nemotron-retrieval-recipes` |
| `nemotron-speech` | `skills/nemotron-speech` |
| `nemotron-voice-agent-builder` | `skills/nemotron-voice-agent-builder` |
| `nv-generate-ct-rflow` | `skills/nv-generate-ct-rflow` |
| `nv-generate-mr` | `skills/nv-generate-mr` |
| `nv-generate-mr-brain` | `skills/nv-generate-mr-brain` |
| `nv-generate-mr-brain-finetune` | `skills/nv-generate-mr-brain-finetune` |
| `nv-generate-vae-finetune` | `skills/nv-generate-vae-finetune` |
| `nv-reason-ct` | `skills/nv-reason-ct` |
| `nv-reason-cxr` | `skills/nv-reason-cxr` |
| `nv-segment-ct` | `skills/nv-segment-ct` |
| `nv-segment-ct-finetune` | `skills/nv-segment-ct-finetune` |
| `nv-segment-ctmr` | `skills/nv-segment-ctmr` |
| `nvflare-autofl` | `skills/nvflare-autofl` |
| `nvflare-autofl-report` | `skills/nvflare-autofl-report` |
| `nvflare-convert-huggingface` | `skills/nvflare-convert-huggingface` |
| `nvflare-convert-lightning` | `skills/nvflare-convert-lightning` |
| `nvflare-convert-pytorch` | `skills/nvflare-convert-pytorch` |
| `nvflare-diagnose-job` | `skills/nvflare-diagnose-job` |
| `nvflare-fed-stats` | `skills/nvflare-fed-stats` |
| `nvflare-orient` | `skills/nvflare-orient` |
| `nvflare-shared` | `skills/nvflare-shared` |
| `nvidia-app` | `skills/nvidia-app` |
| `nvidia-broadcast` | `skills/nvidia-broadcast` |
| `nvidia-ontology-management` | `skills/nvidia-ontology-management` |
| `nvidia-ontology-query` | `skills/nvidia-ontology-query` |
| `nvidia-ontology-setup` | `skills/nvidia-ontology-setup` |
| `nvidia-skill-finder` | `skills/nvidia-skill-finder` |
| `nvshmem-collect-performance-data` | `skills/nvshmem-collect-performance-data` |
| `nvshmem-configure-nic-pe-mapping` | `skills/nvshmem-configure-nic-pe-mapping` |
| `nvshmem-docs` | `skills/nvshmem-docs` |
| `nvshmem-enable-tma` | `skills/nvshmem-enable-tma` |
| `nvshmem-get-started` | `skills/nvshmem-get-started` |
| `nvshmem-install` | `skills/nvshmem-install` |
| `nvshmem-select-remote-transport` | `skills/nvshmem-select-remote-transport` |
| `nvshmem-troubleshoot-and-report-bugs` | `skills/nvshmem-troubleshoot-and-report-bugs` |
| `nvshmem-tune-performance` | `skills/nvshmem-tune-performance` |
| `omniverse-cad-to-simready` | `skills/omniverse-cad-to-simready` |
| `omniverse-content-services` | `skills/omniverse-content-services` |
| `omniverse-realtime-viewer` | `skills/omniverse-realtime-viewer` |
| `omniverse-usd-performance-tuning` | `skills/omniverse-usd-performance-tuning` |
| `ovnewton-usage` | `skills/ovnewton-usage` |
| `paidf-cosmos-predict` | `skills/paidf-cosmos-predict` |
| `physical-ai-defect-image-generation` | `skills/physical-ai-defect-image-generation` |
| `physical-ai-infrastructure-setup-and-resilient-scaling` | `skills/physical-ai-infrastructure-setup-and-resilient-scaling` |
| `physical-ai-neural-reconstruction` | `skills/physical-ai-neural-reconstruction` |
| `physical-ai-physics-selection` | `skills/physical-ai-physics-selection` |
| `physical-ai-video-data-augmentation` | `skills/physical-ai-video-data-augmentation` |
| `physicsnemo-discover` | `skills/physicsnemo-discover` |
| `physicsnemo-shard-tensor` | `skills/physicsnemo-shard-tensor` |
| `portfolio-optimization` | `skills/portfolio-optimization` |
| `pva-build-test` | `skills/pva-build-test` |
| `pva-debug` | `skills/pva-debug` |
| `pva-doctor` | `skills/pva-doctor` |
| `pva-opgen` | `skills/pva-opgen` |
| `pva-setup` | `skills/pva-setup` |
| `pva-spec` | `skills/pva-spec` |
| `rag-blueprint` | `skills/rag-blueprint` |
| `rag-eval` | `skills/rag-eval` |
| `rag-perf` | `skills/rag-perf` |
| `rtvi-cv-customize-model` | `skills/rtvi-cv-customize-model` |
| `rtvi-cv-scaffold-vss-service` | `skills/rtvi-cv-scaffold-vss-service` |
| `rtvi-vlm-customize-model` | `skills/rtvi-vlm-customize-model` |
| `rtx-remix-modding` | `skills/rtx-remix-modding` |
| `runai-audit-assigned-quota` | `skills/runai-audit-assigned-quota` |
| `runai-diagnose-failed-workload` | `skills/runai-diagnose-failed-workload` |
| `runai-get-cluster-health-summary` | `skills/runai-get-cluster-health-summary` |
| `runai-get-utilization-overview` | `skills/runai-get-utilization-overview` |
| `runai-get-workload-health-summary` | `skills/runai-get-workload-health-summary` |
| `skill-card-generator` | `skills/skill-card-generator` |
| `tao-analyze-changenet-rca` | `skills/tao-analyze-changenet-rca` |
| `tao-analyze-detection-kpi` | `skills/tao-analyze-detection-kpi` |
| `tao-analyze-gaps-od-map` | `skills/tao-analyze-gaps-od-map` |
| `tao-analyze-gaps-visual-changenet` | `skills/tao-analyze-gaps-visual-changenet` |
| `tao-analyze-gaps-vlm-bcq` | `skills/tao-analyze-gaps-vlm-bcq` |
| `tao-artifacts` | `skills/tao-artifacts` |
| `tao-convert-dataset-format` | `skills/tao-convert-dataset-format` |
| `tao-data-io` | `skills/tao-data-io` |
| `tao-finetune-anomalygennext` | `skills/tao-finetune-anomalygennext` |
| `tao-finetune-clip` | `skills/tao-finetune-clip` |
| `tao-finetune-cosmos-embed` | `skills/tao-finetune-cosmos-embed` |
| `tao-finetune-cosmos-reason` | `skills/tao-finetune-cosmos-reason` |
| `tao-finetune-huggingface-model` | `skills/tao-finetune-huggingface-model` |
| `tao-finetune-kumo-anomaly` | `skills/tao-finetune-kumo-anomaly` |
| `tao-finetune-kumo-forecast` | `skills/tao-finetune-kumo-forecast` |
| `tao-finetune-video-clip` | `skills/tao-finetune-video-clip` |
| `tao-generate-anomalies` | `skills/tao-generate-anomalies` |
| `tao-generate-image-embeddings` | `skills/tao-generate-image-embeddings` |
| `tao-generate-image-grounding` | `skills/tao-generate-image-grounding` |
| `tao-generate-od-defects` | `skills/tao-generate-od-defects` |
| `tao-generate-referring-expressions` | `skills/tao-generate-referring-expressions` |
| `tao-generate-video-reasoning-annotations` | `skills/tao-generate-video-reasoning-annotations` |
| `tao-launch-workflow` | `skills/tao-launch-workflow` |
| `tao-list-capabilities` | `skills/tao-list-capabilities` |
| `tao-mine-aoi-images` | `skills/tao-mine-aoi-images` |
| `tao-mine-nearest-neighbors` | `skills/tao-mine-nearest-neighbors` |
| `tao-mine-od-images` | `skills/tao-mine-od-images` |
| `tao-port-huggingface-model` | `skills/tao-port-huggingface-model` |
| `tao-prepare-anomalygennext-inputs` | `skills/tao-prepare-anomalygennext-inputs` |
| `tao-route-visual-changenet-samples` | `skills/tao-route-visual-changenet-samples` |
| `tao-run-automl` | `skills/tao-run-automl` |
| `tao-run-automl-deft-pipeline` | `skills/tao-run-automl-deft-pipeline` |
| `tao-run-deft-aoi` | `skills/tao-run-deft-aoi` |
| `tao-run-deft-aoi-cosmos3` | `skills/tao-run-deft-aoi-cosmos3` |
| `tao-run-deft-cr-its-mining` | `skills/tao-run-deft-cr-its-mining` |
| `tao-run-deft-object-detection` | `skills/tao-run-deft-object-detection` |
| `tao-run-deft-od-aoi` | `skills/tao-run-deft-od-aoi` |
| `tao-run-deft-pas` | `skills/tao-run-deft-pas` |
| `tao-run-dinov3-ssl-deft` | `skills/tao-run-dinov3-ssl-deft` |
| `tao-run-inference-service` | `skills/tao-run-inference-service` |
| `tao-run-on-brev` | `skills/tao-run-on-brev` |
| `tao-run-on-docker` | `skills/tao-run-on-docker` |
| `tao-run-on-kubernetes` | `skills/tao-run-on-kubernetes` |
| `tao-run-on-slurm` | `skills/tao-run-on-slurm` |
| `tao-run-on-virtualenv` | `skills/tao-run-on-virtualenv` |
| `tao-setup` | `skills/tao-setup` |
| `tao-setup-nvidia-gpu-host` | `skills/tao-setup-nvidia-gpu-host` |
| `tao-token-efficient-execution` | `skills/tao-token-efficient-execution` |
| `tao-train-action-recognition` | `skills/tao-train-action-recognition` |
| `tao-train-bevfusion` | `skills/tao-train-bevfusion` |
| `tao-train-centerpose` | `skills/tao-train-centerpose` |
| `tao-train-codetr` | `skills/tao-train-codetr` |
| `tao-train-deformable-detr` | `skills/tao-train-deformable-detr` |
| `tao-train-depth-anything-v2` | `skills/tao-train-depth-anything-v2` |
| `tao-train-dino` | `skills/tao-train-dino` |
| `tao-train-dinov3` | `skills/tao-train-dinov3` |
| `tao-train-fast-foundation-stereo` | `skills/tao-train-fast-foundation-stereo` |
| `tao-train-foundation-stereo` | `skills/tao-train-foundation-stereo` |
| `tao-train-grounding-dino` | `skills/tao-train-grounding-dino` |
| `tao-train-image-classification` | `skills/tao-train-image-classification` |
| `tao-train-mask-auto-encoder` | `skills/tao-train-mask-auto-encoder` |
| `tao-train-mask-auto-label` | `skills/tao-train-mask-auto-label` |
| `tao-train-mask-grounding-dino` | `skills/tao-train-mask-grounding-dino` |
| `tao-train-mask2former` | `skills/tao-train-mask2former` |
| `tao-train-metric-learning-recognition` | `skills/tao-train-metric-learning-recognition` |
| `tao-train-nvdinov2` | `skills/tao-train-nvdinov2` |
| `tao-train-nvpanoptix3d` | `skills/tao-train-nvpanoptix3d` |
| `tao-train-ocdnet` | `skills/tao-train-ocdnet` |
| `tao-train-ocrnet` | `skills/tao-train-ocrnet` |
| `tao-train-oneformer` | `skills/tao-train-oneformer` |
| `tao-train-optical-inspection` | `skills/tao-train-optical-inspection` |
| `tao-train-pointpillars` | `skills/tao-train-pointpillars` |
| `tao-train-pose-classification` | `skills/tao-train-pose-classification` |
| `tao-train-reid` | `skills/tao-train-reid` |
| `tao-train-rtdetr` | `skills/tao-train-rtdetr` |
| `tao-train-segformer` | `skills/tao-train-segformer` |
| `tao-train-single-step` | `skills/tao-train-single-step` |
| `tao-train-sparse4d` | `skills/tao-train-sparse4d` |
| `tao-train-visual-changenet` | `skills/tao-train-visual-changenet` |
| `tao-validate-dataset-format` | `skills/tao-validate-dataset-format` |
| `tao-validate-recipe-transfer` | `skills/tao-validate-recipe-transfer` |
| `tilegym-adding-cutile-kernel` | `skills/tilegym-adding-cutile-kernel` |
| `tilegym-converting-cutile-to-julia` | `skills/tilegym-converting-cutile-to-julia` |
| `tilegym-converting-cutile-to-triton` | `skills/tilegym-converting-cutile-to-triton` |
| `tilegym-cutile-autotuning` | `skills/tilegym-cutile-autotuning` |
| `tilegym-cutile-python` | `skills/tilegym-cutile-python` |
| `tilegym-improve-cutile-kernel-perf` | `skills/tilegym-improve-cutile-kernel-perf` |
| `tilegym-monkey-patch-kernels-to-transformers` | `skills/tilegym-monkey-patch-kernels-to-transformers` |
| `vss-build-vision-ai` | `skills/vss-build-vision-ai` |
| `warp-compile-time-optimizer` | `skills/warp-compile-time-optimizer` |
| `warp-debug-gradients` | `skills/warp-debug-gradients` |
| `warp-eval` | `skills/warp-eval` |
| `nvidia-skill-finder` | `plugins/nvidia-skills/skills/nvidia-skill-finder` |

### amd-skills

- 上游：https://github.com/amd/skills.git
- 本地 pin：`9eba3192f6793237eaeacfabd61fb0bd557e53cf`（submodule）
- 上游 URL：`https://github.com/amd/skills.git/tree/9eba3192f6793237eaeacfabd61fb0bd557e53cf/<路径>`
- 安装前检查其 skills/ 目录；本仓库只安装 skills/，不安装其他目录

| 安装名 | 上游相对路径 |
| --- | --- |
| `flydsl-kernel-authoring` | `skills/flydsl-kernel-authoring` |
| `flydsl-kernel-trace-analysis` | `skills/flydsl-kernel-trace-analysis` |
| `hyperloom-workload-optimizer` | `skills/hyperloom-workload-optimizer` |
| `lemonade-router-builder` | `skills/lemonade-router-builder` |
| `local-ai-app-integration` | `skills/local-ai-app-integration` |
| `local-ai-use` | `skills/local-ai-use` |
| `magpie-kernel-evaluator` | `skills/magpie-kernel-evaluator` |
| `quark-install` | `skills/quark-install` |
| `quark-torch-llm-ptq` | `skills/quark-torch-llm-ptq` |
| `rocm-doctor` | `skills/rocm-doctor` |
| `rocprofiler-pc-sampling` | `skills/rocprofiler-pc-sampling` |
| `rocprofiler-systems-choosing-a-profiling-preset` | `skills/rocprofiler-systems-choosing-a-profiling-preset` |
| `rocprofiler-systems-discovering-profiling-options` | `skills/rocprofiler-systems-discovering-profiling-options` |
| `rocprofiler-systems-instrumenting-binaries` | `skills/rocprofiler-systems-instrumenting-binaries` |
| `serving-llms-on-epyc` | `skills/serving-llms-on-epyc` |
| `serving-llms-on-instinct` | `skills/serving-llms-on-instinct` |
| `tracelens-analysis-orchestrator` | `skills/tracelens-analysis-orchestrator` |

### ccfa-skills

- 上游：https://github.com/mikubaka88/CCFA-Skills.git
- 本地 pin：`91ca9e2801698466f934b557a75ce46db8ba46b5`（submodule）
- 上游 URL：`https://github.com/mikubaka88/CCFA-Skills.git/tree/91ca9e2801698466f934b557a75ce46db8ba46b5/<路径>`
- ccf-* 含 SKILL.md 者为 runtime skill（安装名 = 目录名）；不含者为支撑目录（直接链接）

| 安装名 | 上游相对路径 |
| --- | --- |
| `ccf-common` | `ccf-common` |
| `ccf-experiment-designer` | `ccf-experiment-designer` |
| `ccf-humanization` | `ccf-humanization` |
| `ccf-idea-optimizer` | `ccf-idea-optimizer` |
| `ccf-idea-reviewer` | `ccf-idea-reviewer` |
| `ccf-integrity-auditor` | `ccf-integrity-auditor` |
| `ccf-literature-monitor` | `ccf-literature-monitor` |
| `ccf-literature-searcher` | `ccf-literature-searcher` |
| `ccf-paper-reviewer` | `ccf-paper-reviewer` |
| `ccf-paper-to-exemplar` | `ccf-paper-to-exemplar` |
| `ccf-paper-writer` | `ccf-paper-writer` |
| `ccf-pipeline-orchestrator` | `ccf-pipeline-orchestrator` |
| `ccf-project-scaffolder` | `ccf-project-scaffolder` |
| `ccf-rebuttal-writer` | `ccf-rebuttal-writer` |
| `ccf-research-engineer` | `ccf-research-engineer` |
| `ccf-skill-forger` | `ccf-skill-forger` |
| `ccf-submission-checker` | `ccf-submission-checker` |
| `ccf-visual-composer` | `ccf-visual-composer` |

### cursor-skills

- 上游：https://github.com/Saddss/cursor-skills.git
- 本地 pin：`400c165e7a6afac09e0f84aab85bcc20f37252f5`（submodule）
- 上游 URL：`https://github.com/Saddss/cursor-skills.git/tree/400c165e7a6afac09e0f84aab85bcc20f37252f5/<路径>`
- 同名时本地 catalog 优先（install.sh 先到先得）

| 安装名 | 上游相对路径 |
| --- | --- |
| `academic-figure-workflow` | `skills/academic-figure-workflow` |
| `brainstorming` | `skills/brainstorming` |
| `codebase-onboarding` | `skills/codebase-onboarding` |
| `diagnose` | `skills/diagnose` |
| `docx` | `skills/docx` |
| `executing-plans` | `skills/executing-plans` |
| `grill-with-docs` | `skills/grill-with-docs` |
| `handoff` | `skills/handoff` |
| `high-performance-code-review` | `skills/high-performance-code-review` |
| `improve-codebase-architecture` | `skills/improve-codebase-architecture` |
| `karpathy-guidelines` | `skills/karpathy-guidelines` |
| `kernel-cute-writing` | `skills/kernel-cute-writing` |
| `kernel-triton-writing` | `skills/kernel-triton-writing` |
| `llm-pipeline-analysis` | `skills/llm-pipeline-analysis` |
| `llm-serving-auto-benchmark` | `skills/llm-serving-auto-benchmark` |
| `llm-serving-capacity-planner` | `skills/llm-serving-capacity-planner` |
| `llm-torch-profiler-analysis` | `skills/llm-torch-profiler-analysis` |
| `model-compute-simulation` | `skills/model-compute-simulation` |
| `model-perf-binary-search` | `skills/model-perf-binary-search` |
| `parallel-exploring` | `skills/parallel-exploring` |
| `pdf` | `skills/pdf` |
| `perf-analysis` | `skills/perf-analysis` |
| `perf-host-analysis` | `skills/perf-host-analysis` |
| `perf-host-optimization` | `skills/perf-host-optimization` |
| `perf-nsight-compute-analysis` | `skills/perf-nsight-compute-analysis` |
| `perf-nsight-systems` | `skills/perf-nsight-systems` |
| `perf-optimization` | `skills/perf-optimization` |
| `perf-workload-profiling` | `skills/perf-workload-profiling` |
| `pp-separation-benchmark` | `skills/pp-separation-benchmark` |
| `pptx` | `skills/pptx` |
| `prototype` | `skills/prototype` |
| `review` | `skills/review` |
| `search-first` | `skills/search-first` |
| `setup-matt-pocock-skills` | `skills/setup-matt-pocock-skills` |
| `simplify-code` | `skills/simplify-code` |
| `tdd` | `skills/tdd` |
| `to-issues` | `skills/to-issues` |
| `verification-before-completion` | `skills/verification-before-completion` |
| `writing-plans` | `skills/writing-plans` |

### ponytail

- 上游：https://github.com/DietrichGebert/ponytail.git
- 本地 pin：`96608e078b2d5ea607c5c3b8b5a54d1c1fba6846`（submodule）
- 上游 URL：`https://github.com/DietrichGebert/ponytail.git/tree/96608e078b2d5ea607c5c3b8b5a54d1c1fba6846/<路径>`
- 安装名与本地数组一致

| 安装名 | 上游相对路径 |
| --- | --- |
| `ponytail` | `skills/ponytail` |
| `ponytail-audit` | `skills/ponytail-audit` |
| `ponytail-debt` | `skills/ponytail-debt` |
| `ponytail-gain` | `skills/ponytail-gain` |
| `ponytail-help` | `skills/ponytail-help` |
| `ponytail-review` | `skills/ponytail-review` |

### tilelang-cuda-skills

- 上游：https://github.com/sablin39/tilelang-cuda-skills.git
- 本地 pin：`842d1dc7d684861d2da241cb3e51137a47d5cad6`（submodule）
- 上游 URL：`https://github.com/sablin39/tilelang-cuda-skills.git/tree/842d1dc7d684861d2da241cb3e51137a47d5cad6/<路径>`
- 嵌套 submodule external/TileFoundry 由 update-repos.sh --recursive 维护；其余上游 skill 未安装（见 skills/other 与 skills/kernel 的本地/vendored 对应项）

| 安装名 | 上游相对路径 |
| --- | --- |
| `tilelang-optimization-loop` | `skills/tilelang-optimization-loop` |
| `designing-with-tilefoundry` | `skills/designing-with-tilefoundry` |
| `debugging-tilelang-programs` | `skills/tilelang/debugging-tilelang-programs` |
| `optimizing-tilelang-programs` | `skills/tilelang/optimizing-tilelang-programs` |
| `profiling-tilelang-programs` | `skills/tilelang/profiling-tilelang-programs` |
| `testing-fwd-bwd-kernels` | `skills/tilelang/testing-fwd-bwd-kernels` |
| `torch-profiling-tilelang-programs` | `skills/tilelang/torch-profiling-tilelang-programs` |
| `writing-tilelang-design-docs` | `skills/tilelang/writing-tilelang-design-docs` |
| `writing-tilelang-kernels` | `skills/tilelang/writing-tilelang-kernels` |

### kernel-design-agents

- 上游：https://github.com/OpenDarrenlu/kernel-design-agents.git
- 本地 pin：`2e49985e277f65ad6c07d31b943cf78628e06e98`（submodule）
- 上游 URL：`https://github.com/OpenDarrenlu/kernel-design-agents.git/tree/2e49985e277f65ad6c07d31b943cf78628e06e98/<路径>`
- 只安装 KernelWiki；其嵌套 skills（ncu-report-skill）已在上游合并为目录内引用

| 安装名 | 上游相对路径 |
| --- | --- |
| `KernelWiki` | `skills/KernelWiki` |

## 4. 外部来源（sources/external-sources.conf，5 个）

| name | 上游仓库 | ref | runtime | 安装 skill |
| --- | --- | --- | --- | --- |
| `mattpocock-skills` | https://github.com/mattpocock/skills | `4ae2317ddb22` | `none` | `ask-matt`, `code-review`, `codebase-design`, `diagnosing-bugs`, `domain-modeling`, `grill-with-docs`, `implement-spec`, `implement`, `improve-codebase-architecture`, `pr`, `prototype`, `research`, `r |
| `rea` | https://github.com/morluto/rea | `160d821ac9a0` | `npm:rea-agents@6.8.0:rea` | `reverse-engineer-anything` |
| `sepia` | https://github.com/Nanako0129/sepia.git | `0eb11d6c64c9` | `none` | `sepia-hemingway`, `sepia-recreate`, `sepia-refactor`, `sepia-review`, `sepia-write`, `sepia` |
| `supervisor-skills` | https://github.com/HKUSTDial/Supervisor-Skills.git | `4bd3fef393e5` | `none` | `benchmark-paper-template`, `deep-research`, `drawio-reconstruction`, `figure-designer`, `idea-evaluator`, `intro-drafter`, `paper-polish`, `paper-writer`, `pre-submission-reviewer`, `rebuttal-guidanc |
| `wx-cli` | https://github.com/Steady775/wx-cli | `08af894594b4` | `npm:@jackwener/wx-cli@0.3.0:wx` | `wx-cli` |

### mattpocock-skills

- 上游：https://github.com/mattpocock/skills
- ref：`4ae2317ddb22b046e26455e4f9c9b3aaaf62212d`
- mode：`skills`；skills_dir：`skills/engineering,skills/productivity,skills/misc`
- runtime：`none`
- 安装 skill：

| 安装名 | 上游相对路径 |
| --- | --- |
| `ask-matt` | `skills/engineering/ask-matt` |
| `code-review` | `skills/engineering/code-review` |
| `codebase-design` | `skills/engineering/codebase-design` |
| `diagnosing-bugs` | `skills/engineering/diagnosing-bugs` |
| `domain-modeling` | `skills/engineering/domain-modeling` |
| `grill-with-docs` | `skills/engineering/grill-with-docs` |
| `implement-spec` | `skills/engineering/implement-spec` |
| `implement` | `skills/engineering/implement` |
| `improve-codebase-architecture` | `skills/engineering/improve-codebase-architecture` |
| `pr` | `skills/engineering/pr` |
| `prototype` | `skills/engineering/prototype` |
| `research` | `skills/engineering/research` |
| `retro` | `skills/engineering/retro` |
| `setup-matt-pocock-skills` | `skills/engineering/setup-matt-pocock-skills` |
| `tdd` | `skills/engineering/tdd` |
| `to-spec` | `skills/engineering/to-spec` |
| `to-tickets` | `skills/engineering/to-tickets` |
| `triage` | `skills/engineering/triage` |
| `wayfinder` | `skills/engineering/wayfinder` |
| `wizard` | `skills/engineering/wizard` |
| `grill-me` | `skills/productivity/grill-me` |
| `grilling` | `skills/productivity/grilling` |
| `handoff` | `skills/productivity/handoff` |
| `teach` | `skills/productivity/teach` |
| `to-questionnaire` | `skills/productivity/to-questionnaire` |
| `wait-what` | `skills/productivity/wait-what` |
| `writing-for-agents` | `skills/productivity/writing-for-agents` |
| `git-guardrails-claude-code` | `skills/misc/git-guardrails-claude-code` |
| `migrate-to-shoehorn` | `skills/misc/migrate-to-shoehorn` |
| `scaffold-exercises` | `skills/misc/scaffold-exercises` |
| `setup-pre-commit` | `skills/misc/setup-pre-commit` |

provenance：MIT（LICENSE, Copyright 2026 Matt Pocock）。插件版本 1.4.0（2026-11 更新：CHANGELOG 增 software-factory 等 changeset；新增 skills/in-progress/software-factory 为未完成技能，同 in-progress 不安装）。安装 engineering、productivity、misc 三类；skills/in-progress 为上游标注的未完成技能，不安装。与 cursor-skills 的同名改写版（tdd、grill-with-docs、improve-codebase-architecture、setup-matt-pocock-skills、prototype、handoff）冲突时 cursor-skills 优先（先到先得，保留本地推理触发词改写）；其余同源名称直接安装上游版本

### rea

- 上游：https://github.com/morluto/rea
- ref：`160d821ac9a0310afcb8d1fc427862dd6508add5`
- mode：`skills`；skills_dir：`runtime-skills`
- runtime：`npm:rea-agents@6.8.0:rea`
- 安装 skill：

| 安装名 | 上游相对路径 |
| --- | --- |
| `reverse-engineer-anything` | `runtime-skills/reverse-engineer-anything` |

provenance：MIT。源码版本 6.8.0（tag rea-agents-6.8.0）（.release-please-manifest.json）。元数据差异：已安装的 rea-agents@6.8.0 npm 包 engines 为 22 系列 ^22.23.4、24 系列 ^24.22.1、以及 >=26.12.1（注意：字段分隔符不能出现在此处）；源码 package.json 同版本号的 engines 更新为 22.23.4 起、24.22.1 起与 26.12.1 起；运行时按 npm 包检查，不以源码为准（node-preflight.sh 已按 6.8.0 包的 engines 更新）。仓库内 skills/ 与 dist/ 为构建产物不存在于 clone 中，而 rea-agents@6.8.0 npm 包的 files 含 skills/reverse-engineer-anything（由 .agents/skills 模板经 generate-skill-metadata.mjs 生成，并固定到该版本）。因此 skill 来自 npm 包，不从仓库模板直接安装。.agents/skills/rea-* 为维护者用技能，不安装。安装器只为 codex 注册 REA MCP（install-rea-mcp.sh：rea setup --client codex --skill=false --yes，使用 REA 自身的备份与保留逻辑；--no-rea-mcp 或 --no-integrations 可跳过）。Hopper/Ghidra/IDA 等原生 provider 与 Frida 类安装脚本不在安装器范围内，未经验证

### sepia

- 上游：https://github.com/Nanako0129/sepia.git
- ref：`0eb11d6c64c962fcc35f596c167c6e38045d6cb8`
- mode：`skills`；skills_dir：`skills`
- runtime：`none`
- 安装 skill：

| 安装名 | 上游相对路径 |
| --- | --- |
| `sepia-hemingway` | `skills/sepia-hemingway` |
| `sepia-recreate` | `skills/sepia-recreate` |
| `sepia-refactor` | `skills/sepia-refactor` |
| `sepia-review` | `skills/sepia-review` |
| `sepia-write` | `skills/sepia-write` |
| `sepia` | `skills/sepia` |

provenance：MIT（LICENSE, Copyright 2026 Nanako Tsai）。plugin 版本 0.13.0（v0.13.0 release）。skills/sepia 为总路由（references/ 下有风格规则）；skills/sepia-write、sepia-review、sepia-refactor、sepia-recreate、sepia-hemingway 为操作入口，按 ../sepia/SKILL.md 相对路径读取总路由，必须与 sepia 作为同级目录安装。无运行时 CLI

### supervisor-skills

- 上游：https://github.com/HKUSTDial/Supervisor-Skills.git
- ref：`4bd3fef393e5ffc891c74309a5ef6c5ad3ad12c8`
- mode：`skills`；skills_dir：`skills`
- runtime：`none`
- 安装 skill：

| 安装名 | 上游相对路径 |
| --- | --- |
| `benchmark-paper-template` | `skills/benchmark-paper-template` |
| `deep-research` | `skills/deep-research` |
| `drawio-reconstruction` | `skills/drawio-reconstruction` |
| `figure-designer` | `skills/figure-designer` |
| `idea-evaluator` | `skills/idea-evaluator` |
| `intro-drafter` | `skills/intro-drafter` |
| `paper-polish` | `skills/paper-polish` |
| `paper-writer` | `skills/paper-writer` |
| `pre-submission-reviewer` | `skills/pre-submission-reviewer` |
| `rebuttal-guidance` | `skills/rebuttal-guidance` |
| `tech-paper-template` | `skills/tech-paper-template` |
| `vibe-research-workflow` | `skills/vibe-research-workflow` |

provenance：检视 2026-11-08（本地 checkout detached 于 4bd3fef，origin 与 url 一致；2026-11-03 更新仅 README 增补说明，12 个 skill 内容未变）。根 LICENSE 为 CC BY-NC-SA 4.0（Copyright (c) 2026 Yuyu Luo (methodology) and contributors），安装的 skill 副本不含根 LICENSE，归属与许可记在本条与 skills/research/README.md。12 个 SKILL.md 位于 skills/<name>/，frontmatter name 与目录名一致。逐 skill 的 frontmatter license：CC-BY-4.0（figure-designer、idea-evaluator、vibe-research-workflow、tech-paper-template、benchmark-paper-template、pre-submission-reviewer），CC-BY-NC-SA-4.0（deep-research、paper-writer、intro-drafter、paper-polish），rebuttal-guidance 与 drawio-reconstruction 无 frontmatter license。skills/drawio-reconstruction/LICENSE 为 MIT（Copyright (c) 2026 VCG-Bench Authors），随该目录一起安装。skill 之间的依赖均为各自目录内的相对路径（references/、scripts/、agents/、examples/、assets/、requirements.txt）；无 CLI 运行时：安装器不装任何命令，drawio 脚本仅标准库，crop_assist.py 的 Pillow 为可选、需用户批准后自行安装，export_drawio.py 需本机 draw.io 桌面应用，安装器不提供。handbook/ 为仓库根目录的教材，不随任何 skill 安装，仅被 idea-evaluator 与 vibe-research-workflow 作为可选深入阅读引用

### wx-cli

- 上游：https://github.com/Steady775/wx-cli
- ref：`08af894594b4afd468e23e17dbd783f15403f13b`
- mode：`skills`；skills_dir：`.`
- runtime：`npm:@jackwener/wx-cli@0.3.0:wx`
- 安装 skill：

| 安装名 | 上游相对路径 |
| --- | --- |
| `wx-cli` | `.` |

provenance：Apache-2.0。这是 Steady775 fork（origin 为 Steady775/wx-cli，提交 08af894）；README、Cargo.toml、npm 包均为 jackwener/wx-cli 上游 0.3.0。SKILL.md（唯一 frontmatter name=wx-cli）来自 fork；可执行文件来自 npm 官方包 @jackwener/wx-cli@0.3.0（fork 未发布 npm 包，本机也无 cargo）。安装只执行 npm 安装与 wx --version；不执行 sudo wx init、codesign、tccutil、重启微信或读取聊天数据

安装 skill 总数（含各 catalog 枚举与 external，可能有同名先到先得）：628。

## 检查与更新流程

```bash
python3 scripts/build_sources.py --fetch   # 查询上游 tip 并写入本文件的更新状态列
python3 scripts/build_sources.py --check   # 校验本文件与 checkout 一致
bash update-repos.sh all                   # 更新全部 submodule / 外部 checkout
```

逐来源的检查结论记录在 `sources/update-log.md`（手工维护）。

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

