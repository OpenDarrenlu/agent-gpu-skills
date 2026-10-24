# Agent Guidance

This repository is an Agent Skill bundle. Local skills are organized under
`skills/kernel`, `skills/framework`, `skills/other`, and `skills/routers`;
external catalogs remain under `repos/` and are installed without being copied
into this repository.

## Routing

When the user does not name a skill, first decide whether the task is GPU-only.
A named skill always wins; check that it is installed before relying on it.

For GPU-only work, start with `gpu-development-catchall` and choose the
narrowest route:

- `gpu-kernel-authoring-router`: write or change CUDA, PTX, Triton, Gluon,
  CUTLASS, CuTe, GEMM, attention, reduction, or fused kernels.
- `gpu-performance-router`: a kernel is slow, NCU/NSYS evidence, occupancy,
  warp stalls, SM/TC utilization, roofline, persistent-kernel diagnosis, or
  optimization planning.
- `gpu-framework-router`: framework/runtime behavior, SGLang/vLLM serving,
  verl RL post-training, PyTorch profiler, end-to-end benchmark,
  launch/scheduling overhead, or framework-level Nsight analysis.
- `llm-serving-router`: LLM serving, KV cache, attention backends, batching,
  capacity planning, TTFT/TBT, deployment, and serving benchmarks.

For mixed or non-GPU work, use `agent-skill-router`. It covers papers and
literature (Zotero, paper review, rebuttal, submission checks), Chinese technical
writing, general code review and software design (TDD, architecture, diagnosis,
issues, plans), WeChat data (only with explicit authorization), and the
sources in `sources/external-sources.conf`. Run its `scripts/route_skill.py` for
the primary skill and at most two companions, then read only the chosen
`SKILL.md` files.

If a router points to a downstream skill, read that skill's `SKILL.md` before
producing detailed advice or code.

## Direct Skill Preferences

### Kernel

- CUDA C++ / PTX / NVIDIA API: `cuda-skill`
- Triton / Gluon / Python GPU kernels: `triton-skill`
- CUTLASS / CuTe / CuTeDSL / template GEMM: `cutlass-skill`
- DeepGEMM / grouped GEMM / MoE GEMM: `deepgemm-skill`
- Blackwell/Hopper kernel wiki and upstream PR references: `KernelWiki`
- Nsight Compute / `.ncu-rep`: `ncu-report-skill`
- Persistent kernel NCU diagnosis: `ncu-persistent-kernel-diagnosis`
- Persistent scheduling / CLC / Stream-K: `persistent-kernel-scheduling`
- Persistent utilization / pipeline bubbles: `persistent-kernel-utilization`
- IKET CuTe DSL timelines: `iket-profiling`
- Analytical kernel modeling: `nv-gpu-kernel-performance-modeling`
- TileLang kernel authoring: `writing-tilelang-kernels`
- TileLang debugging/lowering: `debugging-tilelang-programs`
- TileLang optimization loop: `tilelang-optimization-loop`
- TileLang kernel profiling: `profiling-tilelang-programs`
- TileLang PyTorch baseline profiling: `torch-profiling-tilelang-programs`
- TileLang forward/backward validation: `testing-fwd-bwd-kernels`
- TileFoundry HIR and placement design: `designing-with-tilefoundry`
- CUTLASS/CuTe/FlashAttention research: `colfax-research-skill`

### Framework

- SGLang runtime and backend integration: `sglang-skill`
- verl/HybridFlow RL post-training (PPO/GRPO, rollout, reward, agent loop,
  train-inference mismatch): `verl-skill`
- Bitwise train-inference consistency / true on-policy RL (vLLM batch
  invariance, logprob 0 diff, BI kernels, backward/STE, C0–C10 gates):
  `train-inference-consistency` (from vllm-project/vllm `.agents/skills`)
- Serving design and capacity planning: `llm-serving-router`
- Whole-application timeline: `nsys-profile-analysis` or `perf-nsight-systems`
- Serving workload benchmark: `llm-serving-auto-benchmark`
- PyTorch/runtime profiling: `llm-torch-profiler-analysis`

### Other

- 简化实现、避免过度设计、通用代码审查与仓库审计：`ponytail` /
  `ponytail-review` / `ponytail-audit`（来自 `repos/ponytail/skills`；GPU 事实与性能结论仍使用领域 skill）。

- AMD/ROCm/HIP docs: `amd-gpu-docs`
- AMD Instinct/CDNA ISA: `amd-instinct-isa`
- NCCL/NVSHMEM/RDMA/collectives: `gpu-communication-libraries` or `nccl-skill`
- Local technical article retrieval: `zhihu-knowledge-base` or
  `giantpandallm-knowledge`
- Research-paper workflow: the matching `ccf-*` skill in `repos/ccfa-skills`
- Salary/offer negotiation: `salary-negotiation-skill`
- Chinese technical writing (kernel/inference/training 原理报告, design docs,
  performance reports, source walkthroughs, screenshot-to-article, API docs,
  runbooks): `tech-doc-style-chinese` (local GPU-customized override of
  Fenng/Tech-Doc-Style-Chinese; upstream clone in `repos/tech-doc-style-chinese`)

## Operating Rules

- Infer the skill from domain words, symptoms, tools, and file types; do not
  require the user to name it.
- For performance work, prefer profile evidence over speculation. If evidence is
  missing, propose the smallest profiling step.
- Verify correctness before performance tuning.
- Keep answers grounded in the selected skill's local references, scripts, and
  source maps.
- For Chinese reports and articles, combine the domain skill (facts, evidence)
  with `tech-doc-style-chinese` (structure, evidence levels, typography).

<!-- BEGIN zotero-cli skill (managed by `zotero-mcp install-skill`) -->

## Zotero library access

Read and write a Zotero library from the shell with the `zotero-cli` command - search papers by keyword or meaning, read PDF full text and page ranges, get and set metadata, manage collections, tags, notes and annotations, add items by DOI/URL/ISBN, and export bibliographies. Use whenever the user asks about their Zotero library, references, citations, papers they have saved, or their reading notes.

Read `skills/other/zotero-cli/SKILL.md` before using it. Prefer it over the Zotero MCP server when you have shell access: the MCP tool schemas cost ~13k tokens of context on every request, this costs only what you run.

Quick check that it is set up: `zotero-cli config`. Pass `--json` whenever you will parse the output.

<!-- END zotero-cli skill -->
