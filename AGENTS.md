# Agent Guidance

This repository is a skill bundle for GPU development agents. When working here,
prefer the most specific installed skill before answering detailed technical questions.

## Skill Routing

Use a router first when the user does not name a skill:

- `gpu-development-catchall`: any GPU/CUDA/ROCm/kernel/performance/communication/serving question where the right domain is unclear; it routes to the most specific skill.
- `gpu-performance-router`: GPU profiling, "why is this slow", NCU/NSYS, SM/TC utilization, stalls, occupancy, roofline, B200/H100/A100 optimization.
- `gpu-kernel-authoring-router`: writing or changing CUDA, PTX, Triton, Gluon, CUTLASS, CuTe, GEMM, attention, or fused GPU kernels.
- `llm-serving-router`: SGLang, LLM inference, KV cache, attention backends, FlashInfer, MLA, MoE, throughput, latency, benchmarks, capacity planning.

If a router points to a more specific skill, read that downstream skill before producing the detailed answer or code.

## Direct Skill Preferences

- CUDA C++ / PTX / NVIDIA API / architecture docs: `cuda-skill`
- AMD/ROCm/HIP documentation and broad local knowledge-base queries: `amd-gpu-docs`
- AMD Instinct / CDNA4 ISA / AMD GPU assembly: `amd-instinct-cdna4-isa`
- Triton / Gluon / Python GPU kernels: `triton-skill`
- CUTLASS / CuTe / CuTeDSL / template GEMM: `cutlass-skill`
- DeepGEMM / grouped GEMM / MoE GEMM / contiguous or masked expert layouts: `deepgemm-skill`
- SGLang serving/runtime/kernel integration: `sglang-skill`
- Nsight Compute / `.ncu-rep` / kernel bottleneck reports: `ncu-report-skill`
- Persistent kernel NCU diagnosis: `ncu-persistent-kernel-diagnosis`
- Persistent kernel scheduling / CLC / Stream-K: `persistent-kernel-scheduling`
- Persistent kernel utilization tuning: `persistent-kernel-utilization`
- Analytical GPU performance modeling: `nv-gpu-kernel-performance-modeling`
- NCCL / NVSHMEM / GPUDirect / RDMA / collectives: `gpu-communication-libraries`
- Colfax Research articles and reading order: `colfax-research-skill`

## Operating Rules

- Do not require the user to mention a skill explicitly. Infer from domain words, symptoms, tools, and file types.
- For performance work, prefer profile evidence over speculation. If evidence is missing, propose the smallest profiling step.
- For implementation work, verify correctness before performance tuning.
- Keep answers grounded in the chosen skill's local references, scripts, and repo paths when available.
