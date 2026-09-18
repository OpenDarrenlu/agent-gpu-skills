---
name: gpu-framework-router
description: Route GPU framework, runtime, serving, and framework-level profiling questions to the right skill. Use for SGLang, vLLM-like serving, PyTorch runtime behavior, attention backends, KV cache, end-to-end latency, workload benchmarks, Nsight Systems traces, framework-level Nsight Compute profiling, or deciding whether a problem belongs to the framework or the underlying kernel.
---

# GPU Framework Router

Use this entry point when the question is about a GPU framework or an end-to-end
runtime rather than a standalone kernel.

## Routing order

1. **SGLang internals, model/backend integration, KV cache, scheduler, or serving
   regressions**: use `sglang-skill`.
2. **LLM serving design, vLLM/SGLang/LMDeploy/TensorRT-LLM comparison, batching,
   capacity, TTFT/TBT, or deployment strategy**: use `llm-serving-router`.
3. **Framework-level workload profiling or benchmark design**: use
   `llm-serving-auto-benchmark` or `perf-workload-profiling` when installed.
   For a PyTorch baseline that will become a TileLang kernel, use
   `torch-profiling-tilelang-programs`.
4. **Whole-application CPU/GPU timeline, launch causality, overlap, or NVTX
   analysis**: use `perf-nsight-systems` or `nsys-profile-analysis`.
5. **A framework dispatches a slow individual CUDA/Triton/CUTLASS kernel**:
   hand off to `gpu-performance-router`, then use the most specific kernel skill.
6. **PyTorch profiler, compiler, or host-side runtime behavior**: use
   `llm-torch-profiler-analysis`, `torch-profiling-tilelang-programs`,
   `perf-host-analysis`, or `perf-host-optimization` when installed.

## Framework versus kernel

Ask one separating question before optimizing:

- Is end-to-end latency dominated by scheduling, batching, memory management,
  synchronization, launch overhead, or a backend choice? Stay in this router.
- Is one kernel or one dispatch path dominant after measurement? Hand off to
  `gpu-performance-router`.

Do not infer a framework bottleneck from a single kernel metric. Start with a
representative workload and a framework-level timeline, then drill down into
kernel profiling only after the hot path is established.
