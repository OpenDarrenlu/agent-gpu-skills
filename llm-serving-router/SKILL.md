---
name: llm-serving-router
description: >
  Router for LLM serving, inference systems, SGLang, vLLM-like serving issues, KV cache, attention
  backends, FlashInfer, MLA, MoE, expert parallelism, TP/PP/EP, speculative decoding, continuous
  batching, chunked prefill, CUDA Graph, throughput, latency, capacity planning, and serving benchmarks.
  Casual entry point: anything about LLM serving, inference service, deploy LLM, serving framework,
  vLLM, SGLang, LMDeploy, TensorRT-LLM, TGI, ktransformers, KV cache issue, attention backend,
  throughput benchmark, latency benchmark, TTFT, TBT, inter-token latency, capacity planning, GPU memory
  for serving, OOM serving, batching strategy, scheduling strategy, prefill decode, disaggregated
  serving, spec decode. 中文触发词：LLM 服务、推理服务、SGLang、KV cache、attention
  backend、FlashInfer、MLA、MoE、吞吐、延迟、压测、容量规划。
triggers:
  - "KV cache"
  - "LLM 服务"
  - "OOM"
  - "SGLang"
  - "TBT"
  - "TTFT"
  - "attention backend"
  - "batching"
  - "decode"
  - "prefill"
  - "speculative decoding"
  - "vLLM"
  - "吞吐"
  - "容量规划"
  - "延迟"
  - "推理服务"
  - "部署大模型"
---

# LLM Serving Router

Use this router for LLM inference and serving questions, especially when GPU kernels and serving behavior interact.

## Route

- **SGLang implementation, runtime, model/backend changes, serving bugs**: use `sglang-skill`.
- **Serving benchmark or automated latency/throughput testing**: use `llm-serving-auto-benchmark` when installed.
- **Capacity planning, tokens/sec, GPU count, concurrency, SLA sizing**: use `llm-serving-capacity-planner` when installed.
- **Pipeline/runtime bottleneck analysis**: use `llm-pipeline-analysis` when installed.
- **Torch profiler traces**: use `llm-torch-profiler-analysis` when installed.
- **DeepGEMM grouped/MoE kernel implementation details**: use `deepgemm-skill`.
- **Other attention/MoE kernel implementation details**: use `triton-skill`, `cutlass-skill`, or `cuda-skill` depending on stack.
- **Multi-GPU communication or parallelism bottlenecks**: use `gpu-communication-libraries`.

## Workflow

1. Separate serving symptoms from kernel symptoms: latency/throughput/OOM/scheduler vs single-kernel performance.
2. Read the most specific downstream skill before proposing changes.
3. Ask for or infer the serving shape when needed: model, GPUs, batch/concurrency, prefill/decode mix, sequence lengths, backend.
4. Prefer measurement-backed recommendations: benchmark, trace, or profiler evidence before deeper rewrites.
