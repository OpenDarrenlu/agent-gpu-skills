---
name: gpu-performance-router
description: >
  Router for NVIDIA GPU performance, profiling, and bottleneck diagnosis. Use when
  the user says a CUDA/Triton/CUTLASS kernel is slow, asks "why is this slow",
  wants ncu/nsys/Nsight Compute/Nsight Systems analysis, SM/TC utilization,
  warp stalls, occupancy, memory throughput, persistent kernel bubbles, roofline,
  B200/H100/A100 performance, or an optimization plan. 中文触发词：为什么慢、性能瓶颈、
  ncu 报告、nsys 时间线、SM 利用率、TC 利用率、warp stall、occupancy、GPU 优化计划。
---

# GPU Performance Router

Use this router to pick the most specific GPU performance skill before answering.

## Route

- **NCU / Nsight Compute / .ncu-rep / single-kernel bottleneck**: use `ncu-report-skill`.
- **Persistent kernel NCU diagnosis**: use `ncu-persistent-kernel-diagnosis`.
- **Persistent scheduling choice, CLC, Stream-K, tail effect**: use `persistent-kernel-scheduling`.
- **Persistent utilization tuning, SM/TC utilization, pipeline bubbles**: use `persistent-kernel-utilization`.
- **Analytical latency prediction or cross-architecture modeling**: use `nv-gpu-kernel-performance-modeling`.
- **General CUDA/PTX correctness, architecture, or API details**: use `cuda-skill`.
- **Triton kernel performance**: use `triton-skill`, then use `ncu-report-skill` if profiling evidence is needed.
- **CUTLASS/CuTe kernel performance**: use `cutlass-skill`, then use `ncu-report-skill` if profiling evidence is needed.

## Workflow

1. Identify whether the user has profiling evidence, source code, or only symptoms.
2. Select the downstream skill above and read its `SKILL.md` before doing detailed analysis.
3. If no profiling evidence exists and the task is about performance, prefer a profile-first plan over speculative fixes.
4. Keep the final answer evidence-ranked: observed signal, likely cause, concrete next action.
