---
name: deepgemm-skill
description: >
  DeepGEMM source reference and workflow for developing NVIDIA grouped GEMM and MoE kernels. Use for
  DeepGEMM, M-grouped or K-grouped GEMM, contiguous or masked grouped layouts, FP8/FP4/BF16 GEMM, Mega
  MoE, SM90/Hopper, SM100/Blackwell, JIT kernel generation, TMA, WGMMA, tcgen05, grouped GEMM
  scheduling, alignment, scaling-factor packing, or adapting DeepGEMM kernels into another project.
  Common queries: DeepGEMM install, DeepGEMM grouped GEMM, M-grouped, K-grouped, contiguous grouped
  GEMM, masked grouped GEMM, DeepGEMM JIT, gemm.hpp, DeepGEMM scheduler, DeepGEMM test, test_fp8_fp4,
  test_mega_moe, Mega MoE, DeepGEMM SM90, DeepGEMM SM100, FP8 FP4 BF16 GEMM, scale factor, UE8M0,
  swapAB, TMA multicast, DeepGEMM build, DeepGEMM integration, DeepGEMM contiguous layout, DeepGEMM
  masked layout, expert parallelism kernel. 中文触发词：DeepGEMM、grouped GEMM、分组 GEMM、MoE GEMM、M grouped、K
  grouped、contiguous layout、masked layout、Mega MoE、专家算子。
triggers:
  - "DeepGEMM"
  - "DeepGEMM JIT"
  - "DeepGEMM scheduler"
  - "K-grouped"
  - "M-grouped"
  - "Mega MoE"
  - "MoE GEMM"
  - "contiguous layout"
  - "gemm.hpp"
  - "grouped GEMM"
  - "masked layout"
  - "test_fp8_fp4"
  - "test_mega_moe"
  - "专家算子"
  - "分组 GEMM"
---

# DeepGEMM Grouped GEMM Development

Use the installed DeepGEMM checkout as a read-only implementation reference.
Implement changes in the user's target project unless the user explicitly asks
to modify DeepGEMM itself.

## Locate the source

DeepGEMM lives under this skill's `repos/deepgemm/` directory:

- Cursor: `~/.cursor/skills/deepgemm-skill/repos/deepgemm/`
- Claude Code: `~/.claude/skills/deepgemm-skill/repos/deepgemm/`
- Codex: `~/.codex/skills/deepgemm-skill/repos/deepgemm/`
- Gemini CLI: `~/.gemini/skills/deepgemm-skill/repos/deepgemm/`

Set `DEEPGEMM_REPO` to the applicable path before running search commands. If
the checkout is missing, run `bash update-repos.sh deepgemm` from the
`agent-gpu-skills` repository, then reinstall the skill.

Before using exact paths, read
[`references/grouped-gemm-code-map.md`](references/grouped-gemm-code-map.md).
It records the current host-to-device call chain and the best starting points.

## Choose the kernel family

1. Use **M-grouped contiguous** for training forward or prefill when tokens from
   all experts are concatenated and every expert shares fixed N and K.
2. Use **M-grouped masked** for decode/CUDA Graph flows with `[G, max_M, K]`
   storage and a device-side valid-M value per expert.
3. Use **K-grouped contiguous** for weight-gradient-style work where M and N are
   fixed, K varies by group, and each result may accumulate C.
4. Use **Mega MoE** only for the fused multi-rank SM100 path that combines EP
   communication, two GEMMs, and SwiGLU.
5. Use the dense GEMM path when group scheduling or expert-specific weights are
   not required.

Do not treat DeepGEMM as a general CUTLASS grouped GEMM: its M-grouped kernels
hold N and K fixed, and its K-grouped kernels hold M and N fixed.

## Trace before adapting

Follow this order instead of copying a device kernel in isolation:

1. Read the matching test in `tests/test_bf16.py`, `tests/test_fp8_fp4.py`, or
   `tests/test_mega_moe.py` to establish the tensor contract and reference math.
2. Read `tests/generators.py` to understand padding, empty groups, cumulative
   layouts, scaling factors, and architecture-dependent variants.
3. Trace the Python export in `deep_gemm/__init__.py` through
   `csrc/apis/gemm.hpp` or `csrc/apis/mega.hpp`.
4. Trace the selected JIT wrapper in `csrc/jit_kernels/impls/`; record its
   generated template parameters, runtime arguments, and heuristic inputs.
5. Read the corresponding device implementation in
   `deep_gemm/include/deep_gemm/impls/`, then inspect
   `scheduler/gemm.cuh`, MMA/PTX helpers, and epilogue code only as needed.
6. Port the smallest complete vertical slice: contract checks, layout
   transformation, scheduling, mainloop, and epilogue.

## Preserve critical invariants

- Query `get_theoretical_mk_alignment_for_contiguous_layout()` and set the
  chosen alignment; never hard-code the current result.
- Distinguish per-row expert IDs, per-group valid-M values, per-group K sizes,
  and prefix-sum end offsets. They are not interchangeable `grouped_layout`
  encodings.
- Keep padding and empty-group behavior from `tests/generators.py`, including
  `-1` M-group padding and aligned prefix-sum gaps.
- Preserve architecture-specific scaling: SM90 uses FP32 scale factors, while
  SM100 uses packed UE8M0 where required.
- Preserve layout restrictions. In particular, SM90 FP8 paths are more
  restrictive than SM100 and commonly require K-major A/B.
- Include C/D accumulation semantics for K-grouped kernels.
- Treat `deep_gemm/legacy/` as SM80 Triton compatibility code, not the primary
  implementation for current SM90/SM100 work.

## Search efficiently

```bash
rg -n "m_grouped|k_grouped" "$DEEPGEMM_REPO/tests" "$DEEPGEMM_REPO/csrc/apis"
rg -n "MGrouped|KGrouped" "$DEEPGEMM_REPO/csrc" "$DEEPGEMM_REPO/deep_gemm/include"
rg -n "grouped_layout|use_psum_layout" "$DEEPGEMM_REPO/deep_gemm/include/deep_gemm/scheduler"
rg -n "get_.*config|expected_m|num_groups" "$DEEPGEMM_REPO/csrc/jit_kernels/heuristics"
rg -n "wgmma|tcgen05|TMA" "$DEEPGEMM_REPO/deep_gemm/include/deep_gemm"
```

## Verify in risk order

1. Reproduce the target tensor contract with a small per-group PyTorch
   reference before tuning.
2. Cover zero-sized groups, nonuniform group sizes, alignment tails, padding,
   accumulation, and every enabled layout/dtype combination.
3. Run the narrow matching upstream test selection on supported hardware.
4. Benchmark only after correctness passes; compare against the same shapes and
   byte/operation accounting used by the upstream tests.
5. For performance claims, collect NCU/NSYS evidence and inspect JIT output with
   `DG_JIT_WITH_LINEINFO=1`, `DG_JIT_DUMP_SASS=1`, or
   `DG_JIT_PTXAS_VERBOSE=1` as needed.
