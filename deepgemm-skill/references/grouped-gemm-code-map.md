# DeepGEMM Grouped GEMM Code Map

Verified against DeepGEMM 2.6.1 at commit
`559d79fb6994a58b8a15b4b93bf13ccc16edf247` (2026-07-15). Re-run the search
commands after updating the checkout because upstream paths can change.

## Call chain

| Layer | Primary path | Purpose |
|:--|:--|:--|
| Python exports | `deep_gemm/__init__.py` | Public grouped GEMM and Mega MoE names |
| Test contracts | `tests/test_bf16.py`, `tests/test_fp8_fp4.py` | Correctness, aliases, shapes, alignment, performance accounting |
| Input generators | `tests/generators.py` | Layout construction, quantization, empty groups, prefix sums |
| Host API | `csrc/apis/gemm.hpp` | Validation, scale transformation, architecture dispatch, pybind docs |
| Layout API | `csrc/apis/layout.hpp` | M/K alignment and scale-factor packing |
| JIT wrapper | `csrc/jit_kernels/impls/*.hpp` | Heuristics, template rendering, launch arguments, runtime compilation |
| Heuristics | `csrc/jit_kernels/heuristics/{config,sm90,sm100}.hpp` | Tile, cluster, pipeline, launch configuration |
| Device kernels | `deep_gemm/include/deep_gemm/impls/*.cuh` | Mainloops and architecture-specific implementations |
| Scheduler | `deep_gemm/include/deep_gemm/scheduler/gemm.cuh` | Persistent grouped work assignment and group offsets |
| MMA/PTX | `deep_gemm/include/deep_gemm/{mma,ptx}/` | WGMMA, tcgen05, TMA, load/store primitives |
| Epilogue | `deep_gemm/include/deep_gemm/epilogue/` | Accumulator transformation and C/D stores |

## Family entry points

| Family | Python/API names | Tests and implementation starting points |
|:--|:--|:--|
| M-grouped contiguous FP8/FP4 | `m_grouped_fp8_fp4_gemm_{nt,nn}_contiguous` and FP8 aliases | `tests/test_fp8_fp4.py::test_m_grouped_gemm_contiguous`; `csrc/apis/gemm.hpp`; `sm90_fp8_gemm_1d2d.hpp`; `sm100_fp8_fp4_gemm_1d1d.hpp` |
| M-grouped masked FP8/FP4 | `m_grouped_fp8_fp4_gemm_nt_masked` and FP8 alias | `tests/test_fp8_fp4.py::test_m_grouped_gemm_masked`; same JIT wrapper families |
| K-grouped FP8 | `k_grouped_fp8_gemm_{nt,tn}_contiguous` | `tests/test_fp8_fp4.py::test_k_grouped_gemm_contiguous`; `sm90_fp8_gemm_1d1d.hpp`; `sm100_fp8_fp4_gemm_1d1d.hpp` |
| M-grouped BF16 | `m_grouped_bf16_gemm_{nt,nn}_contiguous`, `m_grouped_bf16_gemm_nt_masked` | `tests/test_bf16.py`; `sm90_bf16_gemm.hpp`; `sm100_bf16_gemm.hpp` |
| K-grouped BF16 | `k_grouped_bf16_gemm_tn_contiguous` | `tests/test_bf16.py::test_k_grouped_gemm_contiguous`; BF16 JIT wrappers |
| Mega MoE | `fp8_fp4_mega_moe`, `bf16_mega_moe` | `tests/test_mega_moe.py`; `csrc/apis/mega.hpp`; `heuristics/mega_moe.hpp`; `scheduler/mega_moe.cuh` |

The device headers corresponding to a JIT wrapper have the same base filename
under `deep_gemm/include/deep_gemm/impls/`.

## `grouped_layout` encodings

| Mode | Encoding | Important behavior |
|:--|:--|:--|
| M-grouped contiguous, row map | `int32[M]`; expert ID for valid rows, `-1` for padding | Each expert segment begins at the selected M alignment |
| M-grouped contiguous, prefix sum | `int32[G]`; actual end offset for each group | Next group begins at `align(previous_end, alignment)` |
| M-grouped masked | `int32[G]`; valid M for each `[G, max_M, ...]` slice | `expected_m` is a scheduling hint and undersizing it can reduce efficiency |
| K-grouped contiguous, size list | `int32[G]`; K length per group | Host `ks_cpu` and device layout describe the same aligned groups |
| K-grouped contiguous, prefix sum | `int32[G]`; actual end offset for each group | Next group begins at `align(previous_end, k_alignment)`; supports empty groups |

Read `generate_m_grouped_contiguous`, `generate_m_grouped_masked`,
`generate_k_grouped_contiguous`, and `generate_k_grouped_contiguous_psum` before
constructing any of these layouts.

## Architecture split

| Concern | SM90 | SM100 |
|:--|:--|:--|
| Tensor core path | WGMMA | tcgen05/UMMA |
| FP8 scaling | FP32 scale-factor path | Packed UE8M0 path where selected |
| FP8 grouped implementation | Primarily 1D2D for M-grouped, 1D1D for K-grouped | FP8/FP4 1D1D family |
| A/B layout | FP8 is generally K-major restricted | Supports more MN/K-major combinations |
| K-grouped prefix-sum path | Not the primary path | Supported by the SM100 scheduler and packing helpers |
| Mega MoE | Not supported | SM100 fused communication/compute path |

Confirm current guards in `csrc/apis/gemm.hpp` and
`csrc/utils/compatibility.hpp`; do not infer support from filenames alone.

## High-value searches

```bash
rg -n "m_grouped.*contiguous|m_grouped.*masked|k_grouped" csrc/apis/gemm.hpp
rg -n "GemmType::MGrouped|GemmType::KGrouped" deep_gemm/include/deep_gemm/scheduler/gemm.cuh
rg -n "grouped_layout" csrc/jit_kernels/impls deep_gemm/include/deep_gemm/impls
rg -n "get_theoretical_mk_alignment|set_mk_alignment" csrc/apis tests
rg -n "assert diff|ref_d" tests/test_bf16.py tests/test_fp8_fp4.py
```
