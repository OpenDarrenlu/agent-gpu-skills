# 17 - Python Bindings (`nccl4py`) and NCCL EP Expert Parallelism Extension

## Primary source files

- `repos/nccl/bindings/nccl4py/README.md`
- `repos/nccl/bindings/nccl4py/setup.py`
- `repos/nccl/bindings/nccl4py/pyproject.toml`
- `repos/nccl/bindings/nccl4py/nccl/core/communicator.py`
- `repos/nccl/bindings/nccl4py/nccl/core/team.py`
- `repos/nccl/bindings/nccl4py/nccl/core/resources.py`
- `repos/nccl/bindings/nccl4py/nccl/core/typing.py`
- `repos/nccl/bindings/nccl4py/nccl/core/device/cute/*`
- `repos/nccl/bindings/nccl4py/examples/cute/*`
- `repos/nccl/bindings/ir/*`
- `repos/nccl/contrib/nccl_ep/README.md`
- `repos/nccl/contrib/nccl_ep/include/nccl_ep.h`
- `repos/nccl/contrib/nccl_ep/include/ep_enums.h`
- `repos/nccl/contrib/nccl_ep/nccl_ep_env.h`
- `repos/nccl/contrib/nccl_ep/nccl_ep.cc`
- `repos/nccl/contrib/nccl_ep/ep_test.cu`
- `repos/nccl/contrib/nccl_ep/ep_bench.cu`
- `repos/nccl/contrib/nccl_ep/ep_test.py`

## nccl4py overview

`nccl4py` provides Python bindings for NCCL with both low-level Cython bindings and a higher-level
Pythonic API for collective operations.

Requirements from the README:

| Component | Requirement |
|---|---|
| CUDA Toolkit | CUDA 12.x or 13.x |
| NCCL Library | matching CUDA package (`nvidia-nccl-cu12` or `nvidia-nccl-cu13`) |
| Python | 3.10+ |

Experimental Cython support:

```python
from nccl.bindings cimport cynccl
```

The `cynccl.pxd` file is included for direct Cython integration.

The version in this checkout is 0.5.0 (`repos/nccl/bindings/nccl4py/nccl/core/_version.py`).

`nccl` is a PEP 420 implicit namespace package: nccl4py owns `nccl.bindings` and `nccl.core`, while NCCL
extension distributions (e.g. NCCL EP) add their own `nccl.<ext>` subpackage. Per the README, this means
nccl4py must be installed into a real environment rather than with `pip install --target DIR`.

## nccl4py development setup

Set CUDA path:

```bash
export CUDA_HOME=/usr/local/cuda
```

Build with Makefile and `uv`:

```bash
cd repos/nccl/bindings/nccl4py
make dev
make build
make clean
```

Manual setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[cu12]   # CUDA 12.x
# or
pip install -e .[cu13]   # CUDA 13.x
pip install build
python -m build
```

The Makefile detects CUDA version from `CUDA_HOME`, installs CUDA-specific dependencies, and builds Cython
extensions.

## nccl4py teams and device communicator requirements

The Pythonic API in `nccl.core` mirrors the NCCL Device API host calls:

- `NCCLTeam` (`repos/nccl/bindings/nccl4py/nccl/core/team.py`) is an `(n_ranks, rank, stride)` view over a
  communicator, produced by the `Communicator` properties `team_world`, `team_lsa`, and `team_rail`
  (`repos/nccl/bindings/nccl4py/nccl/core/communicator.py`).
- `Communicator.team_rank_to_world(team, team_rank)` and `Communicator.team_rank_to_lsa(team, team_rank)`
  translate team-relative ranks to world or LSA rank space.
- `Communicator.create_dev_comm(requirements)` takes an `NCCLDevCommRequirements` dataclass and returns a
  `DevCommResource`. Beyond scalar fields (`barrier_count`, `gin_context_count`, `gin_signal_count`,
  `gin_counter_count`, `gin_connection_type`, `gin_traffic_class`, `gin_type`, `cft_caps`,
  `cft_barrier_count`, ...), requirements are declared structurally:
  - `teams`: tuple of `TeamRequirement(team, multimem)`; a team requested with `multimem=True` yields a
    multicast handle retrievable via `DevCommResource.multimem_handle`.
  - `resources`: tuple of `LsaBarrierRequirement`, `GinBarrierRequirement`, or `LLA2ARequirement`; each
    entry yields, in order, a handle in `DevCommResource.resource_handles`.
- `RegisteredWindowHandle.get_multimem_device_pointer(multimem, offset=0)`
  (`repos/nccl/bindings/nccl4py/nccl/core/resources.py`) resolves the multicast device pointer for a
  window against an explicit `MultimemHandle` from device communicator creation, complementing
  `get_lsa_multimem_device_pointer(offset)` which uses the LSA team's multimem.

## nccl4py per-call collective config

Every collective on `Communicator` (`allreduce`, `broadcast`, `reduce`, `allgather`, `reduce_scatter`,
`alltoall`, `gather`, `scatter`) accepts a keyword-only `config: NCCLCollConfig | None` argument that tunes
that single call. `NCCLCollConfig` (`repos/nccl/bindings/nccl4py/nccl/core/communicator.py`) mirrors
`ncclCollConfig_t`:

| Field | Meaning |
|---|---|
| `min_ctas` / `max_ctas` | per-call channel/CTA bounds (`NCCL_MIN_CTAS` / `NCCL_MAX_CTAS` take precedence) |
| `nvls_ctas` | NVLS-pool channel cap |
| `cga_cluster_size` | thread-block-cluster size 0-8 (Hopper+) |
| `alg_selection` / `force_alg_selection` | algorithm filter string, e.g. `"ring"`, `"tree,ring"`, `"^symk"`; whether an unsatisfiable selection is an error |
| `cta_policy` | `CTAPolicy` scheduling policy |
| `user_profiler_tag` | opaque value delivered to profiler plugins |
| `vendor_options` | tuple of `VendorOption` |

`VendorOption` mirrors one `ncclConfigExt_t` node, keyed by `(vendor_id, option_id)`; exactly one of
`int_value`, `str_value`, `raw_value` must be set. The official NCCL library ignores all extensions, so a
vendor option only affects vendor libraries that recognize its `vendor_id`.

## nccl4py CFT (Compute Fabric Transport) support

CFT setup (NCCL 2.31+) is exposed on both the host and device sides:

- `NCCLConfig.host_cft_mode: NcclHostCftMode` (`repos/nccl/bindings/nccl4py/nccl/core/typing.py`) controls
  whether the communicator creates the CUDA fabric logical endpoints backing host-side CFT queries
  (`DEFAULT`, `ENABLE`, `DISABLE`, `FALLBACK`).
- `Communicator.team_cft(mode)` returns the CFT team in the requested `NcclCftTeamMode` layout (`FLAT`,
  `HIER_MULTIMEM`, `HIER_LSA`); `Communicator.team_cft_multimem` returns the CFT multimem team.
- `NCCLDevCommRequirements.cft_caps` takes an `NcclCftCap` bitmask and `cft_barrier_count` allocates CFT
  barriers for device-side use.
- Logical endpoint addresses are returned as `CftLeInfo` (`le_id`, `le_offset`)
  (`repos/nccl/bindings/nccl4py/nccl/core/resources.py`) from `RegisteredWindowHandle.get_cft_le_info(
  peer_cft, cft_team, offset=0)`, `get_peer_le_info(peer, offset=0)`, and
  `get_multimem_le_info(offset=0)`.

## nccl4py communicator capability introspection

`Communicator.properties` returns an `NCCLCommProperties` dataclass
(`repos/nccl/bindings/nccl4py/nccl/core/communicator.py`) whose fields report what the communicator can do:
`device_api_support`, `multimem_support`, `host_rma_support`, `gin_type`, `railed_gin_type`,
`available_gin_types`, `gin_connection_type`, `gin_min_stride`, `n_lsa_teams`, and `comm_hash` (fields
marked NCCL 2.31+ are `None` when nccl4py was built against an older NCCL). The most common checks are also
exposed directly as `Communicator` properties: `device_api_support`, `multimem_support`, `gin_type`,
`railed_gin_type`, `host_rma_support`, `n_lsa_teams`.

## CuTe DSL device bindings (`nccl.core.device.cute`)

`repos/nccl/bindings/nccl4py/nccl/core/device/cute/` provides experimental CuTeDSL bindings over the NCCL
device API; it requires the `nvidia-cutlass-dsl` package (installed via the `nccl4py[cu12]` / `[cu13]`
extras) and NCCL 2.30.7 or newer (`repos/nccl/bindings/nccl4py/examples/cute/README.md`). Main classes:

- `DevComm` (`comm.py`): wraps a `DevCommResource` as a `@cute.jit` argument. Exposes `rank`, `n_ranks`,
  `lsa_rank`, `lsa_size`, barrier handle properties (`lsa_barrier`, `rail_gin_barrier`,
  `hybrid_lsa_barrier`, `hybrid_rail_gin_barrier`, `world_gin_barrier`), `lsa_multimem`, team factories
  (`team_world`, `team_lsa`, `team_rail`, `team_cft(mode)`, `team_cft_multimem`), rank translation
  (`team_rank_to_world`, `team_rank_to_lsa`), resource-buffer addressing (`resource_buffer_local_pointer`,
  `resource_buffer_lsa_pointer`, `resource_buffer_peer_pointer`, `resource_buffer_multimem_pointer`,
  `resource_buffer_lsa_multimem_pointer`, plus `resource_buffer_cft_le_info`,
  `resource_buffer_peer_le_info`, `resource_buffer_multimem_le_info`), and the `gin(...)` factory.
- `Window` (`window.py`): wraps a `RegisteredWindowHandle`; address translation via `local_pointer`,
  `lsa_pointer`, `peer_pointer`, `multimem_pointer`, `lsa_multimem_pointer`, and CFT logical endpoints via
  `cft_le_info`, `peer_le_info`, `multimem_le_info` (returning `CftLeInfo` from `types.py`).
- `Gin` (`gin.py`): GIN operations `put`, `put_value`, `get` (NCCL 2.31.1+), `signal`, `read_signal`,
  `wait_signal`, `read_counter`, `wait_counter`, `reset_counter`, `reset_signal`,
  `signal_shadow_pointer`, and `flush`.
- `Coop` (`coop.py`): cooperative scopes from `cta()`, `warp()`, `thread()`, `lanes(lane_mask)`,
  `warp_span(warp0, n_warps, id)`.
- Barrier sessions (`barrier.py`): `lsa_session`, `gin_session`, `hybrid_session` factories with
  `arrive`/`wait`/`sync` methods.
- Compile-only arguments (`runtime.py`): `make_fake_dev_comm()`, `make_fake_window()`,
  `make_fake_multimem_handle()`, `make_fake_lsa_barrier_handle()`, `make_fake_gin_barrier_handle()` let an
  application compile a `@cute.jit` function with type-only arguments before creating NCCL resources, then
  invoke it with real ones.

Runnable examples covering this surface live in `repos/nccl/bindings/nccl4py/examples/cute/`
(`00_basic.py` through `07_compile_with_fake_args.py`); they run under MPI with one GPU per rank.

## Bindings/IR

`bindings/ir` contains wrappers around NCCL device headers for IR/device integration. Files include:

- `nccl_device_wrapper.h`
- `nccl_device_wrapper__impl.h`
- CMake/Make build files.

Use this area when integrating NCCL device functionality into compiler/IR flows rather than normal Python
host bindings.

## NCCL EP overview

NCCL EP is a high-performance NCCL API extension for Mixture-of-Experts Expert Parallelism communication.
It provides dispatch and combine primitives implemented on top of NCCL Device API using:

- LSA (Load-Store Accessible) operations for local/NVLink communication,
- GIN (GPU-Initiated Networking) operations for RDMA/network communication.

It targets MoE token dispatch and expert output combine patterns in modern sparse LLMs.

The in-tree version is v0.1.0 (`NCCL_EP_MAJOR`/`NCCL_EP_MINOR`/`NCCL_EP_PATCH` in
`repos/nccl/contrib/nccl_ep/include/nccl_ep.h`, `NCCL_EP_API_VERSION 1`). The README notes that `nccl_ep`
development has moved to the [NVIDIA/nccl-extensions](https://github.com/NVIDIA/nccl-extensions)
repository.

## NCCL EP algorithms

| Algorithm | Target | Communication pattern |
|---|---|---|
| Low-Latency (LL) | small batch / inference | direct point-to-point all-to-all with experts |
| High-Throughput (HT) | training / prefill / large batch | hierarchical NVLink intra-node aggregation + RDMA inter-node |

HT mode leverages Hopper features such as warp-specialized pipelines and TMA operations according to the
README.

## NCCL EP key features

- Staged execution in LL mode through send-only flag.
- Automatic tuning of buffer sizes, queue pairs, and channels.
- Restricted type-conversion/scaling support.
- C and Python APIs.
- Benchmark and test tools: `ep_test`, `ep_bench`.

## NCCL EP dependencies

From README:

| Component | Version/notes |
|---|---|
| CUDA | 13+ |
| NCCL | 2.29+ with Device API and GIN support |
| MPI | any OpenMPI/MPICH-style MPI for multi-process launch |
| GPU | Hopper H100 or Blackwell tested |

## NCCL EP environment setup

```bash
export COMPUTE_CAP=<discovered_compute_cap>   # e.g. 90 for H100
export CUDA_HOME=/path/to/cuda
export MPI_HOME=/path/to/openmpi
export NCCL_HOME=/path/to/nccl/build
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:${NCCL_HOME}/lib:$LD_LIBRARY_PATH"
export PATH="${CUDA_HOME}/bin:${NCCL_HOME}/bin:${MPI_HOME}/bin:$PATH"
```

Build NCCL:

```bash
cd /path/to/nccl-source
make -j src.build BUILDDIR=${NCCL_HOME}
```

Build NCCL EP:

```bash
make -C contrib/nccl_ep MPI=1 BUILDDIR=${NCCL_HOME} \
  NVCC_GENCODE="-gencode=arch=compute_${COMPUTE_CAP},code=sm_${COMPUTE_CAP}"
```

Outputs include:

- `${NCCL_HOME}/lib/libnccl_ep.a`
- `${NCCL_HOME}/lib/libnccl_ep.so`
- `${NCCL_HOME}/include/nccl_ep.h`
- `${NCCL_HOME}/test/nccl_ep/ep_test`
- `${NCCL_HOME}/test/nccl_ep/ep_bench`

For multi-node RDMA GIN:

```bash
export NCCL_GIN_TYPE=3  # GDAKI
```

Debug:

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

NCCL EP-specific variables (`repos/nccl/contrib/nccl_ep/nccl_ep_env.h`):

```bash
export NCCL_EP_TOKENS_PER_CHUNK=128  # HT dispatch/combine tokens-per-chunk override
export NCCL_EP_ENV_VERBOSE=1         # dump every resolved NCCL EP env var at group creation
export NCCL_EP_TIMEOUT_MS=...        # overrides ncclEpGroupConfig_t::timeout_ns
```

## NCCL EP C API quick shape

From `repos/nccl/contrib/nccl_ep/include/nccl_ep.h`:

```c
// Group management; custom allocator (if any) is set via config.alloc
// (ncclEpAllocConfig_t). Zero-init uses cudaMalloc/cudaFree.
ncclEpCreateGroup(&ep_group, comm, &config);
ncclEpGroupDestroy(ep_group);

// Handle management. ncclEpCreateHandle combines ncclEpInitHandle (buffer
// allocation, optionally caller-owned via handle_mem sized by
// ncclEpHandleMemSize) and ncclEpUpdateHandle (bind top-k routing).
ncclEpCreateHandle(&handle, ep_group, layout, &topk_idx, layout_info, &handle_cfg, stream);
ncclEpUpdateHandle(handle, &new_topk_idx, layout_info, stream);  // optional: refresh routing
ncclEpHandleDestroy(handle);

// Communication. inputs/outputs are named-struct pointers
// (ncclEpDispatchInputs_t / ncclEpDispatchOutputs_t /
// ncclEpCombineInputs_t / ncclEpCombineOutputs_t); each cross-boundary
// tensor lives in a named field as an ncclEpTensor_t*.
ncclEpDispatch(handle, &dispatch_in, &dispatch_out, layout_info, &dispatch_cfg, stream);
ncclEpCombine(handle, &combine_in, &combine_out, &combine_cfg, stream);
ncclEpComplete(handle, &complete_cfg, stream);  // LL mode only
```

LL mode also offers fault-tolerance entry points: `ncclEpMaskQuery` / `ncclEpMaskUpdate` /
`ncclEpMaskClean` (active-rank mask, enabled via `ncclEpGroupConfig_t::enable_mask`) and
`ncclEpGetAsyncError` / `ncclEpErrorClear`.

## NCCL EP Python API quick shape

The NCCL EP Python bindings are distributed separately from nccl4py (this checkout carries no
`contrib/nccl_ep/python` directory); both distributions contribute packages to the implicit `nccl`
namespace, so they install together without either owning `nccl/__init__.py`. Import the low-level binding:

```python
from nccl.ep import NCCLLibrary, NCCL_EP_ALGO_LOW_LATENCY

nccl_lib = NCCLLibrary()
# nccl_lib.ncclEpDispatch(...)
# nccl_lib.ncclEpCombine(...)
```

A higher-level Pythonic API exists in the same `nccl.ep` package and is exercised end to end by
`repos/nccl/contrib/nccl_ep/ep_test.py`:

```python
import nccl.ep as nccl_ep

config = nccl_ep.GroupConfig(
    algorithm=nccl_ep.Algorithm.LOW_LATENCY,   # or HIGH_THROUGHPUT
    num_experts=num_experts,
    max_dispatch_tokens_per_rank=num_tokens,
    max_recv_tokens_per_rank=num_tokens * n_ranks,
    max_token_bytes=hidden * 2,                # bfloat16
    alloc=nccl_ep.AllocConfig(alloc_fn=..., free_fn=...),  # optional custom allocator
)
ep_group = nccl_ep.Group.create(comm, config)  # comm is an nccl.core.Communicator

ep_handle = ep_group.create_handle(
    nccl_ep.Layout.EXPERT_MAJOR,               # LL; ep_test.py uses FLAT for HT
    topk_idx.tensor,                           # nccl_ep.Tensor(data_ptr, dtype=..., shape=...)
    layout_info=None,                          # optional nccl_ep.LayoutInfo
    config=nccl_ep.HandleConfig(),
    stream=stream,
)
ep_handle.dispatch(nccl_ep.DispatchInputs(tokens=...), nccl_ep.DispatchOutputs(tokens=...),
                   layout_info=..., config=nccl_ep.DispatchConfig(send_only=...), stream=stream)
ep_handle.combine(nccl_ep.CombineInputs(tokens=...), nccl_ep.CombineOutputs(tokens=...),
                  config=nccl_ep.CombineConfig(send_only=...), stream=stream)
ep_handle.complete(stream=stream)              # LL mode only
```

Custom allocators are plain functions decorated with `@nccl_ep.AllocFn` / `@nccl_ep.FreeFn`.

## NCCL EP core data structures

### `ncclEpTensor_t`

Plain tensor descriptor struct (`repos/nccl/contrib/nccl_ep/include/nccl_ep.h`), not an opaque handle:
caller-initialized with `NCCL_EP_TENSOR_INIT` and filled with `ndim`, `datatype`, `data` (device pointer or
`NULL` for window-backed tensors), `win_hdl`/`win_offset`, and a caller-owned `sizes` array. No explicit
destruction is needed; library-allocated descriptors come from `ncclEpTensorAlloc` and are released with
`ncclEpTensorDestroy`.

### `ncclEpGroup_t`

Opaque handle created from an NCCL communicator by `ncclEpCreateGroup`. `ncclEpGroupConfig_t` carries the
algorithm (`NCCL_EP_ALGO_LOW_LATENCY` / `NCCL_EP_ALGO_HIGH_THROUGHPUT` from
`repos/nccl/contrib/nccl_ep/include/ep_enums.h`), sizing fields (`num_experts`,
`max_dispatch_tokens_per_rank`, `max_recv_tokens_per_rank`, `max_token_bytes`, `rdma_buffer_size`), tuning
fields (`num_qp_per_rank`, `num_channels`, `max_num_sms`), the allocator (`alloc`), and LL fault-tolerance
fields (`enable_mask`, `timeout_ns`, `zero_copy`).

### `ncclEpHandle_t`

Opaque handle representing prepared routing/metadata for dispatch/combine. Created by
`ncclEpCreateHandle` with a receive buffer layout (`ncclEpLayout_t`: HT supports `NCCL_EP_LAYOUT_FLAT` and
`NCCL_EP_LAYOUT_EXPERT_MAJOR`; LL supports `NCCL_EP_LAYOUT_EXPERT_MAJOR` and `NCCL_EP_LAYOUT_RANK_MAJOR`)
and a top-k indices tensor; the routing is cached in the handle and refreshed with `ncclEpUpdateHandle`.

### `ncclEpLayoutInfo_t`

Named optional device-side metadata tensors passed to `ncclEpCreateHandle` / `ncclEpUpdateHandle` and
`ncclEpDispatch`: `expert_counters`, `src_rank_counters` (LL rank-major only), `expert_offsets` (HT
expert-major only), `recv_total_counter`, plus `recv_topk_idx_kind` selecting LOCAL/GLOBAL expert ids in
`recv_topk_idx`. LL mode requires `layout_info == NULL` at handle time.

## NCCL EP tensor fields and dimensions

The old tag constants (`NCCL_EP_TENSOR_TAG_*`) no longer exist; each cross-boundary tensor now lives in a
named field of one of the API's struct types (`ncclEpDispatchInputs_t`, `ncclEpDispatchOutputs_t`,
`ncclEpCombineInputs_t`, `ncclEpCombineOutputs_t`, `ncclEpLayoutInfo_t`).

Notation from README:

- `B`: batch size
- `H`: hidden dimension
- `S`: scales dimension
- `L`: local experts
- `K`: top-k
- `R`: number of ranks
- `N(r)`: number of tokens targeting rank `r`

### LL mode, same datatype

| Operation | Struct | Field | Dims |
|---|---|---|---|
| Dispatch | dispatch_inputs | tokens | `[B x H]` |
| Dispatch | dispatch_outputs | tokens | `[L x R x B x H]` |
| Dispatch | layout_info | expert_counters | `[L]` |
| Combine | combine_inputs | tokens | `[L x R x B x H]` |
| Combine | combine_outputs | tokens | `[B x H]` |
| Combine | combine_outputs | topk_weights | `[B x K]` |

### HT mode, same datatype

HT supports `NCCL_EP_LAYOUT_FLAT` (contiguous `[N(r) x H]` receive buffer) and
`NCCL_EP_LAYOUT_EXPERT_MAJOR` (grouped by local expert, optionally padded via
`ncclEpHandleConfig_t::dispatch_output_per_expert_alignment`). `topk_idx` is supplied once at handle
creation and cached; forward pass:

| Operation | Struct | Field | Dims |
|---|---|---|---|
| Create | layout_info | expert_counters | `[L]` |
| Dispatch | dispatch_inputs | tokens | `[B x H]` |
| Dispatch | dispatch_inputs | topk_weights | `[B x K]` |
| Dispatch | dispatch_outputs | tokens | `[N(r) x H]` |
| Dispatch | dispatch_outputs | topk_weights | `[N(r) x K]` |
| Dispatch | dispatch_outputs | topk_idx | `[N(r) x K]` |
| Combine | combine_inputs | tokens | `[N(r) x H]` |
| Combine | combine_outputs | tokens | `[B x H]` |

Backward pass adds `combine_inputs.topk_weights` `[N(r) x K]` and `combine_outputs.topk_weights`
`[B x K]`, as described in the README.

## NCCL EP test commands

Single-node:

```bash
mpirun -np 8 ./build/test/nccl_ep/ep_test -a ll -t 128 -d 7168
mpirun -np 8 ./build/test/nccl_ep/ep_test -a ht -t 4096 -d 7168
```

Multi-node example:

```bash
mpirun -np 16 \
  --map-by ppr:8:node \
  -x NCCL_GIN_TYPE=3 \
  -x LD_LIBRARY_PATH \
  ./build/test/nccl_ep/ep_test -a ll -t 128 -d 7168
```

`ep_test` accepts `-a {ll,ht}`, `-t <tokens>`, `-d <hidden>`, `-s {none,dispatch,combine,both}`
(send-only), `-c` (cached mode, HT only), and `-r` (random routing). `ep_test.py` replicates the same
scenarios through the `nccl.ep` Pythonic API.

## When to recommend NCCL EP

Recommend NCCL EP when the user is implementing MoE expert-parallel dispatch/combine and wants NCCL-native
Device API/GIN/LSA integration. Do not recommend it for ordinary data-parallel allreduce, where standard
NCCL collectives are simpler.
