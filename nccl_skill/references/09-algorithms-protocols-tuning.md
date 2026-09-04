# 09 - Algorithms, Protocols, Tuning, and Cost Model

## Primary source files

- `repos/nccl/src/tuning/tuning.cc`: tuning orchestration and entry points (`ncclTuningInit`,
  `ncclTuningCompute`).
- `repos/nccl/src/tuning/cost_model.cc`: unified cost model (`modelMap[]`), algorithm/protocol parsing,
  cost estimates. Per-model files live alongside it: `ring.cc`, `tree.cc`, `collnet.cc`, `nvls.cc`,
  `pat.cc`, `sym_model.cc` (symmetric device-API kernels), `ce_model.cc` (Copy Engine collectives).
- `repos/nccl/src/graph/search.cc`: topology graph search.
- `repos/nccl/src/graph/connect.cc`: channel/ring/tree connection construction.
- `repos/nccl/src/include/collectives.h`: algorithm/protocol/function enums and constants.
- `repos/nccl/src/include/tuning.h`: tuning id space, `ncclTuningInput_t`/`ncclTuningResult_t`, and
  the `NCCL_TUNING_MASK_*` masks.
- `repos/nccl/src/include/nccl_tuner.h`: tuner interface used internally.
- `repos/nccl/src/config/algorithm_registry.cc`: registry mapping selectable algorithm names
  (including symmetric kernels) to tuning ids; used by `NCCL_ALGO`-style selection and per-call
  `algSelection`.
- `repos/nccl/src/config/collconfig.cc`: `ncclCollConfig_t` resolution, per-call algorithm mask
  (`ncclCollConfigGetAlgMask`), and CTA policy resolution.
- `repos/nccl/src/ce_coll.cc`: Copy Engine (zero-SM) collectives, including the hierarchical
  multi-node variants.
- `repos/nccl/plugins/tuner/README.md`: external tuner plugin docs.
- `repos/nccl/src/plugin/tuner/tuner.cc`: tuner plugin loading/integration.

## NCCL operation selection dimensions

For each collective, NCCL chooses across several dimensions:

1. **Collective function**: AllReduce, Broadcast, Reduce, ReduceScatter, AllGather, AlltoAll, Gather,
   Scatter, P2P, RMA.
2. **Algorithm**: Ring, Tree, CollNet direct/chain, NVLS, NVLS tree, PAT, or specialized schedules.
3. **Protocol**: LL, LL128, SIMPLE.
4. **Channels**: number of channels/rings and per-peer channel allocation.
5. **Threads/CTAs**: kernel launch resource choices.
6. **Transport**: P2P, SHM, NET, CollNet, NVLS, plugin/IB/socket.
7. **Registration/window mode**: normal, user-registered, symmetric, device API.

Performance tuning is about identifying which dimension is wrong for the workload/topology.

## Algorithms

`collectives.cc` exposes string names for algorithms:

| Internal algorithm | String | Typical role |
|---|---|---|
| `NCCL_ALGO_TREE` | `TREE` | latency-friendly reductions/broadcast patterns |
| `NCCL_ALGO_RING` | `RING` | bandwidth-oriented large collectives |
| `NCCL_ALGO_COLLNET_DIRECT` | `COLLNET_DIRECT` | in-network/multi-node collective acceleration |
| `NCCL_ALGO_COLLNET_CHAIN` | `COLLNET_CHAIN` | CollNet chain variant |
| `NCCL_ALGO_NVLS` | `NVLS` | NVLink/NVSwitch local collectives |
| `NCCL_ALGO_NVLS_TREE` | `NVLS_TREE` | NVLS tree variant |
| `NCCL_ALGO_PAT` | `PAT` | ReduceScatter/AllGather only, `SIMPLE` protocol; see "PAT enablement" |

Do not assume every algorithm is valid for every collective/topology/message size. NCCL computes
availability and estimated time.

## Protocols

`collectives.cc` exposes string names:

| Protocol | String | Typical role |
|---|---|---|
| `NCCL_PROTO_LL` | `LL` | low-latency protocol for small messages |
| `NCCL_PROTO_LL128` | `LL128` | low-latency protocol using 128-bit-oriented paths |
| `NCCL_PROTO_SIMPLE` | `SIMPLE` | bandwidth-oriented protocol for larger messages |

Protocol thresholds and thread counts are tuned per topology and architecture. Environment overrides can
force bad choices; use them experimentally.

## Algorithm/protocol filter syntax

`tuning/cost_model.cc` (`parseList`) parses a mapping string of operation prefixes to element lists.

Examples from source comments:

```bash
NCCL_ALGO="ring,collnetdirect;allreduce:tree,collnetdirect;broadcast:ring"
NCCL_PROTO="LL,Simple;allreduce:^LL"
NCCL_PROTO="^LL128;allreduce:LL128"
```

Semantics:

- `;` separates mapping entries.
- `prefix:list` applies only to a collective prefix.
- A first entry without prefix applies to all prefixes.
- `,` separates elements.
- A leading `^` excludes the listed elements.
- All entries after the first must have a prefix if the first was prefix-less.
- Unknown prefixes/elements return `ncclInvalidUsage`.

`NCCL_SYM_KERNEL` uses the same syntax to filter symmetric (device-API) kernels by name
(`ncclSymKernelStr`); setting `NCCL_ALGO` or `NCCL_SYM_KERNEL` resets the other dimension's enable
set, so a forced `NCCL_ALGO` also disables symmetric kernels unless they are re-enabled.

Use filters for diagnosis or controlled deployments, not as first-line tuning.

## Cost model inputs

`tuning/cost_model.cc` initializes default tuner constants with tables for:

- base latencies by algorithm/protocol,
- hardware latencies for NVLINK, PCI, NET,
- max bandwidths for LL,
- per-channel LL128 ring/tree bandwidths,
- per-channel tree bandwidths,
- per-channel NVLS tree bandwidths,
- architecture-specific factors for Volta, Ampere, Hopper, Blackwell.

The model also considers:

- number of ranks,
- number of nodes,
- min/max compute capability,
- CPU architecture/vendor for network overhead defaults,
- graph bandwidths and channel counts,
- net device type,
- plugin-provided tuner constants or cost modifications.

## Unified cost model

`tuning/cost_model.cc` organizes all cost estimation behind a single `modelMap[]` of
`ncclTuningModelEntry_t` entries (init/sim/finalize per model), indexed by a unified tuning id
space defined in `include/tuning.h`:

- `[0, NCCL_TUNING_SYM_KERNEL_ID_OFFSET)`: legacy general models — one id per algorithm x protocol
  pair (ring, tree, CollNet, NVLS, NVLS tree, PAT), modeled in `tuning/ring.cc`, `tree.cc`,
  `collnet.cc`, `nvls.cc`, `pat.cc`.
- `[NCCL_TUNING_SYM_KERNEL_ID_OFFSET, NCCL_TUNING_CE_METHOD_ID_OFFSET)`: symmetric registered-memory
  kernels built on the device API (`ncclSymkKernelId_*`), modeled by `ncclTuningSymkModelSim` in
  `tuning/sym_model.cc`.
- `[NCCL_TUNING_CE_METHOD_ID_OFFSET, NCCL_TUNING_COUNT)`: Copy Engine collective methods (zero-SM
  path), modeled by `ncclTuningCeModelSim` in `tuning/ce_model.cc`.

Both the legacy kernel path and the device-API symmetric path query the same interface:
`ncclTuningCompute` (declared in `include/tuning.h`) takes an `ncclTuningInput_t` with a
`tuningMask` (`NCCL_TUNING_MASK_GENERAL_KERNELS`, `NCCL_TUNING_MASK_SYM_KERNELS`,
`NCCL_TUNING_MASK_CE`, `NCCL_TUNING_MASK_ALL`) and returns the best `ncclTuningResult_t`
(estimated time, channels, warps, kernel id). The symmetric scheduler
(`scheduler/symmetric_sched.cc`) and the CE multicast decision (`ce_coll.cc`,
`ncclCeAllGatherUseMulticast`) both select implementations through this same cost query.

## PAT enablement

Source behavior in `ncclPatEnable` (`tuning/pat.cc`):

- Requires SM60 or higher for CUDA atomics.
- `NCCL_PAT_ENABLE` defaults to auto (`2`); an explicit value overrides auto.
- PAT is only wired up for ReduceScatter and AllGather, and only with the `SIMPLE` protocol;
  other collectives/protocols are hard-disabled in `ncclTuningPatModelInit` and `modelMap[]`.
- Multi-rank-per-node (hierarchical) PAT is inter-node only: it requires `nNodes >= 2` and NVLS
  support, uses NVLS for the intra-node phase, and requires every node to have exactly as many
  local ranks as the NVLS head count. PAT does not support net device offload
  (`netDeviceType` must be host).
- The hierarchical variant is opt-in: `ncclTuningPatModelSim` reports `FLT_MAX/2` for multi-RPN
  communicators, so automatic selection never picks it. Enable it explicitly with `NCCL_ALGO=PAT`
  or per-call via `ncclCollConfig_t.algSelection` (see "Per-call algorithm selection").

So PAT is not a generic all-topologies algorithm; it is constrained.

## Zero-SM (Copy Engine) collectives

The `NCCL_CTA_POLICY_ZERO` flag (`nccl.h.in`, value `0x02`) requests collectives that consume no SM
resources: data movement runs on Copy Engines (CE) and host-side RMA instead of a GPU kernel. It can
be set at comm level via `ncclConfig_t.CTAPolicy`, per call via `ncclCollConfig_t.CTAPolicy`, or
process-wide via `NCCL_CTA_POLICY=ZERO` (parsed in `init.cc`; ZERO takes precedence over
`NCCL_CTA_POLICY_EFFICIENCY` when both are set).

Routing happens in `enqueue/enqueue.cc`: when the resolved policy includes ZERO and a CE path is
available (`ncclCeAvailable` / `ncclHierCeAvailable`), the collective is appended as a CE task
(`ceCollTaskAppend`) instead of a kernel.

From `ce_coll.cc`:

- Single-node CE collectives cover AllGather, AlltoAll, Scatter, and Gather (CUDA driver 12.5+).
- Hierarchical CE collectives (`ncclHierCeAvailable`) cover AllGather and AlltoAll on multi-node
  communicators where the LSA team spans the node but not the whole comm: inter-node transfers go
  through the host RMA proxy (`rma/rma_proxy.cc`, internal RMA contexts provisioned by
  `ncclRmaWantInternalCtx` under exactly the zero-CTA + multi-node conditions), and intra-node
  distribution uses Copy Engine writes via LSA pointers. Both buffers must be in registered
  symmetric windows; symmetric support must be enabled.
- CE AllGather unicast vs multicast is itself chosen through the unified cost model
  (`tuning/ce_model.cc`, per-arch latency floors and NVLink/D2H bandwidth caps for SM90/SM100).

## Symmetric TMA kernels on Blackwell

The symmetric (device-API) kernel set includes TMA-based kernels
(`ncclSymkKernelId_AllGather_TmaST`, `..._TmaSTMC`, `ncclSymkKernelId_ReduceScatter_TmaLD`,
`ncclSymkKernelId_AllReduce_RSxTmaLD_AGxTmaST`). In `sym_kernels.cc`:

- `NCCL_SYM_TMA_ENABLE` defaults to `1`, so TMA kernels are enabled by default on Blackwell
  (`ncclSymkTmaAvailable` requires `minCompCap >= 100`).
- TMA kernels are masked out when buffers are not 16-byte symmetric-aligned.

TMA kernels are first-class citizens of the cost model: they have entries in `modelMap[]` and
per-architecture latency/bandwidth parameters in `tuning/sym_model.cc`, so symmetric
registered-memory collectives pick between TMA and non-TMA variants by estimated time.

## Per-call algorithm selection

`ncclCollConfig_t` (initialize with `NCCL_COLLCONFIG_INITIALIZER`) supports per-collective
algorithm selection through the `nccl*Config` API variants (`ncclAllReduceConfig`, etc.):

- `algSelection`: a selection string parsed by `ncclAlgParse` (`config/algorithm_parser.cc`) into a
  mask over the registry in `config/algorithm_registry.cc`; NULL/empty means automatic. Names are
  matched case-insensitively as prefixes of registry entries, e.g. `PAT` selects the `PAT_SIMPLE`
  row and `RING` selects all `RING_*` rows; symmetric kernel names like `SYMK_TmaST` are also
  selectable.
- `forceAlgSelection` (default 1): an unsatisfiable selection (bad syntax, or nothing valid for
  this collective) is a hard `ncclInvalidArgument` error; set to 0 to log and fall back to
  automatic selection.

The mask is validated up front in `enqueue/enqueue.cc` (`ncclCollConfigGetAlgMask`) and applied by
both the legacy and symmetric schedulers. Environment filters (`NCCL_ALGO`/`NCCL_PROTO`/
`NCCL_SYM_KERNEL`) win over per-call `algSelection` for any function they forced
(`scheduler/symmetric_sched.cc`).

## Thread-count tuning

Variables:

```bash
NCCL_NTHREADS
NCCL_LL128_NTHREADS
```

`getNthreads` validates:

- multiple of warp size,
- maximum bound,
- minimum bound.

Invalid values are logged and clamped/defaulted. If a user sets thread counts, advise validating logs
because NCCL may not use the exact requested value.

## Channel/ring tuning

Important variables:

| Variable | Meaning |
|---|---|
| `NCCL_MIN_NCHANNELS` | lower bound for channels |
| `NCCL_MAX_NCHANNELS` | upper bound for channels |
| `NCCL_MIN_NRINGS` | legacy/ring lower bound |
| `NCCL_MAX_NRINGS` | legacy/ring upper bound |
| `NCCL_NCHANNELS_PER_NET_PEER` | per-network-peer channel count |
| `NCCL_NVLS_NCHANNELS` | NVLS channel count |
| `NCCL_P2P_SCHEDULE_GROUP_SIZE` | P2P schedule group size |

Increasing channels can improve bandwidth but also increases resource use and can worsen small-message
latency or contend with compute kernels.

## Tuner plugin interface

External tuner plugins modify NCCL's algorithm/protocol selection by changing cost tables and channel
counts without recompiling NCCL.

From `plugins/tuner/README.md`, the interface includes:

```c
ncclResult_t (*init)(size_t nRanks, size_t nNodes,
                     ncclDebugLogger_t logFunction, void **context);

ncclResult_t (*getCollInfo)(void* context, ncclFunc_t collType, size_t nBytes,
                            int numPipeOps, float** collCostTable,
                            int numAlgo, int numProto,
                            int regBuff, int* nChannels);

ncclResult_t (*destroy)(void* context);
```

Tuner plugins can:

- set cost to `0.0` to prefer a combination,
- set cost to `NCCL_ALGO_PROTO_IGNORE` to disable a combination,
- adjust `nChannels`,
- implement topology/workload-aware strategies.

Loading:

```bash
export LD_LIBRARY_PATH=/path/to/plugin:$LD_LIBRARY_PATH
export NCCL_TUNER_PLUGIN=example
# or
export NCCL_TUNER_PLUGIN=libnccl-tuner-example.so
# or
export NCCL_TUNER_PLUGIN=/absolute/path/libnccl-tuner-example.so
```

Debugging tuner behavior:

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=TUNING
```

## Tuner plugin best practices

1. Keep `getCollInfo` lightweight; it can run on critical paths.
2. Cache expensive topology/workload decisions in plugin context.
3. Return `ncclSuccess` for ignored/no-op cases.
4. Avoid returning errors from `getCollInfo`; initialization is a safer failure point.
5. Test across message sizes and rank/node counts.
6. Compare against no plugin with `nccl-tests` and representative application workloads.
7. Document every forced ignore/preference because it can become wrong on new hardware.

## Manual tuning workflow

1. Baseline with no overrides.
2. Capture `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=GRAPH,TUNING,NET`.
3. Identify chosen algorithm/protocol/channels in logs.
4. Run `nccl-tests` for target collective/message sizes.
5. Try one override at a time:
   - algorithm filter,
   - protocol filter,
   - channel bounds,
   - transport-specific variable.
6. Measure median and tail, not just best run.
7. Remove overrides that help microbenchmarks but hurt application overlap.
8. Prefer tuner plugin for systematic deployment-specific policy.

## When not to tune manually

Avoid manual overrides when:

- the root cause is mismatched rank/device mapping,
- NCCL is falling back to sockets due to network setup,
- CUDA stream synchronization serializes communication,
- message sizes vary widely and one forced algorithm hurts other phases,
- the cluster has mixed GPU/NIC topology,
- a framework already has NCCL/tensor-parallel scheduling assumptions.

Fix the topology/usage issue first.
