# 01 - NCCL Overview, Build, Install, and Repository Orientation

## Scope

This reference summarizes what NCCL is, how the source tree is organized, how to build/install it,
and how the example programs are intended to be used. It is based primarily on:

- `repos/nccl/README.md`
- `repos/nccl/makefiles/version.mk`
- `repos/nccl/docs/examples/README.md`
- `repos/nccl/docs/examples/Makefile`
- `repos/nccl/CMakeLists.txt`
- `repos/nccl/Makefile`
- `repos/nccl/contrib/README.md`
- `repos/nccl/docs/dev_guide/nccl_coding_style.md`

NCCL version in this source checkout: **2.31.2** (`NCCL_MAJOR=2`, `NCCL_MINOR=31`, `NCCL_PATCH=2`).

## What NCCL provides

NCCL, pronounced "Nickel", is a standalone library of communication routines for GPUs. It implements
collectives and P2P communication patterns used by distributed machine learning, HPC, and GPU runtime
systems. The public README describes the core operations as:

- AllReduce
- AllGather
- Reduce
- Broadcast
- ReduceScatter
- send/receive based communication patterns

The current public header additionally exposes AlltoAll, Gather, Scatter, one-sided signal/RMA-like
operations, parameter APIs, memory/window registration, and Device API entry points through headers
included by `nccl_device.h`.

Since NCCL 2.31, the Device API also includes Compute Fabric Transport (CFT) helpers for kernels
that communicate through CUDA fabric logical endpoints (put, get, reduction, and barrier operations);
see `repos/nccl/docs/userguide/source/usage/cft.rst`.

## Hardware and topology targets

NCCL is optimized for high-bandwidth GPU paths:

| Path | Typical use | NCCL implementation areas |
|---|---|---|
| PCIe peer-to-peer | GPU-to-GPU within a node when CUDA P2P is available | `src/transport/p2p.cc`, topology paths |
| NVLink / NVSwitch | Fast local GPU fabric | graph search, NVLS transport, P2P transport |
| Shared memory | Intra-node fallback/staging paths | `src/transport/shm.cc` |
| InfiniBand Verbs / RDMA | Multi-node GPU clusters | `src/transport/net_ib/*`, net plugin API |
| TCP/IP sockets | Portable network fallback and bootstrap | `src/transport/net_socket.cc`, `src/misc/socket.cc` |
| GIN / device networking | GPU-initiated network paths for Device API and EP | `src/gin/*`, `src/transport/net_ib/gdaki/*` |

NCCL uses topology discovery and graph search to choose rings, trees, CollNet, NVLS, and other patterns.
Application code should usually not hard-code these algorithms unless debugging or tuning.

## Source tree map

```text
repos/nccl/
├── README.md                         # project overview, build/install/test commands
├── Makefile                          # top-level make targets
├── CMakeLists.txt                    # CMake build entry
├── .clang-format                     # clang-format rules applied to src/ (see docs/dev_guide)
├── makefiles/                        # make fragments, version, common compiler settings
├── docs/examples/                    # pedagogical C++/CUDA examples
├── docs/dev_guide/                   # developer guide, incl. nccl_coding_style.md
├── src/
│   ├── nccl.h.in                     # public C API template
│   ├── collectives.cc                # public collective/P2P API wrappers -> ncclInfo
│   ├── group.cc                      # group semantics and async job execution
│   ├── enqueue/                      # task scheduling, launch planning, kernel upload
│   ├── init.cc                       # communicator initialization and lifecycle APIs
│   ├── dev_runtime.cc                # Device API host entry points and runtime support
│   ├── cft_dev_runtime.cc            # Compute Fabric Transport (CFT) host-side runtime
│   ├── allocator.cc                  # ncclMemAlloc/ncclMemFree
│   ├── mem_manager.cc                # tracked/suspendable communicator memory
│   ├── proxy.cc                      # CPU proxy progress engine
│   ├── bootstrap.cc                  # bootstrap connections and rank exchange
│   ├── device/                       # GPU kernel implementations and protocol primitives
│   ├── nccl_device/                  # device-side Device API implementations (LSA/GIN/CFT)
│   ├── gin/                          # GPU-initiated networking host runtime
│   ├── graph/                        # topology discovery, graph search, tuning, channels
│   ├── transport/                    # P2P, SHM, NET, CollNet, NVLS, IB, socket transports
│   ├── plugin/                       # dynamic plugin loading wrappers
│   ├── scheduler/                    # symmetric and allgatherv scheduling extensions
│   ├── register/                     # user buffer and collective registration helpers
│   ├── rma/                          # one-sided RMA/signal implementation
│   ├── ras/                          # reliability/availability/serviceability subsystem
│   └── include/                      # internal and plugin headers
├── plugins/                          # example external plugins (net, tuner, profiler, env)
├── bindings/nccl4py/                 # Python/Cython binding package
└── contrib/                          # community/partner extensions on public APIs (contrib/README.md):
                                      # custom_algos, pace, nccl4rust, nccl_checkpoint, nccl_ubx, niin;
                                      # nccl_ep and nccl_m2n are developed in NVIDIA/nccl-extensions
```

## Build from source

From the repository root:

```bash
make -j src.build
```

If CUDA is not installed at `/usr/local/cuda`:

```bash
make -j src.build CUDA_HOME=/path/to/cuda
```

By default, NCCL builds into `build/`. Override with `BUILDDIR`:

```bash
make -j src.build BUILDDIR=/tmp/nccl-build
```

To reduce build time and binary size, restrict GPU architectures with `NVCC_GENCODE`:

```bash
make -j src.build NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
```

One feature-specific toolchain requirement: CFT device code requires CUDA Toolkit 13.3 and SM_100
(Blackwell) architectures, as documented in
`repos/nccl/docs/examples/06_device_api/04_cft_barrier/README.md`.

The top-level README points to official tested builds, but the source build is the right path when
working on internals, Device API, plugin interfaces, or contrib extensions.

## Package targets

NCCL can build OS packages from source:

### Debian/Ubuntu
```bash
sudo apt install build-essential devscripts debhelper fakeroot
make pkg.debian.build
ls build/pkg/deb/
```

### RedHat/CentOS
```bash
sudo yum install rpm-build rpmdevtools
make pkg.redhat.build
ls build/pkg/rpm/
```

### OS-agnostic txz
```bash
make pkg.txz.build
ls build/pkg/txz/
```

## Tests and benchmarks

The source README explicitly says NCCL tests are maintained separately in `nccl-tests`. Typical flow:

```bash
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make
./build/all_reduce_perf -b 8 -e 256M -f 2 -g <ngpus>
```

Use `nccl-tests` for performance and correctness testing. The in-repository examples are educational
templates and are not intended to maximize performance for individual communication patterns.

## Examples build and run

The examples live in `repos/nccl/docs/examples/` and are grouped by feature:

```text
01_communicators/          # init/destroy/query patterns: single process, pthread, MPI
02_point_to_point/         # ncclSend/ncclRecv ring pattern
03_collectives/            # basic collective examples, e.g. AllReduce
04_user_buffer_registration/
05_symmetric_memory/
06_device_api/             # LSA allreduce, GIN/hybrid alltoall, and CFT barrier examples
07_kernel_fusion/          # fused compute+communication kernels (RMSNorm over LSA/multimem/GIN)
08_ras/                    # RAS fault-detection example
common/                    # shared utilities for advanced examples
```

Python variants are provided where `nccl4py` exposes the required host-side APIs, with per-example
`requirements-cu12.txt` / `requirements-cu13.txt`; pthread-specific and CUDA device-side examples
remain C/CUDA-only.

Build examples while building NCCL:

```bash
make -j examples
make -j examples MPI=1
```

Build examples against an existing NCCL installation:

```bash
cd docs/examples
make NCCL_HOME=/path/to/nccl MPI=1
```

Build-stage variables:

| Variable | Meaning |
|---|---|
| `NCCL_HOME` | local base directory of NCCL installation |
| `MPI` | `0` or `1`, controls MPI-enabled examples |
| `MPI_HOME` | MPI installation prefix when non-standard |
| `CUDA_HOME` | CUDA installation prefix |

Run-stage variables:

| Variable | Meaning |
|---|---|
| `NTHREADS` | number of threads for threaded examples; defaults to visible GPU count |
| `CUDA_VISIBLE_DEVICES` | list of GPUs visible to the app |
| `NCCL_*` | all NCCL environment variables apply |

For runtime issues, start with:

```bash
NCCL_DEBUG=INFO ./example_or_app
```

## CMake vs Make

The source tree includes both `Makefile` and `CMakeLists.txt`. Most NCCL developer docs and examples
use Make targets, while package and integration environments may use CMake. When answering build questions,
prefer the documented Make command unless the user is explicitly integrating NCCL into a CMake-based build.

## Practical build/debug checklist

1. Confirm CUDA compiler and runtime versions: `nvcc --version`, `nvidia-smi`.
2. Confirm target GPU architecture and set `NVCC_GENCODE` when build time matters.
3. Build `src.build` before examples or contrib libraries that depend on headers/libs.
4. Set `LD_LIBRARY_PATH` to include the built NCCL library when running non-installed builds.
5. Use `NCCL_DEBUG=INFO` before changing transport/tuning variables.
6. Validate with `nccl-tests` before assuming an application-level distributed training bug is in NCCL.
