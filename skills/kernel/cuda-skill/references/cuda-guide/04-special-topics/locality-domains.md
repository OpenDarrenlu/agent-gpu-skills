---
url: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/locality-domains.html
---

# 4.7. Locality Domains

A **locality domain** is a portion of a GPU that contains streaming multiprocessors (SMs) and device memory. An application can allocate device memory in a locality domain and create a green context with SM resources in the same locality domain. Co-locating computation near the memory it accesses can improve performance on devices that have more than one locality domain. The compute localization API is described in [Localizing Compute Resources](#locality-domains-compute) and memory localization in [Allocating Localized Device Memory](#locality-domains-memory).

CUDA identifies the locality domains of a device by zero-based **locality domain IDs**. A locality domain ID is meaningful only together with its device ID. The same ID is used when localizing SM resources and memory, allowing an application to request matching placement for both.

## 4.7.1. Discovering Locality Domains

Use `cudaDeviceGetAttribute` with the following `cudaDeviceAttr` values:

  * `cudaDevAttrLocalityDomainCount` returns the number of locality domains on the device.

  * `cudaDevAttrLocalityDomainMultiprocessorCount` returns the number of SMs in each locality domain.


Applications should query these attributes rather than assume a particular topology. The number of locality domains available on a device is affected by the system configuration. The same device type may be presented with a different number of locality domains under different scenarios ([Section 4.7.5.3](#locality-domains-support)). All devices, including those with a single locality domain available, can use the locality domain APIs. On devices with a single locality domain, these APIs are the same as targeting the full device. Use the device attributes to programmatically query the locality domain properties of the device:


    int localityDomainCount = 0;
    int smsPerLocalityDomain = 0;

    cudaDeviceGetAttribute(&localityDomainCount, cudaDevAttrLocalityDomainCount, device);
    cudaDeviceGetAttribute(&smsPerLocalityDomain, cudaDevAttrLocalityDomainMultiprocessorCount, device);


## 4.7.2. Localizing Compute Resources

To localize SM resources to a specific locality domain, use the CUDA green context API. SMs placement is expressed through the green context resource APIs described in [Section 4.6](green-contexts.html#green-contexts). The main steps to construct a localized SM partition are:

  1. Obtain the device’s `cudaDevResourceTypeSm` resource with `cudaDeviceGetDevResource`.

  2. Describe a localized SM group with `cudaDevSmResourceGroupParams` ([details](#locality-domains-localized-sm-group)).

  3. Pass the group parameters to `cudaDevSmResourceSplit`.

  4. Generate a resource descriptor with `cudaDevResourceGenerateDesc`, create a green context with `cudaGreenCtxCreate`, and finally create a stream via `cudaExecutionCtxStreamCreate`.

  5. Launch kernels on the stream.


Typical green context creation is described in [Section 4.6.2](green-contexts.html#green-contexts-ease-of-use). The additions necessary to localize SM resources are illustrated below:

To describe a localized SM group, set `cudaDevSmResourceGroupLocalityDomainId` in `flags` and set the ID in `localityDomainId`. Valid IDs are between `0` and `cudaDevAttrLocalityDomainCount - 1`.


    cudaDevResource smResource;
    cudaDeviceGetDevResource(device, &smResource, cudaDevResourceTypeSm);

    {
        cudaDevSmResourceGroupParams params = {};
        // all fields in params must be zero-initialized
        params.flags = cudaDevSmResourceGroupLocalityDomainId;
        params.localityDomainId = localityDomainId;

        cudaDevResource result;
        cudaDevSmResourceSplit(&result, 1, &smResource, nullptr, 0, &params);
    }


Multiple constraints may be specified in `params` in combination with requesting a locality domain. For example:

The `smCount` constraint may be `0` (discovery mode) or positive. In discovery mode the API returns the number of SMs in the locality domain in `params.smCount`. On a full, un-partitioned, device resource, this value is the same as `cudaDevAttrLocalityDomainMultiprocessorCount`. When `smCount` is positive, the API will allocate up to the `cudaDevAttrLocalityDomainMultiprocessorCount` value. Larger `smCount` requests will fail.

When `coscheduledSmCount` is `0` the API does not restrict SMs by coscheduled capability and the maximum number of localized SMs are available. The cluster capability of the partition can be dynamically queried later with `cudaOccupancyMaxPotentialClusterSize`. When `coscheduledSmCount` is nonzero, the the API will select only localized SMs satifsying the requested `coscheduledSmCount`.

By default, all SMs in the `result` must satisfy the locality domain constraint and any additional constraints specified. To relax this behavior, add `cudaDevSmResourceGroupBackfill` to `params.flags`. With backfill, the API fills the group from the requested locality domain first, then from SMs not attributed to a locality domain, and finally from other locality domains. The call fails with `cudaErrorInvalidResourceConfiguration` if no SM can be found in the requested locality domain.

Any unused SMs not returned in the `result` are returned in the API’s `remainder` parameter.

The `result`’s `cudaDevResource.sm.flags` and `cudaDevResource.sm.localityDomainId` report the resource placement. When combining SM resources in one descriptor, the resources must have matching constraints, including `flags` and `localityDomainId` values.

The following code queries the number of locality domains, and constructs one stream per locality domain.


    std::vector<cudaExecutionContext_t> contexts(localityDomainCount);
    std::vector<cudaStream_t> streams(localityDomainCount);

    {
        std::vector<cudaDevSmResourceGroupParams> params(localityDomainCount);
        std::vector<cudaDevResource> result(localityDomainCount);

        for (int i = 0; i < localityDomainCount; i++) {
            params[i] = {};
            params[i].flags = cudaDevSmResourceGroupLocalityDomainId;
            params[i].localityDomainId = i;
        }

        // A single split produces non-overlapping resources for all domains.
        cudaDevSmResourceSplit(result.data(), localityDomainCount, &smResource, nullptr, 0, params.data());

        for (int i = 0; i < localityDomainCount; i++) {
            cudaDevResourceDesc_t desc;
            cudaDevResourceGenerateDesc(&desc, &result[i], 1);
            cudaGreenCtxCreate(&contexts[i], desc, device, 0);
            cudaExecutionCtxStreamCreate(&streams[i], contexts[i], cudaStreamDefault, 0);
        }
    }


Each stream in `streams` executes on SMs from the locality domain with the corresponding index. Destroy the streams with `cudaStreamDestroy` and then destroy their contexts with `cudaExecutionCtxDestroy` when they are no longer needed.

## 4.7.3. Allocating Localized Device Memory

CUDA represents a localized memory location with a `cudaMemLocation` whose `type` is `cudaMemLocationTypeDeviceLocalityDomain`. For this location type, initialize both members of `cudaMemLocation.localized`:

  * `deviceId` identifies the GPU.

  * `localityDomainId` identifies a locality domain on that GPU.


The runtime API uses this location with stream-ordered memory pools ([Section 4.3](stream-ordered-memory-allocation.html#stream-ordered-memory-allocator)). The driver API additionally uses the corresponding `CUmemLocation` with virtual memory management allocations ([Section 4.17](virtual-memory-management.html#virtual-memory-management)).

Allocations created through other mechanisms, for example `cudaMalloc` or unified memory, can not be localized. [MPS (Multi-Process Service)](https://docs.nvidia.com/deploy/mps/index.html) in MLOPart mode can be used to transparently localize applications that rely on `cudaMalloc`.

### 4.7.3.1. Creating a Localized Memory Pool

To create a localized memory pool, set the location in `cudaMemPoolProps` and call `cudaMemPoolCreate`:


    cudaMemPool_t pool {};
    cudaMemPoolProps props {};

    props.allocType = cudaMemAllocationTypePinned;
    props.location.type = cudaMemLocationTypeDeviceLocalityDomain;
    props.location.localized.deviceId = device;
    props.location.localized.localityDomainId = localityDomainId;

    cudaMemPoolCreate(&pool, &props);


Allocate from this pool with `cudaMallocFromPoolAsync` and release the allocation with `cudaFreeAsync`. The location can also be passed to `cudaMemGetDefaultMemPool`, `cudaMemGetMemPool`, and `cudaMemSetMemPool` when selecting the default or current pinned-memory pool for that locality domain. For the general memory-pool programming model, see [Memory Pools](stream-ordered-memory-allocation.html#stream-ordered-memory-pools).

### 4.7.3.2. Creating a Localized VMM Allocation

The driver API can place physical memory created by `cuMemCreate` in a locality domain. Set `CUmemAllocationProp::location.type` to `CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN` and initialize `location.localized.deviceId` and `location.localized.localityDomainId`. The resulting physical allocation is reserved, mapped, and granted access in the same way as other VMM allocations. Access control set via `cuMemSetAccess` operates at full device granularity, not individual locality domains. See the [localized VMM allocation section](virtual-memory-management.html#vmm-locality-domains) for the complete allocation sequence and granularity requirements.

Managed memory can not be localized: Requesting `cudaMemAllocationTypeManaged` with `cudaMemLocationTypeDeviceLocalityDomain` will return `CUDA_ERROR_NOT_SUPPORTED`. Similarly, `cudaMemPrefetchAsync` and `cudaMemAdvise` do not accept a device locality domain as a location and will return `CUDA_ERROR_NOT_SUPPORTED`.

See [Locality Domains and Unified Memory](unified-memory.html#um-locality-domains) for the relationship between locality domains and Unified Memory.

## 4.7.4. Querying Memory Placement

Use `cudaMemPoolGetAttribute` with `cudaMemPoolAttrLocalityDomainId` to query a pool. It returns the locality domain ID for a localized pool and `-1` for a pool that is not localized.

Use `cudaPointerGetAttributes` to query an allocation through its pointer. The returned `cudaPointerAttributes.localityDomainOrdinal` field contains the allocation’s locality domain ordinal, or `-1` when the allocation is not localized:


    cudaPointerAttributes attributes {};
    cudaPointerGetAttributes(&attributes, ptr);
    int localityDomainId = attributes.localityDomainOrdinal;


## 4.7.5. Programming Guidance

To localize a workload, launch kernels using compute resources from the same locality domain as their memory allocations.

Localizing only compute or only memory resources typically does not improve performance.

If a kernel must use memory from several locality domains, measure whether compute localization is beneficial.

### 4.7.5.1. Localized Kernels With Fork/Join Pattern

The **fork/join pattern** uses events and stream waits to express a parallel region. The application creates one stream from each localized green context by using `cudaExecutionCtxStreamCreate`. A separate main stream coordinates those localized streams. Create one fork event for the main stream and one join event per localized stream with `cudaEventCreateWithFlags`; timing should be disabled when the events are used only for ordering.

To fork, record one event on the parent stream and make every localized stream wait for that event. Work previously submitted to the parent is therefore ordered before the localized work. Submit independent work to the localized streams, using memory from their corresponding locality domains.

To join, record one event on each localized stream after its work and make the parent stream wait for all of those events. Work subsequently submitted to the parent is then ordered after the localized work.


    produceInputs<<<grid, block, 0, parentStream>>>(...);
    cudaEventRecord(forkEvent, parentStream);

    // localizedStreams[i] belongs to the green context for locality domain i.
    for (int i = 0; i < localityDomainCount; i++) {
        cudaStreamWaitEvent(localizedStreams[i], forkEvent);
        processPartition<<<grid, block, 0, localizedStreams[i]>>>(...);
        cudaEventRecord(joinEvents[i], localizedStreams[i]);
    }

    for (int i = 0; i < localityDomainCount; i++) {
        cudaStreamWaitEvent(parentStream, joinEvents[i]);
    }
    consumeResults<<<grid, block, 0, parentStream>>>(...);


### 4.7.5.2. CUDA Graphs

CUDA Graphs ([Section 4.2](cuda-graphs.html#cuda-graphs)) support localization through stream capture and manual graph construction. For some applications, CUDA graphs may offer improved performance over stream-based programming.

#### 4.7.5.2.1. Stream Capture

A fork/join pattern can be captured to a CUDA graph. Begin capture on the parent stream, record the fork event, and wait on the fork event in each localized stream. Waiting on the fork event enrolls the localized streams in the same capture graph as the parent stream. Operations submitted to the localized streams become parallel graph branches. After launching all work on the localized streams, record the join event and wait on all the join events in the parent stream. These waits reconnect all branches to the parent stream. Every participating stream must be joined back to the origin stream before `cudaStreamEndCapture` is called.


    cudaStreamBeginCapture(parentStream, cudaStreamCaptureModeGlobal);

    // Record forkEvent, submit work to each localized stream, record the
    // joinEvents, and make parentStream wait as in the fork/join example above.

    cudaGraph_t graph;
    cudaStreamEndCapture(parentStream, &graph);


The locality restriction of each child stream is retained by the captured kernel nodes. Graph replay uses the original localized SMs even when the executable graph is launched on a different stream. The graph launch stream supplies ordering but does not replace the execution resources recorded in the nodes.

#### 4.7.5.2.2. Graph Construction

The same graph topology can be constructed explicitly. Add one kernel node for each locality domain, using the corresponding localized execution context in `cudaKernelNodeParamsV2.ctx`.


    cudaGraph_t graph = nullptr;
    cudaGraphCreate(&graph, 0);

    std::vector<cudaGraphNode_t> localizedNodes(localityDomainCount);
    for (int i = 0; i < localityDomainCount; i++) {
        cudaGraphNodeParams params = { CU_GRAPH_NODE_TYPE_KERNEL };
        params.kernel.kern = kernel;
        params.kernel.gridDim = grid;
        params.kernel.blockDim = block;
        params.kernel.kernelParams = kernelArgs[i];
        params.kernel.ctx = localizedContexts[i];

        cudaGraphAddNode(&localizedNodes[i], graph, nullptr, nullptr, 0, &params);
    }


### 4.7.5.3. Compatibility Notes

vGPU and MIG devices do not support localization and will report only 1 locality domain.

CUDA on Windows does not support localization and will report only 1 locality domain.

CUDA Checkpoint does not guarantee that localized green contexts retain their localization properties when restored on a different device. The application may exhibit reduced performance after a restore.

[MPS (Multi-Process Service)](https://docs.nvidia.com/deploy/mps/index.html) in MLOPart mode presents each MLOPart device as a CUDA device with a single locality domain. MPS static partitioning treats the full device as a single locality domain.