---
url: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/compute-fabric-transport.html
---

# 4.18. Compute Fabric Transport

Compute fabric transport provides large-scale GPU-to-GPU communication over the NVLink fabric for communication libraries such as NCCL and NVSHMEM, through the [NCCL Device API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html) and [NVSHMEM](https://docs.nvidia.com/nvshmem/api/index.html). Most developers should prefer those libraries when they match the needed communication pattern. Compute fabric transport is primarily intended for developers writing such communication libraries, or for applications that need fine-grained control when the pattern is not well captured by those libraries. Authors of custom communication kernels can also combine fabric operations with the NCCL and NVSHMEM device APIs when they need even more direct access than those device APIs alone provide.

The [Unicast](virtual-memory-management.html#unicast-memory-sharing) and [Multicast](virtual-memory-management.html#multicast-obj-mem) memory sharing APIs let an application access GPU memory across the NVLink fabric by mapping remote physical memory into a local virtual address space. Once a peer allocation is mapped and given access rights, kernels access it like local memory using loads and stores (plus `multimem` instructions for multicast mappings). That model is both _address-centric_ and _memory-centric_ : the unit of sharing is a virtual-to-physical mapping, and every byte a kernel touches on a peer must be reachable through an address that the issuing process has reserved and mapped.

On a large NVLink fabric, this model has two limitations, and both affect how an application is written.

Virtual addresses provide no error propagation channel. When a kernel issues a load or store to a peer address, the hardware delivers either the data or a memory fault — there is no intermediate result that the issuing thread can inspect to detect and programmatically handle a failure. On large fabric systems, transient errors such as packet loss or link resets are a possibility, and at sufficient scale they become a consideration for resilient application design. A fabric error encountered during a remote access typically faults the access or terminates the process, and a GPU kernel has no way to respond to such a failure: no path to retry or reroute.

Virtual addresses also name only memory. A mapping is defined in terms of physical memory, so a resource on the fabric that is not memory cannot be reached through a mapping at all.

Compute fabric transport provides a complementary, _resource-centric_ model for GPUs connected by the NVLink fabric. Instead of importing a peer’s memory into a local address space, an application creates a **logical endpoint** : a named transport object that represents a resource reachable over the fabric. A logical endpoint is identified by a 32-bit integer, a **logical endpoint id** , and a target within it is named by that id together with a 64-bit offset rather than by a virtual address. Because the id names the resource itself, the model is not tied to memory, though memory is currently the only kind of resource that logical endpoints expose.

An imported endpoint also remains valid when the owner changes the endpoint’s backing memory. A peer imports the endpoint, associates it with one of its own reserved endpoint ids, and does not import the owner’s allocation. The owner can therefore grow, shrink, or replace the buffer behind the endpoint by rebinding it locally, and the peer can keep using its local endpoint id without re-importing the endpoint. The owner and its peers must still synchronize across the rebind so that peers do not access the endpoint while the owner is rebinding it to the new memory. Additionally, issuing an operation that lands outside the currently bound range is undefined behavior. By contrast, a VMM IPC import maps a specific allocation, so resizing that allocation on the owner side requires the peer to import and map the new one.

Once an endpoint has been created, shared with the peers that need it, and backed by a resource, asynchronous fabric operations — put, get, atomics, and reductions — can be issued against a peer’s logical endpoint id and offset. Unlike a load or store to a mapped peer address, a fabric operation reports an explicit completion status that the issuing code can inspect. On failure, the application can retry, reroute, or take other recovery action instead of faulting the process.

## 4.18.1. Prerequisites and Scope

Using logical endpoints has the following requirements:

  * **CUDA driver API.** The logical endpoint APIs are part of the low-level [CUDA Driver API](../03-advanced/driver-api.html#driver-api).

  * **NVLink fabric connectivity and device support.** The participating GPUs must be able to reach one another over an NVLink fabric. Whether a device supports logical endpoints depends on its architecture, the driver, and the system configuration, so an application should determine support by querying the device attributes described in [Section 4.18.2.2](#cft-query-for-support) rather than inferring it from the hardware present.

  * **IMEX daemon and channels.** Sharing an endpoint between processes uses fabric IPC handles, which require the NVIDIA IMEX daemon to be running and IMEX channels to be configured.


Endpoints can be shared with processes on the same node or on different nodes within the same NVLink fabric. The fabric IPC handle mechanism is similar to the one used by the VMM APIs. See [Virtual Memory Management](virtual-memory-management.html#virtual-memory-management) for details on fabric handles and IMEX channels.

## 4.18.2. Preliminaries

The following definitions establish the key objects and operations used in the rest of this section.

### 4.18.2.1. Definitions

**Logical Endpoint Id:** A logical endpoint id (`CUlogicalEndpointId`) is a 32-bit value that names a logical endpoint within a process. Endpoint ids are process-local: each process reserves its own range with `cuLogicalEndpointIdReserve` and chooses which reserved id to associate with a given endpoint. Fabric operations address an endpoint by this id together with an offset. A single endpoint id names one endpoint, but multiple ids may name the same endpoint. An endpoint id is associated with an endpoint through `cuLogicalEndpointCreate` or `cuLogicalEndpointImport`. This association must be removed with `cuLogicalEndpointDestroy` before the endpoint id is released with `cuLogicalEndpointIdRelease`. Releasing an endpoint id while it remains associated with an endpoint is undefined behavior. A later reservation may reuse a released id. [Section 4.18.6](#cft-logical-endpoint-ids) describes the lifecycle of a logical endpoint id.

**Logical Endpoint:** A logical endpoint is a transport object that represents a resource reachable over the NVLink fabric, exposed as a bounded offset space. Creating the endpoint allocates resources used to route fabric operations addressed to it. The application must then explicitly bind a target resource to one or more ranges of the endpoint’s offset space. Fabric operations name their target using an associated endpoint id together with an offset into that space. An endpoint is associated with a reserved id when the process creates it with `cuLogicalEndpointCreate` or imports a peer endpoint with `cuLogicalEndpointImport`. Destroying an endpoint association before explicitly unbinding its resources unbinds those resources but does not destroy the resources themselves. Destroying the resources before unbinding them from the endpoint is undefined behavior. [Section 4.18.3](#cft-api-overview) outlines an endpoint’s lifecycle.

**Unicast Logical Endpoint:** A unicast logical endpoint represents a point-to-point transport target. It has a single owner device, specified when the endpoint is created, on which the endpoint’s bound memory resides. Peer devices issue put or get operations against the endpoint id to move data into or out of the owner’s bound memory.

**Multicast Logical Endpoint:** A multicast logical endpoint represents a transport target that exposes multiple resources at each offset. Before binding its resource, each participating process must add that resource to the multicast endpoint. The process then binds the resource to the endpoint. Together, these bindings create one replica per team member. Fabric operations targeting the multicast endpoint access the corresponding offset in every replica. A multicast logical endpoint is the compute fabric transport counterpart of the VMM multicast object described in [Multicast Memory Sharing](virtual-memory-management.html#multicast-obj-mem).

**Compute Fabric Transport Handle:** A compute fabric transport handle is a resource-centric alternative to a traditional pointer. It is the pair of a logical endpoint id and an offset: the logical endpoint id selects a destination on the NVLink fabric — a unicast endpoint’s owner device or a multicast team — and the offset selects a location within that destination’s bound memory. Fabric operations use this handle in place of a peer virtual address.

**Logical Endpoint Clique:** A logical endpoint clique is a dynamic group of GPUs that are mutually reachable over Compute Fabric Transport operations at a required level of capability. The _clique type_ describes the desired level of capability. There are two logical endpoint clique types: `CU_CLIQUE_TYPE_UNICAST_LOGICAL_ENDPOINT` and `CU_CLIQUE_TYPE_MULTICAST_LOGICAL_ENDPOINT`. The _clique id_ is a 32-bit value naming the dynamic group of GPUs. Attempting to import a logical endpoint onto a device that is not a member of the same clique as the exporter device will fail. Attempting to create a multicast logical endpoint by adding devices that don’t belong to the same clique will fail. A single device can be a member of several cliques. The program may query the cliques a device is part of using `cuDeviceGetCliqueCount` and `cuDeviceGetCliqueInfo`.

**Endpoint Binding:** Endpoint binding is the association between a range in a logical endpoint’s offset space and a physical memory allocation supplied by the application. The application can bind memory using a mapped virtual address with `cuLogicalEndpointBindAddr` or by allocation handle with `cuLogicalEndpointBindMem`. Fabric operations are valid only for endpoint ranges that have been bound, and bindings must satisfy the endpoint’s alignment and size limits. See [Section 4.18.8](#cft-binding-memory) and [Section 4.18.6.1](#cft-limits-alignment).

All fabric operations are asynchronous: issuing one only initiates the transfer. Their completion and status are reported explicitly through the completion mechanism provided by each operation (see [Section 4.18.10](#cft-completion-status)).

**Put and Get Operations:** Put and get are data-movement primitives that compute fabric transport provides. A put operation moves data from the issuing GPU to a target logical endpoint. The multimem variant, `cuda::ptx::fabric_try_put_multimem`, performs a broadcast to a multicast logical endpoint, writing the source data to the corresponding offset in every replica. A get operation moves data from a target logical endpoint to the issuing GPU. In both directions, the remote side is identified by a logical endpoint id and offset rather than by a peer virtual address, and the remote allocation does not need to be mapped into the issuing process’s virtual address space.

**Reduction and Pull-Reduction Operations:** A reduction operation (_red_) combines data from the issuing GPU into a target logical endpoint using a reduction operator. A pull-reduction operation (_pull_red_) reads from a target multicast logical endpoint, combines values across the multicast team’s replicas with a reduction operator, and returns the result to the issuing GPU. It is the fabric counterpart of `multimem.ld_reduce`.

**Atomic Operations:** An atomic operation (_atom_) performs an atomic read-modify-write on a target logical endpoint. It reads the original value, overwrites it with a new value, and returns the original value. In contrast, a reduction operation does not return the original value. Operators for computing the new value include `add`, `min`, `max`, bitwise `and`, `or`, and `xor`. An exchange operation stores a user-provided value, and compare-and-swap stores a new value only if the target holds the expected compare value.

**Counted Operation:** A counted operation is a fabric operation that increments a target counter by the number of bytes written to the destination as data arrives. The receiver waits for the counter to reach an _expected byte count_ before consuming the data. The receiver does not know which bytes have been written until this expected byte count is reached. Once the counter reaches the expected byte count, the receiver can safely consume the data. The counter must be aligned to a 256-byte boundary in endpoint memory. Endpoints must be created with counted operation support enabled (`CU_LOGICAL_ENDPOINT_FLAG_COUNTED_OPS`), and applications must query device support (`CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_COUNTED_OPS_SUPPORTED`) before using this capability.

**Completion Status and Error Reporting:** A fabric operation is tracked by a _completion object_ , which signals when the operation has completed and records the operation’s status. Waiting on that object indicates whether the tracked operations completed successfully or whether a fabric error occurred. When a failure is reported, the application can query the detailed per-operation error status and then retry, reroute, or otherwise recover instead of faulting the process.

Because fabric operation updates are delivered out of order, observing that one destination location has been written — for example, the last byte of the destination region — does not imply that the rest of the transfer has landed. Observing that every destination location has been written does not indicate that the operation completed successfully and that the destination resource can be reused. For example, the producer may receive an error status if the response is dropped, causing it to retry the operation. Therefore, applications must wait for completion and inspect the reported status before treating the operation as complete. See [Section 4.18.10](#cft-completion-status).

### 4.18.2.2. Query for Support

Applications should query support before using logical endpoint APIs, because availability depends on the GPU architecture, the driver, and the system configuration. The following device attributes describe the relevant capabilities. The features are independent, so an application should query exactly the ones it intends to use.

**Unicast Logical Endpoint Support**

Query whether a device supports unicast logical endpoints:


    int unicastSupported = 0;
    cuDeviceGetAttribute(&unicastSupported,
                         CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_SUPPORTED,
                         device);
    if (unicastSupported != 0) {
        // `device` supports unicast logical endpoints
    }


**Multicast Logical Endpoint Support**

Query whether a device supports multicast logical endpoints:


    int multicastSupported = 0;
    cuDeviceGetAttribute(&multicastSupported,
                         CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_MULTICAST_SUPPORTED,
                         device);
    if (multicastSupported != 0) {
        // `device` supports multicast logical endpoints
    }


**Logical Endpoint IPC Handle Support**

Before exporting or importing an endpoint, query the IPC handle types that the device supports for logical endpoints. The attribute returns a bitmask of `CUlogicalEndpointIpcHandleType` values. The following checks whether the fabric IPC handle type for logical endpoints is supported:


    int supportedHandleTypes = 0;
    cuDeviceGetAttribute(
        &supportedHandleTypes,
        CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_SUPPORTED_HANDLE_TYPES,
        device);
    if ((supportedHandleTypes &
         CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC) != 0) {
        // `device` supports fabric IPC handles for logical endpoints
    }


**Counted Operations Support**

Counted operations are an optional capability. Query whether a device supports them before requesting `CU_LOGICAL_ENDPOINT_FLAG_COUNTED_OPS` on an endpoint:


    int countedOpsSupported = 0;
    cuDeviceGetAttribute(&countedOpsSupported,
                         CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_COUNTED_OPS_SUPPORTED,
                         device);
    if (countedOpsSupported != 0) {
        // `device` supports counted operations via logical endpoints
    }


**Unicast Access from Owner Device**

The owner device of a unicast endpoint is specified in the endpoint properties passed to `cuLogicalEndpointCreate`. The `CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_ACCESS_ON_OWNER_DEVICE_SUPPORTED` device attribute indicates whether this device can access the unicast logical endpoints it owns. If this access is not supported, the device can access the bound memory through pointers, which currently provides the best performance for the owner device. Devices other than the owner can access a unicast endpoint through a logical endpoint id associated with that endpoint.


    int ownerAccessSupported = 0;
    cuDeviceGetAttribute(&ownerAccessSupported,
                         CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_ACCESS_ON_OWNER_DEVICE_SUPPORTED,
                         device);
    if (ownerAccessSupported != 0) {
        // `device` supports unicast logical endpoint access on the owner device
    }


## 4.18.3. API Overview

The logical endpoint APIs are part of the low-level [CUDA Driver API](../03-advanced/driver-api.html#driver-api), and working with a logical endpoint follows a well-defined lifecycle. The steps below outline the workflow, and the sections that follow cover each step in detail. They fall into the following phases:

  * **Setup phase (Steps 1-11):** Verify participating devices are members of the same clique, reserve an id, describe and create the endpoint, share it with peers, confirm that it is ready, then allocate and bind its backing memory before data movement begins.

  * **Use phase (Step 12):** Issue fabric operations against the endpoint from the device.

  * **Cleanup phase (Step 13):** Unbind and destroy the endpoint, release its ids, and free the backing memory after all uses have completed.


  1. **Verify clique membership.** Verify that participating devices belong to the same clique with `cuDeviceGetCliqueCount` and `cuDeviceGetCliqueInfo` as described in [Section 4.18.4](#cft-clique-verification).

  2. **Reserve ids.** Reserve a range of logical endpoint ids with `cuLogicalEndpointIdReserve`. Reservation is per process. The id lifecycle and the rules for reuse and release are described in [Section 4.18.6](#cft-logical-endpoint-ids).

  3. **Describe the endpoint.** Populate a `CUlogicalEndpointProp` with the endpoint type (unicast or multicast), the owner device for unicast or group size for multicast, the size, the requested IPC handle types (`ipcHandleTypes`), and any flags — for example, `CU_LOGICAL_ENDPOINT_FLAG_COUNTED_OPS` to request support for counted operations (see [Counted Operation](#cft-counted-operation)).

  4. **Check alignment and size limits.** Call `cuLogicalEndpointGetLimits` for the proposed properties to obtain the required `bindAlignment` and `maxSize`. `maxSize` is the largest endpoint size the configuration permits. The endpoint size and all bindings must satisfy these limits.

  5. **Create the endpoint.** Call `cuLogicalEndpointCreate` to associate one of the reserved ids with a newly created endpoint described by the `CUlogicalEndpointProp` populated above.

  6. **Share the endpoint.** Export the endpoint to a fabric IPC handle with `cuLogicalEndpointExport`, transfer the handle through any IPC mechanism to the peer processes that need it, and have each peer process import it with `cuLogicalEndpointImport`. Exporting and importing the endpoint do not require memory to be bound to it.

  7. **Add devices (multicast only).** For a multicast endpoint, every participating device must be added to the group with `cuLogicalEndpointAddDevice`. Each process adds its own device after it holds the endpoint locally, so for multicast the endpoint is shared before the devices are added.

  8. **Confirm readiness.** Both `cuLogicalEndpointCreate` and `cuLogicalEndpointImport` are non-blocking. Use `cuLogicalEndpointQuery` to confirm readiness before use. Confirm a locally created endpoint before binding its memory and an imported endpoint before a kernel issues operations against it.

  9. **Allocate backing memory.** Allocate backing memory with the Virtual Memory Management APIs (`cuMemCreate`) or via `cudaMallocAsync`. The bound memory must be externally shareable if the endpoint will be shared with another process. The same requirement applies when memory is bound through an imported multicast endpoint. Configure the allocation for external sharing by requesting the fabric handle type (`CU_MEM_HANDLE_TYPE_FABRIC`) when creating it, and size the allocation to satisfy both `bindAlignment` and the allocation granularity. This step and endpoint creation are independent, but both must complete before binding.

  10. **Bind memory.** Associate physical memory with the endpoint using `cuLogicalEndpointBindMem` (by allocation handle) or `cuLogicalEndpointBindAddr` (by mapped address). Memory cannot be bound to an imported unicast endpoint, but it can be bound to an imported multicast endpoint. Fabric operations are valid only for ranges that have been bound.

  11. **Synchronize after binding.** Before any process issues fabric operations, participating processes must synchronize after binding, as described in [Section 4.18.8](#cft-binding-memory).

  12. **Use the endpoint.** Launch kernels that issue fabric operations against peer endpoint ids.

  13. **Clean up.** Unbind memory, destroy the endpoints, release the id range, and free the backing memory.


## 4.18.4. Verify clique membership

Before we can issue any compute fabric transport operations, we need to make sure that all participating devices are members of a clique that supports the desired set of compute fabric transport operations, and that they are all members of the same clique for a particular clique type. Otherwise, the devices cannot reach each other over the network with that particular type of operation.

First, we query how many cliques this device is a member of:


      size_t cliqueCount = 0; // not modified if there are errors
      cuDeviceGetCliqueCount(&cliqueCount, cuDevice);


Then, we query the clique information for all cliques this device is a member of and find the clique id of the clique we will be using. In this example we will be using the clique with type `CU_CLIQUE_TYPE_UNICAST_LOGICAL_ENDPOINT`.


      std::vector<CUcliqueInfo> cliqueInfos(cliqueCount);
      cuDeviceGetCliqueInfo(cliqueInfos.data(), &cliqueCount, cuDevice);
      unsigned int localCliqueId = 0;
      bool found = false;
      for (size_t idx = 0; idx < cliqueCount; ++idx) {
        if (cliqueInfos[idx].type == CU_CLIQUE_TYPE_UNICAST_LOGICAL_ENDPOINT) {
          localCliqueId = cliqueInfos[idx].id;
          found = true;
          break;
        }
      }


Verify that all participating devices have the same clique id for that clique type.


      std::vector<unsigned int> allCliqueIds(numRanks);
      MPI_Allgather(&localCliqueId, sizeof(unsigned int), MPI_BYTE,
                              allCliqueIds.data(), sizeof(unsigned int), MPI_BYTE, MPI_COMM_WORLD);
      for (int i = 0; i < numRanks; i++) {
        if (allCliqueIds[i] != localCliqueId) {
          fprintf(stderr, "Rank %d: Clique id %u does not match expected %u\n", myRank, allCliqueIds[i], localCliqueId);
          return EXIT_FAILURE;
        }
      }


## 4.18.5. Creating an Endpoint

Creating and importing endpoints begins by determining how many logical endpoint ids the process needs. Then reserve a range of ids with `cuLogicalEndpointIdReserve`, which returns the base id of the range. Choose a count large enough for every endpoint that the process will create or import. The endpoints associated with the range may have different properties. For example, an application may reserve one id per GPU plus one id for a multicast endpoint. The example reserves one id per rank:


      CUlogicalEndpointId leId = 0;
      cuLogicalEndpointIdReserve(&leId, (uint32_t)numRanks);


After reserving the range, associate its ids as endpoints are created or imported. Use `cuLogicalEndpointCreate` for a locally created endpoint and `cuLogicalEndpointImport` when a remote endpoint’s handle becomes available. For each endpoint that the process creates, describe its properties in a `CUlogicalEndpointProp`. A unicast endpoint names the single owner device that will hold its bound memory. Set the endpoint type, the requested IPC handle types (`ipcHandleTypes`), and any flags, opting in to counted operations only when requested:


      CUlogicalEndpointProp endpointProp {};
      endpointProp.type = CU_LOGICAL_ENDPOINT_TYPE_UNICAST;
      endpointProp.unicast = {.device = cuDevice};
      endpointProp.ipcHandleTypes = CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC;
      endpointProp.flags = useCounted ? CU_LOGICAL_ENDPOINT_FLAG_COUNTED_OPS : CU_LOGICAL_ENDPOINT_FLAG_NONE;


The endpoint size is set separately, once its alignment and maximum-size limits are known. See [Section 4.18.6.1](#cft-limits-alignment) for how these are queried and how the size is chosen.

Associate a reserved id with the described properties by calling `cuLogicalEndpointCreate`. A process that creates a local endpoint and imports peer endpoints typically reserves one id per participant so that a peer is addressed as the base id plus its rank. Each rank creates its own endpoint at `leId + myRank`:


      cuLogicalEndpointCreate(leId + myRank, &endpointProp);


`cuLogicalEndpointCreate` is non-blocking, so the endpoint is not usable the moment it returns. Confirm that it is fully constructed with `cuLogicalEndpointQuery` before binding memory to it, as described in [Section 4.18.7.1](#cft-confirming-readiness).

**Multicast endpoint.** A multicast endpoint follows the same steps, differing in how it is described and in requiring a group of devices to be added. It is described with `type = CU_LOGICAL_ENDPOINT_TYPE_MULTICAST` and `multicast.numDevices` set to the number of devices in the multicast group. Once it has been created and shared, every participating device must join the group with `cuLogicalEndpointAddDevice`. A process adds its own device only after it holds the endpoint locally — the creator after `cuLogicalEndpointCreate` and each peer after `cuLogicalEndpointImport`. The endpoint is not ready for use until the multicast group is complete. Membership is permanent for the lifetime of the endpoint.

## 4.18.6. Logical Endpoint Ids

A logical endpoint id is a process-local name whose lifecycle is separate from that of the endpoint it identifies. An endpoint id is always in one of the following states:

  1. **Not reserved.** The id is not owned by the process and cannot be associated with an endpoint. `cuLogicalEndpointIdReserve` reserves it.

  2. **Reserved.** The id is owned by the process but names nothing. Calling `cuLogicalEndpointCreate` or `cuLogicalEndpointImport` with a reserved id associates it with an endpoint and moves it to the associated state. `cuLogicalEndpointIdRelease` returns it to the not-reserved state.

  3. **Associated.** The id names an endpoint, and fabric operations that use the id target that endpoint. `cuLogicalEndpointDestroy` removes the association and returns the id to the reserved state.


`cuLogicalEndpointIdReserve` reserves a range of `count` endpoint ids and returns its base, `baseLeId`, so the reserved range is `[baseLeId, baseLeId + count)`. The base is an output: the caller does not choose which ids it receives, and reservation fails if no range of `count` ids is available, for example because another library in the process holds reservations.

An endpoint may be associated with several ids at once. For instance, it can be associated with the owner’s id through `cuLogicalEndpointCreate` and with additional ids through `cuLogicalEndpointImport`. Such ids are _aliases_ for the same endpoint, and an operation issued against any of them reaches the same destination.

`cuLogicalEndpointDestroy` removes the association for one id. That id can then be associated with a different endpoint by a later create or import, without being reserved again. Destroy also unbinds any memory this process bound through the id. The endpoint itself is freed only when the last of its aliases is destroyed.

`cuLogicalEndpointIdRelease(baseLeId, count)` releases up to `count` ids in the range `[baseLeId, baseLeId + count)`. Every id in the range must have been reserved, and every endpoint associated with an id in the range must first have been destroyed.

Because reservation is per process, an endpoint need not be imported with the same id in every importing process. A convenient convention is for each process to reserve a range of the same size, one id per participant, and address a given peer’s endpoint as `baseLeId + peer` locally. Each process has its own reserved `baseLeId`, which need not match across processes. CUDA does not detect or report a mismatch if the same peer endpoint is imported with different ids in different processes. Maintaining the correct local peer-to-endpoint-id mapping is the application’s responsibility.

### 4.18.6.1. Limits and Alignment

The alignment and maximum-size requirements of an endpoint depend on its properties, so they must be queried for the specific configuration being created rather than assumed. `cuLogicalEndpointGetLimits` returns both values for a given `CUlogicalEndpointProp`:


    cuuint64_t bindAlignment = 0;
    cuuint64_t maxSize = 0;
    cuLogicalEndpointGetLimits(&bindAlignment, &maxSize, &prop);


The endpoint size and every bind offset must be a multiple of the returned `bindAlignment`. `maxSize` is the maximum size of the logical endpoint. If `maxSize` is less than `CUlogicalEndpointProp::size`, the application must adjust the request to that smaller value. To expose the entire requested range, the application may partition it across multiple endpoints, each no larger than `maxSize`.

A single allocation covering the whole endpoint is the simplest arrangement, but it is not a requirement. An endpoint’s offset space may equally be backed by several bindings that map distinct, possibly non-contiguous physical allocations, as long as every bound range is a multiple of `bindAlignment`.

## 4.18.7. Sharing an Endpoint

Sharing an endpoint with a peer follows an export-transfer-import exchange: the owner exports the endpoint to a fabric IPC handle, transfers that handle to the peers that need it, and each peer imports the endpoint and associates it with one of its own reserved logical endpoint ids that is not already associated with an endpoint. Sharing uses fabric IPC handles, so the endpoint must be bound to externally shareable memory, and the environment must meet the fabric IPC requirements described in [Section 4.18.1](#cft-prerequisites-scope).

Export the endpoint with `cuLogicalEndpointExport`, requesting the same IPC handle type that the endpoint was described with (`ipcHandleTypes`). This produces a `CUlogicalEndpointFabricHandle`, the fabric IPC handle representing the logical endpoint. Transfer it to the peers using an inter-process communication mechanism of the application’s choice, and then have each peer import it with `cuLogicalEndpointImport`. The importing id is chosen locally import it with `cuLogicalEndpointImport`. The importer can chose the logical endpoint id at which to import the endpoint. This id must have been successfully reserved, and must not be assocaited with any other logical endpoint, but it does not have to match the id the owner used to create the endpoint. Every device can import the same endpoint at a different id, and even at multiple different ids (i.e. importing the same endpoint multiple times is allowed).

The unicast point-to-point example below shares endpoints on a ring. Each rank exports its own endpoint and, with a single `MPI_Sendrecv`, sends the handle to the receive neighbor while collecting the send neighbor’s handle. It imports that handle as `leId + sendRank`, the id it later uses to issue fabric operations against the imported endpoint’s bound memory:


      // Export our own endpoint to a fabric IPC handle, hand it to the peers that
      // need it, and import each peer's handle to a locally reserved id.
      CUlogicalEndpointFabricHandle localHandle;
      CUlogicalEndpointFabricHandle sendRankHandle;
      cuLogicalEndpointExport(&localHandle, leId + myRank, CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC);
      MPI_Sendrecv(
          &localHandle, sizeof(CUlogicalEndpointFabricHandle), MPI_BYTE, recvRank, 0,
          &sendRankHandle, sizeof(CUlogicalEndpointFabricHandle), MPI_BYTE, sendRank, 0,
          MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      cuLogicalEndpointImport(leId + sendRank, &sendRankHandle, CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC);


### 4.18.7.1. Confirming Readiness

`cuLogicalEndpointCreate` and `cuLogicalEndpointImport` return as soon as the request is accepted. Using a logical endpoint id with an API that requires a fully constructed endpoint before `cuLogicalEndpointQuery` reports readiness is undefined behavior. `cuLogicalEndpointQuery` is itself non-blocking: it returns 0 if _any_ id in the queried range is not yet ready, and a non-zero value once _all_ ids in the given range are ready, so it is typically called in a polling loop.

Before querying readiness, reserve the logical endpoint id and associate it with an endpoint through `cuLogicalEndpointCreate` or `cuLogicalEndpointImport`. Multicast endpoints do not become ready until all devices are added with `cuLogicalEndpointAddDevice`, requiring multi-process programs to export them (`cuLogicalEndpointExport`) and share them before blocking on their readiness.

Before performing any of the following operations on the logical endpoint, call `cuLogicalEndpointQuery` and wait until it reports that the logical endpoint is fully constructed:

  * Bind memory with `cuLogicalEndpointBindAddr` or `cuLogicalEndpointBindMem`.

  * Unbind memory with `cuLogicalEndpointUnbind`.

  * Issue any `cuda::ptx::fabric_try_*` operation.


The following APIs do not require `cuLogicalEndpointQuery` to have reported readiness:

  * `cuLogicalEndpointExport`

  * `cuLogicalEndpointDestroy`

  * `cuLogicalEndpointIdRelease`


These APIs do not require readiness, but their other lifecycle preconditions still apply.

## 4.18.8. Binding Memory

An endpoint owns no memory. Binding associates a range of the endpoint’s offset space with a physical allocation, and fabric operations are valid only for ranges that have been bound. A binding is established per device, so a unicast binding applies to the owner device, while a multicast binding applies to an individual device in the multicast team.

Memory can be bound in either of two ways, depending on how the application holds the allocation:

  * `cuLogicalEndpointBindMem` binds by allocation handle, using the `CUmemGenericAllocationHandle` returned by `cuMemCreate`.

  * `cuLogicalEndpointBindAddr` binds by mapped virtual address. Examples include a pointer returned by `cudaMallocAsync` and an address in a range mapped with `cuMemMap`. See the CUDA Memory Management APIs for the currently supported allocation types.


Note

Not every CUDA-accessible allocation can be bound to a logical endpoint. See the CUDA Driver API docs for the allocation types currently supported by the binding APIs.

Both take the endpoint id, the device to which the binding applies, an offset into the endpoint’s offset space, the allocation to bind, and the size of the binding. `cuLogicalEndpointBindMem` additionally takes a `memOffset` into the allocation, letting a sub-range of the handle be bound. The endpoint offset, the size, and (for `cuLogicalEndpointBindMem`) the `memOffset` must all be multiples of `bindAlignment`, and the bound range must lie within the endpoint’s size (see [Section 4.18.6.1](#cft-limits-alignment)).


      // Bind the whole backing allocation at endpoint offset 0, by allocation handle.
      cuLogicalEndpointBindMem(leId + myRank, cuDevice, 0, exportHandle, 0, exportSize, 0);
      // No rank may issue an operation until every destination has been bound.
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));


After establishing the initial bindings, participating processes must synchronize before any process issues a fabric operation. For a unicast endpoint, each importer must synchronize with the exporting process after the owner binds its backing memory. For a multicast endpoint, all participating processes must wait until every participant has bound its backing memory. The application may use any inter-process synchronization mechanism that establishes this ordering.

Because a peer imports the endpoint rather than the owner’s allocation, the owner can replace the memory behind an endpoint without any peer re-importing it. Unbind the current range with `cuLogicalEndpointUnbind` and bind a new allocation through the same endpoint id to the same range of the endpoint’s offset space. The owner and its peers must synchronize across the rebind so that no peer issues an operation against the endpoint while it is unbound. Issuing an operation that lands outside the currently bound range is undefined behavior.

## 4.18.9. Fabric Operations

Threads access endpoint resources by executing _fabric operations_ that accept a compute fabric transport handle — a logical endpoint id and offset pair. Fabric operations are asynchronous, that is, programs must explicitly wait on their completion to observe their effects. Fabric operations are prefixed with a `try_` to indicate that they may fail. That is, after waiting for completion, threads must check the operation’s completion mechanism (see [Section 4.18.10](#cft-completion-status)) to determine whether the operation succeeded. This distinguishes fabric operations from ordinary pointer operations: failures from pointer-based accesses are reported to the application by destroying either the context or the entire application. Applications that want to handle these errors - for example, to recover from pointer failures - can only do so at the context or process level. Fabric operations allow applications to handle errors at the program thread level.

Device code issues fabric operations through the following `cuda::ptx` instructions:

Instruction | Operation  
---|---  
`cuda::ptx::fabric_try_put` | Write to a compute fabric transport handle.  
`cuda::ptx::fabric_try_get` | Read from a compute fabric transport handle.  
`cuda::ptx::fabric_try_red` | Reduction into a compute fabric transport handle.  
`cuda::ptx::fabric_try_pullred` | Pull-reduction from a compute fabric transport handle.  
`cuda::ptx::fabric_try_atom` | Atomic read-modify-write on a compute fabric transport handle.  

The operations above - with the exception of `try_atom` \- move data in 16-byte units, so both ends of a transfer must be 16-byte aligned: the source pointer and the destination offset into the endpoint must each be a multiple of 16 bytes.

The example issues fabric puts. A put reads data from a shared-memory buffer and writes it to a destination handle. Multiple threads write the input data to shared memory, then use `__syncthreads` to synchronize with the thread that issues the put. The put reads shared memory via the async-proxy, so the issuing thread calls `cuda::ptx::fence_proxy_async(cuda::ptx::space_shared)` to synchronize the generic-proxy shared memory writes to the async proxy. The put takes the destination handle, the source buffer base address, a byte count, and the `mbarrier` that will track completion:


        // Make the staged shared-memory source visible to the async proxy before the put reads it.
        cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
        cuda::ptx::fabric_try_put(cuda::ptx::space_shared, cuda::ptx::sem_relaxed,
                                  cuda::ptx::scope_sys, putLeId, offset, smemBuf,
                                  FABRIC_CHUNK_SIZE, &putBar);


The call sets several instruction qualifiers whose full set — the access sizes, memory-ordering modes, and scopes each instruction accepts — is specified by the PTX ISA (see [Fabric Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#fabric-operations)). Two aspects of the operation are worth calling out here because they connect it to the rest of the program: how its completion is tracked and how memory ordering is specified.

**Transaction count.** Fabric operations that use an `mbarrier` for completion follow the same `mbarrier` transaction-accounting model as the pointer-addressed `cuda::ptx::cp_async_bulk` copies described in [Using TMA to transfer data](async-copies.html#async-copies-tma-one-dim). As the put completes, it decrements its `mbarrier`’s transaction count by 1 per 16 bytes delivered. The barrier phase completes after all expected arrivals have occurred and the transaction count reaches zero. The put does not set the expected transaction count. A separate `cuda::ptx::mbarrier_arrive_expect_tx`, issued after the operation is submitted, records an arrival and increments the transaction count by the byte count divided by 16, because that is the amount the put will decrement it by. Fetching operations use byte-based transaction accounting instead. `cuda::ptx::fabric_try_get` and `cuda::ptx::fabric_try_pullred` decrement the transaction count by 1 for every byte delivered, so `cuda::ptx::mbarrier_arrive_expect_tx` adds the full byte count. [Section 4.18.10](#cft-completion-status) covers the mbarrier layout requirement and how the fabric status is read back.

A **counted** put additionally increments a destination-side byte counter as its data arrives, so the receiver can poll on the counter before accessing the local data instead of exchanging a separate completion message. The code below does this with `cuda::ptx::fabric_try_put_counted` — the plain put’s `.counted::bytes` variant — passing a second destination offset, the endpoint offset of the counter:


        // Make the staged shared-memory source visible to the async proxy before the put reads it.
        cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
        cuda::ptx::fabric_try_put_counted(cuda::ptx::space_shared, cuda::ptx::sem_relaxed,
                                          cuda::ptx::scope_sys, putLeId, offset, counterOffset,
                                          smemBuf, FABRIC_CHUNK_SIZE, &putBar);


Counted operations require an endpoint created with counted-operation support (`CU_LOGICAL_ENDPOINT_FLAG_COUNTED_OPS`, see [Section 4.18.2.2](#cft-query-for-support)), and the target counter must be aligned to a 256-byte boundary in endpoint memory. An application satisfies this when it lays out the endpoint’s backing memory, reserving the counter in a suitably aligned region.

Although the example issues only puts, the get, reduction, and pull-reduction operations use the same submission, status-reporting, and wait sequence described in [Section 4.18.10](#cft-completion-status). Their transaction-counting units are operation-specific, as described above.

**Memory ordering.** The example issues its fabric puts with `relaxed` ordering at `sys` scope. A fabric put spans several proxies: it accesses the remote data (and, for a counted put, the counter) through the _fabric-proxy_ , reads its `.shared::cta` source through the _async-proxy_ , and updates the completion `mbarrier` through the _generic-proxy_. Because the mbarrier is updated through the generic-proxy, completion is observed just by waiting at the barrier with block-scope operations. However, unlike `cuda::ptx::cp_async_bulk` operations, observing completion of a fabric operation does not order fabric-proxy accesses to the generic-proxy. Before accessing the data through the generic-proxy with a pointer, `cuda::ptx::fence_proxy_generic_fabric_alias` must be used to synchronize the fabric-proxy with the generic-proxy.

An endpoint id alone does not provide pointer access to a remote destination. Importing an endpoint does not map its bound resource into the importer’s virtual address space. The following example therefore uses a unicast endpoint owned by the issuing thread’s device. `leId` is the endpoint id, and the memory bound at `dataOffset` is also mapped into the device’s virtual address space. `dataPtr` points to the same location.

After issuing the put, the thread waits for completion. Before reading the destination through the generic-proxy with `dataPtr`, the thread must issue `cuda::ptx::fence_proxy_generic_fabric_alias` first with `cuda::ptx::sem_release` and then with `cuda::ptx::sem_acquire`. The completion wait alone does not order the fabric-proxy writes before the generic-proxy pointer load. `waitForSuccessfulCompletion` represents the `mbarrier` wait and status check described in [Section 4.18.10](#cft-completion-status). Initially the destination is 0, and `smemSrcData` is a 16-byte shared-memory source whose first `uint32_t` element holds 42.

Issuing and loading from same thread (Thread 0)  
---  


    namespace ptx = cuda::ptx;

    ptx::fabric_try_put(
        ptx::space_shared,
        ptx::sem_relaxed,
        ptx::scope_sys,
        leId, dataOffset,
        smemSrcData, 16, &mBar);
    ptx::fabric_submit();
    ptx::mbarrier_arrive_expect_tx(
        ptx::sem_acquire,
        ptx::scope_cta,
        ptx::space_shared,
        &mBar, 1);
    waitForSuccessfulCompletion(&mBar);

    // Order the subsequent generic-proxy load after the fabric put.
    ptx::fence_proxy_generic_fabric_alias(ptx::sem_release);
    ptx::fence_proxy_generic_fabric_alias(ptx::sem_acquire);

    assert(*dataPtr == 42);


After the mbarrier wait returns, the put has completed, but fabric-proxy writes are still unordered with respect to generic-proxy loads. The release and acquire fences establish that ordering, so the pointer load observes the value 42.

The same release/acquire proxy-fence is also required when one thread publishes data with fabric operations and another thread on the endpoint owner’s device consumes the bound destination through pointers. In the following schematic message-passing example, the sender writes the data and signaling flag with fabric operations through the _fabric-proxy_. The receiver polls the flag and reads the data through pointers using the _generic-proxy_. `dataPtr` and `flagPtr` address the destination memory bound to the receiver-owned endpoint. Initially, the destination data and flag are both 0. `smemSrcData` and `smemSrcFlag` are 16-byte shared-memory sources composed of four `uint32_t` elements. The first element of `smemSrcData` holds 42, and the first element of `smemSrcFlag` holds 1.

Sender (Thread 0) | Receiver (Thread 1)  
---|---  


    namespace ptx = cuda::ptx;

    ptx::fabric_try_put(
        ptx::space_shared,
        ptx::sem_relaxed,
        ptx::scope_sys,
        leId, dataOffset,
        smemSrcData, 16, &mBar);
    ptx::fabric_submit();
    ptx::mbarrier_arrive_expect_tx(
        ptx::sem_acquire,
        ptx::scope_cta,
        ptx::space_shared,
        &mBar, 1);
    waitForSuccessfulCompletion(&mBar);

    ptx::fence_proxy_generic_fabric_alias(
        ptx::sem_release);

    ptx::fabric_try_put(
        ptx::space_shared,
        ptx::sem_relaxed,
        ptx::scope_sys,
        leId, flagOffset,
        smemSrcFlag, 16, &mBar);
    ptx::fabric_submit();
    ptx::mbarrier_arrive_expect_tx(
        ptx::sem_acquire,
        ptx::scope_cta,
        ptx::space_shared,
        &mBar, 1);
    waitForSuccessfulCompletion(&mBar);


| 


    namespace ptx = cuda::ptx;

    cuda::atomic_ref<uint32_t,
    cuda::thread_scope_system> flag(*flagPtr);

    while (flag.load(cuda::memory_order_relaxed) != 1);

    // Order the data load after the observed flag update.
    ptx::fence_proxy_generic_fabric_alias(
        ptx::sem_acquire);

    assert(*dataPtr == 42);


After the data put completes, the sender issues `cuda::ptx::fence_proxy_generic_fabric_alias(cuda::ptx::sem_release)` before issuing the flag put. When the receiver’s flag load observes the value 1 written by that put, the receiver knows that the sender has published the data. Before consuming the data, the receiver must issue `cuda::ptx::fence_proxy_generic_fabric_alias(cuda::ptx::sem_acquire)`. This acquire fence prevents the subsequent pointer load of the data from being ordered before the observed flag update. Together, the sender’s release fence and the receiver’s acquire fence order the sender’s data put before the receiver’s data load. In the absence of any other writes to the data, the load therefore observes the value 42.

### 4.18.9.1. Per-Thread and Warp-Collective Operations

Each fabric instruction defines its execution granularity. `fabric.try_get`, `fabric.try_get.tensor`, `fabric.try_put`, `fabric.try_put.tensor`, and `fabric.try_red` are per-thread: each execution initiates an independent operation for the issuing thread. `fabric.try_pullred` is warp-collective: all 32 lanes of a warp must execute the instruction together. The PTX ISA specifies the required participation for each instruction.

Execution granularity describes only how threads issue a fabric operation. Completion tracking is independent of that granularity; see [Section 4.18.10.1](#cft-configuring-barrier).

## 4.18.10. Completion Status and Error Reporting

A fabric operation reports its status through its completion object. The operations described here use a shared-memory `mbarrier` for completion, so the same `mbarrier` signals completion and records status. Initialize this `mbarrier` by calling `cuda::ptx::mbarrier_init` with `cuda::ptx::layout_v1`. This layout extends the object with the field that holds the fabric operation status. The default `cuda::ptx::layout_v0` tracks only arrival and transaction counts and cannot carry that status, so it cannot be used with the reporting mechanism:


        cuda::ptx::mbarrier_init(cuda::ptx::layout_v1, &putBar, 1);


`cuda::ptx::fabric_try_put` takes the mbarrier that tracks the operation. As the operation’s data is delivered, the fabric engine completes the corresponding transaction on the barrier and records the operation’s status there. The signal that the transfer has landed therefore comes from the put itself, not from the issuing thread.

The put delivers its bytes but does not tell the barrier how many to expect. For the per-thread put shown here, the issuing thread supplies that count in a separate step. It first submits the operation with `fabric_submit` so the fabric engine eventually begins consuming it, then calls `cuda::ptx::mbarrier_arrive_expect_tx` to record an arrival and add the expected transaction count — the number of bytes the operation will deliver, counted in 16-byte units. Each fabric operation must be submitted before the barrier phase that tracks its completion advances. After submitting the operation, the issuing thread either performs a barrier operation required for the phase to advance (for example, `cuda::ptx::mbarrier_arrive_expect_tx` or `cuda::ptx::mbarrier_expect_tx`), or it synchronizes with a different thread that does so. The issuing thread need not wait on the barrier itself. Another thread may perform that wait. Once all expected arrivals have occurred and the transaction count reaches zero, the barrier phase advances and the wait returns. The program must wait for every fabric operation to complete before the grid exits. Leaving an operation outstanding at grid exit is undefined behavior.

The following sequence submits that put and waits for its completion:


        cuda::ptx::fabric_submit();
        cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_relaxed, cuda::ptx::scope_cta,
                                             cuda::ptx::space_shared, &putBar,
                                             (FABRIC_CHUNK_SIZE / FABRIC_TX_GRANULARITY_BYTES));
        // Spin while the barrier's phase hasn't flipped yet. The report outputs are
        // only valid once the barrier completes, so inspect them after.
        bool isReportSeen = false;
        uint8_t reportValue = 0;
        bool complete = waitWithDebugTimeout(
            [&]() {
              return !cuda::ptx::mbarrier_try_wait_parity(cuda::ptx::mbarrier_phase_primary, cuda::ptx::sem_relaxed,
                                                          cuda::ptx::scope_cta, isReportSeen, reportValue, &putBar,
                                                          /*phaseParity=*/0);
            },
            "cftTryPutKernel: mbarrier try_wait parity");
        if (!complete) { __trap(); }
        if (isReportSeen) {
          reportFabricError(&reportValue, offset);
        }


The `phase_type::primary` form of `cuda::ptx::mbarrier_try_wait_parity` has _report predicate_ and _report value_ destination operands alongside its _completion predicate_. If the _completion predicate_ is false, the _report predicate_ and _report value_ contain unspecified values. When the `try_wait` observes phase completion — that is, when the _completion predicate_ is true — the _report predicate_ and _report value_ together contain additional information about the operations tracked by that barrier phase. If all operations tracked by the barrier phase were fabric operations, a false _report predicate_ indicates that all fabric operations succeeded. Otherwise, if the _report predicate_ is true, the _report value_ may contain more information about the failures, and the program may use `cudaFabricOpErrorStatusCount` and `cudaFabricOpErrorStatusGet` to inspect this information.


    inline __device__ void reportFabricError(uint8_t *reportValue, uint64_t offset) {
      const unsigned long long off = offset;
      unsigned int errCount = 0;
      if (cudaFabricOpErrorStatusCount(reportValue, cudaFabricOpStatusSourceMbarrierV1, &errCount) != cudaSuccess) {
        printf("cudaFabricOpErrorStatusCount failed at offset %llu\n", off);
        return;
      }
      for (unsigned int i = 0; i < errCount; ++i) {
        cudaFabricOpStatusInfo info = cudaFabricOpStatusInfoSuccess;
        if (cudaFabricOpErrorStatusGet(reportValue, cudaFabricOpStatusSourceMbarrierV1, i, &info) == cudaSuccess) {
          printf("fabric put error[%u/%u] at offset %llu: status=%d\n", i, errCount, off, (int)info);
        } else {
          printf("cudaFabricOpErrorStatusGet failed for error[%u/%u] at offset %llu\n", i, errCount, off);
        }
      }
    }


The decoded status identifies why an operation failed, which lets the application recover instead of faulting the process. A transient fabric error such as a dropped packet or a link reset can be retried, rerouted, or handled by an application-specific fallback. The example prints the decoded errors, whereas a communication library would typically act on them to preserve forward progress.

### 4.18.10.1. Configuring Barrier Arrivals and Transaction Counts

The completion sequence above records the arrival and expected-transaction contribution from the thread that issued the put. Which thread may record them depends on the operation’s execution granularity. For a per-thread operation, the issuing thread must record them for its own operation. For a warp-collective `fabric.try_pullred`, either a single lane of the participating warp records them on the warp’s behalf or every lane records its own arrival. `cuda::ptx::mbarrier_arrive_expect_tx` is a single fused operation: it counts one arrival and adds to the expected transaction count in the same step, so one thread must perform both. It must run after the operation is submitted with `fabric.submit` and before the wait observes completion. The thread that records the arrival and expected transaction count need not be the thread that waits on the barrier.

Several independent operations can share one barrier. When a warp-collective `fabric.try_pullred` uses a single arrival, any one lane adds the expected transaction count equal to the number of bytes the warp delivers. When every lane records an arrival instead, the other 31 contribute zero transactions, so the barrier’s initialized arrival count must be the full warp width.

Recording one arrival per thread inflates the barrier’s arrival count when many operations share it. A `layout::v1` mbarrier holds its arrival count in a 9-bit field, so the count is capped at 511; exceeding it overflows the barrier. A block of 512 threads each arriving on one shared barrier would already cross that cap. Prefer one arrival per group where the granularity allows it: synchronize the warp and elect a single lane, or synchronize the block and record one arrival on a block-wide shared barrier.

## 4.18.11. Cleaning Up

Cleaning up requires synchronization between any process that may issue an operation against an endpoint and any process that has bound a resource to it. For a unicast endpoint, this means that the owner must not unbind or destroy the endpoint, or free its backing resource, while an importer might still issue an operation against it. For a multicast endpoint, each participating process owns its bound replica, so no participant may unbind or free its replica while another process might still issue an operation against the endpoint. What matters is the ordering, not the mechanism that enforces it: any inter-process synchronization that establishes this ordering before bindings are removed is sufficient.


      // ---- Clean up endpoints and free the backing allocation. ----
      // Once every rank's stream has synchronized, all fabric puts
      // have completed and no rank will issue another operation. Making that global
      // with a barrier ensures no peer still targets an endpoint we clean up below.
      MPI_Barrier(MPI_COMM_WORLD);

      // Destroy the imported endpoint (the peer target our puts addressed). An import
      // is only a local reference to the peer's endpoint id and maps no peer memory,
      // so once our puts have drained this just drops that reference.
      cuLogicalEndpointDestroy(leId + sendRank);

      // Unbind and destroy our owned endpoint, then release the reserved id range.
      cuLogicalEndpointUnbind(leId + myRank, cuDevice, 0, exportSize);
      cuLogicalEndpointDestroy(leId + myRank);
      cuLogicalEndpointIdRelease(leId, numRanks);

      // Free the backing allocation in VMM cleanup order: unmap the range, release
      // the physical handle, then free the reserved virtual address range.
      cuMemUnmap(exportPtr, exportSize);
      cuMemRelease(exportHandle);
      cuMemAddressFree(exportPtr, exportSize);


**Wait for fabric operations and all issuers to stop.** A completion object that tracks fabric operations must be waited on before that object is deallocated. For fabric operations issued by a grid, a thread in the grid must wait for completion before the grid exits. Exiting without performing that wait is undefined behavior. In the example, each kernel waits for completion before returning, and stream synchronization confirms that those kernels and their waits have completed.

Fabric operations issued by another process that target resources bound to the same logical endpoint may still be in flight. Before removing a unicast binding, the owner must therefore wait until every importer has stopped issuing. Before removing any multicast binding, every participant that owns a bound replica must know that all processes have stopped issuing. The example does this with a barrier after all ranks synchronize their CUDA streams. An application may use a narrower synchronization scheme when it preserves the required ordering.

**Destroy imported endpoint ids.** Each importing process destroys the logical endpoint id associated with its imported endpoint by calling `cuLogicalEndpointDestroy`. This drops only that local alias and unbinds nothing on the owner, so the destroy itself needs no cross-process coordination.

**Destroy the owner’s endpoint id.** Destroying the owner’s logical endpoint id also unbinds any memory this process bound through that id, so calling `cuLogicalEndpointUnbind` first is optional but recommended. The endpoint’s resources are released once every alias — the owner’s id and any imports — has been destroyed.

The same path applies to unicast and multicast endpoints. Because multicast membership is permanent, changing a multicast group requires destroying the endpoint and creating a new one. A device cannot leave the group while the endpoint exists.

**Reuse or release the endpoint ids.** After an endpoint association is destroyed with `cuLogicalEndpointDestroy`, its id remains reserved but unassociated and can be reused immediately. To return ids to the system, call `cuLogicalEndpointIdRelease` on the range reserved earlier with `cuLogicalEndpointIdReserve`.

**Free the backing allocation.** The endpoint does not own its backing memory, so that allocation is freed separately. Use the deallocation mechanism that corresponds to how the memory was allocated. For a VMM allocation, the usual order is `cuMemUnmap` to remove the mapping, `cuMemRelease` to release the physical handle, and `cuMemAddressFree` to return the reserved virtual address range. The program must not free an allocation while it is bound to an endpoint.