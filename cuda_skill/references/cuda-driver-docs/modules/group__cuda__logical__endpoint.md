# 6.17. Logical Endpoint

**Source:** group__CUDA__LOGICAL__ENDPOINT.html#group__CUDA__LOGICAL__ENDPOINT


### Classes

struct

CUlogicalEndpointFabricHandle


struct

CUlogicalEndpointProp



### Typedefs

typedef cuuint32_t CUlogicalEndpointId


### Enumerations

enum CUlogicalEndpointFlag

enum CUlogicalEndpointIpcHandleType

enum CUlogicalEndpointType


### Functions

CUresult cuLogicalEndpointAddDevice ( CUlogicalEndpointId leId, CUdevice dev )


Associates a device to a multicast logical endpoint.

######  Parameters

`leId`
    Logical endpoint id representing a multicast logical endpoint.
`dev`
    Device that will be associated with the multicast logical endpoint.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED

###### Description

Associates a device to a logical endpoint. The type of the logical endpoint must be CU_LOGICAL_ENDPOINT_TYPE_MULTICAST. The added device will be a part of the multicast team of size specified by CUlogicalEndpointProp::multicast::numDevices during cuLogicalEndpointCreate. The association of the device to the multicast logical endpoint is permanent during the life time of the multicast logical endpoint. All devices must be added to the multicast logical endpoint before any memory can be bound to any device in the team. A multicast logical endpoint will not be ready for use until all devices have been added. User can query whether the logical endpoint is ready for use via cuLogicalEndpointQuery.

CUresult cuLogicalEndpointBindAddr ( CUlogicalEndpointId leId, CUdevice dev, cuuint64_t offset, void* ptr, cuuint64_t size, unsigned long long flags )


Bind a memory allocation represented by a virtual address to a logical endpoint.

######  Parameters

`leId`
    Logical endpoint to which memory will be associated.
`dev`
    Device on which the memory will be bound to the logical endpoint
`offset`
    Offset into the logical endpoint space.
`ptr`
    Virtual address of the memory allocation.
`size`
    Size of memory that will be bound to the logical endpoint.
`flags`
    Flags for future use, must be zero for now.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED, CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_SYSTEM_NOT_READY, CUDA_ERROR_ILLEGAL_STATE

###### Description

Binds the memory allocation specified by its mapped address `ptr` to a logical endpoint represented by `leId` at the offset `offset`. The memory must have been allocated via cuMemCreate or cudaMallocAsync. The intended `size` of the bind, the `offset` in the logical endpoint range and `ptr` must be multiples of the value for `bindAlignment` as returned by cuLogicalEndpointGetLimits.

The `size` cannot be larger than the size of the allocated memory. Similarly the `size` \+ `offset` cannot be larger than the total size of the logical endpoint.

For device memory, i.e., type CU_MEM_LOCATION_TYPE_DEVICE, the memory allocation must have been created on the device specified by `dev`. For host NUMA memory, i.e., type CU_MEM_LOCATION_TYPE_HOST_NUMA, the memory allocation must have been created on the CPU NUMA node closest to `dev`. That is, the value returned when querying CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID for `dev`, must be the CPU NUMA node where the memory was allocated.

For multicast endpoints, the device named by `dev` must have been added to the multicast team via cuLogicalEndpointAddDevice.

For unicast endpoints the device named by `dev` must be the owner device specified during cuLogicalEndpointCreate via CUlogicalEndpointProp::unicast::device.

Externally shareable as well as imported multicast endpoints can be bound only to externally shareable memory. Imported unicast endpoints cannot be bound to any memory.

This call will return CUDA_ERROR_INVALID_VALUE if cuLogicalEndpointQuery has not been called for the logical endpoint to ensure that the endpoint is ready for memory binding.

Note that this call will return CUDA_ERROR_OUT_OF_MEMORY if there are insufficient resources required to perform the bind. This call may also return CUDA_ERROR_SYSTEM_NOT_READY if the necessary system software is not initialized or running. This call may return CUDA_ERROR_ILLEGAL_STATE if the system configuration is in an illegal state. In such cases, to continue using logical endpoints, verify that the system configuration is in a valid state and all required driver daemons are running properly.

CUresult cuLogicalEndpointBindMem ( CUlogicalEndpointId leId, CUdevice dev, cuuint64_t offset, CUmemGenericAllocationHandle memHandle, cuuint64_t memOffset, cuuint64_t size, unsigned long long flags )


Binds memory object represented by a handle to the logical endpoint.

######  Parameters

`leId`
    Logical endpoint to which memory will be associated.
`dev`
    Device on which the memory will be bound to the logical endpoint
`offset`
    Offset into the logical endpoint space.
`memHandle`
    Handle representing a memory allocation.
`memOffset`
    Offset into the memory for the attachment
`size`
    Size of memory that will be bound to the logical endpoint.
`flags`
    Flags for future use, must be zero for now.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED, CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_SYSTEM_NOT_READY, CUDA_ERROR_ILLEGAL_STATE

###### Description

Binds the memory allocation specified by `memHandle` to a logical endpoint represented by `leId` at the offset `offset`. The memory must have been allocated via cuMemCreate. The intended `size` of the bind, the offset in the logical endpoint range `offset` and the offset in the memory handle `memOffset` must be multiples of the value for `bindAlignment` as returned by cuLogicalEndpointGetLimits.

The `size` \+ `memOffset` cannot be larger than the size of the allocated memory. Similarly the `size` \+ `offset` cannot be larger than the total size of the logical endpoint.

For device memory, i.e., type CU_MEM_LOCATION_TYPE_DEVICE, the memory allocation must have been created on the device specified by `dev`. For host NUMA memory, i.e., type CU_MEM_LOCATION_TYPE_HOST_NUMA, the memory allocation must have been created on the CPU NUMA node closest to `dev`. That is, the value returned when querying CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID for `dev`, must be the CPU NUMA node where the memory was allocated.

For multicast endpoints, the device named by `dev` must have been added to the multicast team via cuLogicalEndpointAddDevice.

For unicast endpoints the device named by `dev` must be the owner device specified during cuLogicalEndpointCreate via CUlogicalEndpointProp::unicast::device.

Externally shareable as well as imported multicast endpoints can be bound only to externally shareable memory. Imported unicast endpoints cannot be bound to any memory.

This call will return CUDA_ERROR_INVALID_VALUE if cuLogicalEndpointQuery has not been called for the logical endpoint to ensure that the endpoint is ready for memory binding.

Note that this call will return CUDA_ERROR_OUT_OF_MEMORY if there are insufficient resources required to perform the bind. This call may also return CUDA_ERROR_SYSTEM_NOT_READY if the necessary system software is not initialized or running. This call may return CUDA_ERROR_ILLEGAL_STATE if the system configuration is in an illegal state. In such cases, to continue using logical endpoints, verify that the system configuration is in a valid state and all required driver daemons are running properly.

CUresult cuLogicalEndpointCreate ( CUlogicalEndpointId leId, const CUlogicalEndpointProp* prop )


Creates a logical endpoint with the requested properties and associates it with the logical endpoint id.

######  Parameters

`leId`
    Logical endpoint id that will be associated with the newly created logical endpoint.
`prop`
    Properties of the logical endpoint to create.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED, CUDA_ERROR_OUT_OF_MEMORY

###### Description

This creates a logical endpoint as described by `prop`. The number of participating devices is determined by the CUlogicalEndpointProp::type. If the type is CU_LOGICAL_ENDPOINT_TYPE_UNICAST then CUlogicalEndpointProp::unicast::device specifies the owner device of the unicast logical endpoint. If the type is CU_LOGICAL_ENDPOINT_TYPE_MULTICAST then CUlogicalEndpointProp::multicast::numDevices specifies the number of devices in the multicast logical endpoint team.

Devices can be added to a multicast logical endpoint via cuLogicalEndpointAddDevice. After all the participating devices have been added, a call to cuLogicalEndpointQuery must be made to ensure that the logical endpoint is ready for memory binding and access.

A unicast logical endpoint does not have a notion of adding devices via cuLogicalEndpointAddDevice. However, a call to cuLogicalEndpointQuery must still be made to ensure that the logical endpoint is ready for memory binding and access.

Memory is bound to the logical endpoint via either cuLogicalEndpointBindAddr or cuLogicalEndpointBindMem, and can be unbound via cuLogicalEndpointUnbind. The total amount of memory that can be bound per device is specified by CUlogicalEndpointProp::size. This size must be a multiple of the value for `bindAlignment` as returned by cuLogicalEndpointGetLimits. The maximum size for the logical endpoint cannot exceed the value for `maxSize` as returned by cuLogicalEndpointGetLimits. The bind alignment and maximum size depend on the properties of the logical endpoint.

CUresult cuLogicalEndpointDestroy ( CUlogicalEndpointId leId )


Removes the association of the logical endpoint from the logical endpoint id.

######  Parameters

`leId`
    Logical endpoint id of the logical endpoint to be destroyed.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED

###### Description

Removes the association between the logical endpoint id and the logical endpoint resources. Any memory bound by this process to any device associated with the logical endpoint will be unbound. If this was the last reference to the logical endpoint, all associated resources will be destroyed.

CUresult cuLogicalEndpointExport ( void* handle, CUlogicalEndpointId leId, CUlogicalEndpointIpcHandleType handleType )


Exports a logical endpoint associated with leId to an IPC handle.

######  Parameters

`handle`
    Pointer to the location in which to store the requested handle type.
`leId`
    Logical endpoint id of logical endpoint.
`handleType`
    Type of shareable handle requested. Defines type and size of the handle output parameter.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED, CUDA_ERROR_OUT_OF_MEMORY

###### Description

Given a logical endpoint id `leId`, create a shareable handle `handle` that can be used to share the logical endpoint with other processes. The recipient process can convert the shareable handle back into a logical endpoint id using cuLogicalEndpointImport. The implementation of what this `handle` is and how it can be transfered is defined by the requested handle type in `handletype`.

CUresult cuLogicalEndpointGetLimits ( cuuint64_t* bindAlignment, cuuint64_t* maxSize, const CUlogicalEndpointProp* prop )


Calculates the minimum alignment and the maximum size for the given logical endpoint properties.

######  Parameters

`bindAlignment`
    Minimum alignment granularity of the proposed logical endpoint.
`maxSize`
    Maximum size of the logical endpoint.
`prop`
    Properties of the logical endpoint.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED, CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_SYSTEM_NOT_READY, CUDA_ERROR_ILLEGAL_STATE

###### Description

The `bindAlignment` can be used as a multiple for size and bind offset values. The `maxSize` is the maximum size of the logical endpoint. If `maxSize` is less than CUlogicalEndpointProp:size the user must adjust the request to the smaller value.

CUresult cuLogicalEndpointIdRelease ( CUlogicalEndpointId baseLeId, cuuint32_t count )


Releases a range of logical endpoint ids.

######  Parameters

`baseLeId`
    First logical endpoint id to be released back to the system.
`count`
    Number of logical endpoint ids to release back to the system.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED, CUDA_ERROR_OUT_OF_MEMORY

###### Description

Releases up to `count` logical endpoint ids starting at `baseLeId`. The range of ids represented by `baseLeId`, `baseLeId` \+ `count`) must all be previously reserved. All logical endpoints in the range must be destroyed before they can be released.

[CUresult cuLogicalEndpointIdReserve ( CUlogicalEndpointId* baseLeId, cuuint32_t count )


Reserves a range of logical endpoint ids.

######  Parameters

`baseLeId`
    If cuLogicalEndpointIdReserve returns CUDA_SUCCESS, *baseLeId contains the base logical endpoint id of the reserved logical endpoint id range.
`count`
    The number of logical endpoint ids to reserve.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED, CUDA_ERROR_OUT_OF_MEMORY

###### Description

Reserves a range of logical endpoint ids starting at `*baseLeId` and extending for `count`. The reserved ids can be used to create or import logical endpoints via cuLogicalEndpointCreate or cuLogicalEndpointImport respectively.

CUresult cuLogicalEndpointImport ( CUlogicalEndpointId leId, const void* handle, CUlogicalEndpointIpcHandleType handleType )


Imports a logical endpoint from the given IPC handle and associates it with a logical endpoint id.

######  Parameters

`leId`
    Logical endpoint id that will be used to access the exported logical endpoint.
`handle`
    Shareable handle representing the logical endpoint that is to be imported.
`handleType`
    Handle type of the exported handle

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED, CUDA_ERROR_OUT_OF_MEMORY

###### Description

Imports a logical endpoint from the given IPC `handle` and associates it with the logical endpoint id specified by `leId`.

If the current process cannot support the logical endpoint described by the shareable handle, this API will error as CUDA_ERROR_NOT_SUPPORTED. If `handle` is of type CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC and the importer process does not have access permissions, then CUDA_ERROR_NOT_PERMITTED will be returned

CUresult cuLogicalEndpointQuery ( CUlogicalEndpointId leId, cuuint32_t count, int* queryStatus )


Determines if all logical endpoints in the range have been successfully constructed.

######  Parameters

`leId`
    First logical endpoint ID to be queried.
`count`
    Number of logical endpoints IDs to be queried.
`queryStatus`
    Status of the logical endpoints. Returns 0 if any logical endpoint in the given range is not fully constructed, and non-zero if all logical endpoints in the given range are fully constructed.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED, CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_SYSTEM_NOT_READY, CUDA_ERROR_ILLEGAL_STATE

###### Description

Queries the driver to determine if all logical endpoints in the given range starting at `leId` and extending for `count` have been successfully constructed.

Provides a mechanism to ensure that it is safe to begin using a logical endpoint ID. Using a logical endpoint ID before verifying that it is fully constructed can result in undefined behavior.

This is not a blocking API, it returns immediately with a `queryStatus` of 0 if any logical endpoint ID in the given range is not fully constructed, and a non-zero value otherwise.

CUresult cuLogicalEndpointUnbind ( CUlogicalEndpointId leId, CUdevice dev, cuuint64_t offset, cuuint64_t size )


Unbinds any binding at offset from the logical endpoint.

######  Parameters

`leId`
    Logical endpoint id representing a logical endpoint.
`dev`
    Device on which the memory is bound to the logical endpoint
`offset`
    Offset into the logical endpoint.
`size`
    Desired size to unbind.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED, CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_SYSTEM_NOT_READY, CUDA_ERROR_ILLEGAL_STATE

###### Description

Unbinds any memory allocations bound to the logical endpoint on `dev` at `offset` and up to the given `size`. The intended `size` of the unbind and the offset in the logical endpoint range `offset` must be multiples of the value for `bindAlignment` as returned by cuLogicalEndpointGetLimits.

The `offset` must correspond to a value specified during a bind call. The `size` must either match the bind call of the offset or be the combined `size` of multiple bind calls. The `size` \+ `offset` must fully enclose all bindings that are covered.
