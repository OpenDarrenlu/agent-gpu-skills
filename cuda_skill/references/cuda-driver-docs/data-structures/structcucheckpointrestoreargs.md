# 7.8. CUcheckpointRestoreArgs

**Source:** structCUcheckpointRestoreArgs.html#structCUcheckpointRestoreArgs


### Public Variables

CUcheckpointGpuPair * gpuPairs

unsigned int gpuPairsCount

char reserved[64-sizeof(CUcheckpointGpuPair *)-sizeof(unsigned int)]


### Variables

CUcheckpointGpuPair * CUcheckpointRestoreArgs::gpuPairs


Pointer to array of gpu pairs that indicate how to remap GPUs during restore

unsigned int CUcheckpointRestoreArgs::gpuPairsCount


Number of gpu pairs to remap

char CUcheckpointRestoreArgs::reserved[64-sizeof(CUcheckpointGpuPair *)-sizeof(unsigned int)]


Reserved for future use, must be zeroed

* * *
