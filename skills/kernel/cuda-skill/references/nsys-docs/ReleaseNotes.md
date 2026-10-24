---
url: https://docs.nvidia.com/nsight-systems/ReleaseNotes/index.html
---

# Release Notes

Release notes and known issues.

## What’s New

### Nsight Systems 2026.5.1 Highlights:

  * NVIDIA Rubin GPU architecture support

    * Performance metrics support for GR100: General Metrics and Compute Triage

  * Windows on Arm Support

    * Trace arm64 native and x64 emulated applications

    * NVIDIA RTX Spark CPU core PMU metrics sampling

    * NVIDIA RTX Spark GPU power and temperature metrics

    * NVIDIA Video Codec SDK tracing

  * CUDA improvements

    * CUDA 13.4 support

    * CUDA work queue information — Added channel count and channel ID details

    * Multi-process capture ranges — Improved CUPTI buffer flushing when profiling ends through cudaProfilerStop()

    * CUDA kernel details now show if execution model was Tile IR or SIMT

    * NVTX ranges can now be projected onto the “All Streams” row, in addition to individual stream rows.

    * Nsight Systems now displays readable names for cuTile kernels.

    * CUDA workloads submitted through Direct3D 12 CUDA-in-Graphics (CiG) streams are now traced and displayed in the timeline.

  * PyTorch profiling

    * The new `--pytorch=functions-trace-shapes` option captures PyTorch function calls and stores tensor shapes efficiently as binary payloads.

    * Use the existing `--pytorch=functions-trace` option when call ranges are sufficient. It captures calls without shapes, reducing profiling overhead and trace size.

  * Graphics profiling improvements

    * Added interception and display support for Direct3D 12 CiG streams.

    * Added support for tracing Vulkan Reflex applications that use `VK_NV_low_latency2`.

    * Added support for tracing applications that use Direct3D 12 Runtime Bypass.

    * Added support for the `ID3D12Device15`, `ID3D12ApplicationIdentity`, and `ID3D12RuntimeValidationControl` Direct3D 12 interfaces.

    * Added support for the `VK_KHR_maintenance10`, `VK_EXT_memory_decompression`, and `VK_NV_compute_occupancy_priority` Vulkan extensions.

  * Agentic AI Skill Pack

    * Preview release focused on graphics performance analysis and frame stutter analysis.

    * Delivers faster answers with lower token consumption and improved answer accuracy.

    * Agent-agnostic design for use with leading AI agents and models.

  * Network profiling improvements

    * Added collection of Spectrum-X high frequency performance counters. Counters are read via DTS (DOCA Telemetry Service) and the switch Cumulus OS. This enables users without root privileges to collect high-frequency NIC metrics on systems where DTS is installed.

    * Improved NCCL recipes, including a new NCCL straggler recipe to pinpoint slow MPI ranks.

  * Storage profiling improvements

    * Added an S3 object storage access summary recipe.

    * Added NVIDIA SCADA metrics.

  * CPU profiling improvements

    * Extended CPU uncore profiling on NVIDIA Vera and NVIDIA Thor with support for uncore events, uncore derived metrics, and uncore event filters.

    * Added CPU core single-pass metric sets based on the Arm Topdown methodology, enabling related metrics to be collected concurrently on supported CPUs.

  * Time synchronization

    * Nsight Systems now includes vClock, an unprivileged virtual clock designed to provide accurate time synchronization across multiple cluster nodes. This enables Nsight Systems to align distributed trace files collected across different cluster or cloud nodes.

  * Export

    * NVTX binary payloads can now be exported as columnar fields using a dynamic schema.

  * Plugins

    * DCGM — Added `--power-profile` to see if it is supported, requested, or enforced

  * NVIDIA Nsight Cloud

    * Nsight Streamer updates.

    * Docker updates.

    * Kubernetes updates.

    * Nsight Operator updates.

    * Improvements for analysis, OpenTelemetry, and Dynamo.

    * Learn more through the new docs site.


## Known Issues

### General Issues

  * When profiling Spectrum-X switches there is an approximate delay of 25s until the first sample is sent from the switches. Hence, profile for more than 25s when collecting Ethernet switch data.

  * If you see high branch mis-predicts and instruction TLB refills, we suggest you try [NVIDIA/cpu-code-locality-tool](https://github.com/NVIDIA/cpu-code-locality-tool) to further optimize your code for NVIDIA Grace’s code caches.

  * RoCE counters for ConnectX NICs are not available with version 2025.6.

  * Nsight Systems trace features that require process injection (e.g. OSRT, NVTX, CUDA trace) may fail to collect data and cause unstable behavior when profiling applications that use seccomp to restrict system calls, such as Linux’s file utility. The injection library may violate the process’s seccomp policy, causing thread/process termination and/or other unstable behaviors like leaving the application hanging in a zombie process state. Disable seccomp in the target application if possible or use only non-injection-based profiling features (e.g. CPU sampling, GPU metrics sampling) for those applications.

  * The current release of Nsight Systems CLI doesn’t support naming a session with a name longer than 127 characters. Profiling an executable with a name exceeding 111 characters is also unsupported by the `nsys profile` command. Those limitations will be removed in a future version of the CLI.

  * Nsight Systems 2020.4 introduces collection of thread scheduling information without full sampling. While this allows system information at a lower cost, it does add overhead. To turn off thread schedule information collection, add `--cpuctxsw=none` to your command line or turn off in the GUI.

  * Profiling greater than 5 minutes is not officially supported at this time. Profiling high activity applications, on high performance machines, over a long analysis time can create large result files that may take a very long time to load, run out of memory, or lock up the system. If you have a complex application, we recommend starting with a short profiling session duration of no more than 5 minutes for your initial profile. If your application has a natural repeating pattern, often referred to as a frame or an iteration, you will typically only need a few of these. This suggested limit will increase in future releases.

  * Attaching or re-attaching to a process from the GUI is not supported with the x86_64 Linux target. Equivalent results can be obtained by using the interactive CLI to launch the process and then starting and stopping analysis at multiple points.

  * To reduce overhead, Nsight Systems traces a subset of API calls likely to impact performance when tracing APIs rather than all possible calls. There is currently no way to change the subset being traced when using the CLI. See respective library portion of this documentation for a list of calls traced by default. The CLI limitation will be removed in a future version of the product.

  * There is an upper bound on the default size used by the tool to record trace events during the collection. If you see the following diagnostic error, then Nsight Systems hit the upper limit.

        Reached the size limit on recording trace events for this process.
               Try reducing the profiling duration or reduce the number of features
               traced.


  * When profiling a framework or application that uses CUPTI, like some versions of TensorFlow(tm), Nsight Systems will not be able to trace CUDA usage due to limitations in CUPTI. These limitations will be corrected in a future version of CUPTI. Consider turning off the application’s use of CUPTI if CUDA tracing is required.

  * Tracing an application that uses a memory allocator that is not thread-safe is not supported.

  * Tracing OS Runtime libraries in an application that preloads glibc symbols is unsupported and can lead to undefined behavior.

  * Nsight Systems cannot profile applications launched through a virtual window manager like GNU Screen.

  * Using Nsight Systems MPI trace functionality with the Darshan runtime module can lead to segfaults. To resolve the issue, unload the module.




    module unload darshan-runtime


  * Profiling MPI Fortran APIs with MPI_Status as an argument, e.g. MPI_Recv, MPI_Test[all], MPI_Wait[all], can potentially cause memory corruption for MPICH versions 3.0.x. The reason is that the MPI_Status structure in MPICH 3.0.x has a different memory layout than in other MPICH versions (2.1.x and >=3.1.x have been tested) and the version (3.3.2) we used to compile the Nsight Systems MPI interception library.

  * Using `nsys export` to export to an SQLite database will fail if the destination filesystem doesn’t support file locking. The error message will mention:




    std::exception::what: database is locked


  * On some Linux systems when VNC is used, some widgets can be rendered incorrectly, or Nsight Systems can crash when opening Analysis Summary or Diagnostics Summary pages. In this case, try forcing a specific software renderer: `GALLIUM_DRIVER=llvmpipe nsys-ui`

  * Due to [a known bug in Open MPI 4.0.1](https://github.com/open-mpi/ompi/issues/6648), target application may crash at the end of execution when being profiled by Nsight Systems. To avoid the issue, use a different Open MPI version, or add `--mca btl ^vader` option to `mpirun` command line.

  * The multiprocessing module in Python is commonly used by customers to create new processes. On Linux, the module defaults to using the “fork” mode where it forks new processes, but does not call exec. According to the POSIX standard, fork without exec leads to undefined behavior and tools like Nsight Systems that rely on injection are only allowed to make async-signal-safe calls in such a process. This makes it very hard for tools like Nsight Systems to collect profiling information. See <https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods>

Use the set_start_method in the multiprocessing module to change the start method to “spawn” which is much safer and allows tools like Nsight Systems to collect data. See the code example given in the link above.

The user needs to ensure that processes exit gracefully (by using close and join methods, for example, in the multiprocessing module’s objects). Otherwise, Nsight Systems cannot flush buffers properly and you might end up with missing traces.

  * When the CLI sequence launch, start, stop is used to profile a process-tree, LinuxPerf does a depth first search (DFS) to find all of the threads launched by the process-tree before programming the OS to collect the data. If, during the DFS, one or more threads are created by the process tree, it is possible those threads won’t be found and LinuxPerf would not collect data for them.

Note that once a thread is programmed via perf_event_open, any subsequent children processes or threads generated by that thread will be tracked since the perf_event_open inherit bit is set.

No other CLI command sequence suffers from this possible issue. Also, if a systemwide mode is used, the issue does not exist.


### vGPU Issues

  * When running Nsight Systems on vGPU you should always use the profiler grant. See [Virtual GPU Software Documentation](https://docs.nvidia.com/grid/latest/grid-vgpu-user-guide/index.html#enabling-cuda-toolkit-profilers-vgpu) for details on enabling NVIDIA CUDA Toolkit profilers for NVIDIA vGPUs. Without the grant, unexpected migrations may crash a running session, report an error and abort. It may also silently produce a corrupted report which may be unloadable or show inaccurate data with no warning.

  * Starting with vGPU 13.0, device level metrics collection is exposed to end users even on vGPU. Device level metrics will give info about all the work being executed on the GPU. The work might be in the same VM or some other VM running on the same physical GPU.

  * As of CUDA 11.4 and R470 TRD1 driver release, Nsight Systems is supported in a vGPU environment which requires a vGPU license. If the license is not obtained after 20 minutes, the tool will still work but the reported GPU performance metrics data will be inaccurate. This is because of a feature in vGPU environment which reduces performance but retains functionality as specified in [Grid Licensing User Guide](https://docs.nvidia.com/grid/latest/grid-licensing-user-guide/index.html#software-enforcement-grid-licensing).


### Docker Issues

  * In a Docker, when a system’s host utilizes a kernel older than v4.3, it is not possible for Nsight Systems to collect sampling data unless both the host and Docker are running a RHEL or CentOS operating system utilizing kernel version 3.10.1-693 or newer. A user override for this will be made available in a future version.

  * When `docker exec` is called on a running container and stdout is kept open from a command invoked inside that shell, the exec shell hangs until the command exits. You can avoid this issue by running with `docker exec --tty`. See the bug reports at:

  * [moby/moby#33039](https://github.com/moby/moby/issues/33039)

  * [drud/ddev#732](https://github.com/drud/ddev/issues/732)


### CUDA Trace Issues

  * When using CUDA graphs that start and end on different GPUs, Nsight Systems may show incorrect end timestamp for the CUDA graph activity, or encounter “Wrong Event Order” error when generating the report, if CUDA graph-level trace is enabled (which is the default mode for CUDA trace) and CUDA driver version < 12.0. To avoid the issue, upgrade CUDA driver to at least 12.0, or use `--cuda-graph-trace=node` to switch to node-level trace mode, or add the following settings in the config.ini file:

        CuptiUseRawGpuTimestamps=false


  * When using “cudaGraphNodeSetEnabled()” to enable/disable CUDA graphs in the middle of collection, Nsight Systems may encounter “Wrong Event Order” error when generating the report, if CUDA graph-level trace is enabled (which is the default mode for CUDA trace) and CUDA driver version < 12.0. To avoid the issue, upgrade CUDA driver to at least 12.0, or use `--cuda-graph-trace=node` to switch to node-level trace mode.

  * CUDA hardware trace, also called Hardware Event System (HES) trace, is the default CUDA trace method on supported systems. Hardware trace is not supported for MPS workloads, MIG partitions, vGPU or virtual machine environments, some Confidential Compute configurations, unsupported GPUs, or incompatible driver or CUPTI combinations. In these cases, Nsight Systems may automatically fall back to the legacy software-instrumented CUDA trace and report the trace method in the Diagnostics Summary page. Use `--trace=cuda-sw` to request software CUDA trace explicitly.

  * If a system is in the CC-DevTools mode (CC stands for Confidential Compute) and Nsight Systems is used to trace CUDA in an application using libcrypto, Nsight Systems may crash when the application exits. The crash occurs during the application teardown and causes profiler data loss. To avoid losing CUDA tracing data in this situation, a few options exist.

1\. Add a cudaDeviceSynchronize call to the application immediately before the application exits. Nsight Systems flushes all available data on a synchronization and data loss will be avoided.

2\. Add a cudaProfilerStop call to the application immediately before the application exits and set the Nsight Systems `--flush-on-cudaprofilerstop` switch to true. In this case, Nsight Systems will flush all available data at this point.

3\. End the profile before the application exits using one of many Nsight Systems mechanisms to end a profile. For example;

>     * Set a collection duration that ends before the application exits (see the `--duration` switch).
> 
>     * Use a capture range to only collect data during a specific period of the application’s execution (see the `--capture-range` switch).
> 
>     * Set the CUDA flush interval to frequently flush data during a profile. Any data collected after the last flush and before the application’s exit will likely be lost. Note that frequent CUDA flushes will increase profiling overhead.
> 
>     * Use the Nsight Systems CLI’s `start`, `launch`, `stop` commands to manually start and stop a collection before the application exits.

  * The cudaMemPrefetchAsync() API allows the user to specify a stream to enqueue a memory prefetch operation. However, Nsight Systems does not get the stream information for UVM page migrations from the UVM backend. Thus, Nsight Systems cannot show stream information correctly correlated with a cudaMemPrefetchAsync() API call. This will be fixed in a future version.

  * When using CUDA Toolkit 10.X, tracing of DtoD memory copy operations may result in a crash. To avoid this issue, update CUDA Toolkit to 11.X or the latest version.

  * Nsight Systems will not trace kernels when a CDP (CUDA Dynamic Parallelism) kernel is found in a target application on Volta devices or later.

  * On Tegra platforms, CUDA trace requires root privileges. Use the **Launch as root** checkbox in project settings to make the profiled application run as root.

  * If the target application uses multiple streams from multiple threads, CUDA event buffers may not be released properly. In this case, you will see the following diagnostic error:

        Couldn't allocate CUPTI bufer x times. Some CUPTI events may
               be missing.


Please contact the Nsight Systems team.

  * In this version of Nsight Systems, if you are starting and stopping profiling inside your application using the interactive CLI, the CUDA memory allocation graph generation is only guaranteed to be correct in the first profiling range. This limitation will be removed in a future version of the product.

  * CUDA GPU trace collection requires a fraction of GPU memory. If your application utilizes all available GPU memory, CUDA trace might not work or can break your application. As an example cuDNN application can crash with `CUDNN_STATUS_INTERNAL_ERROR` error if GPU memory allocation fails.

  * For older Linux kernels, prior to 4.4, when profiling very short-lived applications (~1 second) that exit in the middle of the profiling session, it is possible that Nsight Systems will not show the CUDA events on the timeline.

  * When more than 64k serialized CUDA kernels and memory copies are executed in the application, you may encounter the following exception during profiling:

        InvalidArgumentException: "Wrong event order detected"


Please upgrade to the CUDA 9.2 driver at minimum to avoid this problem. If you cannot upgrade, you can get a partial analysis, missing potentially a large fraction of CUDA events, by using the CLI.

  * On Vibrante, when running a profiling session with multiple targets that are guest VMs in a CCC configuration behind a NAT, you may encounter an error with the following text during profiling:

        Failed to sync time on device.


Please edit the group connection settings, select **Targets on the same SoC** checkbox there and try again.

  * When using the 455 driver, as shipped with CUDA Tool Kit 11.1, and tracing CUDA with Nsight Systems you many encounter a crash when the application exits. To avoid this issue, end your profiling session before the application exits or update your driver.


### Multi Report Analysis Issues

  * Be aware that setting up Dask analysis on your workstation requires some additional work on the system. For small data inputs, running the recipes without Dask may be faster.


## Deprecated Features

  * Nsight Systems versions, starting with 2026.4 do not provide support for the legacy `json` and `text` export options. Use `jsonlines` instead. If you need to use the old exports, we recommend you use an older version, downloadable from <https://developer.nvidia.com/gameworksdownload>.

  * Nsight Systems versions, starting with 2026.2 have changed the available options for `nic-metrics` on the command line from `true` and `false` to `lf`, `hf` and `none`. Currently the old options continue to work with the old behavior, but they will be removed in a future version of the product.

  * Nsight Systems versions, starting with 2026.1 do not provide support for the legacy `--nvprof` CLI option. If you need to convert a script that uses this option, see an archived version of the documentation for the Nsight Systems CLI equivalent options.

  * Nsight Systems versions, starting with 2025.4 do not provide support for Pascal or Volta architectures, we recommend you use an older version, downloadable from <https://developer.nvidia.com/gameworksdownload>.

  * Nsight Systems versions, starting with 2024.2 do not provide support for Power PC, we recommend you use an older version, downloadable from <https://developer.nvidia.com/gameworksdownload>.

  * Nsight Systems versions, starting with 2024.4 do not provide support for cuBLAS versions prior to 11.4. If you cannot update your cuBLAS, we recommend you use an older version, downloadable from <https://developer.nvidia.com/gameworksdownload>.