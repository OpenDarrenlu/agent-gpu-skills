# 来源、用途与限制

检索/阅读日期：2026-10-24。通过联网读取 NVIDIA 官方文档、用户提供的六个项目，并使用 GitHub repository search 补充检索。本文是工作流综合，不是上游文档副本。GitHub HEAD 与在线文档会变化；实际使用工具时固定 commit/版本，复查目标 SM 和指令形式的覆盖。

## 首选官方依据

| 来源 | 本 skill 采用的内容 |
|---|---|
| [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html) | cubin/fatbin、cuobjdump/nvdisasm 输入差别、真实选项、CFG/liveness、JSON schema 与地址缺口 |
| [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) | 位宽/舍入、memory consistency、warp MMA/WGMMA、cp.async/mbarrier/TMA/tcgen05、target notes；PTX 不是完整 SASS 规范 |
| [CUDA Driver API: Execution Control](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html) | launch 参数传递、function attributes、参数 offset/size 查询、扩展 launch |
| [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) | replay、serialization、clock/cache control、metric 的解释边界 |
| [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) | memcheck/racecheck/initcheck/synccheck 的用途与范围 |
| [CUTLASS 3.x GEMM API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html) | collective mainloop/epilogue、builder、kernel/device 层级 |
| [CUTLASS Blackwell Functionality](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html) | Blackwell 具体 target、功能和 kernel family 的适用条件 |

本次本地工具为 CUDA 13.1.115，实际核对了 `cuobjdump --help`/`nvdisasm --help`；在线资料中涉及更新版本的功能必须再次核对本地支持。创建 skill 时没有可连接的 GPU driver，因此本地静态验证不包含 GPU 正确性/性能证明。

## 用户提供的来源

### 1. cuobjdump-lowering-inspector

[rdspring1/thrashl skill](https://github.com/rdspring1/thrashl/blob/HEAD/claude/skills/cuobjdump-lowering-inspector/SKILL.md)：采用受控 snippet→PTX/SASS、记录 target/commands、分开观察与推断的思路。

没有照搬其中的 `cuobjdump --list-symbols`，本地实际选项为 `--dump-elf-symbols`；也不直接将 executable 传给 nvdisasm，不以 debug 重编译作为原性能证据。文中 dtype/架构概括需回到官方 instruction target notes。

### 2. CTF CUDA reverse

[chhaynee/ctf-skills](https://github.com/chhaynee/ctf-skills/blob/HEAD/ctf-reverse/cuda-gpu-kernels.md)：采用先恢复 host/device 边界、参数/缓冲区/launch 表、位宽和 CPU scalar model 的方法。CTF 的逐线程模型不能直接覆盖生产 tensor core 与跨 CTA 协议；本 skill 增加并行布局和 GPU 性能验证。

### 3. nvdisasm-re

[项目](https://github.com/edmcman/nvdisasm-re)、[semantics](https://github.com/edmcman/nvdisasm-re/blob/HEAD/docs/semantics.md)、[emulation](https://github.com/edmcman/nvdisasm-re/blob/HEAD/docs/emulation.md)：目前最直接相关的语义 lifting/Ghidra/assembler/decoder 路径。阅读时 README 描述基于 CUDA 13.4 的 machine descriptions；这不是对本地 13.1 的验证。

区分可解码、native/opaque 语义、可仿真、真实硬件验证。阅读时硬件证据主要为 SM89/SM90，其余不能当已在硬件验证。readable FP 模式会简化舍入/FTZ/NaN 等；CPU emulator 的跨 CTA 全局状态和弱内存模型有边界，不能验证任意 tensor/async kernel 或预测真实性能。语义覆盖以对应 commit 的 coverage 和具体 modifier 为准。

### 4. svod tk-kernels

[npatsakula/svod skill](https://github.com/npatsakula/svod/blob/HEAD/.claude/skills/tk-kernels/SKILL.md)：采用离线编译对比、按主循环步比较指令、区分模型预测与实测、先正确再调优的思路。它是 **SVOD tile-kernel 工作流**，不是通用反编译器，也不能因名字 tk 将它当成 ThunderKittens skill；不移植项目专用环境变量和性能数字。

### 5. SASS King

[florianmattana/sass-king](https://github.com/florianmattana/sass-king)：采用“一次改变一个变量”的 microkernel corpus、带证据边界的 signature、生产 kernel pattern matching。当前重点 SM120/SM120a；跨 SM 使用必须重新验证，不能把路线图当成已实现能力。

### 6. sass-analyzer

[alexander-hm/sass-analyzer](https://github.com/alexander-hm/sass-analyzer)：用于多入口解析、指令分类、CFG 和 FA 类 pattern 索引。HGMMA/EX2/warp split 等启发式有助缩小调查范围，但并不证明 QK/PV/softmax 或完整布局。

## 补充检索

- [NVIDIA NVBit](https://github.com/NVlabs/NVBit)：研究型动态二进制插桩，适合 memory trace、动态指令与分支观测；按安装版本确认架构支持，不用插桩时间作为原始性能。
- [cloudcores/CuAssembler](https://github.com/cloudcores/CuAssembler)：SASS assembler 与受控实验工具；阅读时明确说明主要支持 Pascal/Volta/Turing/Ampere，不能默认覆盖 Hopper/Blackwell，更不负责生成 CUDA 源码。
- [redplait/denvdis](https://github.com/redplait/denvdis)：由 SASS King 的 related work 进一步找到并阅读；编码字段、predicate/register tracking、constant-bank 线索用于交叉核对，项目自身说明解码/解析并不完整。

部分搜索引擎返回了脚本跳转页，因此工具选型以实际读取的官方文档/仓库内容为依据，不把未能打开的搜索结果当作证据。
