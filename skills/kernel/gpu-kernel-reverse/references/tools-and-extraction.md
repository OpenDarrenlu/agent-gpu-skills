# 工具、提取与证据保存

## 版本与路径

记录 `nvcc`、`ptxas`、`cuobjdump`、`nvdisasm` 的 `--version`，profiling 时也记录 `ncu`。用本地 `--help` 核对选项，在线最新版不代表本地支持。无 GPU 时通常仍可编译和反汇编。

基础脚本只用 Python 标准库和已安装工具，不安装解码器、不启动目标二进制：

```bash
python3 /path/to/gpu-kernel-reverse/scripts/collect_evidence.py --help
python3 /path/to/gpu-kernel-reverse/scripts/collect_evidence.py /data/target.cubin \
  --out /work/reverse/original --function exact_mangled_entry
python3 /path/to/gpu-kernel-reverse/scripts/collect_evidence.py /data/kernels.fatbin \
  --out /work/reverse/container --extract selected_image_name
```

以下路径/入口是需替换的示例；在与输入分离的新目录操作。

`--out` 必须尚不存在。脚本写 `manifest.json` 和逐命令原始输出，返回码 `0/3/1/2` 分别表示收集 complete/partial/failed/调用无效；`0` 也不代表逆向成功。函数筛选只作用于 cuobjdump SASS，nvdisasm 仍覆盖整个 cubin；提取的文件需另选并单独分析。默认逐命令超时 60 秒，可按输入规模调整，超时保留已产生的输出。

## 容器与裸 cubin

```bash
cuobjdump --list-elf /absolute/path/library.so
cuobjdump --list-ptx /absolute/path/library.so
# 在新 extraction 目录运行；selector 来自实际 --list-elf 输出。
cuobjdump --extract-elf selected_image_name /absolute/path/library.so
```

selector 是子串，可能提取多个文件，记录实际输出及 hash，再按 caller/profile 选择。大型库优先提取所需 image；`all` 用于确需全部枚举的任务。处理 relocatable fatbin 时核对 `--all-fatbin`，未链接对象不等于可启动程序。

- cuobjdump 支持 host executable/fatbin/cubin；nvdisasm 普通输入是单个 cubin，不要直接传 `.so`。
- `--dump-ptx` 只提取保留的 PTX，**不从 SASS 恢复 PTX**；没有 PTX 不等于 cubin 损坏。
- list/extract 选项会覆盖其他选项的行为，按独立命令执行，保存 stderr/返回码。
- 使用 `--dump-elf-symbols`，不要照搬不存在的 `--list-symbols`。
- cuobjdump `--function` 接收名称；nvdisasm `--cuda-function-index` 接收 symbol index，不能互换。
- 同名 kernel 可能有多个 SM/特化；名字相同不意味着同一基准。

## 单个 cubin

```bash
cuobjdump --dump-elf target.cubin
cuobjdump --dump-elf-symbols target.cubin
cuobjdump --dump-resource-usage target.cubin
cuobjdump --dump-sass --function exact_mangled_entry target.cubin
nvdisasm --print-code --print-instruction-encoding target.cubin
nvdisasm --output-control-flow-graph-with-basic-blocks target.cubin
nvdisasm --print-life-ranges target.cubin
```

支持时另存 `nvdisasm --emit-json target.cubin`。记录 SchemaVersion/Producer，按真实 schema 解析。当前官方 JSON 的 PC 为 `start + instruction_index * 16`，**空 opcode 代表地址缺口，不能先删掉再编号**。不要硬套到旧架构文本/控制字格式。JSON 的 SM 数字未必足够区分所有特性后缀，结合 ELF flags、特性与 build/caller 证据。

CFG/liveness 失败时保存错误，可用 `--no-dataflow` 获取原始反汇编，但不能声称得到完整 CFG/liveness。缺重定位、间接跳转、device call 保持待解。`-G` 只可辅助单独实验，不能用它重编译后的结构替代原优化 kernel。

保留 `.nv.info*`、`.nv.constant*`、`.nv.shared*`、symbols、relocations、debug/lineinfo。参数元数据无法独自证明 pointer 指向类型；最大寄存器编号不等于分配或活跃寄存器数。

## 辅助工具

| 缺口 | 工具 | 边界 |
|---|---|---|
| 语义 lifting/Ghidra | `edmcman/nvdisasm-re` | 核对 SM、opcode **和 modifier** 覆盖；opaque 保持未解；readable FP 不可用于 bitwise 结论 |
| 编译器模式 | `florianmattana/sass-king` | 证据主要来自 SM120/120a；跨架构需重做实验 |
| SASS/CFG/FA 结构索引 | `alexander-hm/sass-analyzer` | QK/PV/softmax 等 flag 是启发式，不是证明 |
| 编码/控制字段 | `redplait/denvdis` | 架构表和模型可能不完整，与官方 dump/实验交叉核对 |
| 动态地址/分支/指令数 | NVIDIA `NVBit` / caller 跟踪 | 核对兼容性；插桩扰动时序，原始性能另测 |
| 改 SASS 的受控实验 | `cloudcores/CuAssembler` | 公开说明以 Pascal–Ampere 为主，不默认支持 Hopper/Blackwell；不是源码重建结果 |

第三方工具固定 commit，核对覆盖，只安装当前步骤需要的工具；不复制整个 opcode 库或上游 skill。来源见 [sources.md](sources.md)。

nvdisasm-re emulator 可用于已覆盖语义的差分，但内存/调度模型不等于 GPU。检查跨 CTA 全局共享、atomic、弱内存、异步操作与未映射读取行为；仿真一致不能证明无 race、无越界或同等性能。独立 oracle 不应共享被测 lifter 的全部实现。

## 失败时

- 解码失败：核对真实 SM、容器/压缩层、toolkit 支持，再用兼容版本；不能换 SM 直到不报错。
- 缺 lineinfo：转向 CFG、数据流、参数和受控编译，不生成不存在的源码定位。
- 原 kernel 不能执行：查 ABI/依赖/硬件，不用候选自比替代。
- profiler 缺权限：仍可 CUDA event 计时，明确缺 counter，不伪造 stall 或缓存命中率。
