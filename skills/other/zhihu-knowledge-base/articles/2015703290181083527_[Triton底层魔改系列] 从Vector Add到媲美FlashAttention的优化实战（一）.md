# [Triton底层魔改系列] 从Vector Add到媲美FlashAttention的优化实战（一）

**作者**: ZhangZVibe Coding忠实拥趸手撸醋打蒜籽

**原文链接**: https://zhuanlan.zhihu.com/p/2015703290181083527

---

我一直有个问题：市面上的DSL都声称比Triton快好几倍。这几倍怎么来的？Triton要达到这个性能真的很难吗？

我打算用一个系列来回答这些问题。系列会包含两部分：

Triton编译器的内部工作原理
更通用的kernel优化思路

同时我基于Triton开发一套自己的DSL和compiler: TeraXlang，目的是：1. 增加底层工作流程的分析工具，2. 提升Triton性能的优化器。这套工具目前能让Triton在attention效率上媲美FlashAttention，还能辅助实现高效的MLA和NSA代码。支持Hopper和Blackwell。(欢迎查看repo https://github.com/deciding/txl， pip install teraxlang 进行安装)

第一课：Vector Add

从最基础的例子开始——Vector Add，GPU编程的Hello World。

代码
import teraxlang as txl
import torch
import triton
import triton.language as tl

@txl.jit()
def add_kernel(
    x_ptr, y_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    bid = txl.bid(0)
    block_start = bid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    
    # 减去sum是一个额外的reduction操作，目的是让大家看看底层的ir是如何处理这些op的
    s = tl.sum(output)
    output -= s
    
    tl.store(output_ptr + offsets, output, mask=mask)

运行方式：Modal云服务（每月30美元额度），代码在 我的repo的docker/tutorials/vector_add.py。使用modal最大的好处是不需要安装任何其他包，只需要 pip install modal即可，所有的安装都发生在container里面。而且也不需要担心每次创建container会花时间，他会把你的container镜像保存好，下次基本是秒开（不排队等b200的情况下）

运行后会在volume里生成IR文件（ttir、ttgir、llir、ptx），我的代码会自动调用 txl.tools.generate_htmls 生成HTML查看器。这也是为什么需要安装 teraxlang。

为什么要讲Vector Add？

因为这是理解Triton工作流最简单的入口。通过这个例子可以讲清楚：

Python代码怎么变成TTIR
TTIR怎么变成TTGIR
TTGIR怎么变成LLIR，PTX
每一步分别做了什么优化
Triton编译流程
Python → TTIR → TTGIR → LLIR → PTX → SASS
各阶段作用
阶段	含义	主要工作
TTIR	Triton Intermediate Representation	类型推导、基本算子融合
TTGIR	Triton GPU IR	GPU特有优化、shared memory分配、warp同步
LLIR	LLVM IR	通用优化、寄存器分配、循环展开
PTX	Parallel Thread Execution	虚拟ISA，线程级并行、内存层次
SASS	NVIDIA Assembly	实际GPU指令

那么如果我想了解每个阶段具体都优化了啥呢？

Option1: 你可以访问TeraXLang IR Viewer， 或者打开https://deciding.github.io/txl选 Tools -> IR VIewer. 看优化前后的代码对照（支持ttir ttgir llir ptx）

TeraXLang IR Viewer (https://deciding.github.io/txl/tools/ir-viewer.html)

Option2: 采用我tutorial的方式modal跑完了之后内嵌teraxlang的generate_htmls的工具调用

Option3:

在txl.jit（原tl.jit）上添加参数看diff

@txl.jit(diff_mode='ttir', diff_select=0)

这里 diff_mode可以是 ttir/ttgir/llir，diff_select就是里面的分支。

统一一下术语：ttir/ttgir/llir 是优化过程中不同的stage，而每个stage不是单独的一步优化，它是由一堆pass组成的。比如ttir是这些（带txl后缀的是teraxlang独有的）

ttgir passes：

llir passes：

这个功能方便的地方就是会输出每一个具体的pass，到底什么被优化了，比如这个ttgir的第一步就是会给每个tensor加上推理出来的layout信息（以后再讲）：

这里希望不要吓到大家，虽然信息看上去很多，但是我尽量只讲最重要的。而且这篇文章作为系列的试水，我不可能直接上强度。

其实大量的pass其实主要是编译器优化，并不是专门的kernel性能优化，比如Dead Code Elimination（DCE），Common Subexpression Elimination（CSE），这些除非你和我一样要自行修改compiler，不然不必了解。

重点说下TTGIR。这个阶段是Triton真正发挥价值的地方：

把数据放到shared memory
做TMA（Tensor Memory Accelerator）优化
Warp-level reduction
Persistent kernel调度

FlashAttention为什么快？其实是可以在TTGIR阶段做了大量手动优化（这个以后再说）

工具介绍
1. generate_htmls

自动扫描目录下所有IR文件，生成HTML查看器：

python -m teraxlang.tools.build_binding_view <ir目录> <py源文件>

支持：.ttir、.ttgir、.ptx

2. 在线IR Viewer

网页版，不需要安装，直接上传IR文件： http://deciding.github.io/txl/tools/ir-viewer.html

或者 TeraXLang API 点击 IR Viewer

功能：

Drag & Drop上传
左面板IR代码，右面板Python代码
点击任意行跳转到对应绑定
绿线=IR绑定到Python，橙线=Python绑定到IR

这个工具对分析编译过程特别有用。比如你想知道某行Python代码对应的PTX是什么，直接点一下就看到了。

3. Vector add 实战

大家跑完了 modal run docker/tutorials/vector_add.py之后会看到下面的输出，可以看到3个html生成了。

但是这个时候文件还在云端需要复制一下最后一行的modal volume get {VOLUME_NAME} {DUMP_DIR}来下下来。

我们先打开以viewer_ttir.html为后缀的文件

有颜色的线是有ttir <-> py对应的，你点一下就能跳转到对应行。这样你就知道triton的底层是会被转换成啥了。反过来也很管用，一些高级的kernel一转可能就几百上千行了，你点一下就可以直接看到对应的是那行triton，很方便debug。

举例来说tl.load(y_ptr_offsets)对应的ttir是

splat会把一个ptr散布到整个1024长的vector上，通过addptr把每一个元素的指针offset加上，再通过load来读取指针中的数据。

我们切到ttgir的html。

发现同样的一行tensor类型会有一个额外的标注#blocked。这个#blocked其实是一个layout变量的名字。代表这个tensor有一个被规定好的layout，这个layout是被定义在一个叫blocked的layout变量里。具体的定义就在文件头：

可以看到有两个layout变量 blocked和blocked1，这两个layout都是一个ttg.blocked的类型（在triton cpp的代码里类型是BlockedLayout）。triton有若干layout比如专门服务mma operand的NVMMASharedLayout，有SwizzledSharedLayout等等。BlockedLayout比较简单一些，看定义就能明白：blocked是一个分布在4个warp，每个32个thread，每个thread有4个register存储。

细心的朋友应该能发现一个问题：vector长度不是1024吗，但是根据blocked的定义他不是只存储了4x32x4=512个值，另外512哪里去了？

说到这里我们就可以看一下ptx文件了：

这里数据的加载使用的是ld.global.v4 也就是一次load 4个值，这也就对应了为什么blocked的sizePerThread会是4。如果一次没有分完那么每个thread就再分一次呗，所以这里有两个ld.global.v4 语句。

blocked的定义并不一定是把全部的vector都包括进去，他可以只是定义一个切分方式，然后可以自动scale到整个tensor上。

写到这里，我自然想到同样的工具能不能用到cuteDSL上呢？

大家可以试试modal run docker/fa4_benchmark.py

开html的话用‘cutlass___call___flash_attn_local...'开头的这个，也就是带local字样的，这个是联通本地可tune的fa4 py文件。

然后就可以在fa4和ptx之间跳转玩耍了。

但是这里有个问题，就是cuteDSL的优化层级做的太高了，即使我只开了o1，能找到的python映射还是少之又少。我能发现的有帮助的地方，举个例子就是cute.copy 这一句：

我在写compiler的时候很少会注意L2::cache_hint，不知道这个对性能的影响有多少？

后续内容计划
Matmul
Persistent kernel怎么写
Warp-level tiling
什么时候用swizzle，什么时候不用
跟cuBLAS对比性能差距在哪
Flash Attention
FlashAttention和FlashAttention2、3的区别
Triton实现有哪些坑
怎么做才能超过FlashAttention3
理解FA4
MLA（Multi-Latent Attention）
KV cache压缩
推理性能瓶颈在哪
如何设计硬件友好的layout
NSA（Native Sparse Attention）
动态稀疏pattern
Block sparsity vs token sparsity
Hopper上的特殊优化
总结

这个系列的目标：

搞懂Triton编译器每一阶段在做什么
理解GPU kernel优化的本质
学会用工具分析IR绑定关系
具备独立优化高性能kernel的能力
介绍一下我的profiler大杀器

我打算在我的这套工具上给大家讲明白Triton的优化原理，以及我是怎么把它魔改到可以和cutlass cuteDSL一战的。同时为了对照，我还打算出一个cuteDSL的简易指南。最终目的就是从更多的角度，更高的维度讲明白cuda算子优化这点事。

有什么问题评论区见。

相关资源

Triton官方文档：http://triton-lang.org
TeraXLang：http://github.com/deciding/teraxlang
Modal：http://modal.com
