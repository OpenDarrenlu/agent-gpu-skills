# OSDI '20 | RAMMER (NNFusion) 如何进一步压榨加速器性能

**作者**: 乔枫惜Ph.D. Student @ Stanford CS

**原文链接**: https://zhuanlan.zhihu.com/p/275837455

---

简要介绍

传统的深度学习框架（TensorFlow，PyTorch，MXNet, etc.）通常把神经网络计算抽象为由算子（operator）与依赖关系构建而成的数据流图（data flow graph，DFG），并按照拓扑序将算子逐个调度给硬件（GPU or other accelerators）或像MXNet中dependency engine可以将多个算子同时调度到GPU（inter operator parallelism）不同的stream执行。而在此之下又存在另一层调度器（block dispatcher, warp scheduler）负责充分发掘单个算子中的并行性（intra operator parallelism）并将计算任务映射给更小粒度的处理单元（SMs, CUDA Cores）。

这样的两层调度的方法尽管为系统设计带来了一些简洁，但在实际的部署中，两层调度器互相不感知会导致几个问题：runtime的调度开销很大（prepare kernel context，kernel launch，redundant IO, etc.），inter operator parallelism不能够被有效的利用，忽视了inter和intra operator两种并行性的相互影响。

为了能够打破这种僵局，我们将原数据流图中的算子解析为rOperator并将其分解为更小的调度单元，称之为rTask。将底层的硬件抽象为由多个virtualized execution units（vEU）组成的vDevice。在这套新的抽象下，我们可以以更细的rTask粒度，将数据流图调度到多个vEU之上，兼顾计算任务中的两种并行性与底层计算资源的协调。而整个schedule plan是在compile time生成并“静态”映射到硬件上的，因此可以天然地消除掉许多原本存在的调度开销。

尽管上述介绍中用了不少CUDA的概念，但是不难发现整个抽象的设计是硬件中立（hardware neutral）的（甚至可以用来描述在多核CPU上的执行，只是CPU可以利用起来的parallelism相对比较有限）。因此我们在NVIDIA GPU，AMD GPU和GraphCore上面都评估了我们这套编译技术所能取得的性能收益，在有的神经网络模型上甚至实现了对TensorRT最高3.1x的加速。

Rammer这项工作背后，是MSRA过去一年多时间里打造的一套名为NNFusion的DNN编译器，能够将现有的神经网络模型编译为对应硬件的可高效运行的源码（如CUDA，ROCm等），同时支持用户自行替换kernel实现或自动从外部导入高性能的kernel实现（如TVM[1]生成）。为了方便地与现有的codebase和GPU programming model兼容，Rammer采用的是source code transformation（数据流图中的算子预先提供好kernel implementation）这种方式而不是像TVM，Tensor Comprehensions[2]一样定义新的计算抽象需要用户提供算子的计算逻辑。

Rammer填补了一个什么样的空白？

一篇好的System文章，不仅优化性能，更要阐明这个问题。

和很多其他的工作一样，一开始我们就只是想改善一个具体的神经网络推理时（尤其出于对延迟的保障，很多场景下小的batch size其实是标准做法）GPU利用率偏低的问题，而除了优化算子实现以外，朴素的想法就是将多个算子一同交给GPU同时执行（concurrent execution），这显然不是一个新问题。CUDA很早就引入了stream的概念对其提供支持，虽然也经历了从Fermi架构上false-serialization[3]到Volta上支持多进程硬件资源分配的MPS的较长的发展历程。GPU社区中很早就有相当一些效果不错的工作（concurrent kernel execution[4]，elastic kernel[5]等），那么在DNN的场景下，为什么大家对这个问题认知不足？

像上文提到的MXNet中有dependency engine一样，TensorFlow开发早期也有支持多个stream的尝试。但是到后来都接近弃置了，我认为主要有这么几个原因：

不同的CUDA stream在runtime时采用spatial multiplexing的方法来调度不同stream queue上的kernel，粒度更粗而彼此之间又极易产生相互干扰影响最终性能[6]。
这几年GPU在堆料的路上一去不返，今天Ampere GA100中有128个SM但仅仅回溯几年，Kepler GK180中仅有15个SM而已，所以在早期无论是GPU community还是DNN的框架开发，在现有的GPU programming model下都已经形成了硬件对于inter operator parallelism并没有太多加速潜力的印象。
早期的神经网络结构比较简单，如AlexNet等本身在inter operator parallelism上也没有什么发挥空间。但是现在随着AutoML的出现，网络结构趋于复杂，此外也有ResNext[7]，ResNeSt[8]等工作引入了新的神经网络设计模式，这个问题正变得更重要。

只是将inter operator parallelism挖掘起来会是一个好的性能优化，但不足以成为一个好的system工作。事实上在SOSP 2019的投稿中我们已经完成了初步的实现并且在一些模型上也有比较好的加速效果，但是并没有定义清楚我们面对的到底是什么样的问题，加之因为没有整个NNFusion codebase的支持实验做的比较粗糙和简陋，没有取得很好的反馈。

重新定义一个问题和定位一个工作并不是在用不同的写法来写“茴”。之前我们只是在做一个广义上的kernel fusion，也没有设立起rTask和vEU的抽象。而在弄明白本质的问题在于原本系统中两层调度的gap以后，新的抽象很快帮助我们探明了更大的优化空间：

首先是将原本的通过cost model来选择子图进行fusion的问题，转变为了以更细粒度下的调度和资源分配问题。而得益于绝大部分情况下，神经网络计算的特征（DFG, 算子和张量）在compile time是已知的，我们因此可以将调度的开销移交给编译器，这既提升了搜索的效率也简化了系统设计。

更重要的是，让inter operator与intra operator parallelism相互影响这个问题走进我们的视野。举一个具体的例子，如果对于同一个算子有两种kernel实现，其中一个相较另一个消耗三倍的资源（CUDA Cores, Shared Memory, etc.），但是只取得两倍的加速，这在并行计算中是很常见的一个现象。而在此前单个算子独占整个硬件的情况下，毫无疑问我们会选择更快的实现。而我们的实验表明，在inter和intra operator两种parallelism协同调度的情况下，选择资源“性价比”最高的实现而非“最快”往往是更优的选择。这其实挑战了之前许多生成高性能算子的工作如AutoTVM[9]等的一个基本假设，单个算子独占整个硬件表现出的计算性能，是否真的是性能调优的金标准？那么显然的，subgraph substitution (TASO[10]) + high performance kernel (TVM)两个“optimal”相结合，并没有带来真的optimal。而我们基于新的抽象，只是浅尝一下简单的policy，就在一些场景下获得了超过现有SOTA的性能。我们也非常欢迎大家基于我们的抽象尝试更多advanced policy来探究对于一个DFG（或者其子图）搜索intra，inter operator parallelism interplay下的更高性能的整体实现。

你可能还会关心的几个问题

究竟什么是rTask？

简而言之就是组成rOperator的互相独立的更小的任务单元，也是我们抽象中最小的调度单元。具体而言，在NVIDIA GPU上，对于用户提供的CUDA kernel implementation，通过我们写的一个小的解析器，可以将每个block转化为一个rTask，其他平台同理。所以不难发现rTask可以利用原本kernel implementation中的语义，虽然因此rTask在实现上与programming model耦合，但是也大幅度降低了所需的工作负担。

如何创建vDevice和vEU？

目前而言是根据硬件的特性再配合简单的heuristic，举例而言V100中有80个SM，每个SM最多能够运行32个block，那么我们可以创建一个包含有2560个vEU的vDevice（或者更少，取决于网络中算子对于硬件的使用情况），而后根据DFG of rTask与vDevice，通过一些简单的policy（譬如直接将众多的rTask以平铺的方式分配给vEU们）就能够生成足够高效的schedule plan。

如何能够将schedule plan“静态”映射给实际的device呢？

因为硬件和programming model的限制，GPU runtime dispatcher和scheduler对用户并不开放可编程接口，而做一个模拟器上的研究不是我们的本意。因此我们借鉴了persistent thread[11]的方式，可以以一个相对小的overhead比较hack一点地将vEU与实际的硬件执行单元（SM等）绑定起来。更多关于设计，实现与实验的细节，欢迎大家参考我们的paper和video。

GPU vendor可能会将类似的想法加进programming model吗？

在MSRA的导师常常会给我说，不要跟vendor抢活干。实事求是的讲，现有的设计和实现因为硬件的限制，没有能够发挥完全的潜力，类似persistent thread的技巧在生产环境下也未被久经考验。所以在NNFusion中Rammer相关的feature也还是比较保守。而vendor修改programming model的事情，其实是一直在发生的，从CUDA 10开始引入了一个新的概念叫CUDA Graph[12]，支持将一整个data flow graph一次性送进GPU中，并在GPU内自动异步执行。这样从理论上来讲，runtime hardware scheduler是有可能感知到DFG详细信息的。但是在硬件上做如此复杂的DFG aware runtime scheduling是有益的吗？这也需要对于microarchitecture做不小的改动。现有的CUDA Graph更多是driver层面的内容，将整个DFG中所有的operator kernel context缓存在GPU中，可以大幅省去重复的launch操作和CPU到GPU的通信开销。在年初的GTC上我也亲口找CUDA Graph的架构师聊过，也确实印证了我们的想法，他们短时间内也没有在CUDA Graph enable DFG计算优化的想法。而其实在TensorRT中我们是已经知道的，存在类似subgraph substitution的机制来协同优化DFG与计算。

支持dynamic DFG和training吗？

目前的设计和实现都是对于静态图做的分析和实验，但是我们同样可以去静态编译它们的条件分支，只是没有做过具体的尝试不是很能确定。同样的道理，尽管没有什么明显的障碍阻止rammer应用在training中，但是我们对此也没有具体尝试过。
另外为了更有效的scheduling，我们也使用了kernel的profiling信息。这里面事实上存在一个假设：kernel在实际执行时的性能表现是deterministic的，而在我们讨论的scenario中这些是得到满足的。

支持对于不同model的多个inference queries融合进一个kernel context同时执行吗？

可能是可行的，我也认为multi tasking可能会是一个重要的场景，但是现有设计和实验仅支持对于不同的模型分别编译与serving。
几个相关的工作

TVM与TASO是两个我非常喜欢的工作，前者是end2end DNN编译的集大成者，如今已经成为重要的infrastructure（ 作者@陈天奇老师和阿里的 @蓝色老师都在知乎上做过很多介绍 ），而后者清楚地定义并且尝试穷尽了一定规模下所有的subgraph substitution搜索空间。但TASO在实际使用中并不足够scalable，而且使用门槛并不低，知乎上阿里的@杨军老师的这篇文章非常详尽地解释了他们在阿里的DNN compiler中是怎样做pattern发掘和替换的，可能会更加practical一点。而在NNFusion中也同时支持从TVM获取高性能kernel implementation和自定义subgraph pattern去进行替换。Zhihao Jia还有另一篇在openreview上面挂着利用Inter Operator Parallelism的工作[13]跟我们也比较相关，但是主要的内容还是关于policy，如果碰巧作者看到这篇文章，非常欢迎来尝试基于Rammer的抽象来写policy。来自UToronto的这个工作[14]则从具体实现上尝试了Horizontal Fusion，这些其实也印证了我们前面的对于趋势的思考。Berkeley的OoO VLIW JIT Compiler[15]也是一个非常有趣的工作，但是在场景和方法上都和我们有比较大的区别。还有一些工作例如BatchMaker[16]，Clipper[17]等已经在不同的scope里就不多赘述了。百度几年前有一个著名的工作叫persistent RNN[18]让我们注意到了persistent thread带来的更多实现上的可能性，后面也陆陆续续参考了GPU community里面的一些讨论，但是persistent thread在多数情况下并不是best practice，需要额外引入sync语义而且对于不同的硬件需要修改配置参数，可能还是需要寄希望于accelerator vendor未来开放更多的可编程接口。

广告时间

NNFusion现已经在Github开源：

目前我们已经发布了0.1 版本, 0.1版本支持TensorFlow和ONNX在内的主流模型格式以及CUDA GPU等设备，提供了丰富的性能优化策略，支持端到端的模型到源代码的AOT编译来消除运行时开销，消除了第三方库或框架的依赖。如果你有更深入的研发需求，可以直接修改NNFusion生成的代码来进行模型的定制化优化。

NNFusion 0.1 是我们在加速模型编译执行上的新的开始，我们会继续研究，不断完善NNFusion的功能，为大家的研究带来便利。非常欢迎大家跟随我们的README去体验NNFusion，也期待你可以在NNFusion中贡献你的真知灼见，和我们一起“压榨加速器的性能”！

另外，与master分支同步发行的还有OSDI paper的复现artifact的分支的0.1版本。大家可以使用docker image进行安装快速试用，也可以根据项目源代码自行编译安装。

MSRA今年共参与了6篇发表在OSDI '20的工作，具体可以参考：

其中Byzantine Ordered Consensus without Byzantine Oligarchy更是获得Best Paper殊荣，我想可以作为MSRA在过去几年厚积薄发与现在野心勃勃状态的一个折射。在NNFusion之前，SPTAG, PAI, NNI等都已经取得了很大的影响力，还有一些尚未开源的项目都有重塑生态的潜力，研究院内求贤若渴，非常欢迎对我们研究感兴趣的同学来实习！

然后是面对上科大校内的同学，欢迎加入GeekPie HPC！几位过去的核心成员今年已经陆续在ISCA，OSDI和SC有所收获，升学和求职上也都有很好的去向，欢迎喜欢折腾的同学加入我们，在备赛中锻炼的能力对今后都会有所裨益。

Finally, I'm looking for a summer research internship in 2021 and a PhD position in 2022 (ML System, Cloud Computing and Computer Architecture). Please drop me any information if you're interested in and feel free to spread this article, thank you!

参考
^TVM https://tvm.apache.org/
^Tensor Comprehensions https://arxiv.org/abs/1802.04730
^False serialization on Fermi https://www.semanticscholar.org/paper/Multi-threaded-Kernel-Offloading-to-GPGPU-Using-on-Wende-Steinke/3a33926319ca66f5273fff3f1ba61e38d17229b1
^Concurrent Kernel Execution https://ieeexplore.ieee.org/document/5999803
^Elastic Kernel http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.649.8875&rep=rep1&type=pdf
^Dynamic Space-Time Scheduling for GPU Inference https://arxiv.org/pdf/1901.00041.pdf
^ResNext https://arxiv.org/abs/1611.05431
^ResNeSt https://arxiv.org/abs/2004.08955
^AutoTVM https://arxiv.org/pdf/1805.08166.pdf
^TASO https://github.com/jiazhihao/TASO
^Persistent Thread https://ieeexplore.ieee.org/document/6339596
^CUDA Graph https://developer.nvidia.com/blog/cuda-10-features-revealed/
^ACCELERATE DNN INFERENCE BY INTER-OPREATOR PARALLELISM https://openreview.net/pdf?id=HJezqlrKvr
^Automatic Horizontal Fusion for GPU Kernels https://arxiv.org/abs/2007.01277
^The OoO VLIW JIT Compiler for GPU Inference https://arxiv.org/abs/1901.10008
^BatchMaker https://dl.acm.org/doi/abs/10.1145/3190508.3190541
^Clipper https://www.usenix.org/system/files/conference/nsdi17/nsdi17-crankshaw.pdf
^Persistent RNN https://github.com/baidu-research/persistent-rnn
