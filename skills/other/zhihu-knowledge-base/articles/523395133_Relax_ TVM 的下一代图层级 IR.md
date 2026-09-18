# Relax: TVM 的下一代图层级 IR

**作者**: 金雨辰机器学习编译器@OctoML; CS PhD@UW

**原文链接**: https://zhuanlan.zhihu.com/p/523395133

---

​
目录
收起
介绍
Relax 的主要目标
Goal 0：构建统一 TVM 各层抽象的接口
Goal 1：支持和优化动态 shape 模型
Goal 2：支持具有高级的数据流语义的“计算图”优化
关键设计原则
Design 0：跨层的统一抽象和优化
Design 1：shape deduction 作为一等计算
Design 2：数据流块（Dataflow Block）是一等公民
Highlights
TVMScript for Relax
快速的原型测试
模块化的编译流程（Modular compilation pipeline）
EmitTE：与 TE/TOPI 直接集成
开发计划
Relax正在公开开发中，欢迎大家参与！

文: 冯思远、蒋子恒、金雨辰

我们代表参与这项工作的许多人（本文中的姓名/ID 按字母顺序列出）写了这篇文章。

Relax 正在公开开发中，我们目前活跃贡献者有: @hypercubestart、@Hzfengsy、@jinhongyii、@junrushao1994、@LeshengJin、@MasterJH5574、@psrivas2、@sunggg、@tqchen、 @yongwww、@YuchenJin、@ZihengJiang。我们也非常期待有更多的小伙伴加入我们。
本文基于 Relax架构概述（由以下 TVM 社区成员共同撰写: @altanh、@electriclilies、@hypercubestart、@jroesch、@junrushao1994、@mikepapadim、@tkonolige、@tqchen、@YuchenJin、@ZihengJiang)。随着大家对项目的贡献，我们还将更新设计文档作者列表。
介绍

2018 年，TVM 社区围绕可表达性、性能和可扩展性的关键原则设计了 Relay。从那时起，Relay 一直是 TVM 中的核心计算图层级中间表示（Graph IR）。

这些年来，无论例如 layout rewrite 之类的上下层协同优化，还是动态 shape 模型的表示和优化，以及用 TVM 做 training 的加速等需求，都正在突破 Relay 最初的设计。这使得我们思考一个问题：我们该如何迭代 Relay 以解决这些新兴的模型和需求，同时保持可表达性、性能和可扩展性的关键原则？

在2021年的 TVMCon 我们提出了演化到新一代深度学习编译系统的核心技术路线 TVM Unity。 TVM Unity 愿景的一个关键点是统一四类抽象，即计算图（Computational Graph）、张量程序（Tensor Programs）、算子库和运行环境（Libraries and Runtimes）、硬件专用指令（Hardware Primitives），并实现这些层之间的有机交互。

Relax（Relay Next）作为对 Relay 进行迭代，是实现 TVM Unity 的关键一环。 Relax 目前正在公开开发中，拥有一个不断壮大的开发社区，并且有来自多个公司以及高校的 TVM 社区成员参与Relax 的每周开发会议。


Relax 的主要目标

我们在总结了 TVM 社区多年使用和开发深度学习编译器的的经验后，以下面三个关键目标设计了 Relax。

Goal 0：构建统一 TVM 各层抽象的接口







目前 TVM 的四层抽象之间有着明确的界限，整个编译流程遵循了一种多层渐进优化（Multi-stage lowering）的方式：图层级的 Relay 一次性 lower 到张量层级的 TIR，TIR 再 lower 到 运行时层级的 TVM FFI PackedFunc。这种 single-shot lowering 的方式导致了如果我们想要在边界上面做一些分步的优化（如把其中一部分子图交给一类编译逻辑，剩下的交给其他的编译逻辑如 BYOC），我们就必须在边界上面引入大量的工程。并且下层的张量计算或者硬件层级的信息往往难以反馈给更高的层级做联合优化。





TVM 社区已经开始表达对这些功能的强烈需求。例如，社区的多位成员建议 TensorIR 中的自动化决策应该在高层为 fusion 和 layout 的决策提供信息。Relax 的首要目标是提供一个整体的解决方案，使得我们能够同时表示和优化高层级和低层级的 IR。

Goal 1：支持和优化动态 shape 模型

动态形状模型（dynamic shape model）的编译和运行一直是 TVM 的一大痛点。

Dynamic shape 指的是模型中张量的形状不是固定的，而是根据运行时的输入决定的。例如，在 transformer 模型中，文本的输入可能具有任意长度，导致 transformer 的输入文本的 “sequence_length” 是动态维度。Relay 只能表示张量形状的某个维度是未知的（通过 Relay.Any），这限制了 IR 表达动态形状张量的能力，并将内存分配和计算全部推到运行时，从而对性能的优化形成了阻碍。

通过对 symbolic shape 的原生支持，Relax 在动态 shape 模型的表达性方面带来了实质性改进。 Relax 社区认为，符号形状（symbolic shape）支持也是在动态形状模型中解锁性能的第一步。

对动态 shape 模型的支持，我们可以分以下两种情况来讨论：

情况0：固定维度，symbolic shape 的情况

在 Relax 中，我们希望对 symbolic 整数形状提供 first-class 的支持，以实现高级优化。

例如，在下面的代码中，模型的输入张量a具有动态的 batch size。 flatten 算子通过获取输入张量 a 的所有维度，将它 reshape 为一维张量 b。

在 Relay 中，输入张量 a 的 batch size 表示为未知维度（以?表示），并且由于 flatten 的计算需要知道所有维度，因此 b 也是未知形状。这会导致表达能力损失，因为已知维度（例如张量a中的224、224、3）一旦与未知维度组合就会被抽象掉。

在 Relax 中，原生地支持 symbolic shape 极大地提高了动态形状模型的可表达性。在输入张量a中，动态批量大小可以表示为符号整数m，输入张量形状可以表示为(m, 224, 224, 3)。对 a 进行 flatten 操作后，输出张量 b 的形状可以表示为 (m * 224 * 224 * 3, )。

# Relay vs. Relax: dynamic batch size on flatten operator

# Relay IR - 没有 symbolic shape 的支持
a: Tensor[(?, 224, 224, 3)]
b: Tensor[(?, )] = flatten(a)

# Relax IR - 有 symbolic shape 的支持
a: Tensor[(m, 224, 224, 3)]
b: Tensor[(m * 224 * 224 * 3, )] = flatten(a)

在编译时知道张量之间形状的关系为我们提供了很好的优化机会。在上面的 Relax 程序中，我们知道张量 a 和张量 b 的形状中的 m 是同一个变量，因此我们可以在编译时推断 a 和 b 占用同样大小的内存，就可以在编译时决定复用这两个张量的内存。

情况1：完全动态的情况

虽然固定维度、symbolic shape 可以包含大多数的情况，但不可避免地我们还需要能处理完全动态的情况。例如，当张量的维度未知，或者遇到数据依赖（data-dependant）类型的算子（比如unique 算子）。Relax 要保证有一个“安全网”（safety net）的策略，使得我们可以处理完全动态的情况。

Goal 2：支持具有高级的数据流语义的“计算图”优化

在传统的深度学习框架中，模型被表示为计算图。 Relay 的很多 pass 目前是根据模型是纯计算图（图中不存在控制流（control flow）和 side effect）的假设来写的。

然而，新的机器学习模型（例如 transformer 、RNN）以及模型的训练比计算图所能表示的要复杂得多。例如，在训练深度学习模型时，反向传播会导致模型的权重随着训练的进行而更新。在计算图中，模型的权重需要在每次反向传递时做一次 copy 到内存中，这会不必要地占用大量的内存和吞吐量。 Relax 为计算图提供了灵活性，可以 in-place 地更新数据，而不需要做一次 copy。

上面的 in-place 更新只是随着机器学习的发展用户可能会遇到的许多不同的高级数据流语义之一。随着机器学习工程师开始使用随机数生成和权重更新，TVM 的图层级 IR 需要能够表示包含控制流、in-place 更新和包含 side effect 的程序。 所以 Relax 的目标是为用户提供最大的表达性，无论他们使用的是传统的计算图语义还是高级的数据流语义。

关键设计原则

Relax 有三个主要的设计原则，它们直接映射到上面的 Relax 的三个目标。

Design 0：跨层的统一抽象和优化

我们做的第一个关键设计是允许高层 Relax IR 能够直接调用更低层次的 TensorIR 和 TVM FFI（PackedFunc）。我们引入了两个 intrinsics 来让 Relax IR 和它们交互。

TensorIR 函数和许多外部库的定义采用 destination-passing style（需要显式分配输出并将其作为参数传递给函数），因此我们引入了 call_tir，它允许用户直接调用 TIR 或者具有 destination-passing style 的第三方库函数。我们引入的第二个 intrinsic 是 call_packed，来对 PackedFunc 函数进行调用。

下面的程序展示了通过 call_tir 和 call_packed，我们可以在高层 Relax IR 程序中直接嵌入并调用 TIR 函数 和 PackedFunc 函数。

from tvm.script import relax as R

@tvm.script.ir_module
class MyIRModule:
    @T.prim_func
    def tir_func(x: T.handle, y: T.handle):
        n = T.var("n")
        X = T.match_buffer(x, (n,), "float32")
        Y = T.match_buffer(y, (n,), "float32")
        with T.grid(n) as i:
            Y[i] = T.exp(X[i])

    @R.func
    def relax_func(x: R.Tensor[(n, k), "float32"]):
        with R.dataflow():
            gv0 = R.call_tir(tir_func, [x], (n, k), dtype="float32")
            R.outputs(gv0)

        R.call_packed("custom_inplace_update", gv0)
        return gv0

这种跨层级的交互解锁了很多之前做不了的事情，比如说：

使用不同的策略 translate/优化程序的不同部分，而不是像现在的 Relay 一样直接从 Relay 一下将全部程序下降为 TIR。
允许自动优化系统（MetaSchedule）分析 call_tir 节点以及被调用的 TIR 程序，执行优化并重写到一个或多个 call_tir 节点，从而将 layout rewrite 等决策直接反馈到高层 IR。
通过将子图转换为对 PackedFunc 的调用，BYOC 将成为编译中自然的一部分。
在一个模型中根据具体情况对不同算子采用不同的 libraries 执行，兼顾 TensorIR 带来的 fusion 灵活性和硬件计算库的高效性。

这也意味着深度学习研究人员、系统工程师、和硬件供应商可以更好地协作，因为我们可以优化和 translate 整个程序的特定部分。 更多细节请参考 Relax 架构设计中的 TIR 和 PackedFunc。

Design 1：shape deduction 作为一等计算

形状推导对于动态模型至关重要。在 dynamic shape 的模型中，我们通常需要在运行时计算张量的形状。此外，我们还需要处理 tensor 形状本身与数据相关的情况（例如 unique 算子）。大多数动态形状模型仍然包含大量的静态形状，我们需要利用这些静态形状信息进行优化。

from tvm.script import relax as R

@R.function
def shape_example(x: R.Tensor[(n, 2, 2), "float32"]):
    with R.dataflow():
        # symbolic and static shape deduction
        lv0: R.Tensor[(n, 4), "float32"] = R.reshape(x, (n, 4))
        lv1: R.Tensor[(n * 4,), "float32"] = R.flatten(lv0)
        lv2: R.Shape = (n * 4,)
        # external opaque shape function
        lv3: R.Shape = R.call_packed("myshape_func", lv2)
        lv4 = R.call_tir(lv3, "custom_func", [lv1], dtype="float32")
        # data dependent case
        lv5: R.Tensor[_, "float32"] = R.unique(lv4)
        # re-match shape
        lv6: R.Tensor[(m,), "float32"] = R.match_shape(lv5, (m,))
        gv0: R.Tensor[(m,), "float32"] = R.exp(lv6)
        R.outputs(gv0)
    return gv0

上面的程序涵盖了 shape deduction 的几种常见情况。重要的是，形状现在与张量值一起成为计算的一部分，也就是说形状的计算可以在运行时发生。

虽然在 TVMScript 中 lv0: R.Tensor[(n, 4), "float32"] 显示了每个 tensor 的形状，但这只是语法糖。从 IR 的角度来看，shape (n, 4) 不包含在 lv0 的 type 中。 lv0 的 type 是 DynTensor(ndim=2, dtype="float32")，而 shape 是附加到每个 Relax 表达式的特殊字段。这区别于在 Relay 中，张量的 shape 是 type 的一部分，也就是一个 shape 是 (2, 3) 的张量和一个 shape 是 （3，2）的张量即便 dtype 相同，它们的类型在 Relay 中是不同的。而在 Relax 中，由于它们维度一样，dtype 也一样，所以这两个张量的类型一致。我们在 Relax 中做出将 shape 和 tensor type 分离开来这个选择的原因是为了简化类型推断，不需要支持依赖类型（dependent type: 数据的类型依赖于值）。

关于 shape 更多的设计和讨论，请参阅 Relax Shape Computation 设计文档。

Design 2：数据流块（Dataflow Block）是一等公民

以本节第一个程序为例，relax_func 中的大部分代码被封装在 with R.dataflow() 的结构中，在这个 dataflow block 中的所有操作都是无副作用的，并且不包含控制流（例如 if-then-else）。

数据流块可以被视为嵌入程序中的纯计算图，因此，数据流块中的大多数绑定变量（lv0、lv1、lv2、lv3）都是局部变量，这意味着它们仅在当前块内可见。这些变量可以被视为计算图的“内部节点”。我们可以将变量标记为输出（gv0），在这种情况下，该变量将在程序的后面部分可见。这些输出变量可以被视为计算图中的输出节点。

请注意，在本节第一个程序示例中的 R.call_packed("custom_inplace_update", gv0) 是在数据流块之外的。数据流块之外的任何东西都可能有副作用。因此，除非我们对它进行更高级的分析，否则我们无法执行优化，例如根据拓扑顺序重新排序这些绑定。

大多数计算图层面的优化都发生在数据流块级别，TVM 中大多数现有的优化也可以转换到数据流块级别。这些优化可以由熟悉计算图概念的 ML engineer/researcher 完成。而对数据流块外部的代码由于具有副作用或者有控制流，相关的优化可以由 ML compiler engineer 来完成。

Highlights

我们总结了一些目前 Relax 开发中的实现亮点：

TVMScript for Relax

在实践中，我们发现一个 round-trippable 的语法会对开发，尤其是初学者的入门非常有帮助。Relax 跟 TensorIR 一样，采用了 TVMScript 作为我们的核心语法，并且实现了图层与算子层 IR 打印在同一个模块（IRModule）中。

借助 TVMScript 语法，我们能够快速做以下几件事：

因为 TVMScript 的 round-trippable 特性，使得其保留了所有 IR 的细节，因此可以用来做输出调试而不会遗漏数据结构中的任何信息；
借助 TVMScript 生成单元测试使用的例子，使得单元测试样例有更好的可读性；
利用 TVMScript，我们可以手动进行 Relax 和 TensorIR 的修改，从而快速验证技术猜想。
快速的原型测试

依托 Relax 以及跨层的统一抽象和优化和 TVMScript 语法，Relax+TIR 首次实现了自上到下全栈的 IR 表达能力。Relax 能够让用户/开发者看到一个 ML Model 是如何从图到算子最后运行在硬件上的全过程，并且能够随意修改其中的任一部分。这种修改能力能够让我们在不改动 pass 和 compilation flow 的前提下生成我们想要的代码和程序。

例如，我们猜测在 BERT 模型中，将 reshape 算子 fuse 到之前的 dense 算子中会有更好的性能。为了验证我们的猜测，在Relay中我们需要做：

更改 FuseOps 的规则；
保证 TE/TOPI 能够生成正确的 PrimFunc；
保证 AutoTVM/Ansor/MetaSchedule 能够生成跟不带 reshape fusion 性能一致的 PrimFunc；
重新 tune 带有新 fusion 规则的程序；
比较性能。

在这当中，我们需要有完整的 Relay、TIR、auto-scheduler 的全面知识，难度可想而知。而在 Relax 中，我们如果目前有不带 reshape fusion 的 tuning 结果，我们只需要：

手动修改 Relax function，把 call_tir(dense) 和 call_tir(reshape) 合并；
手动修改 TIR function，把reshape fuse 进 dense 中，而不需要重新tune（因为 reshape 是 injective 计算，对搜索的依赖程度较低）。
模块化的编译流程（Modular compilation pipeline）

目前添加一个 Relay pass 需要通常常需要 hack Relay 编译器的核心。我们希望在 Relax 中用户可以模块化地定制自己想要的编译流程，去探索一些有意思的想法。比如一些 researcher 想要写一个区别于传统 heuristic 的 Tuning pass，即在 pass 中定义一个搜索空间，不断地生成 candidate IR，build 并 evaluate 这个 candidate 的性能，再将这个性能反馈，尝试下一个 candidate。这种 Tuning pass 在 Relay 中很难很干净地实现为一个独立的 pass。

在 Relax 中我们遵循以下的设计原则：

每一个 pass 都是严格的 IRModule ⇒ IRModule 的转换。在 Relax 中，一个 IRModule 可以包含不同层级的函数。
保持一个 minimum build，这个 minimum build 可以 build 任何 IRModule ⇒ runtime Module，并且运行。也就是在 Relax 中，优化的 pass 和 minimum build 分开，这样上面提到的 Tuning pass 中就可以调用 minimum build 来测试每个 candidate 的性能。

Relax Pass Infrastructure 中有更多关于 pass infrastructure 和 Tuning pass 的设计和讨论。

EmitTE：与 TE/TOPI 直接集成

在 Relay 中，添加自定义算子需要 8个步骤，开发人员通常需要花费比较多的时间同时在 C++ 和 Python 里添加一系列的代码。

在 Relax 中，我们可以重用 TOPI 库来快速创建算子和模型。 Relax 与 TE 的集成非常地自然，因为 TVM 中的 TE DSL 也是基于 symbolic shape 的，而在 Relax 中 symbolic shape 是一等公民，所以 Relax 可以直接与现有的 TE 和 TOPI 库集成。







用上面的代码作为例子：左半部分代码中高亮的一行程序调用了 Relax BlockBuilder 中实现的 emit_te 函数，它可以接受一个 TE 函数，并直接将 Relax Var（例如这里的 input 和 weight） 转换为 TE 张量，并生成一个 call_tir 节点（右半部分代码高亮的部分）。该节点根据 TE 函数调用生成的 TIR 函数 matmul。因为 Relax 中支持 symbolic shape，以及对跨层交互的 call_tir 的支持， Relax 和 TIR 之间的衔接非常简单明了。 感兴趣的小伙伴可以去看 EmitTE Staging Integration 的文档。

开发计划

Relax 处于开发的早期阶段，我们目前的重心在构建和扩展 Relax 的核心基础架构，展示端到端的结果。以下是社区计划遵循的一些关键开发原则：

Relax 尽可能重用现有 Relay 基础设施，同时在 IR 中构建新功能。因此，社区创建了独立于 TVM 的 github repo 来加速开发。在接下来的几个月内，Relax 社区将与更广泛的 TVM 社区讨论我们的 upstreaming 计划。
Relax 致力于提升 TVM 的开发效率以及性能，同时应当与 Relay 现有的功能匹配：这意味着在 Relay 上运行的模型也可以在 Relax 上运行，即使下面的基础设施可能会发生变化。具体请看我们的 Relax Roadmap RFC。
Relax正在公开开发中，欢迎大家参与！
Relax repo:
Relax 设计文档。
我们用 github repo 中的 issue page 进行讨论和 task tracking。
我们使用 Relax discord channel 进行 Relax 相关的讨论，欢迎加入！
我们每周举办一次公开开发会议（并且在北美时区和亚洲时区做切换），具体请查看会议议程。如果感兴趣的话可以添加 Relax 会议 到你的日历上。

感兴趣的小伙伴欢迎观看我们的两次报告和展示：

去年12月的 TVMCon Relax talk
Youtube: https://youtu.be/xVbkjJDMexo
Bilibili: https://www.bilibili.com/video/BV16a411m7mX?share_source=copy_web
最近我们在 Apache TVM Community meeting 上做的 talk 和 demo: https://youtu.be/2aYWGOYmDFY

Relax Upstreaming RFC 已经发出: [RFC] Relax Upstreaming RFC · Pull Request #89 · apache/tvm-rfcs，欢迎大家参与评论！
