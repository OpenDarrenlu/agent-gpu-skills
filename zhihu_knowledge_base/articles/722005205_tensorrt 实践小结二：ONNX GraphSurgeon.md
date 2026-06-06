# tensorrt 实践小结二：ONNX GraphSurgeon

**作者**: weishengying

**原文链接**: https://zhuanlan.zhihu.com/p/722005205

---

​
目录
收起
onnx-graphsurgeon 核心设计
importers
IR
Exporters
onnx-graphsurgeon 的功能
常量折叠
其他功能
结语

tensorrt 实践小结一：onnx-parser

本文在 onnx-parser 的基础上继续学习 onnx graphsurgeon。

onnx-parser 将一个 onnx 计算图转为了 tensorrt 内部的 IR，即 network， trt 内部有大量的 pass，或者说 transformers（这是 onnxruntime 中的叫法，参考之前的这篇 blog 【OnnxRuntime】IR Pass（transform）梳理（一））来优化这个计算图。但是，trt 不开源，我们无法看到所有的详细的图优化细节，并且无法添加自己自定义的图优化 pass。

既然无法基于 trt 内部的 IR 进行图优化，另外一个思路就是直接基于 onnx 计算图上做图优化，不管是 onnx 格式还是 trt 内部的 network IR，亦或是 onnxruntime 内部的定义的 IR，本质都是描述计算的过程（计算图），优化思路上是统一的，实现不同罢了。

onnx-graphsurgeon 提供了一个非常简洁、清晰、明了的工具，能更方便的创建，修改计算图，最终的目的也是让用户能够更方便的优化计算图（如它的名字一样，是一把手术刀）。

TensorRT/tools/onnx-graphsurgeon at release/10.4 · NVIDIA/TensorRT
github.com/NVIDIA/TensorRT/tree/release/10.4/tools/onnx-graphsurgeon




onnx-graphsurgeon 核心设计

从官方文档介绍中看，主要是三个部分：

importers

能将 onnx 模型转为 ONNX GraphSurgeon IR

IR

onnx GraphSurgeon 的 IR 设计和所有的推理引擎框架类似，分为：Tensor 和 Node，Node 就是算子，Tensor 是算子的输入和输出，Tensor 又分为两种，一种是中间变量（Variable），即推理时才能知道具体数据或者shape，两外一种是常量（Constant），如权重。

所有的Node，输入输出Tensor 汇聚在一起，构成一张计算图 graph

Exporters

将 ONNX GraphSurgeon IR 导出为 onnx 模型

onnx-graphsurgeon 的功能

官方提供了一些例子来说明 onnx 手术刀的能力：examples

常量折叠

里面的例子大部分的都比较简单，这里主要介绍一下 常量折叠 功能，在几乎所有的推理引擎中，常量折叠也是一个非常重要的 pass，这里梳理一下原理，不同推理框架只是具体代码不同，整体逻辑都是一致的。

常量折叠前的计算图

如图，第一个 add 算子的两个输入 tensor 都是 constant ，可以折叠为一个 constant tensor，去掉第一个 add 算子，第二个add 同样的处理逻辑，因此，经过常量折叠后，计算图变成：

常量折叠后的计算图

从上面的逻辑可以看出，是否能够折叠，就看这个算子的所有输入，是否都是 constant tensor，这就最重要的一个判据。常量折叠的主要逻辑如下：

常量折叠大致流程

对照代码逐步分析：

首先找到所有的 Constant op， 注意是 Constant op，不是 Constant tensor，constant op的输入形式上是 constant tensor，输出形式上是 variable tensor，因此可以将输出转为 constant tensor
constant op可以直接去掉

代码逻辑上，找到所有的 Constant op后，满足一些条件时，将 Constant Op的输出 variable tensor转换为 constant tensor（to_constant），并剪掉这个输出 constant tensor 前面所有的子图（inputs.clear()）。

2. 去掉一些 shape tensor 连续精度转换的 cast 算子，shape tensor中记录的是输入tensor 的shape，只要这个输入tensor是静态的（没有动态shape），那 shape 算子的输出（shape tensor）就应该是 constant ，pass 4 过程中说明了这一点。

3. 接下来，就是找到更多的 constant tensor，也就是说有一些形式上是 varibale tensor的 node 节点输出，它可能是 constant tensor，

输入如果都是 consant tensor，则输出也是 constant tensor

如代码：遍历所有的 node，如果这个 node 是可以折叠的（最重要的判据就是这个 node 的所有的输入都是 constant tensr），就把这个 node 的输出标记为 constant tensor。（将varaible tensor标记为constant tensor）

4. 第四点是针对 shape tensor 的，如1 中所说，如果一个 variable tensor 是 Shape op 的输出，则找到这个 Shape op的输入 variable tensor，如果输入 variable tensor 的非动态shape的，则Shape op的输出可以标记为 constant tensor，代码如下

标记Shape op的输出是 constant

5. 最后，得到了所有的 “constant value”，里面一部分是本身就是 constant tensor，一部分是 variable tensor，但是经过前面的分析，被标记为了 constant tensor，对于这部分 variable tensor，我们需要借助 onnxruntime 推理引擎，计算出其真实的数值，然后把其变为 constant tensor，然后剪掉该 constant tensor 前面的子图（tensor.inputs.clear）。

将被标记的variable tensor 转换为 constant tensor

使用 onnxruntime 之前，需要将这些被标记的 variable tensor 也标记为图的输出，这样直接运行 onnxruntime session 就可以直接获取想要的推理结果。

常量折叠功能可以尝试多次运行，保证尽可能的折叠掉所有的可以折叠的部分。

其他功能

可以基于此工具开发其他一些 pass，如横向 gemm 融合等，后续再展开讨论~

结语

一般情况下，各家的推理引擎：trt，onnxruntime，paddleInference等，它们内部的图优化逻辑已经非常丰富了，但是想去自定义一个图优化pass，貌似除了修改源码外，并没有提供一些api或者文档来帮助用户完成这件事，onnx-graphsurgeon是一个可以考虑的工具，基于此工具去做一些自定义的图优化，代码基于python，阅读简单，也方便 debug
