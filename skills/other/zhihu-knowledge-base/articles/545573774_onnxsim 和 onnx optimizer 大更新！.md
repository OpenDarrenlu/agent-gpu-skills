# onnxsim 和 onnx optimizer 大更新！

**作者**: 大缺弦昆仑天工 AI Infra 工程师，ONNX 核心成员

**原文链接**: https://zhuanlan.zhihu.com/p/545573774

---

欢迎加入 ONNX QQ 群 1021964010！入群密码 nndab，微信群可加我的微信 daquexian 我来拉入群

更新：我得到了 ONNX 项目的 approver 权限，可以 approve 和合并 ONNX 项目的 PR，之后可以有更大的能力帮助 ONNX 的发展了。欢迎对 ONNX 有想法的小伙伴加群或者加我交流~

onnxsim 推出三年多，可以说已经成为了优化 onnx 模型的标配：star 数量达到了 2.3k，过去的一个月在 PyPI 上有 8 万+ 次的下载量，MMDetection、YOLOv5、YOLOX 等带 onnx 导出功能的模型库官方集成了 onnxsim，MXNet、NCNN、TNN 等框架也都在它们的文档或代码中介绍或使用了 onnxsim。

不过一直以来 onnxsim 有两个痛点：需要传入复杂的参数才能优化输入形状不固定的模型、需要用户提供模型所包含的自定义 OP 的 ONNX Runtime 实现。这两个痛点都来自于 onnxsim 最初的一个选择：那个时候 ONNX 本身的 shape inference 非常不完善，经常 segfault，所以为了能够尽可能得到形状信息来帮助优化，onnxsim 调用了 ONNX Runtime 来推理全图，这就带来了对输入形状以及自定义 OP 的限制。这在 onnxsim 刚写成时不是一个大问题 —— 那个时候 ONNX 本身都还不支持动态形状输入，也就更没有什么包含动态形状输入的模型了。但是现在越来越多的 ONNX 模型自带动态输入形状（至少有动态 batch），自定义 OP 也经常出现，经常有用户在使用 onnxsim 时被这两个问题困扰。

好在相比于三年前，现在 ONNX 本身的 shape inference 鲁棒性已经非常高，也有了 symbolic shape inference 这样可以推导部分形状的好功能。经过一系列的构思和开发，我发布了 onnxsim v0.4，一个脱胎换骨的重写版，彻底消除了上述的两个痛点 —— 不管是静态输入还是动态输入，也不管有没有自定义 OP，在使用新版 onnxsim 时都只需要 onnxsim input.onnx output.onnx 一把梭。此外，新版 onnxsim 是用 C++ 而不是 Python 写的，这使它可以编译为 WebAssembly 并发布到 convertmodel.com —— 一个包含了各种 WebAssembly 格式的模型转换工具、提供开箱即用的模型转换功能的网站（我之后会再专门写一篇文章介绍它）。

同时，onnxsim 的基石之一 —— onnx 的 官方 optimizer 也迎来了大更新，这里要特别感谢社区小伙伴 @小强（知乎同名用户太多了 at 不到，不过已经出现在评论区了~ GitHub 用户名是 HSQ79815）的伟大贡献。onnx optimizer 的更新内容包括：

新增 fuse_concat_and_reshape、eliminate_slice_after_shape、eliminate_shape_gather、replace_einsum_with_matmul、eliminate_nop_expand 等 pass，其中前三个 pass 对于含动态形状的模型（特别是 transformer）有特别大的优化作用。

支持了 >2GB 的 ONNX 模型。我一直觉得 ONNX 因为 protobuf 的限制而对 >2GB 的模型提供专门的处理机制的做法非常不明智，所有的下游工具都要适配这套专门的机制才能支持 >2GB 大模型。接下来我也准备提一个 proposal 干掉这套机制（也许只是嘴炮，但希望能成真）。

适配了 ONNX 的新 IR 版本 —— 在 IR 层面 initializer 终于可以不是 graph input 了。

这些更新内容大部分都是来自社区小伙伴 @小强 的贡献。

让我们来看一下新版 onnxsim 的优化效果吧！

MLPerf 所用的 SSD-MobileNet 模型：

优化前整体结构（只看整体结构就好，分辨率有限，是看不出具体 OP 的）：

优化后整体结构：

优化前后数据对比：

ONNX Model Zoo 的 UltraFace 模型：

优化前整体结构：




优化后整体结构：




优化前后数据对比：

Q & A
问题：现在 PyTorch 导出 ONNX 模型也自带了常量折叠功能，onnxsim 还有必要吗？

回答：有的。onnxsim 的能力比 PyTorch 自带的常量折叠更加强大，模型越复杂效果差异越明显。

2. 问题：为什么我的模型 sim 之后大小反而变大了？

回答：这是正常的，而且不用担心，因为模型的大小和速度无关，模型大小变大的原因是原模型中含有 Tile、ConstantOfShape 等算子，这些算子被消除时会产生很大的 tensor。如果确实不想让模型大小变大，可以使用 --no-large-tensor 参数。

3. 问题：onnxsim 优化后的模型推理速度会更快吗？

回答：一般来说是的，但也要看具体的模型结构。如果模型本身冗余算子不多，可能优化后不会有肉眼可见的速度提升。而冗余算子多的模型效果会很明显，据 @小强 测量，他所用的模型在经过 onnxsim 优化之后，在 ONNX Runtime CUDA EP 上有 6% 左右的速度提升。

最后，欢迎加入 ONNX QQ 群 1021964010，入群密码 nndab。也可以加我的微信 daquexian 加入微信群~
