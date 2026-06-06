# Android NNAPI 即将在 Android Q 迎来大更新

**作者**: 大缺弦昆仑天工 AI Infra 工程师，ONNX 核心成员

**原文链接**: https://zhuanlan.zhihu.com/p/56376295

---

感谢 @Siemon 一起润色和补充这篇文章！

---------------------

2017 年年底的时候，Android 8.1 推出了集成于 Android 系统内的神经网络 API，当时我很快做了一个封装库，也写了一篇专栏 大缺弦：Android 8.1 NNAPI 评测以及可能是全球第一个的 NNAPI 库

一年多过去了，Android 模型部署也有了巨大的变化，ncnn 继续活跃的维护，加入了 int8 和 vulkan 的支持，大量的新框架例如 FeatherCNN 和 Mace 也产生了。另一方面，NPU 越来越火热，几乎成为了每一款新手机芯片的标配。

与 iOS 端模型部署早早被 iOS 内置的 CoreML 统一不同，Android 内置的 NNAPI 因为 Android 系统的碎片化，普及的进度要慢很多，但也有越来越多的设备为 NNAPI 提供额外支持，例如 RK3399、Huawei Mate 20 等等（在这些设备上 NNAPI 可以调用 GPU 或 NPU）。

令人激动的是，根据 AOSP 最新的源码，今年即将随 Android Q 推出的 NNAPI 1.2 有非常多的重量级更新，增加了很多 operations（包括 detection、keypoint 领域需要的 operations）、有了更好的 NPU 支持、新的量化方式、float16、NCHW layout 等等。

新增的 operations

NNAPI 1.0 有 29 个 operations，NNAPI 1.1 有 38 个，而 NNAPI 1.2 有 94 个，新增了 56 个 operations，增加了一倍还多。

新增的 operations 包括 GREATER、LOGICAL_AND、MINIMUM、REDUCE_ANY、SELECT 这些通用操作，还包括为检测网络准备的 BOX_WITH_NMS_LIMIT、ROI_ALIGN、ROI_POOLING 和 GENERATE_PROPOSALS，为 keypoint 准备的 HEATMAP_MAX_KEYPOINT， shufflenet 必需的 CHANNEL_SHUFFLE、GROUP_CONV 等等。

因此在 Android Q 里，NNAPI 的表达能力将会获得极大的提升，原先 NNAPI 只能支持普通的分类网络和部分分割网络，加入这许多的新 operations 之后，Faster R-CNN、Mask R-CNN、ShuffleNet 这些网络都可以支持了。

更好的 NPU 支持

NPU 越来越火热，Kirin 970、Snapdragon 855 等等芯片都集成了 NPU 或类似的加速芯片。NNAPI 从推出时就宣称可以在 CPU、GPU、DSP 上运行，但开发者却无法查看和指定模型运行在什么设备上。我在 Google Pixel 2 上测试时，发现尽管 Pixel 2 有类似于 NPU 的 Vision Core，但实际上仍只有 CPU 可以被使用。

NNAPI 1.2 在这一点上有重大的更新，它新增了一个 ANeuralNetworksDevice 类，表示每个支持 NNAPI 的设备，并分为 UNKNOWN、OTHER、CPU、GPU、ACCELERATOR 几类，还有 getSupportedOperationsForDevices、getVersion、getFeatureLevel 等等细粒度的接口。开发者可以判断设备是否支持给定的模型，也可以指定模型在哪几个设备上运行。这对融合日渐火热的 NPU 生态非常重要。

新的量化方法

NNAPI 从 1.0 版本开始就支持 int8 量化。int8 量化是将原先由浮点数表示的权重和中间特征替换成 8-bit 的整数格式，将 float32 运算替换为 int8 运算，这样减小了内存访问量，也能一次计算更多组数据，所以会大大加速模型的运行速度。但具体如何描述 float 和 8-bit integer 之间的映射关系则有不同的手段，TensorFlow/TensorFlow Lite 一直使用的是包含 zero point 的非对称量化方式，

real\_value = (integer\_value - zero\_point) * scale

，这篇 Google 的 paper 有详细的描述，Tensor RT 使用的是不含 zero point 的对称量化，

real\_value = integer\_value * scale

对称量化因为不含涉及到 zero point 的计算，速度会稍快一些。

NNAPI 1.2 也支持了这种对称量化方式，不过 NNAPI 的实现只会包含某种量化方法下的具体运算的实现，至于如何获取每个 8-bit tensor 的 scale 和 zero point 就需要上游的库的配合，例如 DNNLibrary 提供了一个脚本生成非对称量化需要的 scale 和 zero point

此外，NNAPI 1.2 也加入了对分通道量化的支持，即每个通道都有一个自己的 scale，而不是整个 tensor 使用同一个 scale，在这种分通道的情况下 float 和 int8 之间的映射会更精确，量化模型的准确率也就会更高一些。

其它的新变化

除了上面说到的之外，NNAPI 1.2 还有更多其它的更新：

增加了 TENSOR_FLOAT16 和 FLOAT16 数据类型。float16 相比 float32 既不会损失太多精度，也有很大的加速，但只有 ARM8.2-A 以上的新 ARM 架构才能使用 float16 格式进行计算。NNAPI 1.1 中，开发者可以指定是否允许浮点数以 float16 的精度计算，允许 float16 的精度对某些芯片的加速效果明显。NNAPI 1.2 加入了两个 float16 的数据类型，许多 operations 也做了相应的更新，意味着 NNAPI 对 float16 有了更多的支持。不过需要指出的是 Android 自带的 NNAPI 实现里，对 float16 的数据会转回 float32 进行计算，计算完成再转回 float16（因为大部分芯片不支持 float16 计算），反而会比 float32 慢，所以建议只在支持 float16 计算的设备上使用 float16 数据类型
支持了 NCHW 数据布局。NNAPI 1.1 中，只有 NHWC 数据布局是被支持的，而大部分训练框架常用的是 NCHW 布局。NNAPI 1.2 支持了 NCHW，开发者可以自行选择数据布局。

结尾

如此多的更新是非常令人激动的，有理由相信 NPU 越来越普及之后，NNAPI 将成为 Android 上部署模型的重要选择之一——让开发者去逐个适配不同厂商的 NPU 显然是费时费力又不聪明的办法。

此外，我的 Android NNAPI 封装库 DNNLibrary 今天从我的个人 GitHub 账号 transfer 到我们部门（JDAI Computer Vision）的 GitHub 账号下了。DNNLibrary 支持 onnx 模型，可以弥补 TensorFlow Lite 只能使用 TensorFlow 模型的空缺，并且支持了一些 TensorFlow Lite 在开启 NNAPI 时无法使用的操作（例如 dilated conv 和 prelu），日前也加入了对 8-bit 量化 的支持，DNNLibrary 会第一时间跟进本文所介绍的 NNAPI 1.2 中的新功能，欢迎持续关注和试用 :)

以及，有一个 DNNLibrary & NNAPI 交流群，欢迎 DNNLibrary 用户或者对 NNAPI 有兴趣的小伙伴加入 QAQ，群号 948989771，入群答案：哈哈哈哈

dnnlibrary &amp;amp;amp;amp;amp; NNAPI 交流 QQ 群二维码

和一个纯讨论 Android NNAPI 的微信群：

NNAPI 交流微信群二维码
