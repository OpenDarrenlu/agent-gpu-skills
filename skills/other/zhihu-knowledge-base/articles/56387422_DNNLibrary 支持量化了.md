# DNNLibrary 支持量化了

**作者**: 大缺弦昆仑天工 AI Infra 工程师，ONNX 核心成员

**原文链接**: https://zhuanlan.zhihu.com/p/56387422

---

DNNLibrary 发布了 0.6.10 版，支持了 Android NNAPI 的 QUANT8_ASYMM 类型（也就是 int8 需要的数据类型），并且给出了一个把一个普通的预训练模型转换成 int8 模型的脚本

生成 int8 模型只需两步：

用 quant.py 在一个数据集上跑一下，收集相关数据，生成一个包含 scale 和 zero point 的文本文件和一个存储了量化权重 onnx 模型

2. 然后在用 onnx2daq 把上一步产生的文本文件和 onnx 模型转换成 DNNLibrary 的 daq 模型

dnnlibrary-example 有使用 int8 模型的例子

量化功能还处在初步阶段，还没有正式测试过准确率，欢迎各种尝试和体验，我也会根据反馈不断迭代

另外今年即将推出的 Android Q 会新增新的量化方法，DNNLibrary 也会第一时间跟进
