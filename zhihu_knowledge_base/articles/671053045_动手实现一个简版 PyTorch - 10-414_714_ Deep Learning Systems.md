# 动手实现一个简版 PyTorch - 10-414/714: Deep Learning Systems

**作者**: Alan 小分享​香港科技大学 资讯科技硕士

**原文链接**: https://zhuanlan.zhihu.com/p/671053045

---

今天要推荐的课程是 CMU 10-414 / 717，课程地址：

Deep Learning Systems
dlsyscourse.org/

我完成的课程作业的代码：




这门课主要是讲解了 PyTorch 的整体设计，以及在作业中引导我们实现一个简版的 PyTorch（项目名叫 needle）。

内容包括：

介绍深度学习基础（包括反向传播算法，以及其在两层全连接网络的推导过程）；
如何通过计算图的方式实现 Automatic Differentiation；
PyTorch 是如何做模块化的：
Tensor：用于操作多维数组，同时在前向计算和反向传播时自动构建计算图；
底层通过 device 指定数据放置的位置（CPU / GPU），不同的设备对应不同的
数组操作实现（C++ / CUDA）
nn.Module：封装神经网络的一个功能模块；小的 module 例如 SoftmaxLoss、BatchNorm1d，
大一点的例如 LSTMCell、LSTM 等；
Optimizer：用于反向传播时，更新参数；例如 SGD、Adam 等；
Dataset 和 DataLoader：用于加载和管理数据；
前两个作业中，底层的数组操作都是用 numpy 实现的，后面就会讲解怎么用 C++ 和 Cuda 分别实现；
介绍了 CNN、RNN、LSTM、Transformer、GAN 等经典网络的原理和实现方式；
简单介绍了模型微调、部署、编译等方面的知识。




Homework

作业确实是有一定挑战的，每个作业的工作量都不少。

比如 hw4 中，需要推导 convolution 操作的求导公式，得考虑 padding、stride 等操作，还是得花一点点时间。还有就是有的接口为了实现能简单点，只让我们实现了部分功能，后面使用的时候就得小心了，不然就疯狂报错。




（细节后面陆续更新...）

hw2

梯度更新：

batch 内的梯度是累加起来的。例如，实现 Linear 时，可以看到 batch 内的参数更新过程 -》weight 在 forwad 时 broadcast，backward 时，broadcast 内部会在 batch 的方向做累加。




hw3
