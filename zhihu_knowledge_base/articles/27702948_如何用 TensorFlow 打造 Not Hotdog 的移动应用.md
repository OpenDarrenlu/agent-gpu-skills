# 如何用 TensorFlow 打造 Not Hotdog 的移动应用

**作者**: Max Lv芯片架构师，开源软件开发者

**原文链接**: https://zhuanlan.zhihu.com/p/27702948

---

原文： https://hackernoon.com/how-hbos-silicon-valley-built-not-hotdog-with-mobile-tensorflow-keras-react-native-ef03260747f3

Not Hotdog 官网： Not Hotdog

为了提高移动设备上的执行效率，并减小 model 以及安装包的尺寸，作者做的优化主要包括：

Rounding the weights of our network helped compressed the network to ~25% of its size.
Optimize the TensorFlow lib by compiling it for production with -Os
Removing unnecessary ops from the TensorFlow lib

------

我自己也基于 TensorFlow 的 Android Example 做了一个 YOLOv2 的移植（ madeye/yolo-android ），过程意想不到的顺利。经过 Quantization 后的安装包大小差不多 20MB，识别速度可以达到 2 FPS 左右 （ Google Pixel ）。

感觉可以尝试的事情还有很多，独立开发者们不妨在自己的应用里试试基于 Deep Learning 技术。
