# TensorRt 部署流程 (Nvidia GTC Talk Session)

**作者**: ZZZZJGPU @ RunwayML

**原文链接**: https://zhuanlan.zhihu.com/p/117947139

---

知乎第一篇文章啊，就是这样的广告。

我们今年在GTC 2020 上有一个session，描述了我们(其实也就是我和Jeff)为我司TensorRt 部署做的一些工作。

这个 Session 前半部分是 Nvidia 的合作者介绍TensorRt 7.1的新feature的。后半部分是我们的工作。

本来是准备了一个我司的driving demo video 但是由于变成了网络版于是就算了。这些demo video 基本上断断续续在我司linkedIn 和 twitter上有放出来过。想看的同学私信吧，我问问领导。

还有就是用 TensorRt 的同学尽量用新架构的GPU吧(Volta, Turing) 。。。 我们的经验是Nvidia 对于Pascal 已经不是很上心了各种只有Pascal 上有的小bug 已经慢慢的出来了。。。

GTC 2020: Optimizing TensorRt Conversion for Real-Time Inference On Autonomous Vehicles
developer.nvidia.com/gtc/2020/video/s22198
