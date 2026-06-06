# NV全新的 SIMD VLIW DSP发布

**作者**: maja性别男，爱好女

**原文链接**: https://zhuanlan.zhihu.com/p/6721345137

---

比较好奇，这应该算是首次NV使用 vector SIMD VLIW DSP 处理器（每个核可能实时的）应对边缘计算功耗挑战。




主存采用LPDDR5，计算模型采用vector SIMD 并没有使用，SIMT担任图像及其模型的处理，驾驶任务。产品隶属Drive系列Orin板子上，包括2x独立的处理核心，通过L2 cache 来同步数据。




加入了一个叫做DLUT DSP组件来加速查询操作。




第一次听说这个系统，看了下配置感觉不太像是支持端上大模型的，从流程图上看主要应对CV操作：回顾opencv::gpu下面的图像算子，生态位是数据的预处理和后处理卸载。




https://developer.nvidia.com/blog/optimizing-the-cv-pipeline-in-automotive-vehicle-development-using-the-pva-engine/
