# Virtual Cores and Reconfigurable Pipelines

**作者**: Max Lv芯片架构师，开源软件开发者

**原文链接**: https://zhuanlan.zhihu.com/p/19889811

---

Before reconfigurable processors, we may have reconfigurable pipelines first.

Here is a new startup called Soft Machines. They are developing a processor with “virtual cores”, or just reconfigurable pipelines, to improve IPC by more than 2x.

According to their published document, a typical VISC (trade mark, something like RISC, CISC) architecuture looks like this:

With reconfigurable pipelines, VISC is able to run multiple threads on virtual cores with different numbers of pipelines:

If only considering IPC, their SPEC 2006 result looks impressive:

As a skeptical guy, it's so hard to convince me that they are able to achieve that good performance in real applications. But I really like the ideas: towards a real reconfigurable processor, reconfigurable pipelines would be a good way to start. Also, if combined with ideas of VLIW and DCO (Dynamic Code Optimization) like Tegra K1, we may achieve both wider pipelines and better power efficiency.

All images above are taken from this slides.
