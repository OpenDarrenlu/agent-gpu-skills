# Let's benchmark pandas 2.0.0 RC0！

**作者**: Uranus​清华大学 计算机系博士在读

**原文链接**: https://zhuanlan.zhihu.com/p/608938646

---

pandas 昨天发布了 2.0.0 的第一个 release candidate。从 release notes 上看，几乎全都是性能相关的优化。那我们就来测测它性能究竟如何吧！

环境

测试用的环境为我的工作电脑，具体数据如下：

MacBook Pro (16-inch, 2021)
Chip Apple M1 Max
Memory 32 GB
数据与计算负载

测试所用的数据是常见的 TPC-H SF1，大小大概为 1 GB。测试脚本是从 Xorbits 那里改了改拿来用的。

测试结果

我测试了一下 5 中版本与 options 的组合：

1.5.3
2.0.0rc0
2.0.0rc0 + lazy copy
2.0.0rc0 + pyarrow dtype backend
2.0.0rc0 + lazy copy + pyarrow dtype backend

结论如下，下表中单位为秒：

	round 1	round 2	round 3	average
1.5.3	11.94	11.81	11.94	11.89
2.0.0rc0	17.38	17.50	17.25	17.37
2.0.0rc0 + lazy copy	16.39	16.51	16.52	16.47
2.0.0rc0 + pyarrow dtype backend	51.89	52.55	52.60	52.34
2.0.0rc0 + lazy copy + pyarrow dtype backend	53.51	53.92	54.19	53.87
结论

上面的结果与我的预期相去甚远，甚至可以说是背道而驰。可以看出来的是打开 pyarrow dtype backend 后读取数据的速度变快了不少，但是某些操作慢了很多。考虑到 rc0 还是一个非常早期的版本，让我们期待后续的优化吧～
