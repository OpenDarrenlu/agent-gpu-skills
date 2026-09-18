# LLVM Essential读书笔记之先导篇：从源码开始构建LLVM项目

**作者**: 俯仰AI Framework Engineer

**原文链接**: https://zhuanlan.zhihu.com/p/654311704

---

从源码构建LLVM项目
克隆代码到本地
 git clone https://github.com/llvm/llvm-project.git
 # 需要时间比较长 如果不需要做额外的操作可以只克隆最新的commit
 git clone --depth 1 https://github.com/llvm/llvm-project.git

2. 配置编译选项

 cd llvm-project
 cmake -S llvm -B build -G <generator> [options]

generator一般使用Ninja，而 options 中必须要传入的参数是构建类型CMAKE_BUILD_TYPE

不同的CMAKE_BUILD_TYPE区别如下：

Build Type	Optimization	Debug Info	Assertion
Release	For Speed	No	No
Debug	None	Yes	Yes
RelWithDebInfo	For Speed	Yes	No
MinSizeRel	For Size	No	No
Release适合LLVM和Clang的用户，针对运行速度进行了优化，基本没有任何调试信息和断言信息。
Debug适合LLVM的开发者，输出的信息最丰富，但没有进行任何方面的优化。
RelWithDebInfo适合有部分开发需求的用户，该构建方式针对速度进行了优化，且会输出调试信息。
MinSizeRel适合空间硬盘空间受限的用户，该构建方式同样没有任何调试信息和断言信息，但是针对占用空间大小进行了优化。




综上所述，对于LLVM开发者而言一般的构建可以通过下面的命令进行：

 cmake -S llvm -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
 cmake --build build

LLVM入门系列持续更新中，感兴趣的童鞋可以持续关注。

LLVM Essential读书笔记之一：初识LLVM与LLVM IR - 知乎 (zhihu.com)
