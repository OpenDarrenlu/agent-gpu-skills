# windows下使用vcpkg三分钟搭建openGL运行环境

**作者**: 齐夏LLM / AI infra / GPU

**原文链接**: https://zhuanlan.zhihu.com/p/491066310

---

LearnOpenGl这一章 创建窗口 - LearnOpenGL CN 搭建工程对于很多非科班或者C++不太熟悉的同学来说实在有点劝退，windows下编译各种第三方C++开源库真的是很麻烦，不管是直接编源码还是加载动态静态库，都不是很友好。

目前来说vcpkg是windows下最好的包管理工具了：

可以自动下载大部分开源库
自动安装依赖库
一键集成到Visual Studio 无需手动配置

接下来教你三分钟搭建一个OPENGL程序的基础运行环境

1.装好vcpkg

vcpkg/README_zh_CN.md at master · microsoft/vcpkg

2.到vcpkg.exe的目录下 地址栏里输入cmd拉起命令行

3.安装glad库

vcpkg.exe install glad

4.安装glfw库

vcpkg.exe install glfw3

5.集成到visual studio

vcpkg.exe integrate install 

6.下载demo源码

src/1.getting_started/1.2.hello_window_clear/hello_window_clear.cpp

如图demo会创建一个窗口，如果还是提示找不到库文件的话可以rebuild或者重进一下vs
