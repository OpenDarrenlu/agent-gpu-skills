# 本地VSCode搭建RUST开发环境

**作者**: i4oolishAI框架，编译器，体系结构

**原文链接**: https://zhuanlan.zhihu.com/p/393563450

---

安装C++ build tools: 下载地址： https://visualstudio.microsoft.com/zh-hans/downloads/ 拉到最下面“所有下载”找“Visual Studio 2019工具”，下载工具安装程序。 安装流程：运行安装程序，选择“C++生成工具”
安装RUST：

powershell设置代理：
ENV:HTTP_PROXY=xxx
ENV:HTTPS_PROXY=xxx
下载安装rust开发工具包：rustup-init.exe https://www.rust-lang.org/
安装：默认会安装到当前用户目录下，默认检查环境变量CARGO_HOME/RUSTUP_HOME，分别为.cargo/.rustup设置目录，按需修改（注意：后期包的累积可能会占用大量的空间）
安装完后运行如下命令
rustup --version
cargo --version
配置cargo代理：修改用户主目录下.cargo文件夹中的config文件，添加代理配置：
[http]
proxy = "xxx"
[https]
proxy = "xxx"
安装RUST源码：运行如下命令即可
rustup component add rust-src




安装VSCode插件

安装rust-analyzer和依赖项rust-analyzer server 主要用途：

当前文档符号搜索 [[ctrl+shift+o]]
符号查找 [[ctrl+t]]
输入辅助
代码辅助

详细使用参考：https://www.lmonkey.com/t/87y4QGzL3

安装code-runner
安装CodeLLDB用于调试




VSCode生成RUST工程 在VSCode中teminal中执行“cargo new project_name”即可创建工程。
