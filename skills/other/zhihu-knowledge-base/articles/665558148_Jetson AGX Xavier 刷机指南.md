# Jetson AGX Xavier 刷机指南

**作者**: SnoopyBug划水划呀划~

**原文链接**: https://zhuanlan.zhihu.com/p/665558148

---

准备
Ubuntu主机（虚拟机也行），预留50GB以上空间，提前安装Nvidia SDK Manager
# 先在 https://developer.nvidia.com/sdk-manager 下载安装包（以sdkmanager_2.0.0-11402_amd64.deb为例）
$ sudo apt install ./sdkmanager_2.0.0-11402_amd64.deb

2. 打开Jetson电源，长按恢复按钮后长按电源按钮，电源灯亮，准备刷机。Jetson连接键盘、鼠标、显示器（后续系统配置需要）

3. 连接主机与Jetson（可用原装USB-TypeC数据线，Jetson连电源指示灯旁TypeC口）

刷机

0. 主机打开SDK Manager，登陆Nvidia账号，“LOGIN”登入

$ sdkmanager

1. 选择相应资源（Host Machine资源没有需求可以不选），“CONTINUE”

这里可以看到Jetson显示已连接状态，不然就再次长按恢复与电源按钮

STEP 1

2. 勾选所需资源，设置资源下载位置，勾选同意，“CONTINUE”

STEP 2

3. 系统与资源安装

先等待资源下载完毕。下载完后，会出现以下窗口。

如果第一次拿到Jetson或者需要重装系统，设备选择Jetson AGX Xavier，选择自动安装，IP一般用默认的就行，设置好Jetson上系统的用户名和密码，“TRY”；

无需重装系统的话直接“SKIP”。

准备烧入操作系统

系统刷完之后，会出现准备安装资源的窗口

先别急着安装！！！

刚装完系统的话，先在Jetson上完成系统配置，然后换源（不换源大概率是不行的）。这里贴出Ubuntu20.04-arm64的清华源，其他版本的自行搜索。跳过重装系统的也检查一下Jetson上的源是否正确。

$ sudo vim /etc/apt/sources.list

deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ focal main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ focal main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ focal-updates main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ focal-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ focal-backports main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ focal-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ focal-security main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ focal-security main restricted universe multiverse

p.s. 一定要选择手头机器的对应Ubuntu版本、系统架构的源！

（本人连续在这踩坑，重装了N次）

# 查看操作系统版本、系统架构
$ uname -a
$ lsb_release -a

Jetson操作系统可以正常使用并换源后，接下来装备安装资源，输入已经设置的Jetson系统上的用户名（字母小写）与密码，“INSTALL”

准备安装资源

4. 等待安装完毕后，刷机结束

STEP 4
完毕
