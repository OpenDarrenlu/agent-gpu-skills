# 如何从源码编译OpenCV4Android库

**作者**: 稚晖​​新知答主

**原文链接**: https://zhuanlan.zhihu.com/p/83226948

---

这次的教程比较复杂，只是个人最近项目中刚好遇到的问题，把解决的过程记录下来顺便分享，大家看不懂没关系，可以先马一下以后开发APP项目中使用OpenCV遇到问题再来回顾~
我后面有空也会尝试写一些OpenCV入门的教程，这个做计算机视觉必备的开源库还是有很多值得研究的地方的。
0.前言

OpenCV3.1时代开始，Android平台就已经有官方提供的OpenCV库了，理论上我们是不需要再自行编译的。而且OpenCV的官方建议也是直接使用OpenCV4Android库（也就是预编译的libopencv_java3.so），并提供了两套使用方法：

利用OpenCV提供的全套Java接口， 在Android Java层调用。
利用OpenCV提供的C/C++ 接口， 在JNI层使用（就跟在PC端VC++下使用OpenCV一样一样的）。

但是由于在实际的应用中难免会遇到一些问题，比如在Android工程中如果要同时使用SNPE（一个高性能神经网络加速库）和OpenCV时，由于SNPE使用的STL链接的是libc++，而OpenCV默认使用的是gnu_stl，所以会导致gradle不管怎么配置都无法正常编译过的情况。

这种情况下如果gradle中选择arguments '-DANDROID_STL=c++_shared'的话SNPE可以正常编译，但是在使用像imwrite这样的OpenCV函数时就会报链接错误。相反如果gradle中选择arguments '-DANDROID_STL=gnu_stl'则SNPE无法编译通过。

另外一方面，官方预编译好的OpenCV4Android库是不带contrib模块的，所以无法使用像是`xfeatures2d`这样的类。

以上原因驱使我们需要能够自己从源码编译OpenCV4Android的库。编译方法有几种，可以在Linux下基于NDK编译，也可以在Windows中使用MinGW编译，本文选择的是前者，因为可以生成Docker镜像方便以后部署编译环境。

1.使用现成Dock编译镜像

这里使用的是我已经配置好的编译镜像，由于镜像文件尺寸很大（10GB左右），所以就不上传了，大家可以看完这一节的方法后根据下一节的教程自己生成一样的镜像。（这里就假设大家已经会使用Docker啦，话说Docker 这种神器早用早享受啊~）

Docker镜像环境配置如下：
镜像名称：opencv4android-builder.image
系统版本：Ubuntu 16.04
内部OpenCV版本：3.4.8

启动Docker的命令如下：

docker run -it --name opencv4android-builder  --network host \
-v /home/pengzhihui/_share/OpenCV:/workspace/_share/OpenCV \
-v /etc/timezone:/etc/timezone \
-v /etc/localtime:/etc/localtime \
opencv4android-builder.image

进入Docker容器之后首先设置一下环境变量：

export ANDROID_NDK=/workspace/opencv4android/tools/ndk/android-ndk-r17
export PATH=${PATH}:$ANDROID_NDK

export ANDROID_SDK=/workspace/opencv4android/tools/sdk/android-sdk-linux
export PATH=$ANDROID_SDK/tools:$PATH

export JAVA_HOME=/workspace/opencv4android/tools/jdk/jdk1.8.0_221
export JRE_HOME=${JAVA_HOME}/jre  
export CLASSPATH=.:${JAVA_HOME}/lib:${JRE_HOME}/lib  
export PATH=${JAVA_HOME}/bin:$PATH

source  ~/.bashrc
如果需要修改NDK或者SDK等的版本则也需要更新这些环境变量。

然后进入目录/workspace/opencv4android/build，运行下面的脚本即可开始编译：

python ../opencv/platforms/android/build_sdk.py \
--extra_modules_path=/workspace/_net/opencv_contrib/modules/ \
--config ../opencv/platforms/android/ndk-17.config.py
如需要修改编译选项，则可以修改opencv\platforms\android\android.toolchain.cmake文件。
2.从头开始搭建编译环境

这一节会介绍如何从头开始搭建上面的镜像环境，整个过程踩了很多坑，所以这里记录下来。

2.1 生成Docker基础容器

使用的基础镜像为Ubuntu16.04，因此如果本地没有这个镜像的话，需要先从网上pull一下：

sudo docker pull ubuntu:16.04

然后生成容器：

docker run -it --name opencv4android-builder-tmp  --network host \
-v /home/pengzhihui/_share/OpenCV:/workspace/_share/OpenCV \
-v /etc/timezone:/etc/timezone \
-v /etc/localtime:/etc/localtime \
ubuntu:16.04

进入容器之后先建立在/worksapce/opencv4android工作目录（名字你自己定）：

mkdir /worksapce/opencv4android && cd /worksapce/opencv4android

再根据需要建立如下结构的几个目录（有的目录是后面下载过程产生的）：

接下来需要安装一些基础工具：

先安装vim/nano编辑器用来更新源：
apt update apt install vim

vim /etc/apt/sources.list
把源更新为阿里源并保存：

因为纯净的ubuntu Docker镜像是不带任何文本编辑器的，当然了如果你喜欢用cat命令那也可以直接操作。
deb http://mirrors.aliyun.com/ubuntu/ xenial main
deb-src http://mirrors.aliyun.com/ubuntu/ xenial main
deb http://mirrors.aliyun.com/ubuntu/ xenial-updates main
deb-src http://mirrors.aliyun.com/ubuntu/ xenial-updates main
deb http://mirrors.aliyun.com/ubuntu/ xenial universe
deb-src http://mirrors.aliyun.com/ubuntu/ xenial universe
deb http://mirrors.aliyun.com/ubuntu/ xenial-updates universe
deb-src http://mirrors.aliyun.com/ubuntu/ xenial-updates universe
deb http://mirrors.aliyun.com/ubuntu/ xenial-security main
deb-src http://mirrors.aliyun.com/ubuntu/ xenial-security main
deb http://mirrors.aliyun.com/ubuntu/ xenial-security universe
deb-src http://mirrors.aliyun.com/ubuntu/ xenial-security universe


再次执行apt update即可。

安装zip/unzip工具
apt install zip unzip
安装build工具链
apt install build-essential
安装Python2.7
apt install python
安装wget
apt install wget
安装git
apt install git
安装pip
apt install python-pip
安装ninja
pip install ninja
安装ant
apt install ant
编译Java代码需要用到ant或者gradle，但是我因为是在服务器上进行编译，没有x11环境，也没装Android Studio，所以这里使用ant编译Java。
2.2 编译安装Cmake

如果直接使用apt安装cmake的话，得到的版本会过低（我这里是3.5），而编译OpenCV源码要求的Cmake最低版本是3.6，因此需要从源码编译安装一下。

另一方面，如果所用的cmake版本没有支持HTTPS，则编译OpenCV过程中会报Download failed: 1;"unsupported protocol"错误，导致编译过程中的一些文件无法下载，因此务必安装下面的方法安装Cmake。

下载Cmake3.9.0的源码并编译：

wget --no-check-certificate https://cmake.org/files/v3.9/cmake-3.9.0.tar.gz
tar -zxvf cmake-3.9.0.tar.gz
cd cmake-3.9.0
apt-get install libcurl4-gnutls-dev
apt-get install zlib1g-dev
./bootstrap --system-curl
make && make install

大概需要10分钟时间。

完成后检测安装是否成功：

cmake --version
2.3 安装与配置JDK环境

在官网下载jdk安装包，这里选择的是：

这网站很坑非得要登陆才能下载

然后解压：

tar -zxvf  jdk-8u221-linux-x64.tar.gz

添加环境变量：

export JAVA_HOME=/workspace/opencv4android/tools/jdk/jdk1.8.0_221
export JRE_HOME=${JAVA_HOME}/jre 
export CLASSPATH=.:${JAVA_HOME}/lib:${JRE_HOME}/lib 
export PATH=${JAVA_HOME}/bin:$PATH

source  ~/.bashrc
这里配置的环境变量会在重启后失效，需要在进入docker时重新配置，可以参考第一节中的脚本。（我也不想把环境变量写死以防后面需要更新各种SDK版本）。

运行下面的命令检查JDK是否配置成功：

java -version
2.4 安装与配置Android SDK

在这里下载Android SDK的安装包，这里选择的是android-sdk_r24.4.1-linux.tgz。

将压缩包放置到tools/sdk文件夹并解压：

tar -zvxf android-sdk_r24.4.1-linux.tar

配置环境变量：

export ANDROID_SDK=/workspace/opencv4android/tools/sdk/android-sdk-linux
export PATH=$ANDROID_SDK/tools:$PATH

source  ~/.bashrc

运行下面的命令检查Android SDK是否配置成功：

android -h

运行下面的命令查看安装选项的序列号 ：

android list sdk --all

按序列号安装需要的组件和sdk包：

这里安装以下内容：
24- Android SDK Build-tools, revision 24.0.1
54- SDK Platform Android 7.0, API 24, revision 2
android update sdk -u --all --filter  24,54

然后运行下面的命令安装Android SDK Platform-tools：

这条命令作用是安装所有包，但是这里不需要这么多包，等看到它安装完Android SDK Platform-tools后就可以ctrl-c终止了。
android update sdk --no-ui
2.5 下载OpenCV源码和contrib库

运行下面的命令克隆opencv源码：

git clone https://github.com/opencv/opencv.git
cd opencv && git checkout 3.4

下载opencv的扩展库：

git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib && git checkout 3.4
注意一定要选择切换到合适的分支，否则编译肯定报错的。
这里目前Github仓库的3.4版本为3.4.8。
2.6 开始编译

跟第一节中的方法一样，切换到build目录执行下面的命令即可编译：

python ../opencv/platforms/android/build_sdk.py \
--extra_modules_path=/workspace/_net/opencv_contrib/modules/ \
--config ../opencv/platforms/android/ndk-17.config.py
由于cmake configure可能会失败，多试几次就好了。
所有的指令集都编译完成后，生成的库就在OpenCV-Android-SDK目录下。
3.编译选项分析

编译的配置文件就是上面命令指明的ndk-17.config.py ，里面可以选择需要编译的指令集、STL库等等。

而调用ndk-17.config.py的是build_sdk.py脚本，在里面有ABI类的定义：

class ABI:
    def __init__(self, platform_id, name, toolchain, ndk_api_level = None, cmake_vars = dict()):
        self.platform_id = platform_id # platform code to add to apk version (for cmake)
        self.name = name # general name (official Android ABI identifier)
        self.toolchain = toolchain # toolchain identifier (for cmake)
        self.cmake_vars = dict(
            ANDROID_STL="gnustl_static",
            ANDROID_ABI=self.name,
            ANDROID_PLATFORM_ID=platform_id,
        )
        if toolchain is not None:
            self.cmake_vars['ANDROID_TOOLCHAIN_NAME'] = toolchain
        else:
            self.cmake_vars['ANDROID_TOOLCHAIN'] = 'clang'
            self.cmake_vars['ANDROID_STL'] = 'c++_static'
        if ndk_api_level:
            self.cmake_vars['ANDROID_NATIVE_API_LEVEL'] = ndk_api_level
        self.cmake_vars.update(cmake_vars)
        print ("---> "+self.cmake_vars['ANDROID_TOOLCHAIN'] +" " +self.cmake_vars['ANDROID_ABI']+" " +self.cmake_vars['ANDROID_STL'])
    def __str__(self):
        return "%s (%s)" % (self.name, self.toolchain)
    def haveIPP(self):
        return self.name == "x86" or self.name == "x86_64"

可以看到在ndk-17.config.py中传入的配置参数会在这里被解析更新，比如如果要使用clang编译带NEON支持的armeabi-v7a库并链接c++_shared的话，就改成下面的参数：

ABIs = [
    ABI("2", "armeabi-v7a", None, cmake_vars=dict(ANDROID_ABI='armeabi-v7a with NEON',ANDROID_STL="c++_shared"))
]
就完成啦，得到的OpenCV-Android-SDK库可以在Android Studio中愉快地使用了，不管是Java还是C++接口都很方便，美滋滋~
