# OpenCV源码交叉编译android库

**作者**: 2know​东北大学 信号与信息处理硕士

**原文链接**: https://zhuanlan.zhihu.com/p/58831589

---

OpenCV源码编译android库

之前的文章中介绍过如何源码编译OpenCV库，但是当时编译的是pc的cpu版本，但是在使用中，很多代码最终要运行在手机等便携式设备上，所以本文介绍一下，如何使用OpenCV源码检查编译构建android的opencv库。

交叉编译环境构建

首先需要下载android NDK，下载地址： https://developer.android.google.cn/ndk/downloads；

下载后解压，将文件夹拷贝到相应的目录下，例如/opt/目录下；例如我用的是android-ndk-16b,在android-ndk-r16b/build/tools目录下可以找到make-standalone-toolchain.sh文件，该文件可以帮助我们生成独立编译工具链，可以使用./make-standalone-toolchain --help，命令查看该工具的使用方法：

Valid options (defaults are in brackets):

  --help                   Print this help.
  --verbose                Enable verbose mode.
  --dryrun                 Unsupported.
  --toolchain=<name>       Specify toolchain name
  --use-llvm               No-op. Clang is always available.
  --stl=<name>             Specify C++ STL [gnustl]
  --arch=<name>            Specify target architecture
  --abis=<list>            No-op. Derived from --arch or --toolchain.
  --ndk-dir=<path>         Unsupported.
  --package-dir=<path>     Place package file in <path> [/tmp/ndk-nn]
  --install-dir=<path>     Don't create package, install files to <path> instead.
  --dryrun                 Unsupported.
  --platform=<name>        Specify target Android platform/API level. [android-14]
  --force                  Remove existing install directory.



我们使用如下命令生成独立工具链：

../android-ndk-r16b/build/tools/make-standalone-toolchain.sh\
--toolchain=arm-linux-androideabi-4.9\ #编译工具名字
--platform=android-21\ android api的级别
--install-dir=../android-toolchain-r16b/android-armv7\ 生成工具链的安装目录
--arch=arm\ #编译器版本
--force #使用该选项会先移除安装目录下已经存在的工具 



完成工具链的构建后，就可以进行OpenCV的编译了。

编译OpenCV android库

库的编译非常简单，我们只需要设置OpenCV cmake的编译选项即可，其他和pc端opencv库的编译一样；编译选项设置如下：

export ANDROID_NDK=/home/nn/Project/android-ndk-r14b

cmake -DCMAKE_TOOLCHAIN_FILE=../platforms/android/android.toolchain.cmake \
-DCMAKE_ANDROID_NDK=/home/nn/Project/android-ndk-r14b \
-DANDROID_NATIVE_API_LEVEL=21 \
-DBUILD_ANDROID_PROJECTS=OFF \
-DBUILD_ANDROID_EXAMPLES=OFF \
-DCMAKE_BUILD_TYPE=Release  \
-DBUILD_JAVA=OFF  \
-DCMAKE_ANDROID_ARCH_ABI=armeabi-v7a \
-DCMAKE_INSTALL_PREFIX=/youpath/opencv/install ..


-DCMAKE_TOOLCHAIN_FILE
这个选项是指定交叉编译的cmake路径；在OpenCV工程的platform目录下，android的目录下；
-DCMAKE_ANDROID_NDK
指定NDK的目录，注意在cmake之前使用了export来设置android-ndk环境变量；该选项是为cmake指定android-ndk的路径；
-DANDROID_NATIVE_API_LEVEL
使用该选项指定android的api级别
-DBUILD_ANDROID_PROJECTS
注意我们关掉了android project的编译，同时也关掉了-DBUILD_ANDROID_EXAMPLES因为如果要编译这些，需要我们配置android SDK等更多环境，而我们需要的是opencv的android下的c++计算库，所以不需要编译android project；
-DCMAKE_ANDROID_ARCH_ABI 设置编译的版本，目前设置的是armv7版本，如果需要编译arm64，则设置为arm64-v8a即可。 该选项可以设置为：
“armeabi-v7a”, “armeabi”, “armeabi-v7a with NEON”, “armeabi-v7a-hard with NEON”, “armeabi-v7a with VFPV3”, “armeabi-v6 with VFP”, “arm64-v8a”, “mips”, “mips64”, “x86”, “x86_64”
-DCMAKE_INSTALL_PREFIX
最后设置了android目录；这个是Cmake的选项，如果不设置默认安装在/usr/local目录下。
编译结束后，我们可以在install目录下，看到编译后的库文件及头文件等，如下：
 |-- 3rdparty
 |   `-- libs
 |       `-- armeabi-v7a
 |           |-- libIlmImf.a
 |           |-- libcpufeatures.a
 |           |-- liblibjasper.a
 |           |-- liblibjpeg-turbo.a
 |           |-- liblibpng.a
 |           |-- liblibprotobuf.a
 |           |-- liblibtiff.a
 |           |-- liblibwebp.a
 |           |-- libquirc.a
 |           `-- libtegra_hal.a

...


     `-- staticlibs
            `-- armeabi-v7a
                |-- libopencv_calib3d.a
                |-- libopencv_core.a
                |-- libopencv_dnn.a
                |-- libopencv_features2d.a
                |-- libopencv_flann.a
                |-- libopencv_highgui.a
                |-- libopencv_imgcodecs.a
                |-- libopencv_imgproc.a
                |-- libopencv_ml.a
                |-- libopencv_objdetect.a
                |-- libopencv_photo.a
                |-- libopencv_stitching.a
                |-- libopencv_video.a
                `-- libopencv_videoio.a



以上是库的部分内容，可以使用tree命令查看install目录的结构，查看编译出来的库；OpenCV中编译的时候，编译选项非常多，可以查看CmakeList，关掉一些不需要的选项。

交叉编译android下的OpenCV demo

编译好库以后，便可以尝试构建opencv程序；本片文章使用boxfilter作为示例，在以后的文章中会专门写一个关于OpenCV的filter系列；这里算是一个引子吧。

c++代码如下：

 #include "opencv2/opencv.hpp"
 
 int main()
 {
     //read picture
     cv::Mat img = cv::imread("test.jpg");
     cv::Mat out;
     cv::boxFilter(img, out, -1, cv::Size(5, 5));
     cv::imwrite("result.jpg", out);
 
     return 0;
 }



这是一个简单的boxfilter测试程序；接下来需要构建cmake，交叉编译该代码；
CMakeLists.txt如下：

cmake_minimum_required(VERSION 2.8.3)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(OpenCVHome /youpath/opencv/install/sdk/native)
set(NDK_STANDALONE_TOOLCHAIN /youpath/android-toolchain-r14b/android-armv7)
set(CMAKE_C_COMPILER ${NDK_STANDALONE_TOOLCHAIN}/bin/clang)
set(CMAKE_CXX_COMPILER ${NDK_STANDALONE_TOOLCHAIN}/bin/clang++)
set(CMAKE_FIND_ROOT_PATH ${NDK_STANDALONE_TOOLCHAIN})
add_definitions("--sysroot=${NDK_STANDALONE_TOOLCHAIN}/sysroot")


include_directories(${OpenCVHome}/jni/include)
link_directories(${OpenCVHome}/staticlibs/armeabi-v7a
                 ${OpenCVHome}/3rdparty/libs/armeabi-v7a)

link_libraries(
        opencv_imgcodecs
        opencv_imgproc
        opencv_highgui
        opencv_core
        opencv_video
        opencv_features2d
        opencv_videoio
        cpufeatures
        tegra_hal
        IlmImf
        libjasper
        libjpeg-turbo
        libpng
        libprotobuf
        libtiff
        libwebp
        quirc
        log
        z
        )
add_executable(boxfilter_arm ./src/boxfilter_arm.cpp)
                                                                          


以上是一个加单的CMake，仅仅用于测试效果； 编译完成后在手机上运行可执行文件，结果如下：
test.jpg

result.jpg

可以明显看到图像模糊了。至此，我们在叉编译opencv android库完成；

欢迎关注公众号：计算机视觉与高性能计算(to_know)
