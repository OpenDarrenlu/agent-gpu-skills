# 用C++加速julia：BMP图片读取

**作者**: cherichy宁做我

**原文链接**: https://zhuanlan.zhihu.com/p/60377531

---

最近写了一个程序，需要大批量地读取8位的BMP格式的灰度图，程序写完profiler一看，竟然读图和处理的时间不相上下，这是万万不可接受的。经过一番折腾，最后决定用C++来加速8位BMP图片的读取。目的很简单，读进来以后变成Float32类型就好了。

打个广告：julia什么语言都能call，C，C++，Fortran，python，R，AnyCall！

PyCall.jl + OpenCV方案

想想，读图是多么简单的事情啊，OpenCV走一波岂不美滋滋。然而julia没有OpenCV的封装，那就PyCall吧！于是请来Python轻松call了OpenCV。PyCall就是这么轻松愉快，所有python的库都直接call，你的是我的，他的也是我的，全是我的！

using PyCall
cv2=pyimport("cv2")
imreadcv(fname)=Float32.(cv2."imread"(fname,0))

选了张512×512的8位灰度图来测个速呗。快速计算一下大小。512×512=256KB，换成Float32也就1MB。

  memory estimate:  61.25 MiB
  allocs estimate:  2359352
  --------------
  minimum time:     483.473 ms (1.26% GC)
  median time:      495.222 ms (1.87% GC)
  mean time:        503.798 ms (3.19% GC)
  maximum time:     548.973 ms (15.07% GC)
  --------------
  samples:          10
  evals/sample:     1

然而测试结果大跌眼镜，竟然alloc了61.25M？WTF？？？来看看PyCall怎么说的：

Multidimensional NumPy arrays (ndarray) are supported and can be converted to the native Julia Array type, which makes a copy of the data.

Copy，从Python到Julia的数组是Copy的，反之则不是。就算是Copy也不至于多吃60倍内存对不？果断放弃PyCall。

ImageMagick.jl方案

那要不用julia自己的库？ImageMagick.jl 给大名鼎鼎的ImageMagick库做了封装，速度应该不错。这里有个小坑，对于bmp格式的图，默认是3通道！bmp格式里明明有一位告诉了通道数，就这样被忽略了。无奈，必须先转灰度图，变成了0-1的8位浮点表示。不得不说这是Images.jl 的优秀设计，然而这个格式并不适合做图像处理啊，轻松溢出。。无奈，乘以255f0转成了Float32。

using Images,FileIO
imread(fname)=channelview(Gray.(load(fname)))*255f0

现在性能应该不错？毕竟ImageMagick，确实不错，内存直接降了17倍，速度也提升了20倍，简直强大。仔细一看，这个内存占用3.52M有点意思，分析一下。load进来是RGB表示需要256K*3，转成灰度需要256K，然后变成浮点需要1M，理论上只需要2M内存。另外1.5M可能是用来和库交换数据吧，copy一次0.75M，两次就1.5M了。嗯，就这么骗自己吧。

  memory estimate:  3.52 MiB
  allocs estimate:  291
  --------------
  minimum time:     21.443 ms (0.00% GC)
  median time:      24.656 ms (0.00% GC)
  mean time:        25.802 ms (1.93% GC)
  maximum time:     37.566 ms (13.16% GC)
  --------------
  samples:          194
  evals/sample:     1

TIF方案

还是不爽怎么办，明明直接就是灰度，非要读成RGB很难受啊。然而ImageMagick怎么读都是RGB，就算重新save一下灰度图读经来依然是RGB。就是bmp这个格式的锅！看看人家tif格式就没问题。1.78M内存，基本就是浮点的1M，灰度的256K，加上猜测的两次copy，正好。平均时间减少了8ms的快乐，嗯对，8ms的快乐。

imreadtif(fname)=channelview(load(fname))*255f0
julia> @benchmark imreadtif("ref.tif")
BenchmarkTools.Trial:
  memory estimate:  1.78 MiB
  allocs estimate:  346
  --------------
  minimum time:     14.877 ms (0.00% GC)
  median time:      17.154 ms (0.00% GC)
  mean time:        17.916 ms (1.49% GC)
  maximum time:     28.903 ms (11.29% GC)
  --------------
  samples:          279
  evals/sample:     1

然而手头全是bmp的图怎么办。。预处理转tif？可以是可以，但是过于暴力，不优雅。

CxxWrap.jl + C++方案

于是，想到了以前写的cpp版本的读取bmp格式的代码。bmp作为最简单的图像格式之一，基本可以无脑读取。其中有几个字段特别重要，列表如下：

offset  type    fieldName   
10      uint32  OffsetBits  数据的偏移
18      uint32  Width       宽
22      uint32  Height      高
28      uint16  BitCount    每像素位宽（8表示灰度，24表示RGB）

bmp里面每个像素的数据是倒着存的，即第一个数据为图像的右下角，从右往左，从下往上。如果是灰度图，则每次读一个char，如果是RGB，则每次读3个char，分别为RGB。还有一个规则，由于内存对齐的原因，bmp格式的每一行都需要4字节对齐，所以需要根据图像的宽度来计算一下对齐，将多余的数据扔掉即可。

那么代码就很好写了，先读几个重要的字段，计算出每行实际数据宽度，然后无脑读数据即可。只要记住julia是按列，cpp是按行就好了。让我们来愉快地使用C++读取bmp吧！

julia本身可以非常轻松的ccall，调用 C 和 Fortran 无压力。但是我就想要写C++。于是使用了CxxWrap.jl，这个库对大部分cpp特性进行了封装，可以非常轻松地和julia交换数据。在这里我直接在堆上new了一块内存，然后把这个指针交给julia。julia的gc就能自动帮我释放了，delete都不用写有没有！！

#include "jlcxx/jlcxx.hpp"
#include <fstream>

auto readBMP(std::string filename) {
    std::ifstream file(filename, std::ifstream::binary);

    file.seekg(10, std::ios::beg);
    uint32_t OffBits;
    file.read((char *) &OffBits, sizeof(uint32_t));

    file.seekg(18, std::ios::beg);
    uint32_t Width;
    file.read((char *) &Width, sizeof(uint32_t));

    file.seekg(22, std::ios::beg);
    uint32_t Height;
    file.read((char *) &Height, sizeof(uint32_t));

    file.seekg(28, std::ios::beg);
    uint16_t BitCount;
    file.read((char *) &BitCount, sizeof(uint16_t));

    uint32_t BitDepth = BitCount >> 3;
    uint32_t RowWidth = (Width * BitDepth - 1 >> 2) + 1 << 2;

    float* data=new float[Height*Width];
    float *dp=data+Height*Width-1;
    if (BitDepth == 1) {
        uint8_t bit;
        for (uint32_t r = 0; r < Height; r++) {
            file.seekg(OffBits + r * RowWidth, std::ios::beg);
            for (uint32_t c = 0; c < Width; c++) {
                file.read((char *) &bit, sizeof(bit));
                data[Height - 1 - r + c * Height] = static_cast<float>(bit);
            }
        }
    } else if (BitDepth == 3) {
        uint8_t RGB[3];
        for (uint32_t r = 0; r < Height; r++) {
            file.seekg(OffBits + r * RowWidth, std::ios::beg);
            for (uint32_t c = 0; c < Width; c++) {
                file.read((char *) &RGB, sizeof(char) * 3);
                data[Height - 1 - r + c * Height] = 0.299f * RGB[0] + 0.587f * RGB[1] + 0.114f * RGB[2];
            }
        }
    }
    file.close();
    return jlcxx::make_julia_array(data,Height,Width);
}

JLCXX_MODULE define_julia_module(jlcxx::Module& mod){
    mod.method("load",&readBMP);
}


逻辑非常的C++，只是最后返回时，把指针给了julia托管。感谢C++14，我能用auto做返回值，毕竟我真的不知道也不care返回了什么对不？最后一行轻松创建一个module，服从FileIO.jl的规则，起名为load，把cpp的函数指针给他就好了，参数为string，返回一个julia的array，轻松愉快。

当然，这个是要编译成动态库的，相应的cmake规则如下。需要link julia和cxxwrap_julia两个库。我这里用的是windows+msys2上的mingw，gcc8.3，就是这么潮。linux上也一样，文件路径换一下就好了。

set(CMAKE_CXX_STANDARD 14)

include_directories(D:/Julia/include/julia)
link_directories(D:/Julia/lib)
include_directories(D:/JuliaPkg/Pkgs/packages/CxxWrap/KcmSi/deps/usr/include)
link_directories(D:/JuliaPkg/Pkgs/packages/CxxWrap/KcmSi/deps/usr/lib)

add_definitions(-DJULIA_ENABLE_THREADING)

add_library(testCxx SHARED library.cpp)
target_link_libraries(testCxx cxxwrap_julia julia)

编译结束后，我们得到了一个libtestCxx.dll。为了在julia端调用它，我们还需要稍微打个包，创建一个module BMP，@wrapmodule 和 @initcxx 两个宏会为我们做好一切，只需要改一下库的名字即可。然后就可以用BMP.load()来进行读取啦，把这个文件存为BMP.jl。

module BMP
    using CxxWrap
    @wrapmodule(joinpath(@__DIR__,"libtestCxx"))
    function __init__()
        @initcxx
    end
end

做完了这一切，终于到了紧张而轻松的时刻，见证C++的时刻到了。

include("BMP.jl")
imreadcpp(file)=BMP.load(file)
julia> @benchmark imreadcpp("ref.bmp")
BenchmarkTools.Trial:
  memory estimate:  1.00 MiB
  allocs estimate:  4
  --------------
  minimum time:     9.562 ms (0.00% GC)
  median time:      10.709 ms (0.00% GC)
  mean time:        11.074 ms (0.00% GC)
  maximum time:     33.798 ms (0.00% GC)
  --------------
  samples:          444
  evals/sample:     1

非常完美，内存就是整整的1M，没有任何多余的开销，包括GC。julia不愧是万能call，而且基本没有call的开销啊啊啊，太香了！平均速度11ms，比tif方案又低了6ms，啊，6ms的快乐。没有GC的C++好香甜~

经过一番折腾，终于舒服了，接口优雅，快乐，香甜，具有极佳的食用体验 。大家快来使用万能call机julia，啥都别说了，快上车，call起来！
