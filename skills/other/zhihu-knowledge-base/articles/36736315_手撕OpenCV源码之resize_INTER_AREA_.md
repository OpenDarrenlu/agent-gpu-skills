# 手撕OpenCV源码之resize<INTER_AREA>

**作者**: 2know​东北大学 信号与信息处理硕士

**原文链接**: https://zhuanlan.zhihu.com/p/36736315

---

手撕OpenCV源码之resize<INTER_AREA>

resize在modules/imgproc/src/文件件中,首先看resize API的函数实现:




void resize(InputArray src, OutputArray dst,
            Size dsize, double fx=0, double fy=0, 
            int interpolation=INTER_LINEAR )


参数说明:

src:输入图像
dst:输出图像,dst的数据类型与src相同.
dsize:这个参数是输出图像的尺寸,两种情况,如果该参数设置为0,api会自动计算
输出参数,否则按照输入尺寸.dst的计算公式:
dsize = Size(round(fx×src.cols),round(fy×src.rows))
所以当dsize为0的时候,fx和fy不能为0.
fx:
(double)dsize.width/src.cols
fy:
(double)dsize.height/src.rows
interpolation:插值方法.
在opencv中提供了5中插值方式:
INTER_NEAREST:最邻近插值
INTER_LINEAR:双线性插值,默认情况下使用该方式进行插值.
INTER_AREA:基于区域像素关系的一种重采样或者插值方式.该方法是图像抽取的首选方法,它可以产生更少的波纹,但是当图像放大时,它的效果与INTER_NEAREST效果相似.
INTER_CUBIC:4×4邻域双3次插值
INTER_LANCZOS4:8×8邻域兰索斯插值
5种插值的模板实例化(部分代码)




    static ResizeAreaFastFunc areafast_tab[] =
    {
        resizeAreaFast_<uchar, int, ResizeAreaFastVec<uchar> >,
        0,
        resizeAreaFast_<ushort, float, ResizeAreaFastVec<ushort> >,
        resizeAreaFast_<short, float, ResizeAreaFastVec<short> >,
        0,
        resizeAreaFast_<float, float, ResizeAreaFastNoVec<float, float> >,
        resizeAreaFast_<double, double, ResizeAreaFastNoVec<double, double> >,
        0
    };

    static ResizeAreaFunc area_tab[] =
    {
        resizeArea_<uchar, float>, 0, resizeArea_<ushort, float>,
        resizeArea_<short, float>, 0, resizeArea_<float, float>,
        resizeArea_<double, double>, 0
    };               


从代码中可以看到,opencv中的5种插值方式支持的数据类型有uchar,ushort,short,float,double.

计算缩放系数




if( !dsize.area() )
    {
        dsize = Size(saturate_cast<int>(src.cols*inv_scale_x),
            saturate_cast<int>(src.rows*inv_scale_y));
        CV_Assert( dsize.area() );
    }
    else
    {
        inv_scale_x = (double)dsize.width/src.cols;
        inv_scale_y = (double)dsize.height/src.rows;
    }
    _dst.create(dsize, src.type());
    Mat dst = _dst.getMat();


#ifdef HAVE_TEGRA_OPTIMIZATION
    if (tegra::resize(src, dst, (float)inv_scale_x, (float)inv_scale_y, interpolation))
        return;
#endif

    int depth = src.depth(), cn = src.channels();
    double scale_x = 1./inv_scale_x, scale_y = 1./inv_scale_y;
    int k, sx, sy, dx, dy;


这里是按照dsize的计算规则,计算dsize和缩放的比例.然后为dst创建空间.接下来的几行代码中可以看出,opencv针对NIVIDIA的tegra设备提供了优化.继续往后看,最先出现的是INTER_NEAREST方法.




INTER_NEAREST插值




if( interpolation == INTER_NEAREST )
    {
        resizeNN( src, dst, inv_scale_x, inv_scale_y );
        return;
    }


如果插值方式采用的是最邻近法,那么直接调用resizeNN计算.跳转到resizeNN,可以看到,其源码如下:




INTER_AREA插值




INTER_LINEAR插值处理图像缩小两倍




int iscale_x = saturate_cast<int>(scale_x);
        int iscale_y = saturate_cast<int>(scale_y);

        bool is_area_fast = std::abs(scale_x - iscale_x) < DBL_EPSILON &&
                std::abs(scale_y - iscale_y) < DBL_EPSILON;

        // in case of scale_x && scale_y is equal to 2
        // INTER_AREA (fast) also is equal to INTER_LINEAR
        if( interpolation == INTER_LINEAR && is_area_fast && iscale_x == 2 && iscale_y == 2 )
        {
            interpolation = INTER_AREA;
        }              


当缩小2倍(注意输入输出尺寸是整除的)的时候,INTER_LINEAR插值按照INTER_AREA计算,即当缩小两倍的时候INTER_LINEAR与INTER_AREA是相同的.DBL_EPSILON是c++中的误差系数.




INTER_AREA处理图像缩小

当scale_x >= 1 && scale_y >= 1,从前面的代码中可以知道,此时是图片缩小.地阿妈如下,可以看出,当缩小整数倍的时候,使用resizeAreaFast_函数计算,当缩小非整数倍的时候,使用resizeArea_计算.




if( interpolation == INTER_AREA && scale_x >= 1 && scale_y >= 1 )
        {
            if( is_area_fast )
            {
              ...//resizeAreaFast_
            }
            ...//resizeArea_    
        }               





INTER_AREA处理图像放大

代码如下:




int xmin = 0, xmax = dsize.width, width = dsize.width*cn;
    bool area_mode = interpolation == INTER_AREA;
    bool fixpt = depth == CV_8U;
    float fx, fy;
    ResizeFunc func=0;
    int ksize=0, ksize2;
    if( interpolation == INTER_CUBIC )
        ksize = 4, func = cubic_tab[depth];
    else if( interpolation == INTER_LANCZOS4 )
        ksize = 8, func = lanczos4_tab[depth];
    else if( interpolation == INTER_LINEAR || interpolation == INTER_AREA )//重点在这里
        ksize = 2, func = linear_tab[depth];
    else
        CV_Error( CV_StsBadArg, "Unknown interpolation method" );
    ksize2 = ksize/2;              


代码中注释的地方,可以看到,INTER_AREA处理图像放大的时候,使用的是INTER_LINEAR插值方式.
综上所示,INTER_AREA和INTER_LINEAR是配合使用的,当放大的时候,两种差值方式都是INTER_LINEAR,当缩小2倍的时候,两种插值方式都是INTER_AREA.




resizeAreaFast_解析

首先来看调用resizeAreaFast_函数之前的数据处理,代码如下:




if( is_area_fast )
            {
                int area = iscale_x*iscale_y;
                size_t srcstep = src.step / src.elemSize1();
                AutoBuffer<int> _ofs(area + dsize.width*cn);
                int* ofs = _ofs;
                int* xofs = ofs + area;
                ResizeAreaFastFunc func = areafast_tab[depth];
                CV_Assert( func != 0 );

                for( sy = 0, k = 0; sy < iscale_y; sy++ )
                    for( sx = 0; sx < iscale_x; sx++ )
                        ofs[k++] = (int)(sy*srcstep + sx*cn);

                for( dx = 0; dx < dsize.width; dx++ )
                {
                    int j = dx * cn;
                    sx = iscale_x * j;
                    for( k = 0; k < cn; k++ )
                        xofs[j + k] = sx + k;
                }

                func( src, dst, ofs, xofs, iscale_x, iscale_y );
                return;
            }               


可以看到在调用resizeAreaFast_之前函数求了两个数组,分别是ofs和xofs,ofs中保存了area个值,area个数据的索引.假设area==9,如下图所示,area的9个索引,就是以(a,b)为起点的9个点的索引.也就是说,在src中取任意坐标(a,b)作为基数,都可以通过这9个索引,获得9个位置的值.

(a+0,b+0)(a+1,b+0)(a+2,b+0)(a+0,b+1)(a+1,b+1)(a+2,b+1)(a+0,b+2)(a+1,b+2)(a+2,b+2)

接下来看,xofs中的值,xofs中一共有dsize.width * nc个数据.从第二个for循环中的计算知道,xfos中的值是dst中x方向坐标映射回src后的取值.
经过以上的分析,我们大概能猜出算法的执行过程了,首先是按行计算,所以第一层循环是0到dsiz.height,然后进行dst.width个循环,每次计算dst的一个值.计算方式是,首先按照dst中位置的x方向坐标,读取xofs中相应位置的数据,这个数据便是src中的基坐标,然后按照这个基坐标,和ofs中的9个索引,读出src中的9个位置的值,然后操作这9个值,将计算结果写入dst中,dst中该位置的数值计算结束,循环变量加1,开始计算下一个位置.
OpenCV的整体实现相对复杂,我们一块一块看,首先查找到resizeAreaFast_函数,代码如下所示:




template<typename T, typename WT, typename VecOp>
static void resizeAreaFast_( const Mat& src, Mat& dst, const int* ofs, const int* xofs,
                             int scale_x, int scale_y )
{
    Range range(0, dst.rows);
    resizeAreaFast_Invoker<T, WT, VecOp> invoker(src, dst, scale_x,
        scale_y, ofs, xofs);
    parallel_for_(range, invoker, dst.total()/(double)(1<<16));
}            


这里面需要关注3各部分,首先是VecOp,这是一个向量操作,是出于程序执行效率的考虑做的,但是我们需要他做了什么,来了解程序的执行流程.第二就是parallel_for_,这是一个并行化操作的语句,也是为了提升程序执行的效率,另外这个不影响我们理解程序执行流程,所以不做追究,以后的博文中可能进行详细讲解.第三就是resizeAreaFast_Invoker,这是resizeAreaFast_的主体部分,所以这是我们重点关注的.
resizeAreaFast_Invoker代码如下:(由于代码较长,我们分段进行分析)




virtual void operator() (const Range& range) const
    {
        Size ssize = src.size(), dsize = dst.size();
        int cn = src.channels();
        int area = scale_x*scale_y;
        float scale = 1.f/(area);
        int dwidth1 = (ssize.width/scale_x)*cn;
        dsize.width *= cn;
        ssize.width *= cn;
        int dy, dx, k = 0;

        VecOp vop(scale_x, scale_y, src.channels(), (int)src.step/*, area_ofs*/);
        ....
    }               


前面的计算代码计算了一些变量,获取数据尺寸,图像channel,像素权值(scale)等,需要关注的是vop,查看函数模板的参数:




template <typename T, typename WT, typename VecOp>
class resizeAreaFast_Invoker :
    public ParallelLoopBody
{
  ...
}                  


回顾resizeAreaFast_的模板实例化,代码如下:




static ResizeAreaFastFunc areafast_tab[] =
{
    resizeAreaFast_<uchar, int, ResizeAreaFastVec<uchar> >,
    0,
    resizeAreaFast_<ushort, float, ResizeAreaFastVec<ushort> >,
    resizeAreaFast_<short, float, ResizeAreaFastVec<short> >,
    0,
    resizeAreaFast_<float, float, ResizeAreaFastNoVec<float, float> >,
    resizeAreaFast_<double, double, ResizeAreaFastNoVec<double, double> >,
    0
};             


可以看到第三个参数ResizeAreaFastVec<>,从名称看这是一个向量操作,查找函数源码:




template<typename T>
struct ResizeAreaFastVec
{
    ResizeAreaFastVec(int _scale_x, int _scale_y, int _cn, int _step/*, const int* _ofs*/) :
        scale_x(_scale_x), scale_y(_scale_y), cn(_cn), step(_step)/*, ofs(_ofs)*/
    {
        fast_mode = scale_x == 2 && scale_y == 2 && (cn == 1 || cn == 3 || cn == 4);
    }

    int operator() (const T* S, T* D, int w) const
    {
        if( !fast_mode )
            return 0;

        const T* nextS = (const T*)((const uchar*)S + step);
        int dx = 0;

        if (cn == 1)
            for( ; dx < w; ++dx )
            {
                int index = dx*2;
                D[dx] = (T)((S[index] + S[index+1] + nextS[index] + nextS[index+1] + 2) >> 2);
            }
        ...
        return dx;
    }

private:
    int scale_x, scale_y;
    int cn;
    bool fast_mode;
    int step;
};              


代码比较长,以cn==1为例介绍,首先在初始化的时候计算了fast_mode参数,该参数为真的情况是,图像的宽和高分别缩小2倍,且通道为1/3/4,不支持2通道.若满足该条件,则直接计算.for循环代码很容易理解,每次读取4个值加和取平均.与使用ofs和xofs索引得到的结果是相同的.对于不满足要求的情况直接返回0;也就是说,这段代码完成了width和height缩小两倍的计算.
继续看缩小倍数不是2的情况:




for( dy = range.start; dy < range.end; dy++ )
        {  
          ...
        }   


与之前的推断是一样的,按行进行计算,每次循环计算一行,总共dst.height次循环.




T* D = (T*)(dst.data + dst.step*dy);
           int sy0 = dy*scale_y;
           int w = sy0 + scale_y <= ssize.height ? dwidth1 : 0;

           if( sy0 >= ssize.height )
           {
               for( dx = 0; dx < dsize.width; dx++ )
                   D[dx] = 0;
               continue;
           }

           dx = vop((const T*)(src.data + src.step * sy0), D, w);


首先是调整指针到当前行的行首,指针为D;然后设置w,当sy0 + scale_y <= ssize.height不成立的时候,当前行已经超出计算范围,所以w置零,否则为dwidth1.if语句对D指向的行进行清零;启动vop,若果是缩小2倍,可以在vop中完成当前行的计算,返回dx,若计算完成,dx为dwidth,若不符合条件则返回0,继续后续计算.




for( ; dx < w; dx++ )
            {
                const T* S = (const T*)(src.data + src.step * sy0) + xofs[dx];
                WT sum = 0;
                k = 0;
                #if CV_ENABLE_UNROLLED
                for( ; k <= area - 4; k += 4 )
                    sum += S[ofs[k]] + S[ofs[k+1]] + S[ofs[k+2]] + S[ofs[k+3]];
                #endif
                for( ; k < area; k++ )
                    sum += S[ofs[k]];

                D[dx] = saturate_cast<T>(sum * scale);
            }   


可以看到,如果计算完成,则不会执行;如果不满足,dx依然是0,则这行该代码块;代码块的执行与前面分析的计算方式相同,在此不赘述.
代码块中使用了循环展开,有助于提升代码性能,包括缩小二倍的计算也就是vop中的计算也是进行了循环展开,提升代码执行效率.
再看最后一段代码:




for( ; dx < dsize.width; dx++ )
            {
                WT sum = 0;
                int count = 0, sx0 = xofs[dx];
                if( sx0 >= ssize.width )
                    D[dx] = 0;

                for( int sy = 0; sy < scale_y; sy++ )
                {
                    if( sy0 + sy >= ssize.height )
                        break;
                    const T* S = (const T*)(src.data + src.step*(sy0 + sy)) + sx0;
                    for( int sx = 0; sx < scale_x*cn; sx += cn )
                    {
                        if( sx0 + sx >= ssize.width )
                            break;
                        sum += S[sx];
                        count++;
                    }
                }

                D[dx] = saturate_cast<T>((float)sum/count);
            }


这段代码其实是边界处理.如果w<dwidth,那么就需要执行这段代码,进行最后几个点的计算.可以看到最后几个点由于不够area个,所以各像素的系数采用1/count的形式计算.
到此为止,resizeAreaFast_计算结束,也就是说缩小整数倍的情况计算完了.
总结:
在缩小整数倍的情况下分为缩小2倍和其他倍数,缩小二倍直接计算,不采用xofs和ofs的索引,有利于替升计算效率.同时其他情况也采用for循环展开等方式.




resizeArea解析

接着resize的主程序看,代码如下:




ResizeAreaFunc func = area_tab[depth];
            CV_Assert( func != 0 && cn <= 4 );

            AutoBuffer<DecimateAlpha> _xytab((ssize.width + ssize.height)*2);
            DecimateAlpha* xtab = _xytab, *ytab = xtab + ssize.width*2;

            int xtab_size = computeResizeAreaTab(ssize.width, dsize.width, cn, scale_x, xtab);
            int ytab_size = computeResizeAreaTab(ssize.height, dsize.height, 1, scale_y, ytab);

            AutoBuffer<int> _tabofs(dsize.height + 1);
            int* tabofs = _tabofs;
            for( k = 0, dy = 0; k < ytab_size; k++ )
            {
                if( k == 0 || ytab[k].di != ytab[k-1].di )
                {
                    assert( ytab[k].di == dy );
                    tabofs[dy++] = k;
                }
            }
            tabofs[dy] = ytab_size;

            func( src, dst, xtab, xtab_size, ytab, ytab_size, tabofs );         


这段代码是对于放大倍数为小数的情况的处理,例如输入为10 * 10 输出为3 * 3.看代码,申请了两段内存,xtab/ytab,这是两个结构体类型的数据,定义如下:




struct DecimateAlpha
{
    int si, di;
    float alpha;
};


两个int类型的数据,si是src中的索引,di是dst中与之对应的索引.这个有点例似双线性插值,但是双线性插值涉及的是4个点,area插值涉及的是area个点,所以需要求出宽和高两个方向上的信息.
可能会发现,在ResizeAreaFast_中只求行索引,在ResizeArea_中需要两个方向,因为在ResizeArea_中不同点的权值不同.
代码使用computeResizeAreaTab函数计算索引和权值.查看computeResizeAreaTab代码:




static int computeResizeAreaTab( int ssize, int dsize, int cn, double scale, DecimateAlpha* tab )
{
    int k = 0;
    for(int dx = 0; dx < dsize; dx++ )
    {
        double fsx1 = dx * scale;
        double fsx2 = fsx1 + scale;
        double cellWidth = min(scale, ssize - fsx1);

        int sx1 = cvCeil(fsx1), sx2 = cvFloor(fsx2);

        sx2 = std::min(sx2, ssize - 1);
        sx1 = std::min(sx1, sx2);

        if( sx1 - fsx1 > 1e-3 )
        {
            assert( k < ssize*2 );
            tab[k].di = dx * cn;
            tab[k].si = (sx1 - 1) * cn;
            tab[k++].alpha = (float)((sx1 - fsx1) / cellWidth);
        }

        for(int sx = sx1; sx < sx2; sx++ )
        {
            assert( k < ssize*2 );
            tab[k].di = dx * cn;
            tab[k].si = sx * cn;
            tab[k++].alpha = float(1.0 / cellWidth);
        }

        if( fsx2 - sx2 > 1e-3 )
        {
            assert( k < ssize*2 );
            tab[k].di = dx * cn;
            tab[k].si = sx2 * cn;
            tab[k++].alpha = (float)(min(min(fsx2 - sx2, 1.), cellWidth) / cellWidth);
        }
    }
    return k;
}             


用一个简单的例子来说明代码的执行情况,假设src.width = 26,dst.width = 6,那么scale = 4.33333.

dx012345fsx104.38.612.917.321.6fsx24.38.612.917.321.626cellWidth4.34.34.34.34.34.3sx24812172125sx1058131822

按照上面的数据表格容易计算出tab的值,可以知道,di中保存的是dst的索引,si中保存的是si的索引,但是需要注意的是,相同的di值一般情况下下会对应不同的多个si值,因为这是图像缩小,所以一个dst的像素点是由多个src中像素点共同确定的.只是每个src的权值不同.
k是整个tab的元素数量,也就是结构体数组的size.
继续向下看代码:




AutoBuffer<int> _tabofs(dsize.height + 1);
            int* tabofs = _tabofs;
            for( k = 0, dy = 0; k < ytab_size; k++ )
            {
                if( k == 0 || ytab[k].di != ytab[k-1].di )
                {
                    assert( ytab[k].di == dy );
                    tabofs[dy++] = k;
                }
            }
            tabofs[dy] = ytab_size;


这段代码其实是确定每个映射点的height方向的起始位置.怎么理解呢?每一个dst点映射到src中可能不是又固定的点数确定的,也就是说映射到src中他的邻域大小不一样.所以tabofs中存储的是当前的dst索引,映射到src中索引的起始位置,也就是说映射到src中,当前dst点的邻域为3*4,但是它有一个基坐标,基坐标的height方向索引,就存储在tabofs中.
接下来的代码就是resizeArea的具体实现.代码如下(代码很长,节选cn==1作为例子):




virtual void operator() (const Range& range) const
    {
        Size dsize = dst->size();
        int cn = dst->channels();
        dsize.width *= cn;
        AutoBuffer<WT> _buffer(dsize.width*2);
        const DecimateAlpha* xtab = xtab0;
        int xtab_size = xtab_size0;
        WT *buf = _buffer, *sum = buf + dsize.width;
        int j_start = tabofs[range.start], j_end = tabofs[range.end], j, k, dx, prev_dy = ytab[j_start].di;

        for( dx = 0; dx < dsize.width; dx++ )
            sum[dx] = (WT)0;

        for( j = j_start; j < j_end; j++ )
        {
            WT beta = ytab[j].alpha;
            int dy = ytab[j].di;
            int sy = ytab[j].si;

            {
                const T* S = (const T*)(src->data + src->step*sy);
                for( dx = 0; dx < dsize.width; dx++ )
                    buf[dx] = (WT)0;

                if( cn == 1 )
                //邻域中width方向数据加权累加.
                    for( k = 0; k < xtab_size; k++ )
                    {
                        int dxn = xtab[k].di;
                        WT alpha = xtab[k].alpha;
                        buf[dxn] += S[xtab[k].si]*alpha;
                    }
                    ...
            }
            //area插值,实际就是对映射回src中的点的邻域中的点进行加权累加
            //上一段代码计算的是width方向上邻域数据的加权累加
            //这里需要判断height方向上是否计算结束
            if( dy != prev_dy )
            {
              //如果计算结束,将邻域累加结果又sum写入dst
              T* D = (T*)(dst->data + dst->step*prev_dy);
              for( dx = 0; dx < dsize.width; dx++ )
              {
                //邻域累加结果由sum写入dst
                  D[dx] = saturate_cast<T>(sum[dx]);
                  //width方向上的累加结果写入sum,清理sum中存储的上一次的累加结果.
                  sum[dx] = beta*buf[dx];
              }
              prev_dy = dy;
            }
            else
            {
              //没有计算结束,就将width方向邻域累加结果累加到sum
              //其实就是在做height方向邻域的累加.
              for( dx = 0; dx < dsize.width; dx++ )
                  sum[dx] += beta*buf[dx];
            }
          }
          //最后一次累加,写入dst最后一行
          {
           T* D = (T*)(dst->data + dst->step*prev_dy);
           for( dx = 0; dx < dsize.width; dx++ )
           D[dx] = saturate_cast<T>(sum[dx]);
          }
        }


注释的内容可能比较绕,这里做一个简单的补充:邻域的width方向上的累加容易理解,每次计算一行,计算结束后,先判断height方向上是不是累加结束,没有结束,就在height方向做一次累加.假设当前进行的是邻域的height方向上最后一次累加,那么其实在判断的时候,是false的,然后进入else完成最后一次累加,但是累加结果还是存在sum中的,需要到计算下一个点的邻域累加的时候,才会判断为true,进入if,此时把上一点的邻域累加结果写入dst,然后把当前点的第一个width方向上的累加结果写入sum,注意是写入不是累加.这样一来,就需要在for循环之外增加一次写入.因为dst的最后一行数据计算结束,不会再进入for循环,因此需要在for循环之外,把sum中的结果写入dst.
到此为止,resizeArea的计算结束了.可能会想到之前讲的都是缩小的情况,放大的情况是怎么处理的呢?
看下面的代码:




int xmin = 0, xmax = dsize.width, width = dsize.width*cn;
    bool area_mode = interpolation == INTER_AREA;
    bool fixpt = depth == CV_8U;
    float fx, fy;
    ResizeFunc func=0;
    int ksize=0, ksize2;
    if( interpolation == INTER_CUBIC )
        ksize = 4, func = cubic_tab[depth];
    else if( interpolation == INTER_LANCZOS4 )
        ksize = 8, func = lanczos4_tab[depth];
    else if( interpolation == INTER_LINEAR || interpolation == INTER_AREA )//看这里
        ksize = 2, func = linear_tab[depth];         


看到了吧,放大情况,resizeArea使用的是linear插值.
补充
func声明的时候,使用了参数depth,简单介绍一下:




#define CV_8U   0
#define CV_8S   1
#define CV_16U  2
#define CV_16S  3
#define CV_32S  4
#define CV_32F  5
#define CV_64F  6
#define CV_USRTYPE1 7          


以上是depth的取值,U代表unsigned,S代表signed.而ResizeFastFunc模板的实例化如下:




static ResizeAreaFastFunc areafast_tab[] =
    {
        resizeAreaFast_<uchar, int, ResizeAreaFastVec<uchar> >,
        0,
        resizeAreaFast_<ushort, float, ResizeAreaFastVec<ushort> >,
        resizeAreaFast_<short, float, ResizeAreaFastVec<short> >,
        0,
        resizeAreaFast_<float, float, ResizeAreaFastNoVec<float, float> >,
        resizeAreaFast_<double, double, ResizeAreaFastNoVec<double, double> >,
        0
    };





ResizeAreaFunc func = area_tab[depth];


以上是实例化列表,8个实例化对应8中深度取值.调用对应的实例化函数完成插值.




欢迎关注公众号：计算机视觉与高性能计算(to_2know)

http://weixin.qq.com/r/sShvd-bEIKtfrbIU932j (二维码自动识别)
