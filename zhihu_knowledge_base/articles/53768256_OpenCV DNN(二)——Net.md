# OpenCV DNN(二)——Net

**作者**: 2know​东北大学 信号与信息处理硕士

**原文链接**: https://zhuanlan.zhihu.com/p/53768256

---

OpenCV DNN之Net

好久没有更新了，作为2019年的首发，希望2019年会是腾飞的一年，祝愿大家2019一切都很美好，能在公众号收货更多的干货，大家能一起进步，心想事成。 上一篇博文最后留下了一个尾巴，是关于Net的setInput和forward,当时分别介绍了，这两个函数的定义。本文暂时不深入介绍这两个函数，从OpenCV DNN的Net类入手，拆解OpenCV中DNN的结构。本文主要介绍Net类并且提供googleNet的demo。

Net类的定义

path：opencv/modules/dnn/include/opencv2/dnn/dnn.hpp +365

这个类中定义了创建和操作网络的方法；所谓神经网络其实是一个有向无环图(DAG)，图的顶点是层的实例，边表示输入输出关系。每一个层，在网络中都有唯一的整数ID和字符串名称作为标识；同时，这个类支持副本的引用计数，也就是说副本指向同一个实例。

以下是Net类的源代码：

class CV_EXPORTS_W_SIMPLE Net
{
public:

    CV_WRAP Net();  //!< 默认构造函数
    CV_WRAP ~Net(); //!< 默认析构函数；引用计数为0则析构

     //使用Inter model优化器的中间表示来创建网络；
     //xml是网络拓扑结构的XML配置文件
     //bin是model的model的二进制文件
     //使用Inter model优化器创建网络，OpenCV会使用inter的推理引擎后端进行推理；
    CV_WRAP static Net readFromModelOptimizer(const String& xml, const String& bin);

    //测试网络中是否有layer，是否为空；若没有layer则返回true
    CV_WRAP bool empty() const;

    //向网络中添加新的layer；
    //name是layer的名字，是唯一的；
    //type是网络的类型，卷积层还是relu等；但是必须是OpenCV支持的层，或者自己实现的，在层注册器中注册过的类型；
    //params是层的参数，用于初始化该层；
    //返回值为该层唯一的整数ID；若返回-1表示添加失败
    int addLayer(const String &name, const String &type, LayerParams &params);

    //添加新层，将其第一个输入与上一层第一个输出相连接；
    //参数与addLayer函数相同；
    int addLayerToPrev(const String &name, const String &type, LayerParams &params);

    //转换layer的string name ；返回整数ID；若为-1，则layer不存在
    CV_WRAP int getLayerId(const String &layer);
    //获取layer的string name
    CV_WRAP std::vector<String> getLayerNames() const;

    //字符串和整数的容器
    typedef DictValue LayerId;

    //返回指向网络中指定ID的层的指针
    //ID为整数ID或者字符串ID
    CV_WRAP Ptr<Layer> getLayer(LayerId layerId);

    //返回指向特定层的输入层的指针
    std::vector<Ptr<Layer> > getLayerInputs(LayerId layerId); 

    //连接第一个layer的输出与第二个layer的输入
    //outPin 第一个layer输出的描述.
    //inpPin 第二个layer输入的描述.
    //输入的模板为：<layer_name>.[input_num]
    //模板层名称的第一部分是添加层的sting名称。如果该部分为空，则使用网络输入伪层；
    //模板输入编号的第二个可选部分是层输入编号，或者是标签编号。如果省略此部分，则将使 用第一层输入。
    CV_WRAP void connect(String outPin, String inpPin);

    //第一层的输出与第二层输入相连接
    //outLayerId 第一层的标识符
    //outNum 第一层输出的编号(一个层可能会有多个输出)
    //inpLayerId 第二层的标识符
    //inpNum 第二层输入的编号
    void connect(int outLayerId, int outNum, int inpLayerId, int inpNum);

    //设置网络输入伪层的输出名称
    //每个网络都有自己的输入伪层，id=0
    //该层仅仅存储user的blobs，不进行任何计算
    //这一层提供了用户数据传递到网络中的唯一方法
    //与任何其他层一样，此层可以标记其输出，而此函数提供了一种简单的方法来实现这一点。
    CV_WRAP void setInputsNames(const std::vector<String> &inputBlobNames);

    //下面是Net中的几个forward，上篇博客中介绍过；在此不赘述
    CV_WRAP Mat forward(const String& outputName = String());

    CV_WRAP void forward(OutputArrayOfArrays outputBlobs, 
        const String& outputName = String());

    CV_WRAP void forward(OutputArrayOfArrays outputBlobs,
        const std::vector<String>& outBlobNames);

    CV_WRAP_AS(forwardAndRetrieve) void forward(
        CV_OUT std::vector<std::vector<Mat> >& outputBlobs,
        const std::vector<String>& outBlobNames);

    //编译Halide layers.<Halide是由MIT、Adobe和Stanford等机构合作实现的图像处理语言，它的核心思想即解耦算法和优化>
    //scheduler : 带有scheduler指令的yaml文件的路径
    //@see setPreferableBackend
    //调度Halide后端支持的层，然后编译
    //对于scheduler中不支持的层，或者完全不使用手动调度的层，会采用自动调度
    CV_WRAP void setHalideScheduler(const String& scheduler);

    //指定使用特定的计算平台运行网络
    //输入是backend的标识符
    //如果使用Intel的推理引擎库，DNN_BACKEND_DEFAULT默认表示
    //DNN_BACKEND_INFERENCE_ENGINE 否则是DNN_BACKEND_OPENCV.
    CV_WRAP void setPreferableBackend(int backendId);


    //指定特定的计算设备
    //输入是目标设备的标识符
    /*
     * List of supported combinations backend / target:
     * |                        | DNN_BACKEND_OPENCV | DNN_BACKEND_INFERENCE_ENGINE | DNN_BACKEND_HALIDE |
     * |------------------------|--------------------|------------------------------|--------------------|
     * | DNN_TARGET_CPU         |                  + |                            + |                  + |
     * | DNN_TARGET_OPENCL      |                  + |                            + |                  + |
     * | DNN_TARGET_OPENCL_FP16 |                  + |                            + |                    |
     * | DNN_TARGET_MYRIAD      |                    |                            + |                    |
    */
    CV_WRAP void setPreferableTarget(int targetId);

    //setInput在上篇博客中已经介绍，在此不赘述
    CV_WRAP void setInput(InputArray blob, const String& name = "",
                      double scalefactor = 1.0, const Scalar& mean = Scalar());

    //为layer设置新的参数
    //layer的name
    //layer参数的索引(Layer::blobs array)
    //新的值 Layer::blobs
    //如果新blob的形状与前一个形状不同，则以下正向传递可能失败
    CV_WRAP void setParam(LayerId layer, int numParam, const Mat &blob);

    //返回指定层参数的blob
    //参数同setParam
    CV_WRAP Mat getParam(LayerId layer, int numParam = 0);

    //返回具有未连接输出的层的索引
    CV_WRAP std::vector<int> getUnconnectedOutLayers() const;
    //返回具有未连接输出的层的名字
    CV_WRAP std::vector<String> getUnconnectedOutLayersNames() const;

    //输出网络中所有layer的input和output的shapes
    //netInputShapes 网络输入层中所有输入块的形状
    // layersIds 返回层的ID
    //inLayersShapes 返回输入层形状 顺序与layersIds的顺序相同
    //outLayersShapes 返回输出层形状 顺序与layersIds的顺序相同
    CV_WRAP void getLayersShapes(
        const std::vector<MatShape>& netInputShapes,
        CV_OUT std::vector<int>& layersIds,
        CV_OUT std::vector<std::vector<MatShape> >& inLayersShapes,
        CV_OUT std::vector<std::vector<MatShape> >& outLayersShapes) const;

    /** @重载 */
    CV_WRAP void getLayersShapes(const MatShape& netInputShape,
        CV_OUT std::vector<int>& layersIds,
        CV_OUT std::vector<std::vector<MatShape> >& inLayersShapes,
        CV_OUT std::vector<std::vector<MatShape> >& outLayersShapes) const;

    //输出网络中指定layer的input和output的shapes
    //netInputShape 网络输入的shapes
    //指定layer的ID
    //inLayerShapes返回指定层input的shapes
    //outLayerShapes返回指定层output的shapes
    void getLayerShapes(const MatShape& netInputShape,
                            const int layerId,
                            CV_OUT std::vector<MatShape>& inLayerShapes,
                            CV_OUT std::vector<MatShape>& outLayerShapes) const; // FIXIT: CV_WRAP

    /** @重载 */
    void getLayerShapes(const std::vector<MatShape>& netInputShapes,
                            const int layerId,
                            CV_OUT std::vector<MatShape>& inLayerShapes,
                            CV_OUT std::vector<MatShape>& outLayerShapes) const; // FIXIT: CV_WRAP

    //计算指定input，运行整个网络的FLOPS
    //netInputShapes 所有输入的shapes
    //返回值为FLOP
    CV_WRAP int64 getFLOPS(const std::vector<MatShape>& netInputShapes) const;
    /** 重载 */
    CV_WRAP int64 getFLOPS(const MatShape& netInputShape) const;
    
    //计算指定layer的FLOPS
    CV_WRAP int64 getFLOPS(const int layerId,
                       const std::vector<MatShape>& netInputShapes) const;
    /** 重载 */
    CV_WRAP int64 getFLOPS(const int layerId,
                       const MatShape& netInputShape) const;

    //获取整个model中layer Type的列表
    CV_WRAP void getLayerTypes(CV_OUT std::vector<String>& layersTypes) const;

    //返回网络中指定layer Type的数量
    CV_WRAP int getLayersCount(const String& layerType) const;

    //计算存储模型的权重和中间blob所需的字节数
    //netInputShapes 网络所有输入的shapes
    //weights 输出存储模型中所有层的权重所占用的字节数
    //运行模型，中间blob所需的字节数
    void getMemoryConsumption(const std::vector<MatShape>& netInputShapes,
            CV_OUT size_t& weights, CV_OUT size_t& blobs) const; // FIXIT: CV_WRAP
    /** 重载 */
    CV_WRAP void getMemoryConsumption(const MatShape& netInputShape,
            CV_OUT size_t& weights, CV_OUT size_t& blobs) const;

    //获取模型中指定layer存储权重和中间blob所需的字节数
    CV_WRAP void getMemoryConsumption(const int layerId,
            const std::vector<MatShape>& netInputShapes,
            CV_OUT size_t& weights, CV_OUT size_t& blobs) const;
    /** 重载 */
    CV_WRAP void getMemoryConsumption(const int layerId,
            const MatShape& netInputShape,
            CV_OUT size_t& weights, CV_OUT size_t& blobs) const;

    //计算模型中每一个layer，存储权重和中间blob所需的字节数
    //netInputShapes 网络的所有input的shapes
    //layerIds 输出网络中所有层的layer ID
    //weights 各个层存储权重所需的字节数，与layersIds对应
    //blobs 各个层存储中间blobs所需的字节数，与layersIds对应
    void getMemoryConsumption(const std::vector<MatShape>& netInputShapes,
            CV_OUT std::vector<int>& layerIds,
            CV_OUT std::vector<size_t>& weights,
            CV_OUT std::vector<size_t>& blobs) const; // FIXIT: CV_WRAP
    /** 重载 */
    void getMemoryConsumption(const MatShape& netInputShape,
            CV_OUT std::vector<int>& layerIds,
            CV_OUT std::vector<size_t>& weights,
            CV_OUT std::vector<size_t>& blobs) const; // FIXIT: CV_WRAP
 
    //启用或者禁用网络中的层融合
    //启用为true；禁用为false；
    //默认为启用的
    CV_WRAP void enableFusion(bool fusion);
    
    //返回推理的总时间和layers的时间(in ticks)
    //返回的向量中的索引对应layers ID，有些层可以与其它层融合，在这种情况下，跳过的层计时为0
    //timings 各个层的时间
    //整个model的推理时间
    CV_WRAP int64 getPerfProfile(CV_OUT std::vector<double>& timings);
 
     private:
         struct Impl;
         Ptr<Impl> impl;
     };
}



以上是整个Net中定义的功能，程序中做了简单的注释，某些函数会很清晰，例如addLayer,但是有的函数可能看起来不知所云；没有关系，在后续的文章中会逐步涉及到所有函数，结合函数的定义，会更加清晰。 从对Net的生命的分析，可以看出OpenCV为推理提供了强大的功能；除了对网络的操作(例如添加不同的layers)之外,同时提供了推理平台选择等函数，还提供了丰富的profiling功能，可以方便的分析内存和耗时。

OpenCV运行googleNet

下面给出一个demo，使用openCV完成googleNet的推理；

#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"
#include <iostream>
#include <fstream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

String modelTxt = "bvlc_googlenet.prototxt";
String modelBin = "bvlc_googlenet.caffemodel";
String labelFile = "synset_words.txt";

vector<String> readLabels();
int main(int argc, char** argv) {
    Mat testImage = imread("./space_shuttle.jpg");
    if (testImage.empty()) {
        printf("could not load image...\n");
        return -1;
    }

    //使用caffe model创建Net
    Net net = dnn::readNetFromCaffe(modelTxt, modelBin);
    if (net.empty())
    {
        std::cerr << "Can't load network by using the following files: " << std::endl;
        std::cerr << "prototxt:   " << modelTxt << std::endl;
        std::cerr << "caffemodel: " << modelBin << std::endl;
        return -1;
    }

    // 读取分类数据
    vector<String> labels = readLabels();

    //GoogLeNet accepts only 224x224 RGB-images
    Mat inputBlob = blobFromImage(testImage, 1, Size(224, 224), Scalar(104, 117, 123), false, true);

    Mat prob;
    for (int i = 0; i < 10; i++)
    {
        // 输入
        net.setInput(inputBlob, "data");
        // 分类预测
        prob = net.forward("prob");
    }
    //测试推理时间
    int64 totalTime = Net.getPerfProfile(NULL);
    printf("total forward time is %d\n");

    // 读取分类索引，最大与最小值
    Mat probMat = prob.reshape(1, 1); //reshape the blob to 1x1000 matrix // 1000个分类
     Point classNumber;
     double classProb;
     minMaxLoc(probMat, NULL, &classProb, NULL, &classNumber); // 可能性最大的一个
     int classIdx = classNumber.x; // 分类索引号
     printf("\n current image classification : %s, possible : %.2f \n", labels.at(classIdx).c_str(), classProb);
 
     putText(testImage, labels.at(classIdx), Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2, 8);
     imshow("Image Category", testImage);
 
     waitKey(0);
     return 0;
 }
 
 
 /* 读取图像的1000个分类标记文本数据 */
 vector<String> readLabels() {
     std::vector<String> classNames;
     std::ifstream fp(labelFile);
     if (!fp.is_open())
     {
         std::cerr << "File with classes labels not found: " << labelFile << std::endl;
         exit(-1);
     }
 
     std::string name;
     while (!fp.eof())
     {
         std::getline(fp, name);
         if (name.length())
             classNames.push_back(name.substr(name.find(' ') + 1));
     }
 
     fp.close();
     return classNames;
 }

记得插入图片
