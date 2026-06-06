# CUDA代码实战一matix_mul、reduce

**作者**: 李安渝infra小菜鸡

**原文链接**: https://zhuanlan.zhihu.com/p/4732447803

---

​
目录
收起
CUDA编程模型
代码实践
matirx_mul算子
reduce算子

上一篇文章，总结了一下GPU的体系架构和内存结构

李安渝：GPU的内存体系结构
20 赞同 · 0 评论 文章

这一篇主要是总结一下，如何使用cuda进行gpu编程实战。

CUDA编程模型

CUDA 程序一般使用 .cu 后缀，编译 CUDA 程序则使用 nvcc 编译器。一般而言，典型的CUDA程序的执行流程如下：

分配host内存，并进行数据初始化；
分配device内存，并从host将数据拷贝到device上；
调用CUDA的核函数在device上完成指定的运算；
将device上的运算结果拷贝到host上；
释放device和host上分配的内存。
int main() {
    主机代码;
    核函数调用;
    主机代码;
    核函数调用;
    ......
    return 0;  
}

__global__ void 核函数1(parameters) {
    ......
}

__global__ void 核函数2(parameters) {
    ......
}


CUDA C++ 定义的基本函数执行单元被称为 kernel(核函数) ，kernel 在调用时由 N 个不同的 CUDA 线程并行执行 N 次，而不是像常规 C++ 函数那样只执行一次。

在cuda程序中，我们使用 __global__ 来定义核函数，并且总是将核函数定义为 void类型，这意味着我们不能直接通过设备端的核函数来返回 结果，而是需要在参数 里面 传入一个用来存放运算结果的指针，在运算阶段直接将结果存入指针指向的存储空间中。

__global__ void elementwise_add_f32_kernel(float* a, float* b, float* c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) c[idx] = a[idx] + b[idx];
}


可以在上诉代码中看到，我们传入输入数据指针float*a以及float*b, 还有运算结果的指针float*c 来存储运算结果。

在上一章中，我们也讲述了网格、线程块、线程之间的关系，在代码中我们也是通过线程的全局索引来确定数据在全局中的定位，从而获取数据进行计算。

elementwise_add_f32_kernel<<<grid_size, block_size>>>(a,b,c,N);


grid_size和block_size：grid_size和block_size都可以是一个dim3类型的结构体，或是一个unsigned类型的无符号整数变量。前者表示网格大小，后者表示线程块大小。在 CUDA 的线程组织模型中，线程是最基本的单位，线程块由线程组成，而网格由线程块组成。它们的存在都是为了将线程组织成逻辑上更容易理解的线程单元。

当定义为二维变量时，可以想象，每个网格是由二维排列的块构成的，每个块内部由二维排列的线程构成。如下图：

线程二维组织模式

当定义为三维变量时，可以想象，每个网格是由三维排列的块构成的，每个块内部由三维排列的线程构成。如下图：

线程三维组织模式

显然，参与该核函数计算的线程个数，可以用如下公式表示：

num_threads = grid_size.x * grid_size.y * grid_size.z * block_size.x * block_size.y * block_size.z
blockIdx 指明一个线程所在的线程块在网格中的位置。
blockIdx.x 的范围为 0 到 gridDim.x-1
blockIdx.y 的范围为 0 到 gridDim.y-1
blockIdx.z 的范围为 0 到 gridDim.z-1
threadIdx 指明一个线程在它所在的线程块中的位置。
threadIdx.x 的范围为 0 到 blockDim.x-1
threadIdx.y 的范围为 0 到 blockDim.y-1
threadIdx.z 的范围为 0 到 blockDim.z-1
不论是在 blockIdx 还是 threadIdx 中，x 都是变化最快的分量，其次是 y。

grid 的每个 block 都可以通过一维、二维或三维唯一索引来标识，该索引可通过内置的blockIdx变量在内核中访问。 线程块的尺寸可以通过内置的blockDim变量在内核中访问。如下所示：

线程的索引计算

接着我们来看一下 主机端(cpu端)的操作以及数据内存的拷贝等操作。

__host__ 修饰的函数称为主机函数，它就是主机端的普通 C++ 函数，在主机（CPU）中调用和执行，可以忽略。
__divice__修饰的函数称为设备函数，只能被核函数或是其它设备函数调用，只能在设备中执行。

因为GPU不能直接读取到CPU内存里的数据，所以我们需要在CPU端将数据定义好通过指定的命令将数据传输到GPU的显存(全局内存)中，然后再开始进行核函数调用。

// 函数定义
cudaError_t cudaMalloc(void** d_ptr, unsigned int size);
cudaError_t cudaMemcpy(void* d_ptr, void* h_ptr, unsigned int size, enum cudaMemcpyKind)

// 具体的使用方法
float *h_x = (real*) malloc(M);
float *d_x;
cudaMalloc((void **)&d_x, M);
cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);


cudaMalloc函数中需要注意的是，第一个参数是一个指针的指针，即二级指针，因为这个函数需要在 GPU 显存上分配出一片空间，并且让d_ptr指向这个空间，由于cudaMalloc函数的返回值已经用于返回错误代码，因此需要传入一个二级指针，然后由cudaMalloc函数负责改变这个指针，使它指向刚分配出的那片设备内存。

cudaMemcpy 函数中，第一个参数是指向设备内存的指针，第二个参数是指向主机内存的指针，最后一个参数是 enum 类型的变量，用于指出数据传输的方向，它有五种取值，根据变量名就很容易看出数据传输的方向，比较常用的是前面两种。
cudaMemcpyHostToDevice
cudaMemcpyDeviceToHost
cudaMemcpyHostToHost
cudaMemcpyDeviceToDevice
cudaMemcpyDefault

当我们完成数据的定义和传输之后，开始调用核函数进行运算，在核函数计算完成之后将数据从设备端传输回host端并释放我们使用到的所有内存

cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);
free(h_x);
free(h_z);
cudaFree(d_x);
cudaFree(d_z);

代码实践
matirx_mul算子

首先定义需要用到的数据集，并且由于是矩阵运算所有设定width和height，来表示矩阵的大小(类似于x，y)

// 对于矩阵运算，应该选用grid和block为2-D的
// 矩阵类型，行优先，M(row, col) = *(M.elements + row * M.width + col)
struct Matrix
{
    // int nBytes = width * height * sizeof(float);  可以得到一个martix最大的字节数，也可以看成行列
    int width; 
    int height;
    // 是矩阵的首个元素的地址，这里定义矩阵元素是 行优先
    float *elements;
};


接下来定义核函数和辅助函数

void __global__ martix_mul(Matrix *A, Matrix *B, Matrix *C);// 进行矩阵运算
__device__ float getElement(Matrix *A, int row, int col); // 在设备端获取数据
__device__ void setElement(Matrix *A, int row, int col, float value);// 将运算好的数据重新写回设备端内存
void check(const real *z, const int N);//检查运算结果是否出错


具体代码的解释：

通过设置__device__，来表明这是在设备端(GPU端)获取数据和设置数据

// 获取矩阵A的(row, col)元素
__device__ float getElement(Matrix *A, int row, int col)
{
	return A->elements[row * A->width + col];
}

// 为矩阵A的(row, col)元素赋值
__device__ void setElement(Matrix *A, int row, int col, float value)
{
	A->elements[row * A->width + col] = value;
}


核函数设计

// 矩阵相乘kernel，2-D，每个线程计算一个元素
__global__ void martix_mul(Matrix *A, Matrix *B, Matrix *C)
{
	float Cvalue = 0.0;
	int row = threadIdx.y + blockIdx.y * blockDim.y; //获取 线程在 2-d块儿中的x索引
	int col = threadIdx.x + blockIdx.x * blockDim.x;//获取 线程在 2-d块儿中的y索引 ,to
	for (int i = 0; i < A->width; ++i)
	{
		Cvalue += getElement(A, row, i) * getElement(B, i, col);
	}
    if(row>1000){
        printf("%lf\n",Cvalue);
    }
	setElement(C, row, col, Cvalue);
}


核函数调用：

通过手动设置需要的block size和gridSize，来利用GPU的并行资源，并且设置线程同步来确保计算的正确性。

    // 定义kernel的执行配置
    dim3 blockSize(32, 32); // 一个block 是一个 高有32个线程，宽有32个线程，一共有1024个线程的block
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
        (height + blockSize.y - 1) / blockSize.y);  // 高宽分别需要几个线程块，那么就是一个grid里面需要几个 block 
    // 执行kernel
    martix_mul << < gridSize, blockSize >> >(A, B, C); // 定义这个矩阵运算一共会有这个多个(grid_size * block_size 个线程)执行运算

    // 同步device 保证结果能正确访问
    // 如果不同步设备，那么在核函数执行期间或之后立即访问GPU内存或由GPU生成的数据可能会导致数据不一致或竞态条件。cudaDeviceSynchronize()确保了在主机代码尝试访问这些数据之前，所有相关的GPU工作都已完成。
    // 让 CPU 陷入等待，等 GPU 完成队列的所有任务后再返回。
    cudaDeviceSynchronize();


全部代码：

#include "error.cuh"
#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h> // 包含基本的CUDA运行时API
#include<iostream>
#define printf_sync printf
#ifdef USE_DP
    typedef double real;
    const real EPSILON = 1.0e-15;
#else
    typedef float real;
    const real EPSILON = 1.0e-6f;
#endif
// 对于矩阵运算，应该选用grid和block为2-D的
// 矩阵类型，行优先，M(row, col) = *(M.elements + row * M.width + col)
struct Matrix
{
    // int nBytes = width * height * sizeof(float);  可以得到一个martix最大的字节数，也可以看成行列
    int width; 
    int height;
    // 是矩阵的首个元素的地址，这里定义矩阵元素是 行优先
    float *elements;
};
void __global__ martix_mul(Matrix *A, Matrix *B, Matrix *C);
__device__ float getElement(Matrix *A, int row, int col);
__device__ void setElement(Matrix *A, int row, int col, float value);
void check(const real *z, const int N);
__host__ void say_hello();

int main()
{
    int width = 1 << 10;
    int height = 1 << 10;
    Matrix *A, *B, *C; // 三个矩阵 
    // 申请托管内存
    cudaMallocManaged((void**)&A, sizeof(Matrix));
    cudaMallocManaged((void**)&B, sizeof(Matrix));
    cudaMallocManaged((void**)&C, sizeof(Matrix));
    int nBytes = width * height * sizeof(float); //  矩阵元素占据的内存
    cudaMallocManaged((void**)&A->elements, nBytes); // 分配具体大小给每个矩阵
    cudaMallocManaged((void**)&B->elements, nBytes); 
    cudaMallocManaged((void**)&C->elements, nBytes);

    // 初始化数据
    A->height = height;
    A->width = width;
    B->height = height;
    B->width = width;
    C->height = height;
    C->width = width;
    for (int i = 0; i < width * height; ++i)
    {
        // 赋值所有元素
        A->elements[i] = 1.0;
        B->elements[i] = 2.0;
    }

    // 定义kernel的执行配置
    dim3 blockSize(32, 32); // 一个block 是一个 高有32个线程，宽有32个线程，一共有1024个线程的block
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
        (height + blockSize.y - 1) / blockSize.y);  // 高宽分别需要几个线程块，那么就是一个grid里面需要几个 block 
    // 执行kernel
    martix_mul << < gridSize, blockSize >> >(A, B, C); // 定义这个矩阵运算一共会有这个多个(grid_size * block_size 个线程)执行运算

    // 同步device 保证结果能正确访问
    // 如果不同步设备，那么在核函数执行期间或之后立即访问GPU内存或由GPU生成的数据可能会导致数据不一致或竞态条件。cudaDeviceSynchronize()确保了在主机代码尝试访问这些数据之前，所有相关的GPU工作都已完成。
    // 让 CPU 陷入等待，等 GPU 完成队列的所有任务后再返回。
    cudaDeviceSynchronize();
    // 检查执行结果
    float maxError = 0.0;
    for (int i = 0; i < width * height; ++i)
        maxError = fmax(maxError, fabs(C->elements[i] - 2 * width));
    std::cout << "最大误差: " << maxError << std::endl;

    return 0;
}
// 获取矩阵A的(row, col)元素
__device__ float getElement(Matrix *A, int row, int col)
{
	return A->elements[row * A->width + col];
}

// 为矩阵A的(row, col)元素赋值
__device__ void setElement(Matrix *A, int row, int col, float value)
{
	A->elements[row * A->width + col] = value;
}

// 矩阵相乘kernel，2-D，每个线程计算一个元素
__global__ void martix_mul(Matrix *A, Matrix *B, Matrix *C)
{
	float Cvalue = 0.0;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i = 0; i < A->width; ++i)
	{
		Cvalue += getElement(A, row, i) * getElement(B, i, col);
	}
    if(row>1000){
        printf("%lf\n",Cvalue);
    }
	setElement(C, row, col, Cvalue);
}
__host__ void say_hello(){
    // 
    for(int i=0;i<10;i++){
        std::cout<<"hello"<<std::endl;
    }
}

reduce算子

上一小节，讲述了如何通过线程 组织架构来计算矩阵乘法，这一小节来介绍一下如何通过cuda编程完成reduce运算(sum运算)的并行

reduce算子并行 运算逻辑

核心函数定义：

在reduce的核函数中，我们最大化利用线程块的并行能力，通过分治的加和和线程同步来确保数据计算的正确性

void __global__ reduce_global(real *d_x, real *d_y);
void __global__ reduce_global(real *d_x, real *d_y)
{
    const int tid = threadIdx.x;
    real *x = d_x + blockDim.x * blockIdx.x; // 定义全局索引

// 通过offset来确保每次都能和正确位置的数字进行相➕ 
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset)//确保 不会越界
        {
            x[tid] += x[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        d_y[blockIdx.x] = x[0];
    }
}


调用函数：

real reduce(real *d_x, const int method)
{
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int ymem = sizeof(real) * grid_size;
    const int smem = sizeof(real) * BLOCK_SIZE;
    real *d_y;
    CHECK(cudaMalloc(&d_y, ymem));
    real *h_y = (real *) malloc(ymem);

    switch (method)
    {
        case 0:
            reduce_global<<<grid_size, BLOCK_SIZE>>>(d_x, d_y);
            break;
        default:
            printf("Error: wrong method\n");
            exit(1);
            break;
    }

    CHECK(cudaMemcpy(h_y, d_y, ymem, cudaMemcpyDeviceToHost));

    real result = 0.0;
    for (int n = 0; n < grid_size; ++n)
    {
        result += h_y[n];
    }

    free(h_y);
    CHECK(cudaFree(d_y));
    return result;
}





这一篇只是简单的介绍了cuda的编程模型以及代码实践，但是这显然不是cuda能做到的极限，因为这一节的 代码只是利用cuda并行计算能力，没有对cuda的任何缓存结构(shared mem)和wrap结构做出利用，下一节就写一下如何利用gpu的内存结构来提高算子的运行能力。
