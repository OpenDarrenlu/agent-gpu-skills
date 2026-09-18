# 如何使用cublas计算batched matmul?

**作者**: 俯仰AI Framework Engineer

**原文链接**: https://zhuanlan.zhihu.com/p/721545966

---

​
目录
收起
1. Introduction
2. cublasTgemm 的介绍
2.1 定义
2.2 列主序和行主序
3. CublasTgemmXXXBatched
3.1 cublasSgemmBatch
3.2 CublasSgemmStridedBatched
3.3 CublasSgemmGroupedBatched
4. Reference

本文涉及的内容：

1. cuBLAS 计算库的概念和基本用途介绍。

2. 如何使用 cuBLAS进行单个矩阵乘法计算。

3. 什么是行主序存储与列主序存储？为什么cuBLAS默认列主序存储？

4. 如何使用 cuBLAS 进行带 batch 的矩阵乘法的计算。

5. cuBLAS 提供的多个 batched gemm 函数的基本使用和对比。

其中，3.4. 为本文的重点，1.2. 为所需的先验知识和背景。

本文很少涉及或暂时不涉及：

1. cuBLAS 库的安装。

2. cuBLAS 中矩阵乘法外的其他函数的介绍，如 gemv 或 gevv。

3. 程序的具体的性能分析。

1. Introduction

The cuBLAS library is an implementation of BLAS (Basic Linear Algebra Subprograms) on top of the NVIDIA®CUDA™ runtime. It allows the user to access the computational resources of NVIDIA Graphics Processing Unit (GPU)[1].

BLAS（Basic Linear Algebra Subprograms，基础线性代数程序集）是一个应用程序接口（API）标准，用以规范发布基础线性代数操作的数值库（如矢量或矩阵乘法）。该程序集最初发布于1979年，并用于建立更大的数值程序包[2]。

BLAS按照功能被分为三个级别:

- Level 1：向量-向量运算

- Level 2：矩阵-向量运算

- Level 3：矩阵-矩阵运算

在 CPU 上，常用的 BLAS 实现库包括 Intel 的 MKL（现已成为 oneDNN）。

2. cublasTgemm 的介绍

为了对齐先验信息，作者会在此对 cublasTgemm 接口做一个简单的介绍，如果已经了解的小伙伴可以跳过，直接去第三节阅读。

2.1 定义

cublasTgemm 是 cuBLAS 库中的一个函数，用于执行单精度浮点数的矩阵乘法。这个函数的核心功能是计算两个矩阵的乘积，并可选地将结果与另一个矩阵进行加权求和，形式为：

C = \alpha \times A@B+\beta\times C

其中@表示矩阵乘法运算，cublasTgemm 的接口定义为 ：

cublasStatus_t cublasTgemm(cublasHandle_t handle,
						cublasOperation_t transa,
						cublasOperation_t transb,
						int m, int n, int k,
						const float *alpha,
						const T *A, int lda,
						const T *B, int ldb,
						const float *beta,
						T *C, 
						int ldc)


T 表示支持的数据类型，cublas 中对于不同的数据类型使用不同的接口，如 cublasHgemmBatched 用于计算 half 数据类型的 batched matmul，而 cublasSgemmBatched 则用于计算 float数据类型，S 代表 single precision floating point. 另外 D 表示 double 数据类型，即 double presion；C 表示 complex 复数类型，Z 表示 DoubleComplex 双精度复数。为节省篇幅，下文中均以单精度浮点数为例（S），大家在具体使用时只需要将数据类型进行替换即可。

参数介绍：

handle

cublasHandle_t 类型，是 cuBLAS 库的上下文句柄。在使用 cuBLAS 函数之前，用户需要通过 cublasCreate 创建这个句柄，函数完成后需要调用 cublasDestroy 释放资源。

transa 和 transb:

这两个参数指定矩阵 A 和 B 是否进行转置或共轭转置，参数类型为 cublasOperation_t，可以选择以下三种：

CUBLAS_OP_N: 不进行转置
CUBLAS_OP_T: 转置
CUBLAS_OP_C: 共轭转置（仅适用于复数矩阵）

例如，如果 transa 为 CUBLAS_OP_N，则 A以原始形式参与乘法运算；如果为 CUBLAS_OP_T，则 A被转置后再参与运算。

m:

整数类型，表示矩阵 C 和矩阵 A 的行数。

n:

整数类型，表示矩阵 C 和矩阵 B 的列数。

k:

整数类型，表示矩阵 A 的列数以及矩阵 B 的行数。即当执行矩阵乘法 A * B 时，A 的列数必须等于 B 的行数。

alpha:

指向标量 alpha 的指针，表示矩阵乘积的放缩系数，即执行 A * B 后的结果要乘以 alpha。

A:

指向矩阵 A 数据的指针，数据存储在设备端内存（GPU 内存）中。

lda:

整数类型，指定矩阵 A 的主维度（leading dimension），即 A 的列数或行数，取决于 A 是否转置。lda 是指矩阵 A 中相邻两行（在内存中）的元素间的距离。如果 transa == CUBLAS_OP_N，lda 必须至少为 max(1, m)；如果 transa == CUBLAS_OP_T 或 CUBLAS_OP_C，lda必须至少为 max(1, k)。

B:

指向矩阵 B 数据的指针，同样存储在设备端内存中。

ldb:

整数类型，指定矩阵 B 的主维度，定义方式与 lda 相同。对于 B·，如果 transb == CUBLAS_OP_N，则 ldb 必须至少为 max(1, k)；如果 transb == CUBLAS_OP_T 或 CUBLAS_OP_C, ldb 必须至少为 max(1, n)。

beta:

指向标量 beta 的指针，表示矩阵 C 的放缩系数。计算结果中，原始的 C矩阵将被 beta放缩后再与 alpha * A * B 相加。如果 beta == 0，则相当于忽略原始的 C矩阵。

C:

指向输出矩阵 C 的指针，表示结果矩阵，存储在设备端内存中。

ldc:

整数类型，指定矩阵 C 的主维度。ldc 必须至少为 max(1, m)。

2.2 列主序和行主序

这里必须要提的我们在使用 cuBLAS 的一个大坑是，cuBLAS 对于矩阵的存储方式默认是列主序的，而我们现代一般情况下都默认矩阵是按照行主序存储的。

什么是列主序和行主序呢？行主序就是符合我们直觉的，一行中的相邻的两个元素的物理存储位置也是相邻的，而一列中相邻两个元素的物理存储位置并不相邻，而是相隔一行。而列主序存储则正好相反，其默认一列中相邻两个元素的物理位置连续，而一行中相邻两个元素则在物理位置上相隔一列。

这样的差异会导致一个致命的问题就是，当我们向 cublas 传入一个 m 行 n 列的矩阵时，cublas 会将一行中连续的 m 个元素当作是一列，所以他会把这个矩阵作为一个 n 列 m 行的矩阵来处理，相当于是默认为我们的矩阵做了一个转置，非常违背我们的直觉和习惯。

于是，当我们按照我们的习惯去调用 cuBLAS 函数的时候，我们往往会面临参数传递错误的问题。因为我们传入的一个 m\times k 的矩阵 A 与 k\times n 的矩阵 B 的运算在 cuBLAS 的眼中往往会被看作是 k\times m 与 n\times k 的矩阵乘法运算，而这样的 shape 并不符合矩阵乘法运算的要求，因而报错。

常见的解决方案有两种，一种通过参数来让 cuBLAS 帮我们对输入的两个矩阵进行转置，这种方法的一个问题是虽然对输入矩阵进行了转置，但输出的矩阵却并未转置，所以 cuBLAS 会为我们输出一个 n\times m 的矩阵，需要我们再对输出的矩阵自行进行额外的转置。

另外一种解决方式则不需要 cuBLAS 对我们的矩阵进行转置操作，但是需要调换输入矩阵的位置，也就是将 A@B 换成 B@A ，那么在 cuBLAS 眼中则刚好是一个 n\times k 的矩阵与 k\times m 的矩阵进行矩阵乘法的操作，输出是按照列主序存储的的 n\times m 的矩阵，而正好对应我们想要求得的行主序存储的 m\times n 的矩阵，避免了再对输出进行额外操作。

下面示意如何使用 cublasSgemm 接口对两个矩阵进行矩阵乘法计算：

 #include <cublas_v2.h>
 #include <cuda_runtime.h>
 #include <iostream>
 
 int main() {
     cublasHandle_t handle;
     cublasCreate(&handle);
 
     int m = 16, n = 32, k = 64;
     float alpha = 1.0f;
     float beta = 0.0f;
     std::vector<float> A(m*k, 1);
     std::vector<float> B(k*n, 2);
     std::vecotr<float> C(m*n, 0);
     
     float *d_A, *d_B, *d_C;
     cudaMalloc((void**)&d_A, m*k*sizeof(float));
     cudaMalloc((void**)&d_B, k*n*sizeof(float));
     cudaMalloc((void**)&d_C, m*n*sizeof(float));
 
     cudaMemcpy(d_A, A, m*k*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(d_B, B, k*n*sizeof(float), cudaMemcpyHostToDevice);
 
     cublasSgemm(handle,
                 CUBLAS_OP_N, 
                 CUBLAS_OP_N,
                 n, m, k,
                 &alpha,
                 d_B, 
                 n,
                 d_A, 
                 k,
                 &beta,
                 d_C, 
                 n);
 
     cudaMemcpy(C, d_C, m*n*sizeof(float), cudaMemcpyDeviceToHost);
 
     std::cout << "结果矩阵 C:" << std::endl;
     for (int i = 0; i < m * n; i++) {
         std::cout << C[i] << " ";
         if ((i + 1) % n == 0) std::cout << std::endl;
     }
 
     cublasDestroy(handle);
     cudaFree(d_A);
     cudaFree(d_B);
     cudaFree(d_C);
 
     return 0;
 } 


为什么要设计成列主序存储？

这是因为 cuBLAS 库是与 Fortran 语言的数值计算库 BLAS (Basic Linear Algebra Subprograms) 接口兼容的，而 Fortran 语言中，矩阵的默认存储方式是列主序。

3. CublasTgemmXXXBatched

顾名思义，这类函数主要是用来计算 batched matix multiplication 的。而 cublas 提供了三种用于计算带 batch 的 matmul 的函数，供用户来选择方便自己应用的方式来使用。这三种方式分别是：

1. CublasTgemmBatched ().

2. CblusTgemmStridedBatched ()

3. CblasTgemmGroupedBatched ()

下面作者将和大家挨个学习这三个 API 如何去正确的使用。

3.1 cublasSgemmBatch
cublasStatus_t cublasSgemmBatched(cublasHandle_t handle,
							  cublasOperation_t transa,
							  cublasOperation_t transb,
							  int m, int n, int k,
							  const float  *alpha,
							  const float  *const Aarray[], int lda,
							  const float   *const Barray[], int ldb,
							  const float  *beta,
							  float *const Carray[], int ldc,
							  int batchCount)

CublasTgemmBatched 函数默认你需要处理的这个 batch 中若干的矩阵是 uniform 的，也就是它们的 m, n, k 值以及 lda, ldb, ldc 都是相等的，所以这些值仅需提供一次即可。

而和 cublasTgemm 相比，函数最大的差别实际在传入这些矩阵的值的定义上： const float *const Aarray[]，看起来是是不是很别扭？其实说白了这里就是希望你传入一组指针，每个指针指向一个矩阵。并且用两个 const 对传入数据做了限制，第一个 const 表示不能对传入的这组指针的值进行修改，第二个指针表示不能对传入的这组指针所指向的值进行修改，归根结底就是为了保护输入数据不被修改而已。

对于我们用户而言，我们只需要传入一组指向矩阵的指针即可。

那么这个函数到底能为我们做什么事情呢？让我们用一个数学表达式来说明：

C[i] = \alpha[i]@B[i]+\beta C[i],\qquad i\in[0, batchnum-1]

其中 C [i]表示第 i 个 C 矩阵，@表示矩阵乘法运算， \alpha 和 \beta 表示两个常数。可以看出，这个函数并不需要把所有的 A 矩阵或 BC 矩阵都以物理相邻的方式存储，只需要用户以数据的方式提供这些数据的存储地址即可。

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

// do-while is used to create a local area to avoid duplicated variable names.
#define CUDA_CHECK(EXPRESS)                                                   \
    do {                                                                      \
        cudaError_t error = EXPRESS;                                          \
        if (error != cudaSuccess) {                                           \
            std::cerr << "The cublas api failed in file: " << __FILE__ << ":" \
                      << __LINE__ << std::endl;                               \
            std::cerr << cudaGetErrorName(error) << std::endl;                \
            std::exit(-1);                                                    \
        }                                                                     \
    } while (0)

#define CUBLAS_CHECK(EXPRESS)                                                 \
    do {                                                                      \
        cublasStatus_t status = EXPRESS;                                      \
        if (status != CUBLAS_STATUS_SUCCESS) {                                \
            std::cerr << "The cublas api failed in file: " << __FILE__ << ":" \
                      << __LINE__ << std::endl;                               \
            std::exit(-1);                                                    \
        }                                                                     \
    } while (0)

std::vector<float> CpuMatmul2D(const std::vector<float>& in_a,
                               const std::vector<float>& in_b, int m, int n,
                               int k) {
    std::vector<float> out(m * n, 0);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int l = 0; l < k; l++) {
                out[i * n + j] += in_a[i * k + l] * in_b[l * n + j];
            }
        }
    }
    return out;
}

std::vector<std::vector<float>> CpuKernel(
    const std::vector<std::vector<float>>& in_a,
    const std::vector<std::vector<float>>& in_b, int batch, int m, int n,
    int k) {
    std::vector<std::vector<float>> out;

    for (int b = 0; b < batch; b++) {
        out.emplace_back(CpuMatmul2D(in_a[b], in_b[b], m, n, k));
    }
    return out;
}

int main() {
    int batch = 8;
    int m = 512;
    int n = 128;
    int k = 1024;
    std::vector<std::vector<float>> input_a(batch);
    std::vector<std::vector<float>> input_b(batch);
    std::vector<std::vector<float>> out_c(batch, std::vector<float>(m * n, 0));
    float* a_ptrs[batch];
    float* b_ptrs[batch];
    float* c_ptrs[batch];

    // Initializing host data
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < m * k; i++) {
            input_a[b].emplace_back(rand() % 100 / 1000.);
        }
        for (int i = 0; i < k * n; i++) {
            input_b[b].emplace_back(rand() % 100 / 2024.);
        }
    }

    // Initializing device data
    for (int b = 0; b < batch; b++) {
        void *a_dev, *b_dev, *c_dev;
        CUDA_CHECK(cudaMalloc(&a_dev, m * k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&b_dev, k * n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c_dev, m * n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(a_dev, input_a[b].data(), m * k * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(b_dev, input_b[b].data(), k * n * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(c_dev, out_c[b].data(), m * n * sizeof(float),
                                cudaMemcpyHostToDevice));
        a_ptrs[b] = static_cast<float*>(a_dev);
        b_ptrs[b] = static_cast<float*>(b_dev);
        c_ptrs[b] = static_cast<float*>(c_dev);
    }

    void** a_ptr_dev;
    void** b_ptr_dev;
    void** c_ptr_dev;
    CUDA_CHECK(cudaMalloc(&a_ptr_dev, batch*sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&b_ptr_dev, batch*sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&c_ptr_dev, batch*sizeof(float*)));
    CUDA_CHECK(cudaMemcpy(a_ptr_dev, a_ptrs, batch*sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_ptr_dev, b_ptrs, batch*sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c_ptr_dev, c_ptrs, batch*sizeof(float*), cudaMemcpyHostToDevice));



    // Get Cpu result
    auto cpu_result = CpuKernel(input_a, input_b, batch, m, n, k);

    // Invoke cublasSgemmBatched API
    float alpha = 1.;
    float beta = 1.;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                    &alpha, (float**)b_ptr_dev, n, (float**)a_ptr_dev, k,
                                    &beta, (float**)c_ptr_dev, n, batch));

    // Check result
    std::vector<std::vector<float>> out_host(batch, std::vector<float>(m*n, 0));
    for (int b = 0; b < batch; b++) {
        float* c_dev = (float*)c_ptrs[b];
        CUDA_CHECK(cudaMemcpy(out_host[b].data(), c_dev, m * n * sizeof(float),
                              cudaMemcpyDeviceToHost));

    }

    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < m * n; i++) {
            float diff = abs(cpu_result[b][i] - out_host[b][i]);
            if (std::abs(cpu_result[b][i] - out_host[b][i]) > 1e-6)
            {
                std::cout << "The difference is too big." << std::endl;
                std::exit(-1);
            }
        }
    }

    return 0;
}

作者个人感觉可能对于同一个 batch 内的多个矩阵分散存储的情况更加适合。但是缺点也比较明显，就是需要额外分配一个用于存储这些指针的数组，当 batchsize 特别大的时候，额外占用的存储不容小觑。另外，为了能够让 cuBLAS 能够正常使用我们传入的指针数据，我们还需要将指针数组中的这些指针拷贝至我们的 device 上，占据额外显存先且不论，还会因为额外的数据拷贝开销，增大整体操作的 latency。

3.2 CublasSgemmStridedBatched
cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const float           *alpha,
                                  const float           *A, int lda,
                                  long long int          strideA,
                                  const float           *B, int ldb,
                                  long long int          strideB,
                                  const float           *beta,
                                  float                 *C, int ldc,
                                  long long int          strideC,
                                  int batchCount)

CublasSgemmStridedBatched 函数看起来要和 cublasSgemm 更相似一点，只是额外增加了三个 long long int 类型的参数 strideA, strideB 和strideC 来分别表示 A，B ，C 中矩阵与矩阵之间的 stride（也就是间隔的距离）。由于函数参数的设计，cublasSgemmStridedBatched 函数可能只适合于一个 batch 中的矩阵之间间隔距离想等的存储方式。

此函数的计算用数学公式来表达，即：

C+i*strideC = \alpha(A+i*stride\_A) @(B+i*stride\_B)+\beta (C+i*stride\_C),\qquad i\in[0, batchnum-1]

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

// do-while is used to create a local area to avoid duplicated variable names.
#define CUDA_CHECK(EXPRESS)                                                   \
    do {                                                                      \
        cudaError_t error = EXPRESS;                                          \
        if (error != cudaSuccess) {                                           \
            std::cerr << "The cublas api failed in file: " << __FILE__ << ":" \
                      << __LINE__ << std::endl;                               \
            std::cerr << cudaGetErrorName(error) << std::endl;                \
            std::exit(-1);                                                    \
        }                                                                     \
    } while (0)

#define CUBLAS_CHECK(EXPRESS)                                                 \
    do {                                                                      \
        cublasStatus_t status = EXPRESS;                                      \
        if (status != CUBLAS_STATUS_SUCCESS) {                                \
            std::cerr << "The cublas api failed in file: " << __FILE__ << ":" \
                      << __LINE__ << std::endl;                               \
            std::exit(-1);                                                    \
        }                                                                     \
    } while (0)

std::vector<float> BatchedCpuMatmul(const std::vector<float>& in_a,
                                    const std::vector<float>& in_b, int batch,
                                    int m, int n, int k) {
    std::vector<float> out(batch * m * n, 0);
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int l = 0; l < k; l++) {
                    out[b * m * n + i * n + j] += in_a[b * m * k + i * k + l] *
                                                  in_b[b * k * n + l * n + j];
                }
            }
        }
    }

    return out;
}

int main() {
    int batch = 8;
    int m = 16;
    int n = 64;
    int k = 1024;
    std::vector<float> input_a(batch * m * k);
    std::vector<float> input_b(batch * k * n);
    std::vector<float> out_c(batch * m * n, 0);

    // Initializing host data
    for (int i = 0; i < batch * m * k; i++) {
        input_a.emplace_back(rand() % 100 / 1000.);
    }
    for (int i = 0; i < batch * k * n; i++) {
        input_b.emplace_back(rand() % 100 / 2024.);
    }

    // Initializing device data
    void *a_dev, *b_dev, *c_dev;
    CUDA_CHECK(cudaMalloc(&a_dev, batch * m * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b_dev, batch * k * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c_dev, batch * m * n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(a_dev, input_a.data(), batch * m * k * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_dev, input_b.data(), batch * k * n * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c_dev, out_c.data(), batch * m * n * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Get Cpu result
    auto cpu_result = BatchedCpuMatmul(input_a, input_b, batch, m, n, k);

    // Invoke cublasSgemmBatched API
    float alpha = 1.;
    float beta = 1.;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    int64_t stride_a = m * k;
    int64_t stride_b = k * n;
    int64_t stride_c = m * n;

    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, (float*)b_dev, n,
        stride_b, (float*)a_dev, k, stride_a, &beta, (float*)c_dev, n, stride_c,
        batch));

    // Check result
    std::vector<float> out_host(batch * m * n, 0);
    CUDA_CHECK(cudaMemcpy(out_host.data(), c_dev, batch * m * n * sizeof(float),
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < batch * m * n; i++) {
        if (std::abs(cpu_result[i] - out_host[i]) > 1e-6) {
            std::cout << "The difference is too big." << std::endl;
            std::exit(-1);
        }
    }

    return 0;
}
3.3 CublasSgemmGroupedBatched

前面两个函数 cublasSgemmBatched 和 cublasSgemmStridedBatched 都是要求所有的输入输出矩阵是 uniform 的，也就是这些 batchsize 个 A 矩阵必须具有相同 size，leading dimension ，对 BC 也是。如果需要计算的矩阵的metadata 不同的话，则需要分多次调用函数。

cublasStatus_t cublasSgemmGroupedBatched(cublasHandle_t handle,
                                         const cublasOperation_t transa_array[],
                                         const cublasOperation_t transb_array[],
                                         const int m_array[],
                                         const int n_array[],
                                         const int k_array[],
                                         const float  alpha_array[],
                                         const float *const  Aarray[],
                                         const int lda_array[],
                                         const float *const  Barray[],
                                         const int ldb_array[],
                                         const float  beta_array[],
                                         float *const  Carray[],
                                         const int ldc_array[],
                                         int group_count,
                                         const int group_size[])

cublasSgemmGroupedBatched 的出现很好的解决了这个问题，避免了我们需要调用多个函数的麻烦，只需要将不同属性的矩阵及其对应的属性按照顺序存放在数组中，即可通过一次调用函数来计算需要的函数。

从传参的方面讲，cublasSgemmGroupedBatched 更相似与 cublasSgemmBatched，因为两者都是通过指针数组的方式来传入不同矩阵的指针。而正如之前分析的，这种方式无疑会占用的更多的存储开销，增大程序的复杂性，并带来额外的数据拷贝开销。但是，由于此函数的特殊性及其不可替代行，所以我们也无可厚非，按照规矩来便是。

另外，除了矩阵的 size 等属性可以不同以外，此函数还支持为每个矩阵设置不同的 \alpha 和 \beta 值。这种做法就好像将矩阵进行分组了一下，每个组中包含属性相同的矩阵，属性不同的矩阵位于不同的组中，这可能就是函数名字 grouped的由来把。

cublasTgemmGroupedBatched 函数与之前两个函数还有一点不同的是，目前 NV 只支持两种数据类型，分别是单精度浮点数 float 和双精度浮点数 double。

用官方给出的伪代码来描述这个函数所做的工作，即：

idx = 0;
for i = 0:group_count - 1
    for j = 0:group_size[i] - 1
        gemm(transa_array[i], transb_array[i], m_array[i], n_array[i], k_array[i],
             alpha_array[i], Aarray[idx], lda_array[i], Barray[idx], ldb_array[i],
             beta_array[i], Carray[idx], ldc_array[i]);
        idx += 1;
    end
end

由于这个函数是 cublas 12.5 版本更新后新添加的，作者这里还没能测试这个函数，如果有需要的同学可以参考官方给出的 Example: cublas_gemmGroupedBatched_example。

4. Reference

1. cuBLAS

2. Basic_Linear_Algebra_Subprograms

3. Introducing Grouped GEMM APIs in cuBLAS and More Performance Updates | NVIDIA Technical Blog
