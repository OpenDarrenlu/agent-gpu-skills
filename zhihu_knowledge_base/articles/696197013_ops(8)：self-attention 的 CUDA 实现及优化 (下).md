# ops(8)：self-attention 的 CUDA 实现及优化 (下)

**作者**: 紫气东来​上海交通大学 工学硕士

**原文链接**: https://zhuanlan.zhihu.com/p/696197013

---

​
目录
收起
一、使用 CUDNN 接口实现
1.1 cuDNN 概览及其 attention 实现
1.2 cuDNN 接口调用（V10）
二、self-attention 的反向实现
2.1 反向过程的推导
2.2 反向过程的 CPU 实现
2.3 CUDA 的简单实现（V1）
2.4 反向过程的优化（V2~V8）
参考资料

在上篇中，我们详细讨论了 self-attention 的 CUDA 基本实现及其优化过程

紫气东来：ops(7)：self-attention 的 CUDA 实现及优化 (上)
96 赞同 · 3 评论 文章

本篇为 self-attention 的下篇，主要讨论基于CUDNN 的实现，以及反向的实现。

一、使用 CUDNN 接口实现
1.1 cuDNN 概览及其 attention 实现

cuDNN，即 NVIDIA CUDA Deep Neural Network Library，是深度神经网络算子层级 GPU 加速库集合，提供了深度学习算法中常见算子的高效实现，所以也直接成为了很多上层推理引擎底层调优的算子备选实现，比如 TensorRT、比如 TVM。 cuDNN 则是提供了一系列已经写好的高效的 CUDA C Kernel 的集合，主要是是面向深度神经网络算子的高效实现。

cuDNN 中常见的高效算子实现包括：

前向和反向卷积；
矩阵乘；
前向和反向池化；
前向和反向 Softmax；
前向和反向神经网络激活算子，比如 relu、tanh、sigmoid、rlu、gelu、softplus、swish；
前向和反向归一化计算，比如 BN、IN、LN、LRN、LCN；
基础数学逐点计算；
张量变换计算；

本节则主要关注其 Scaled Dot Product Attention 的实现及接口的用法。cuDNN 中的实现采用了 FlashAttention-2 算法，具体原理不再赘述，在此仅讨论参数及接口的用法。cuDNN 提供了 python 和 C++ 两种接口。

Python 接口的用法

Python 接口的输入输出参数如下所示，具体的用法及案例可以参考这里。

Args:
    q (cudnn_tensor): The query data.
    k (cudnn_tensor): The key data.
    v (cudnn_tensor): The value data.
    is_inference (bool): Whether it is an inference step or training step.
    attn_scale (Optional[Union[float, cudnn_tensor]]): The scale factor for attention. Default is None.
    bias (Optional[cudnn_tensor]): The bias data for attention. Default is None.
    use_alibi_mask (Optional[bool]): Whether to use alibi mask. Default is False.
    use_padding_mask (Optional[bool]): Whether to use padding mask. Default is False.
    seq_len_q (Optional[cudnn_tensor]): The sequence length of the query.
    seq_len_kv (Optional[cudnn_tensor]): The sequence length of the key.
    use_causal_mask (Optional[bool]): Whether to use causal mask. Default is False.
    dropout (Optional[Union[Tuple[(probability: float, seed: cudnn_tensor, offset: cudnn_tensor)], Tuple[mask: cudnn_tensor, scale: cudnn_tensor]]]): Whether to do dropout. Default is None.
    compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
    name (Optional[str]): The name of the operation.

Returns:
    o (cudnn_tensor): The output data.
    stats (Optional[cudnn_tensor]): The softmax statistics in case the operation is in a training step.

C++ API 接口

C++ 接口的输入输出参数如下所示

// returns [output, softmax_stats]
std::array<std::shared_ptr<Tensor_attributes>, 2> 
sdpa(std::shared_ptr<Tensor_attributes> q,
     std::shared_ptr<Tensor_attributes> k,
     std::shared_ptr<Tensor_attributes> v,
     SDPA_attributes options);

其中SDPA_attributes 的参数包括

set_is_inference(bool const value);
set_attn_scale(std::shared_ptr<Tensor_attributes> value);
set_attn_scale(float const value);
set_bias(std::shared_ptr<Tensor_attributes> value);
set_alibi_mask(bool const value);
set_padding_mask(bool const value);
set_seq_len_q(std::shared_ptr<Tensor_attributes> value);
set_seq_len_kv(std::shared_ptr<Tensor_attributes> value);
set_causal_mask(bool const value);
set_dropout(float const probability,
            std::shared_ptr<Tensor_attributes> seed,
            std::shared_ptr<Tensor_attributes> offset);
set_dropout(std::shared_ptr<Tensor_attributes> mask,
            std::shared_ptr<Tensor_attributes> scale);
set_compute_data_type(DataType_t value);

接下来在下一小节中介绍其用法。

1.2 cuDNN 接口调用（V10）
首先构造输入输出 tensor
using graph_tensors_fwd = std::tuple<std::shared_ptr<fe::graph::Graph>,
                                     std::shared_ptr<fe::graph::Tensor_attributes>,  // Q,
                                     std::shared_ptr<fe::graph::Tensor_attributes>,  // K,
                                     std::shared_ptr<fe::graph::Tensor_attributes>,  // V,
                                     std::shared_ptr<fe::graph::Tensor_attributes>,  // Attn_scale,
                                     std::shared_ptr<fe::graph::Tensor_attributes>,  // O
                                     std::shared_ptr<fe::graph::Tensor_attributes>>; // Stats

// Need a cache because graph->build_operation_graph() is slow but everything else seems fast
using cache_type_fwd = std::unordered_map<std::size_t, graph_tensors_fwd>;
基于 cuDNN frontend 构造图，并设置参数以及 tensor mapping
// Loosely based on cuDNN frontend samples functions and massively simplified
template <typename... Args>
auto lookup_cache_or_build_graph_fwd(Args... args) {
    static cache_type_fwd user_maintained_cache_fwd;
    auto [B, H, T, HS, is_inference_only] = std::make_tuple(args...);

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(CUDNN_16BIT)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

    // QKV is (B, T, 3, NH, HS) which cuDNN can handle directly without an external permute
    auto Q = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("Q")
                               .set_dim({B, H, T, HS})
                               .set_stride({3 * H * HS * T,  HS, 3 * H * HS, 1}));
    auto K = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("K")
                               .set_dim({B, H, T, HS})
                               .set_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    auto V = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("V")
                               .set_dim({B, H, T, HS})
                               .set_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    auto attn_scale = graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("attn_scale")
                                .set_dim({1, 1, 1, 1})
                                .set_stride({1, 1, 1, 1})
                                .set_is_pass_by_value(true)
                                .set_data_type(fe::DataType_t::FLOAT));

    auto sdpa_options = fe::graph::SDPA_attributes().set_name("flash_attention");
    sdpa_options.set_is_inference(is_inference_only);
    sdpa_options.set_attn_scale(attn_scale);
    sdpa_options.set_causal_mask(true);

    // Create the graph operation and get the output tensors back
    auto [O, stats] = graph->sdpa(Q, K, V, sdpa_options);

    // Output is (B, T, NH, HS) BF16/FP16 and stats for backward pass is (B, NH, T) FP32
    O->set_output(true).set_dim({B, H, T, HS}).set_stride({H * HS * T, HS, H * HS, 1});

    assert(stats == nullptr || is_inference_only == false);
    if (is_inference_only == false) {
        stats->set_output(true).set_data_type(fe::DataType_t::FLOAT)
                               .set_dim({B, H, T, 1})
                               .set_stride({H * T, T, 1, 1});
    }

    assert(graph->validate().is_good());
    auto key = graph->key();
    auto it = user_maintained_cache_fwd.find(key);
    if (it != user_maintained_cache_fwd.end()) {
        return it->second;
    }

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    assert(graph->build_operation_graph(cudnn_handle).is_good());
    auto plans = graph->create_execution_plans({fe::HeurMode_t::A});
    assert(graph->check_support(cudnn_handle).is_good());
    assert(graph->build_plans(cudnn_handle).is_good());

    auto tuple = std::make_tuple(graph, Q, K, V, attn_scale, O, stats);
    user_maintained_cache_fwd.insert({key, tuple});
    return tuple;
}
完整的 kernel 的实现如下所示
void attention_forward_cudnn(floatX* out,  // output: (B, T, NH, HS)
                             float* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             float* in_fp32,  // fp32 input
                             float* out_fp32, // fp32 output for validation
                             int B, int T, int C, int NH) {
    static bool first_run_validation = true;
    int HS = C / NH; // number of features per head
    bool is_inference_only = (stats == nullptr);

    // Convert from FP32 to FP16/BF16 on 1st run to get correct results
    const int block_size = 64; // smallest full occupancy block size on modern GPUs
    if (first_run_validation) {
        int total_threads = B * T * C * 3;
        assert(total_threads % block_size == 0);
        int num_blocks = total_threads / block_size;
        fp32_to_lowp_kernel<<<num_blocks, block_size>>>(inp, in_fp32);
    }

    // Get graph and tensors from cache (or generate it on first use)
    auto [graph, Q, K, V, attn_scale, O, softmax_stats] =
        lookup_cache_or_build_graph_fwd(B, NH, T, HS, is_inference_only);

    // Prepare all the tensor pointers for executing the graph
    void* devPtrQ = inp;
    void* devPtrK = (inp + C);
    void* devPtrV = (inp + 2 * C);
    float attn_scale_cpu = 1.0 / sqrtf(HS);
    void* devPtrO = out;

    // Build variant pack
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {Q, devPtrQ}, {K, devPtrK}, {V, devPtrV}, {attn_scale, &attn_scale_cpu}, {O, devPtrO}};

    // Add the stats tensor unless we are only doing inference (only needed for backward pass)
    if (is_inference_only == false) {
        variant_pack[softmax_stats] = stats;
    }

    // Reallocate the workspace if the required size is greater than the current workspace
    // By default, cuDNN uses up to 256MiB of workspace, so we don't want to just allocate the maximum
    if (graph->get_workspace_size() > cudnn_workspace_size) {
        if (cudnn_workspace_size > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_size = graph->get_workspace_size();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_size));
    }

    // Execute graph
    assert(graph->execute(cudnn_handle, variant_pack, cudnn_workspace).is_good());
    cudaCheck(cudaGetLastError());

    // Optionally convert back from FP16/BF16 to FP32
    if (first_run_validation) {
        int total_threads = B * T * C;
        assert(total_threads % block_size == 0);
        int num_blocks = total_threads / block_size;
        lowp_to_fp32_kernel<<<num_blocks, block_size>>>(out, out_fp32);
    }
    cudaCheck(cudaGetLastError());
    first_run_validation = false;
}

性能数据如下：

block_size   32 | time 0.169061 ms
block_size   64 | time 0.165807 ms
block_size  128 | time 0.167423 ms
block_size  256 | time 0.165734 ms
block_size  512 | time 0.167426 ms

现在用 RTX 4090 + CUDA12.4 重跑一下V1~V5版本，并与之对比如下，可见 cuDNN 的实现具有显著的优势：

二、self-attention 的反向实现
2.1 反向过程的推导

首先说明矩阵乘法的求导，对于矩阵乘法 \mathbf{Y}=\mathbf{W}\mathbf{X} ，假设其目标函数值为 \phi ，现在用 \text d\mathbf Y, \text d \mathbf W ,\text d \mathbf X 分别表示 \frac{\partial \phi}{\partial \mathbf{Y}}, \frac{\partial \phi}{\partial \mathbf{W}}, \frac{\partial \phi}{\partial \mathbf{X}} 则有(证明过程参见引文[5])： \begin{aligned} \text d \mathbf{W} & =\text d\mathbf{ Y} \cdot \mathbf{X}^T \\ \text d \mathbf{X} & =\mathbf{W}^T \cdot \text d\mathbf{Y} \end{aligned}\\接下来，讨论一下 softmax 的求导过程

设 X=[x_1,x_2,\cdots,x_n],Y=softmax(X)=[y_1,y_2,\cdots,y_n]

即 y_i=\frac{e^{x_i}}{\sum\limits_{j=1}^{n}e^{x_j}} ，且 \sum\limits_{i=1}^ny_i=1 ，现在来求导数 \frac{\partial y_i}{\partial x_j}

(1)当 i=j 时

\begin{align} \frac{\partial y_i}{\partial x_j}&=\frac{\partial y_i}{\partial x_i} \\ &=\frac{\partial}{\partial x_i}(\frac{e^{x_i}}{\sum_ke^{x_k}}) \\&=\frac{(e^{x_i})^{'}(\sum_ke^{x_k})-e^{x_i}(\sum_ke^{x_k})^{'}}{(\sum_ke^{x_k})^2} \\ &=\frac{e^{x_i}\cdot(\sum_ke^{x_k})-e^{x_i}\cdot e^{x_i}}{(\sum_ke^{x_k})^2} \\ &=\frac{e^{x_i}\cdot(\sum_ke^{x_k})}{(\sum_ke^{x_k})^2}-\frac {e^{x_i}\cdot e^{x_i}}{(\sum_ke^{x_k})^2} \\ &=\frac{e^{x_i}}{\sum_ke^{x_k}}-\frac{e^{x_i}}{\sum_ke^{x_k}}\cdot \frac{e^{x_i}}{\sum_ke^{x_k}}\\&=y_i-y_i\cdot y_i\\&=y_i(1-y_i) \end{align}\\

(2)当 i\ne j 时

\begin{aligned} \frac{\partial y_i}{\partial x_j}&=\frac{\partial}{\partial x_j}(\frac{e^{x_i}}{\sum_ke^{x_k}}) \\&=\frac{(e^{x_i})^{'}(\sum_ke^{x_k})-e^{x_i}(\sum_ke^{x_k})^{'}}{(\sum_ke^{x_k})^2} \\&=\frac{0\cdot(\sum_ke^{x_k})-e^{x_i}\cdot e^{x_j}}{(\sum_ke^{x_k})^2} \\&=\frac{-e^{x_i}\cdot e^{x_j}}{(\sum_ke^{x_k})^2} \\&=-\frac{e^{x_i}}{\sum_ke^{x_k}}\cdot \frac{e^{x_j}}{\sum_ke^{x_k}}\\&=-y_i\cdot y_j \end{aligned}\\

综上所述： \frac{\partial y_i}{\partial x_j}=\left\{ \begin{aligned} &y_i(1-y_j) ,~~if~~i=j\\ & y_i(0- y_j) ， ~~if~~ i \ne j \\ \end{aligned} \right.

有了以上准备工作，便可以进行反向过程的计算了。

令 \mathbf{P} = \text{Softmax}(\mathbf S), ~~\mathbf S ={\frac{\mathbf Q\mathbf K^\top}{\sqrt {d_k}}}， \mathbf O = \mathbf P\mathbf V ，则 attention 的反向过程可以描述为

2.2 反向过程的 CPU 实现

在弄清楚了反向的逻辑和流程之后，便可以实现其过程了，其过程对照以上原理非常清晰，不再赘述

// NOTE: Also contains the re-shuffling of the exact position of "scale"
// and when it is applied (after preatt, not "during" preatt)
// also, full matrices are materialized, even the parts that get masked out
void attention_backward_cpu(float* dinp, float* dpreatt, float* datt,
                            float* dout, float* inp, float* att,
                            int B, int T, int C, int NH) {
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;
                float* datt_bth = datt + b*NH*T*T + h*T*T + t*T;
                float* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;
                float* dquery_t = dinp + b * T * C3 + t * C3 + h * hs;
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;

                // backward pass 4, through the value accumulation
                float* dout_bth = dout + b * T * C + t * C + h * hs;
                for (int t2 = 0; t2 < T; t2++) { // ADJUSTED! this was t2 <= t (see note on function)
                    float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C*2;
                    for (int i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // out_bth[i] += att_bth[t2] * value_t2[i];
                        // so now we have:
                        datt_bth[t2] += value_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }

                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                for (int t2 = 0; t2 <= t; t2++) {
                    for (int t3 = 0; t3 <= t; t3++) {
                        float indicator = t2 == t3 ? 1.0f : 0.0f;
                        float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += scale * local_derivative * datt_bth[t2];
                    }
                }

                // backward pass 1, the query @ key matmul
                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    float* dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    for (int i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // preatt_bth[t2] += query_t[i] * key_t2[i]
                        // so now we have:
                        dquery_t[i] += key_t2[i] * dpreatt_bth[t2];
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2];
                    }
                }
            }
        }
    }
}
2.3 CUDA 的简单实现（V1）

简单实现仍然是按照以上的逻辑，同时利用 cublas 库函数计算矩阵乘法

// the sequence of transformations in this compound op is:
// inp (B,T,3C) -> qkvr (B,T,3C) -> preatt (B,NH,T,T) -> att (B,NH,T,T) -> vaccum (B,T,C) -> out (B,T,C)
template<class SoftmaxKernel>
void attention_backward1(float* dinp, float* dqkvr, float* dpreatt, float* datt, float* dvaccum,
                        const float* dout,
                        const float* inp, const float* qkvr, const float* preatt, const float* att, const float* vaccum,
                        int B, int T, int C, int NH,
                        SoftmaxKernel softmax_autoregressive_backward,
                        const int block_size) {
    int HS = C / NH; // head size
    const float alpha = 1.0f;
    const float beta = 1.0f; // note beta = 1.0f so that we accumulate gradients (+=)
    // unpack convenience pointers into q, k, v
    const float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    float *dq, *dk, *dv;
    dq = dqkvr + 0 * B * T * C;
    dk = dqkvr + 1 * B * T * C;
    dv = dqkvr + 2 * B * T * C;

    // backward through the unpermute operation
    int num_blocks = ceil_div(B * T * C, block_size);
    unpermute_kernel_backward<<<num_blocks, block_size>>>(dvaccum, dout, B, T, NH, HS);
    cudaCheck(cudaGetLastError());

    // backward into datt
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            T, T, HS,
                            &alpha,
                            v, HS, T * HS,
                            dvaccum, HS, T * HS,
                            &beta,
                            datt, T, T * T,
                            B * NH));

    // backward into dv
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            HS, T, T,
            &alpha,
            dvaccum, HS, T * HS,
            att, T, T * T,
            &beta,
            dv, HS, T * HS,
            B * NH));

    // backward into preatt
    softmax_autoregressive_backward(dpreatt, datt, att, B, T, C, NH, block_size);
    cudaCheck(cudaGetLastError());

    // backward into q
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            HS, T, T,
                            &alpha,
                            k, HS, T * HS,
                            dpreatt, T, T * T,
                            &beta,
                            dq, HS, T * HS,
                            B * NH));
    // backward into k
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            HS, T, T,
                            &alpha,
                            q, HS, T * HS,
                            dpreatt, T, T * T,
                            &beta,
                            dk, HS, T * HS,
                            B * NH));

    // backward into inp
    num_blocks = ceil_div(B * NH * T * HS, block_size);
    permute_kernel_backward<<<num_blocks, block_size>>>(dinp, dq, dk, dv, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}

在 RTX 4090 上性能数据如下：

block_size   32 | time 7084.399902 ms
block_size   64 | time 7067.519531 ms
block_size  128 | time 7077.885254 ms
block_size  256 | time 9231.899414 ms
block_size  512 | time 8673.948242 ms
block_size 1024 | time 14843.697266 ms
2.4 反向过程的优化（V2~V8）

由于以上的计算中矩阵乘法利用了 cublas 库函数，因此后续的优化主要围绕 softmax 来做

V2：在 t,b,h 维度上并行，性能数据如下
block_size   32 | time 273.880554 ms
block_size   64 | time 271.534363 ms
block_size  128 | time 271.852448 ms
block_size  256 | time 287.613922 ms
block_size  512 | time 339.183411 ms
block_size 1024 | time 388.895203 ms
V3：在 t,b,h 维度上并行，并使用协作组，性能数据如下
block_size   32 | time 17.532518 ms
block_size   64 | time 15.095194 ms
block_size  128 | time 14.864792 ms
block_size  256 | time 14.368258 ms
block_size  512 | time 14.173902 ms
block_size 1024 | time 14.197658 ms
V4：在 V3 基础上增加 UNROLL 操作，性能数据如下
block_size   32 | time 10.701917 ms
block_size   64 | time 11.321651 ms
block_size  128 | time 10.746265 ms
block_size  256 | time 10.383463 ms
block_size  512 | time 10.818865 ms
block_size 1024 | time 10.808935 ms
V5：优化 V4 版本的一些特殊情况，性能数据如下
block_size   32 | time 6.028288 ms
block_size   64 | time 6.770285 ms
block_size  128 | time 6.253971 ms
block_size  256 | time 6.313370 ms
block_size  512 | time 6.458060 ms
block_size 1024 | time 6.442598 ms
V6：该方法比较复杂，主要是通过循环重构和内存访问优化来提高性能，数据如下
block_size   32 | time 3.683847 ms
block_size   64 | time 2.985779 ms
block_size  128 | time 2.968576 ms
block_size  256 | time 3.310278 ms
block_size  512 | time 4.242841 ms
block_size 1024 | time 6.089725 ms
V7：简化数学计算，同时使用协作组的规约操作，性能数据如下
block_size   32 | time 1.766189 ms
block_size   64 | time 1.761990 ms
block_size  128 | time 1.760563 ms
block_size  256 | time 1.758925 ms
block_size  512 | time 1.760154 ms
block_size 1024 | time 1.897673 ms
V8：在V7的基础上新增一些 tricks，性能基本持平
block_size   32 | time 1.781242 ms
block_size   64 | time 1.778285 ms
block_size  128 | time 1.768653 ms
block_size  256 | time 1.762298 ms
block_size  512 | time 1.760768 ms
block_size 1024 | time 1.875651 ms

以上几种方法的性能对比如下：

以上代码更新在 https://github.com/ifromeast/cuda_learning/blob/main/04_transformer/ops/attention_backward.cu。

参考资料

[1] https://github.com/karpathy/llm.c/blob/master/dev/cuda/attention_forward.cu

[2] https://github.com/karpathy/llm.c/blob/master/dev/cuda/attention_backward.cu

[3] https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/operations/Attention.md

[4] https://github.com/NVIDIA/cudnn-frontend/blob/main/samples/cpp/mha.cpp

[5] Not understanding derivative of a matrix-matrix product.

[6] https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf

[7] 无用：反向传播之一：softmax函数

休言万事转头空，未转头时皆梦。——苏轼《西江月·平山堂》
