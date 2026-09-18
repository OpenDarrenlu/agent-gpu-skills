# Pytorch internals - 以add算子为例理解elementwise_kernel和TensorIterator的调用流程

**作者**: 俯仰AI Framework Engineer

**原文链接**: https://zhuanlan.zhihu.com/p/690858698

---

以CUDA add 函数为例，探究pytorch中elementwise kernel的用法

TORCH_IMPL_FUNC(ufunc_add_CUDA)(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, const at::Tensor & out) {
  add_kernel(*this, alpha);
}


其中torch内置的宏定义为

#define TORCH_IMPL_FUNC(name) void structured_##name::impl


将宏定义进行字符串替换，就可以得到

void structured_ufunc_add_CUDA::impl(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, const at::Tensor & out) {
  add_kernel(*this, alpha);
}


在文件UfuncCUDA_add.cu中，结构体structured_ufunc_add_CUDA与其成员函数impl在最开始已经声明：

struct TORCH_API structured_ufunc_add_CUDA : public at::meta::structured_add_Tensor {
void impl(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, const at::Tensor & out);
};


而结构体 structured_ufunc_add_CUDA可以看到是继承自at::meta::structured_add_Tensor

struct TORCH_API structured_add_Tensor : public TensorIteratorBase {

void meta(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha);

};


而结构体structured_add_Tensor又继承自TensorIteratorBase。

这也就解释了为什么在函数structured_ufunc_add_CUDA::impl的实现中调用的add_kernel函数的第一个参数是指向结构体structured_ufunc_add_CUDA自身的指针*this，因为其本身就是结构体TensorIteratorBase的派生。

但是，一个先验常识是，如果向使用TensorIteratorBase，我们需要先实例化一个TensorIteratorConfig，并向其中添加所有参与计算的输入和输出张量，最后显式地调用build方法来得到一个使用输入输出张量进行初始化的TensorIteratorBase实例，这样的实例才是真正有灵魂的，可以使用的。

但在方法structured_ufunc_add_CUDA::impl中，接受的参数是所有的输入输出的张量，但在实际调用kernel 函数add_kernel的时候，只传入了结构体本身*this和alpha两个参数，其余输入输出张量并没有像我们想象中那样用于构造一个有实际意义的TensorIteratorBase。

那么impl方法是如何将传入的多个Tensor参数转换为TensorBaseIterator的呢？那么一个合理的解释就是在实际调用structured_ufunc_add_CUDA::impl这个方法之前，已经在某个方法使用输入输出张量对structured_ufunc_add_CUDA进行了初始化，使其可以直接作为参数被structured_ufunc_add_CUDA: :impl方法调用。

奥秘就在在于结构体structured_add_Tensor中的meta方法，该方法定义在BinaryOps.cpp中：

TORCH_META_FUNC2(add, Tensor) (
  const Tensor& self, const Tensor& other, const Scalar& alpha
) {
  build_borrowing_binary_op(maybe_get_output(), self, other);
  native::alpha_check(dtype(), alpha);
}


结合宏#define TORCH_META_FUNC(name) void structured_##name::meta，就得到了meta的实现

void structured_add_Tensor::meta( const Tensor& self, const Tensor& other, const Scalar& alpha) {
  build_borrowing_binary_op(maybe_get_output(), self, other);
  native::alpha_check(dtype(), alpha);
}


该方法调用了两个方法: build_borrowing_binary_op和alpha_check。其中，前者分别将output、self、other三个参数作为操作数来构造一个用于二元操作的TensorIteratorBase，而结构体structured_add_Tensor本身就继承自结构体TensorIteratorBase，故这三个参数用来对其自身的参数进行初始化赋值和构造。而后者则检查了非Tensor参数alpha的数据类型是否符合预期（什么类型是符合预期的会在上一个方法build_borrowing_binary_op的调用过程中推理得出。这里有个小问题，有时候构造一个TensorIteratorBase需要输入张量和输出张量，为什么meta方法压根没有为输出张量预留参数位置，而是只使用两个操作数和一个常量alpha作为参数？那么输出张量从哪里传入？记住这个问题，我们后面展开。

详细的调用流程：

首先，Dispatcher根据输入tensor的device将add算子派发到CUDA backend，调用add函数在CUDA backend的实现：

TORCH_LIBRARY_IMPL(aten, CUDA, m) {
  ....
m.impl("add.Tensor", TORCH_FN(wrapper_CUDA_add_Tensor));
m.impl("add.out", TORCH_FN(wrapper_CUDA_add_out_out));
m.impl("add_.Tensor", TORCH_FN(wrapper_CUDA_add__Tensor));
  ....


这三个算子都是用来做加法操作的，他们之间的区别在于输出结果的存放位置，具体而言：

add.Tensor：这种形式的加法是创建一个新的张量，将输入张量与另一个张量或标量相加的结果存储在这个新张量中。这种方式不会修改原始张量的值。
add.out：这种形式的加法允许你指定一个输出张量，将输入张量与另一个张量或标量相加的结果存储在提供的输出张量中。如果提供了输出张量，它必须有足够的空间来存储结果。这种方式也不会修改原始张量的值。
add_.Tensor：这种形式的加法是原地（in-place）操作，意味着将输入张量与另一个张量或标量相加的结果直接存储在第一个输入张量中，修改了原始张量的值。这种方式不会创建新的张量。

从三者的形参定义中也可见一斑：

at::Tensor wrapper_CUDA_add_Tensor(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha);

at::Tensor & wrapper_CUDA_add_out_out(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out);

at::Tensor & wrapper_CUDA_add__Tensor(at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha);


三个方法，只有wrapper_CUDA_add_out_out需要传入输出张量的引用out，最终的计算结果就会存入其中。我们就以它为例来看看，一个add算子是如何找到对应的kernel，如何利用Tensor形式的输入输出构造一个TensorIteratorBase，并顺利调用elementwise_kernel来优化计算的。

下面贴出wrapper_CUDA_add_out_out方法的实现：

at::Tensor & wrapper_CUDA_add_out_out(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
  // No device check
  structured_ufunc_add_CUDA_out op(out);
  op.meta(self, other, alpha);
  op.impl(self, other, alpha, op.maybe_get_output(0));
  if (op.proxy_outputs_[0].has_value()) 
      op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);
  return out;
}


wait! structured_ufunc_add_CUDA_out是什么？有点眼熟但是好像又有点不同。是的，结构体structured_ufunc_add_CUDA_out继承自结构体structured_ufunc_add_CUDA（忘了的话可以翻到文章最开始回顾一下这个结构体），并重写了两个新的方法set_output_strided和set_output_raw_strided。光看名字，我们可以得到的信息是，这两个方法设置了一些关于输出的信息。

struct structured_ufunc_add_CUDA_out final : public at::native::structured_ufunc_add_CUDA {
    structured_ufunc_add_CUDA_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    void set_output_strided(
        int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
        TensorOptions options, DimnameList names
    ) override {
       ...
    }
    void set_output_raw_strided(
        int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
        TensorOptions options, DimnameList names
    ) override {
        ...
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
      return proxy_outputs_[output_idx].has_value() ? **proxy_outputs_[output_idx] : outputs_[output_idx].get();
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
    std::array<c10::optional<c10::ExclusivelyOwned<Tensor>>, 1> proxy_outputs_;
    c10::hip::OptionalHIPGuardMasqueradingAsCUDA guard_;
};


用一张图总结一下这一连串的继承关系




这样看起来就比较清晰了，pytorch将一个算子的kernel做了不同层次的抽象，每个层次会做不同的事情。基结构体TensorIterbase以上本文先不做讨论。那么，

structured_add_Tensor实现meta方法。

该方法利用将要执行目标操作的一众操作数，即Tensor类型的输出输出，来完成对自身所具有的，继承自TensorIteratorBase的一众属性进行初始化构造，方便后期计算时使用。

2. structure_ufunc_add_CUDA实现impl方法。

其中调用了针对于特定backend的kernel函数的实现，执行具体的操作。众所周知，不同的backend的kenerl实现不同，所以每个backend都是各自派生出一个结构体，并实现对应的impl方法。对于CUDA backend而言，其impl实现就会对应的包含对CUDA runtime的调用。而对于CPU backend而言，就会使用单指令多数据（Single Instruction Multiple Data）指令，如SSE，AVX512等来进行加速。

3.structured_ufunc_add_CUDA_out重写基类MetaBase的三个方法set_output_strided、set_output_raw_strided和maybe_output。

到了这里，就时为了输出的存储方式而进行分化了，具体而言就是上文中提到的三种不同的存储方式。不同的方式对构造输出信息的方式也不相同，因此需要他们的各自重写相关的函数，即set_output_strided和set_output_raw_strided。那么，剩下的maybe_output函数的作用是什么呢？我们接着往下看。

回到之前的叙述，一个算子经过pytorch的dispatcher分发，找到了指定的CUDA backend的对应的kernel实现，那么这个kernel里面到底做了什么？

at::Tensor & wrapperCUDAaddoutout(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {   
// No device check
      structuredufuncaddCUDAout op(out); 
      op.meta(self, other, alpha); 
      op.impl(self, other, alpha, op.maybegetoutput(0)); 
      if (op.proxyoutputs[0].hasvalue()) op.outputs[0].get().copy(**op.proxyoutputs_[0]); 
       return out; 
}





首先，还记得刚刚说过的结构体structured_ufunc_add_CUDA_out，先是初始化了一个该结构体的实例op，等下！初始化用到了输出Tensor out，难道输出张量是在结构体初始化的时候传进来的？赶快看一下结构体structured_ufunc_add_CUDA_out 的初始化函数：

structuredufuncaddCUDAout(Tensor& out0) : outputs_{ std::ref(out0) } {}


果然，在初始化过程中，`structured_ufunc_add_CUDA_out`将输出Tensor out放在了一个数组中：std::array<c10::optional<c10::ExclusivelyOwned<Tensor>>, 1> proxy_outputs_中，随后，在meta函数中，通过maybe_out这个函数来从相应的数组中取出使用。

const Tensor& maybegetoutput(int64t outputidx) override {       
      return proxyoutputs[outputidx].hasvalue() ? **proxyoutputs[outputidx] : outputs[output_idx].get();     
} 


而对于不需要使用输出Tensor的结构体，输出Tensor则并不会参与其初始化，或者使用self Tensor自身对结构体进行初始化。

大题得证！

那么接下来，op则会调用具体的impl方法执行计算，得到输出结果并返回。

洋洋洒洒写了一大堆，居然只聊完了算子的派发到内核函数的调用，总结下来感觉就是pytorch需要在不同的阶段执行不同的工作。除此之外，针对算子类型以及不同的后端需要对症下药，根据其自身的特定实现一些特化的方法，故此派生了许多不同类型、不同抽象等级的结构体。

[TODO]

接下来会针对TensorIterator和elementwise_kernel的神奇组合实现算子的统一调用尝试深入分析。
