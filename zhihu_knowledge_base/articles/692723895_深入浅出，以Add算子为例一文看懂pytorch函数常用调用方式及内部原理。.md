# 深入浅出，以Add算子为例一文看懂pytorch函数常用调用方式及内部原理。

**作者**: 俯仰AI Framework Engineer

**原文链接**: https://zhuanlan.zhihu.com/p/692723895

---

​
目录
收起
torch.add(a, b) 与 a.add(b)
a.add_(b)
torch.add(a, b, out=c)
Tensor运算符的重载（operator overload）
1. +=运算符的重载
2. +运算符的重载
http://torch.xxx和http://torch.Tensor.xxx

回顾：在上次关于add算子注册和调用派发的实现机制中提到了，对两个Tensor进行加法操作，底层可能会为根据输出Tensor存储位置的不同，派生出三个不同的结构体，它们复用了相同的计算kernel。详细内容可以查看 ：

Pytorch internals - 以add算子为例理解elementwise_kernel和TensorIterator的调用流程

输出Tensor三种不同的存放位置，对应着算子三种不同的操作模式，以add算子为例，它们分别是:

add : 返回的结果存放在一个新的Tensor中
add_：返回的结构存放在第一个输入Tensor中，覆盖其原有数据
add.out：返回的结果存放在用户指定的Tensor中

那么，这三种操作在应用层，也就是我们的python前端分别对应什么样的操作呢？

Talk is cheap, 直接show code。为了方便掩饰，本文以jupyter notebook的形式进行代码展示

torch.add(a, b) 与 a.add(b)
import torch

首先定义两个shape相同的Tensor

a = torch.ones(3, 4)
b = torch.ones_like(a)*2
print(a)
print(b)
​
# output
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]])
tensor([[2., 2., 2., 2.],
        [2., 2., 2., 2.],
        [2., 2., 2., 2.]])

首先来看看，我们最为常用的，也是最符合我们编程习惯的执行add的方式

torch.add(a, b)
print(a)
print(b)
​
# output
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]])
tensor([[2., 2., 2., 2.],
        [2., 2., 2., 2.],
        [2., 2., 2., 2.]])

可以发现，torch.add操作执行后，a和b的值都没有改变，而操作结果也并没有被保存下来，这是因为add函数返回了操作执行的结果，但我们并没有使用变量去接收，所以结果被丢弃了没有保存下来。现在让我们接收返回的结果看看

result = torch.add(b)
print(a)
print(b)
print(result)
​
# output
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]])
tensor([[2., 2., 2., 2.],
        [2., 2., 2., 2.],
        [2., 2., 2., 2.]])
tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]])

从输出可以看到，原有操作数a, b的值并没有被改变，torch.add函数的返回了一个新的Tensor，里面存放着计算结果，我们通过变量result接收了这个计算结果，使其可以在后续的计算中被引用。这种方式可以说是最符合我们直觉的编程范式。

另外，Tensor类的add方法与torch.add方法调用的是同一个c++实现，通过将同一个c++方法分别注册在_VariableFunctionsClass和TensorBase中实现的，在文章的末尾我们会展开，我们在这里可以简单验证一下：

a = torch.ones(3, 4)
b = torch.ones_like(a)*2
print(a)
print(b)
result = a.add(b)
print(a)
print(b)
print(result)
print(id(a))
print(id(result))
​
# ouput
# before operation : a
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]])
# before operation : b
tensor([[2., 2., 2., 2.],
        [2., 2., 2., 2.],
        [2., 2., 2., 2.]])
# after operation : a
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]])
# after operation : b
tensor([[2., 2., 2., 2.],
        [2., 2., 2., 2.],
        [2., 2., 2., 2.]])
# after operation: result
tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]])
# id of a
4750534560
# id of result
4458026448

和torch.add的表现一致，torch.Tensor.add将计算结果保存在一个新的Tensor返回。

a.add_(b)

通过上述add这种范式进行编程，函数的输入输出非常清晰，具有很强的可读性。但是如果考虑到Memory efficiency，那么显然这种方式对于我们的内存资源不是那么的友好。考虑这样一个极端情况：如果你的计算机内存只有1MB大小，而我们的Tensor都非常的巨大，每个都需要占据0.5MB内存，此时仅仅两个输入Tensor a和b就已经占据了所有的内存，此时如果还要为计算结果再申请额外的0.5MB的内存，那肯定会直接报OOM了。

虽然作者上述例子有些极端，显然是刻意为之，但考虑到LLM盛行的今天，各家公司掀起的“百模大战”中，以Open AI为首的AIGC企业都是scaling law的坚信者与奉行者，它们坚信模型的scale直接决定了模型的天赋与上限，似乎无限叠加模型的scale已经被奉为了通网AGI之路的圣经（至少目前是）。说的有点远了，回到我们普通开发者本身，当计算计算设备受限的情况下，我们不得不充分利用现有的每一颗子弹。在大模型训推场景中，限制因素除了算力之外，另一个至关重要的就是显存大小了。尤其是在训练场景中，现有的深度学习优化方法是通过mini-batch的特征去近似整个数据集的统计特征，足够大的显存能够让你把batch_size拉到足够大，让你的训练过程更加稳定的同时，模型的performance也会相应提升。

综上，合理利用内存（显存）在如今的时代背景下非常必要。那么回到我们的topic，当参与操作的某个操作数仅会在本次用到，那么我们就可以把他所占用的内存资源进行回收，用于存放输出的结果，那么在开始的那个例子中，虽然a和b已经占满了整个显存，但是a在之后的操作中不会再被用到那么我们就可以将计算结果存放在a的内存中，覆盖a中原有的数据。这种方式称为原地操作（in-place）。

a = torch.ones(3, 4)
b = torch.ones_like(a)*2
print(a)
print(b)
result_ = a.add_(b)
print(a)
print(b)
print(result_)
​
# output
# before operation : a
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]])
# before operation : b
tensor([[2., 2., 2., 2.],
        [2., 2., 2., 2.],
        [2., 2., 2., 2.]])
# after operation : a
tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]])
# after operation : b
tensor([[2., 2., 2., 2.],
        [2., 2., 2., 2.],
        [2., 2., 2., 2.]])
# after operation : result_
tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]])

可以看到，操作完成后Tensor a中的值发生了改变，计算结果被放在了a原来的内存中，但是新问题来了，返回值Tensor中也有计算结果，这不还是占用了新的内存吗？我们一起看一下：

print(id(a))
print(id(result_))
​
# output
4905729248
4905729248

真相大白，原来两者的内存地址是一样的，也就是返回的result_实际上是Tensor a的一个引用，它和Tensor a指向的是同一个内存区域，并没有额外占用新的内存。如果这样你还是觉得不严谨，那么我们可以追到底层c++的实现中一探究竟。

在上篇文章Pytorch internals - 以add算子为例理解elementwise_kernel和TensorIterator的调用流程中，我们了解到了add算子经过其内部的派发器（Dispatcher）进行派发后，最终会调用三个对应的kenel，我们来看看他们的函数声明:

at::Tensor wrapper_CUDA_add_Tensor(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha);
​
at::Tensor & wrapper_CUDA_add_out_out(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out);
​
at::Tensor & wrapper_CUDA_add__Tensor(at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha);


可以看到，只有对应于add操作的方法wrapper_CUDA_add_Tensor的返回值类型是at::Tensor，其余两个方法的返回值类型都是at::Tensor &，也就是我们所说的引用类型。它们函数实现中的逻辑就是将结果存在某个给定的Tensor（如操作数a）中，然后返回其引用，避免在calling stack返回过程中中重新分配新的内存。如果还想查看更加详细的函数实现，可以跟着作者上篇文章追到底层实现去看，由于这并非本文重点，点到为止，不再赘述。

和add方法不同的是，add_方法仅注册在了TensorBase类中，只可以作为类方法被调用，所以没有torch.add_(a, b)的调用方式❌

torch.add(a, b, out=c)

相信很多同学已经从小标题发现了规律，torch.add(a, b, out)这种用户指定输出结果张量的方式，仅注册在了_VariableFunctionsClass类中，因此仅能够通过上述方式调用。验证一下：

a = torch.ones(3, 4)
b = torch.ones_like(a)*2
result_out = torch.empty_like(a)
print("original id of result_out:", id(result_out))
print(a)
print(b)
result = torch.add(a, b, out=result_out)
print(a)
print(b)
print(result_out)
print("after operation id of result_out:", (id(result_out)))
​
#output
original id of result_out: 4754570624
# before operation:a 
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]])
# before operation: b
tensor([[2., 2., 2., 2.],
        [2., 2., 2., 2.],
        [2., 2., 2., 2.]])
# after operation: a 
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]])
# after operation: b
tensor([[2., 2., 2., 2.],
        [2., 2., 2., 2.],
        [2., 2., 2., 2.]])
# after operation: result_out
tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]])
after operation id of result_out: 4754570624

可以看出，操作（计算）前后原始的输出Tensor并没有被改变，给定的Tensor result_out在计算前后地址未发生改变，但计算结果写入了其中。

Tensor运算符的重载（operator overload）

相信大家在使用pytorch的时候，除了通过调用上述add方法实现张量加法之外，还有一种比较常用的方式，就是以运算符的方式进行Tensor之间的加法操作，如a + b。众所周知，类本身并没有+这种操作，pytorch通过重载运算符的方式实现了这种操作方式，从而提升用户的使用体验。pytorch Tensor中与加法相关的运算符有两个，分别是+和+=:

1. +=运算符的重载

为什么先说+=运算符？不得不说+=运算符的重载是最符合常规的一种实现方式，由于pytorch中Tensor类单继承于TensorBase类，按照常规的思路，只需要挨个去找这两个类中谁实现了对+=重载即可。经过简单的查找，还真找到了，具体的代码实现如下：

class TORCH_API Tensor: public TensorBase {
  ...
  Tensor& operator+=(const Tensor & other) {
    return add_(other);
  }
  Tensor& operator+=(const Scalar & other) {
    return add_(other);
  }
  Tensor& operator-=(const Tensor & other) {
    return sub_(other);
  }
  Tensor& operator-=(const Scalar & other) {
    return sub_(other);
  }
  Tensor& operator*=(const Tensor & other) {
    return mul_(other);
  }
  Tensor& operator*=(const Scalar & other) {
    return mul_(other);
  }
  Tensor& operator/=(const Tensor & other) {
    return div_(other);
    ...
}


意料之中的是，pytorch通过调用上文提到的add_方法来实现对+=运算符的重载，毕竟功能完全一样，重新实现一次完全没有必要。另外，可以看到Tensor类中还重载了大量其他的算数操作符，包括但不限于-=, *=, /=, ~, [], |=, ^=等等，都是通过直接调用对应的算子来实现的。但是，有一个比较诡异的问题是，运算符+并没有这这里重载，那到底是在哪里实现的呢？接着往下看。

2. +运算符的重载

经过笔者这些天对源码的上下求索，终于找到了pytorch对+运算符的重载方式。先说结论：pytorch在python的层面重载了Tensor的+运算符。当我们在实现一个python对象的时候，可以通过实现其magic method __add__来对其+运算符进行重载，举个例子：

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        if isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        
a = Point(3, 4)
b = Point(1, 2)
c = a + b
print(c.x, c.y)
​
#output
4 6

按照这个思路，我们可以直接去pytorch 里面的python source code里面Tensor类找看他有没有实现这个方。首先作者去看了_tensor.py这个文件，发现没有。然后又跑看它的基类，但是跳转过去后确实发现了在TensorBase类中定义了__add__这个方法，但这个文件是一个.pyi为后缀的文件（可能是因为TensorBase是一个用c++实现的python module，.phi文件可能仅定义了相应的接口信息），所以看上去 be like:

# Defined in torch/csrc/autograd/python_variable.cpp
class TensorBase(metaclass=_TensorMeta):
    ...
    _has_symbolic_sizes_strides: _bool
    def __abs__(self) -> Tensor: ...
    def __add__(self, other: Any) -> Tensor: ...
    @overload
    def __and__(self, other: Tensor) -> Tensor: ...
    @overload
    ...

似乎找到了，好像又没找到 x_x。但是pytorch并没有断绝我们的线索，在上面一行的注释中，pytorch开发者明确给出了TensorBase c++实现的具体文件，我们接着往下找：

PyTypeObject THPVariableType = {
    PyVarObject_HEAD_INIT(
        &THPVariableMetaType,
        0) "torch._C._TensorBase", /* tp_name */
    sizeof(THPVariable), /* tp_basicsize */
    0, /* tp_itemsize */
    // This is unspecified, because it is illegal to create a THPVariableType
    // directly.  Subclasses will have their tp_dealloc set appropriately
    // by the metaclass
    nullptr, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    &THPVariable_as_mapping, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE |
        Py_TPFLAGS_HAVE_GC, /* tp_flags */
    nullptr, /* tp_doc */
    // Also set by metaclass
    (traverseproc)THPFunction_traverse, /* tp_traverse */
    (inquiry)THPVariable_clear, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    nullptr, /* tp_methods */
    nullptr, /* tp_members */
    THPVariable_properties, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    // Although new is provided here, it is illegal to call this with cls ==
    // THPVariableMeta.  Instead, subclass it first and then construct it
    THPVariable_pynew, /* tp_new */
};


上面的代码就是定义在python_variable.cpp文件中的使用C++声明的一个Python中的类，具体的细节比较复杂，这里先挖个坑，我们后面接着填。我们可以知道的是这个类的名称是torch._C._TensorBase，继承类THPVariableMetaType，类的属性定义在THPVariable_properties中，使用下面的方法来声明这个类：

bool THPVariable_initModule(PyObject* module) {
  THPVariableMetaType.tp_base = &PyType_Type;
  if (PyType_Ready(&THPVariableMetaType) < 0)
    return false;
  Py_INCREF(&THPVariableMetaType);
  PyModule_AddObject(module, "_TensorMeta", (PyObject*)&THPVariableMetaType);
​
  static std::vector<PyMethodDef> methods;
  THPUtils_addPyMethodDefs(methods, torch::autograd::variable_methods);
  THPUtils_addPyMethodDefs(methods, extra_methods);
  THPVariableType.tp_methods = methods.data();
  if (PyType_Ready(&THPVariableType) < 0)
    return false;
  Py_INCREF(&THPVariableType);
  PyModule_AddObject(module, "_TensorBase", (PyObject*)&THPVariableType);
  torch::autograd::initTorchFunctions(module);
  torch::autograd::initTensorImplConversion(module);
  torch::utils::validate_numpy_for_dlpack_deleter_bug();
  return true;
}


上面的内容先注册了一个_TensorMeta的python，也就是我们的基类。随后通过THPUtils_addPyMethodDefs方法向vector methods中添加了一些方法，并将这些方法赋值给了THPVariableType(TensorBase )的tp_methods，作为类方法来使用。而我们的__add__方法的具体实现，就藏在这些方法中。上面的代码中向vectormethods进行了两次添加，分别是torch::autograd::variable_methods和extra_methods，而我们的方法就在这个torch::autograd::variable_methods中

PyMethodDef variable_methods[] = {
  // These magic methods are all implemented on python object to wrap NotImplementedError
    {"__add__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_add>), METH_VARARGS | METH_KEYWORDS, NULL},
    {"__radd__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_add>), METH_VARARGS | METH_KEYWORDS, NULL},
    {"__iadd__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_add_>), METH_VARARGS | METH_KEYWORDS, NULL},
    {"__rmul__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_mul>), METH_VARARGS | METH_KEYWORDS, NULL},
   ...
}


其中，每个item都包含一个函数名字符串和对应的实现，castPyCFunctionWithKeywords和TypeError_to_NotImplemented_用于进行函数指针的格式转换和NotImplemented的错误检查，真正的函数实现是THPVariable_add

// /app/docker/rocm-torch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp
static PyObject * THPVariable_add(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "add(Scalar alpha, Tensor other)|deprecated",
    "add(Tensor other, *, Scalar alpha=1)",
  }, /*traceable=*/true);
​
  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::add(Tensor self, Scalar alpha, Tensor other) -> Tensor
      
      auto dispatch_add = [](const at::Tensor & self, const at::Scalar & alpha, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.add(other, alpha);
      };
      return wrap(dispatch_add(self, _r.scalar(0), _r.tensor(1)));
    }
    case 1: {
      // aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
      
      auto dispatch_add = [](const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.add(other, alpha);
      };
      return wrap(dispatch_add(self, _r.tensor(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}


终于见到了庐山真面目，原来终究调用的还是torch.Tensor.add这个方法，只不过在外面加上了同步机制包装成了一个匿名函数dispatch_add。

阳光明媚，一切是那么的和谐和美好。

http://torch.xxx和http://torch.Tensor.xxx

本文通篇讲述了一个简单的add算子，但绝不仅仅是为了讲述一个简单的add算子，通过归纳我们可以发现，pytorch中的函数有这样两种使用姿势 （不是仅有这两种），分别是torch.xxx和torch.Tensor.xxx，它们的区别在于注册的在不同的对象中。注册在TensorBase中的方法可以通过torch.Tensor.xxx进行调用而注册在_VariableFunctionsClass的函数可以通过torch.xxx进行调用。

_VariableFunctionsClass定义在pytorch/torch/csrc/autograd/python_torch_functions_manual.cpp中

static PyTypeObject THPVariableFunctions = {
    PyVarObject_HEAD_INIT(
        nullptr,
        0) "torch._C._VariableFunctionsClass", /* tp_name */
    0, /* tp_basicsize */
    0, /* tp_itemsize */
    nullptr, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    nullptr, /* tp_methods */
    nullptr, /* tp_members */
    nullptr, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    nullptr /* tp_new */
}

定义与TensorBase走的是同一种风格，而在初始化的过程中，也为其添加了相应的方法torch_functions，这些方法的数量比起Tensor中就要多了很多了。

void initTorchFunctions(PyObject* module) {
  static std::vector<PyMethodDef> torch_functions;
  gatherTorchFunctions(torch_functions);
  THPVariableFunctions.tp_methods = torch_functions.data();
​
  if (PyType_Ready(&THPVariableFunctions) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPVariableFunctions);
​
  // Steals
  Py_INCREF(&THPVariableFunctions);
  if (PyModule_AddObject(
          module,
          "_VariableFunctionsClass",
          reinterpret_cast<PyObject*>(&THPVariableFunctions)) < 0) {
    throw python_error();
  }
  // PyType_GenericNew returns a new reference
  THPVariableFunctionsModule =
      PyType_GenericNew(&THPVariableFunctions, Py_None, Py_None);
  // PyModule_AddObject steals a reference
  if (PyModule_AddObject(
          module, "_VariableFunctions", THPVariableFunctionsModule) < 0) {
    throw python_error();
  }
}


那么，这个_VariableFunctions到底是什么呢？我们结合两个文件来看，一个是pytorch/torch/_C/_VariableFunctions.pyi和pytorch/torch/__init__.py。其中，前者中定义了所有函数的python接口，后者import了前者的所有这些方法的接口。所以~（长舒一口气），当我们在执行import torch的时候，也会同时importpytorch/torch/_C/_VariableFunctions.pyi中的所有这些方法。

完结！撒花！

BTW，add方法中还可以通过指定一个常数作为其中一个乘数的因子参与运算，类似于:

c = ka * b

个人感觉比较鸡肋，顺便提一下，万一有的同学就有这样的使用场景呢？

感觉更新的速度有点慢，没办法社畜每天还要打工。不过我会尽全力把自己知道的内容做到尽可能地详细与充实，欢迎大家评论点赞，可以评论出你想要了解的内容。

接下来打算更新一下TensorIterator类相关的内容，包括如何进行shape broadcast以及data type promotion等等。
