# LLVM Essential读书笔记之一：初式LLVM与LLVM IR

**作者**: 俯仰AI Framework Engineer

**原文链接**: https://zhuanlan.zhihu.com/p/653970722

---

​
目录
收起
背景介绍
内容简介
LLVM库的集合以及模块化设计
LLVM优化器的模块化设计
LLVM代码生成器的模块化设计
LLVM IR
LLVM编译流程
LLVM IR的设计思想
LLVM的形式
LLVM IR的结构组成
指令格式和变量
示例
参考文献：
背景介绍

LLVM项目于2000年创立于伊利诺斯州大学，原本是一个为了静态及动态编程语言而生的现代的，基于静态单赋值（SSA）编译技术的研究项目。现在已经成长为一个包含许多子项目的大型项目，提供了一系列具有良好定义接口的可重用库。

LLVM采用C++语言实现，其最关键的部分是其提供的LLVM核心库。这些库提供开发者opt工具，与目标机器无关的优化器，以及对多种目标架构的代码生成支持。这些库围绕着LLVM中间表示（Intermediate Representation，IR）进行构建，几乎能够所有的高层级编程语言都能够映射成为LLVM IR。因此，要使用LLVM的优化器和代码生成技术，需要首先编写一个能够将高级语言转化生成LLVM IR的编译器前端。

内容简介

本文简单的总结了 LLVM essentials这本书的第一章内容，主要包括：

LLVM库的集合以及模块化设计。
熟悉了解LLVM IR。
大致了解LLVM工具并在命令行中使用他们。
LLVM库的集合以及模块化设计
LLVM优化器的模块化设计

LLVM的优化器opt，优化器中定义了很多pass。在编译器的优化过程中，"pass" 是指一系列特定的优化步骤或阶段，这些步骤会逐一处理编译器生成的中间表示（例如，抽象语法树、中间代码），以改进程序的性能和效率。每个优化 pass 都有一个具体的任务，它可以执行一种或多种代码变换，以减少计算或内存消耗，提高程序的速度或减少代码大小等。

这些 pass按顺序运行，每个 pass 都在前一个 pass 的基础上进一步优化代码。每个pass都是由c++编写的一个类，该类原始继承自LLVM的Pass类。每个pass能够被编译成动态链接库xxx.o，随后被组合归档整理成为一个静态链接库xxx.a，该静态库包含opt工具中的所有pass，并且他们彼此之间是低耦合的，这些pass会显式地声明彼此之间的dependency（包括glue dependency，chain dependency，data flow dependency等）。

之所以称优化器是模块化设计的，是因为这种设计可以让开发者通过显示地声明依赖关系来控制pass执行的顺序，并能够控制哪些pass需要执行，哪些不需要。随后LLVM PassManger通过这些显式声明的依赖关系来以最优的方式运行这些pass，并且仅仅是需要运行的pass会被链接，而非每次都需要链接整个优化器。

下图引自[1]，说明了不同的pass之间的关系。

图1 优化器pass的模块化设计
LLVM代码生成器的模块化设计

代码生成器也是用了像优化器一样的模块化设计，将整个代码生成过程分成一个个的pass，如指令选择，寄存器分配，调度，代码布局优化，发射汇编代码等等。

在代码生成的过程中，所有的目标架构都具有一些相同的步骤，如需要为虚拟寄存器分配可用的物理寄存器，但是对于不同的目标架构（目标机器），寄存器的集合又不尽相同。因此，编译器开发者可以按需调整每个pass，甚至可以根据目标机器的不同架构，创建自己的自定义pass。通过使用表描述文件.td，Tablegen工具能够帮助实现为不同的目标机构实现代码生成，主要是通过在.td文件中记录目标架构的特性，如寄存器集合等。




LLVM IR
LLVM编译流程

LLVM IR是LLVM项目的核心，对于每个编译器而言，为了方便对不同的高级语言进行优化以及代码生成（codegen），其会根据目标语言产生统一的中间表示（Intermediate Representation，IR），这种中间表示既与源语言无关，也与目标架构无关。随后只需要在统一的IR上实现一套相同的pass即可，避免了为不同的高层级语言设计多套不同的pass和codegen。整个编译的流程可以近似看作图2。




图2 编译器的编译流程示意图

其中，不同的高级语言会通过前端的处理（词法分析、语法分析、语义分析等）生成特定的数据结构（如抽象语法树），随后这些数据结构会被发射（emit）成相应的LLVM IR进行独立于源语言和目标架构的统一表示。接下来，LLVM opt会对这些IR执行指定的pass，常见的如Dead Code Elimination，Loop Invariant Code Motion等。每个pass的输入都是LLVM IR，输出同样也是LLVM IR。最后，经过优化的LLVM IR会送入编译器后端，被转换为Selection Directed Acyclic Graph（Selection DAG），并最终发射为目标架构的汇编指令。

LLVM IR的设计思想

对于一个多前端、多目标架构的编译器而言，选择IR既不能选择过于高层级的IR，因为太过与靠近源语言，很难做一些目标架构相关的优化，也不能选择过于低层级的IR，因为太靠近目标架构的机器指令，不够通用且看不到高层级的流程思想，很多优化无法实施。

LLVM IR在能够清楚的看清高层级语言想法的基础上，处于一个尽可能低层级的位置，通过这样的定位来使自己成为一种通用的R。

理想情况下LLVM R应该是目标架构的无关的，但是并非如此，因为有些语言本身就对目标架构具有依赖性。比如在使用Linux系统的标准C头文件时，这个头文件本身就是与目标架构相关的，可能需要制定一个特定的架构类型来匹配不同的系统调用。

LLVM的形式

LLVM具有一下三种等价的形式：

内存形式
保存在硬盘上的bitcode
适合人类阅读的形式（LLVM汇编）
LLVM IR的结构组成

LLVM Module是LLVM IR中最高等级的数据结构，其中包含所有的输入内容，LLVM Module由函数，全局变量，外部函数原型以及符号表对组成。

除此之外，LLVM IR文件（.ll）中还会显示目标架构中数据的布局和“目标三元组”。

数据布局（Data Layout）即目标机器中的数据存储方式时大端存储（Big Endian）还是小端存储（Little Endian）

目标三元组（target triple）是一个描述目标体系结构和操作系统的字符串，它用于告诉编译器要生成的目标代码的目标环境信息。这个字符串通常采用以下形式：

 <arch>-<vendor>-<os>-<abi>
 其中各部分的含义如下：
 ​
 # <arch>：表示目标的CPU架构，如x86、ARM、MIPS等。
 # <vendor>：通常指CPU供应商，如Intel、AMD等。有时会留空。
 # <os>：表示目标操作系统，如Linux、Windows、macOS等。
 # <abi>：表示应用程序二进制接口，指定了如何与操作系统进行交互，如GNU（通常用于Linux）、MSVC（通常用# 于Windows）等。
指令格式和变量

LLVM将全局变量当作指针来看待，因此在使用load指令时需要显示的解指针操作，存储数据到全局变量时也是一样。

LLVM的局部变量有两类：分配给寄存器的局部变量和分配在栈帧上的局部变量。

分配给寄存器的局部变量往往是临时变量，它们往往被分配给虚拟寄存器，这些分配的虚拟寄存器会在代码生成阶段被替换为物理寄存器。

 %1 = value  # 分配给寄存器的局部变量

还有一类局部变量，它们被分配在当前执行函数的栈帧上，通过alloca指令分配给它们内存地址，这类局部变量需要使用load和store指令来存取变量值。

 %1 = alloca i32 # 分配在栈帧上的局部变量

LLVM采用三地址的指令格式，这种格式有两个源操作数，并使用一个另外的目标操作数。为了方便一些优化操作，LLVM IR 的指令采用静态单赋值（SSA）的形式，如下：

 %4 = add i32 %2, %3
示例

一个简单的LLVM IR demo，将下面的c语言代码转换成LLVM IR。

 int globvar = 12;
 ​
 int add(int a) {
 return globvar + a;
 }

使用如下命令：

 clang -emit-llvm -c -S add.c
 cat add.ll
 ;ModuleID = 'add.c'
 source_filename = "add.c"
 target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
 target triple = "x86_64-pc-linux-gnu"
 ​
 @globvar = global i32 12, align 4
 ​
 ; Function Attrs: noinline nounwind optnone uwtable
 define i32 @add(i32) #0 {
   %2 = alloca i32, align 4
   store i32 %0, i32* %2, align 4
   %3 = load i32, i32* @globvar, align 4
   %4 = load i32, i32* %2, align 4
   %5 = add nsw i32 %3, %4
   ret i32 %5
 }
 ​
 attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
 ​
 !llvm.module.flags = !{!0}
 !llvm.ident = !{!1}
 ​
 !0 = !{i32 1, !"wchar_size", i32 4}
 !1 = !{!"clang version 6.0.0-1ubuntu2 (tags/RELEASE_600/final)"}
 ​

可以看到整个.c文件被发射成了一个LLVM Module，生成的add.ll文件中第一行中定义了LLVM的MoudleID。

从target datalayout = "e-m:e"目标机器的数据可以看出目标机器是小端机器，即数据的高位存放在高地址，低位存放在低地址。

从target triple = "x86_64-pc-linux-gnu"可以看出目标机器cpu架构是x86，操作系统是linux，程序二进制接口是gnu。

接下来定义了一个全局变量globvar，在LLVM IR中，全局变量的命名以@开始，局部变量的命名以%开始。这样规定的原因有二：1. 编译器不必担心变量名与保留的关键字名冲突。2.编译器能够很快提出许多互不冲突的变量名。第二点原因对于编译器将编译出的汇编代码转换成SSA形式很有用，便于后续优化器对代码进行优化。

接下来的部分是关于add函数的定义。

add函数后面的部分是关函数的属性和表示指令：

函数属性（Function Attributes）：生成的LLVM IR代码中包含一个字符串，用于指定函数属性。这些函数属性与C++中的属性（attributes）非常相似。每个在LLVM IR中定义的函数都可以附加一组属性。这些属性可以提供有关函数的元信息，如调用约定、优化提示、内联性等。这些属性对于编译器优化和代码生成非常重要，因为它们可以影响生成的机器代码的行为和性能。

标识指令（Ident Directive）：接下来的部分描述了一个标识指令，该指令用于标识模块（module）和编译器版本信息。通常，这个指令包含关于生成LLVM IR的编译器的版本和其他与编译相关的信息。这个指令的主要目的是提供一些元数据，以便在需要时了解LLVM IR的生成环境和上下文。

参考文献：
Sarda S, Pandey M. LLVM essentials[M]. Packt Publishing Ltd, 2015.
