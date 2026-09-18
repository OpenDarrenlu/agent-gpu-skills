# 万字长文,看这一篇就够了！WebAssembly原理剖析与生产应用

**作者**: 齐夏LLM / AI infra / GPU

**原文链接**: https://zhuanlan.zhihu.com/p/620767652

---

​
目录
收起
第一章 WebAssembly的前世今生：从Mozilla说起
1.1 一家伟大的互联网企业
1.2 脑洞大开的想法：浏览器里跑C++
1.3 另一次失败的尝试：Google Native Client
第二章 asm.js：WebAssembly的前身，一种更快的JS
2.1 C++转换asm.js示例
2.2 asm.js为什么比原生JavaScript 快？
第三章 WebAssembly：绕过JS 直接生成机器码
3.1 WebAssembly是什么？
3.2 为什么需要WebAssembly？
3.3 WebAssembly与JavaScript运行性能详细对比
3.4 如何正确使用WebAssembly？
3.5 使用示例
第四章 WebAssembly在Web端的应用
4.1 常见web端应用概览
第五章 WebAssembly在服务端的应用
5.1 WASI：解决跨平台运行操作系统的差异
5.2 WASI现在的进展？
5.3 WebAssembly在服务端的应用
第六章 总结
第一个问题：你的Web应用性能瓶颈在哪里？先想清楚这个问题再做优化
第二个问题：现在已经是2023年了，WebAssembly到底算成功了吗？
第三个问题：我们如何拥抱WebAssembly这个新技术？

专栏最新文章推荐：

齐夏：[原创长文]2024.10-开源大模型推理引擎现状及常见推理优化方法
512 赞同 · 36 评论 文章

本文全长：20394字 预计阅读时间：5-10min

最新更新时间：2024.3.15 求点赞啊各位大佬~收藏居然是点赞的3倍~

前言：因为工作需要对WebAssembly进行了一些研究，遗憾的是我发现整个中文社区居然没一篇博客能完整地讲清楚这项技术。工作空余时间也不多，大约花了我半年多时间才写完这2万字，你看到某一行字的背后我可能需要花几个小时去看很多的英文原文资料来论证。如果这篇文章对你有所启发，欢迎关注、点赞、收藏以及评论区交流~




WebAssembly，前身技术来自Mozilla和Google Native Client的asm.js，首次发布于2017年3月。并于2019年12月5日正式成为W3C recommendation，至此成为与HTML、CSS以及JavaScript 并列的web领域第四类编程语言。

在web领域，已经有JavaScript这样的利器，而WebAssembly则是打开新世界的大门。WebAssembly并不是要取代JavaScript，而是要在图形图像处理、3D游戏、AR/VR这些应用领域开疆拓土。如今的现代浏览器已经越发朝着微型”第二操作系统“发展，人们希望在浏览器内能完成更多的事情，而WebAssembly作为web端高性能应用的基石，正在让更多的应用场景在浏览器内变为现实。

除了在浏览器内实现高性能应用，WebAssembly也可以脱离web端在搭载了不同硬件和操作系统的各个平台运行，进一步实现当年JAVA所期望的“一次编译，多处运行”。WebAssembly在服务端可用于微服务平台、无服务平台、第三方插件系统等场景。

本文共6个章节，全文逐字阅读约15分钟，具体章节介绍如下：

第一章 WebAssembly的前世今生 : 介绍了asm.js在Mozilla的起源
第二章 asm.js技术 : 底层原理介绍，探讨了asm.js比原生JavaScript更快的原因
第三章 WebAssembly技术 : 底层原理介绍，探讨了WebAssembly .js比asm.js更快的原因
第四章 WebAssembly在Web端的应用 : 公司内外案例，在音视频处理、游戏、web端设计工具等领域中的生产应用
第五章 WebAssembly在服务端的应用 : 如果实现跨平台runtime，在微服务、无服务、容器化等场景的应用。
第六章 总结展望：web应用加速的瓶颈在哪？WebAssembly到底有没有成功？
DOOM3游戏(浏览器内真实效果)

WebAssembly技术体验可直接点击以下链接~

看看你的浏览器能跑多少FPS？浏览器直接玩2D/3D游戏：https://arcadespot.com/game/doom-3/

第一章 WebAssembly的前世今生：从Mozilla说起
Mozilla基金组织LOGO
1.1 一家伟大的互联网企业

说起WebAssembly，那就必须从一家没落而又伟大的互联网公司说起，它就是火狐浏览器的开发者Mozilla。Mozilla的前身是大名鼎鼎的网景公司(Netscape)，也就是JavaScript的开发者。从做浏览器起家一路坎坷至今，Mozilla最近更是频频传出裁员风波，其根源依然是没有找到太好的盈利点。作为互联网开源社区的领跑者，Mozilla在技术上的成就远高于其在商业领域。除了JavaScript和Filefox，Mozilla还留下了Rust、HTML5、MDN（Mozilla Developer Network）以及asm.js这些引领互联网行业发展的重要基石。

1.2 脑洞大开的想法：浏览器里跑C++

2012年Mozilla的工程师在研究LLVM时，突然脑洞大开提出了一个想法：类似游戏引擎这样的高性能应用大多都是C/C++语言写的，如果能将C/C++转换成 JavaScript ，那岂不是就能在浏览器里跑起来了吗？如果可以实现，那么浏览器是不是也就可以直接跑3D游戏之类的C/C++应用？于是Mozilla成立了一个叫做Emscripten的编译器研发项目，Emscripten可以将C/C++代码编译成JavaScript，但不是普通的JS，而是一种被特殊改造的JS，其被命名为asm.js。

Emscripten 的官方描述是：

Emscripten is a toolchain for compiling to asm.js and WebAssembly, built using LLVM, that lets you run C and C++ on the web at near-native speed without plugins.

中文译文：

Emscripten是一个基于LLVM的将C/C++编译到asm.js和WebAssembly的工具链，它可以让你在web上以接近原生的速度运行C/C++而不需要任何插件。

如下图所示：实际上，不只是C/C++代码，只要能转换成LLVM IR的语言，都可以通过Emscripten转换成asm.js。

C++代码转换JS流程
1.3 另一次失败的尝试：Google Native Client




Google在很早之前也一直致力于研究如何让C/C++能够在Chrome里运行起来，并在2009年的安全领域顶级会议IEEE Symposium on Security and Privacy 发表了Google的技术方案NaCl（Google Native Client）以及PNaCl（Portable Google Native Client）。NaCl的本质也是一种沙盒技术，使用工具链编译后的C/C++代码能够以接近原生应用的速度在web端运行，也可以与JS和webapi进行交互。NaCl在安全这块做了大量的设计，其使用了内外双层沙盒，并利用x86内存分段机制来隔离内存，甚至还用上了静态代码分析技术来做沙盒里运行的程序进行检查。

然而在经过了8年的挣扎后，在2017 年5月30日Google宣布弃用NaCl。其根本原因是NaCl这套方案只有自家的Chrome愿意配合支持，所以压根就不具备跨浏览器运行的能力。最终Chrome与Mozilla达成一致，共同推进WebAssembly方案，Chrome也直接用WebAssembly替换掉了NaCl。

第二章 asm.js：WebAssembly的前身，一种更快的JS
2.1 C++转换asm.js示例

一般来说，asm.js并不是直接编写的，而是一个面向JS编译器的中间产物。例如以下的C++代码：

 //计算i+1
int f(int i) {
 return i + 1;
}
//计算字符串长度
size_t strlen(char *ptr) {
  char *curr = ptr;
  while (*curr != 0) {
    curr++;
  }
  return (curr - ptr);
}


使用Emscripten转换后，生成的JS代码如下：

function f(i) {
  i = i|0;
  return (i + 1)|0;
}

function strlen(ptr) {
  ptr = ptr|0;
  var curr = 0;
  curr = ptr;
  while ((MEM8[curr>>0]|0) != 0) {
    curr = (curr + 1)|0;
  }
  return (curr - ptr)|0;
}
 


可以看到这种生成的JS跟普通JS还是区别很大的，就像刚才我们所说，程序员不直接编写asm.js代码，这些看起来怪异的语法都是为了配合编译器生成更高效的机器码。比如在asm.js里反复出现的"按位或"操作，其目的是将原来JavaScript 里的double类型计算转为整形运算(CPU进行整形运算的速度快于浮点型)。而这里被命名为MEM8的数组实际上充当了"堆"的作用。如果只是作为使用者可以不用深究这些优化的具体实现，直接使用Emscripten 来帮助我们完成这一转换过程即可。

2.2 asm.js为什么比原生JavaScript 快？

由于 asm.js 在浏览器中运行，其性能在很大程度上也取决于浏览器和JS引擎的优化支持。2015年6月，Microsoft Edge也开始加入了对asm. js的支持。为了直观展示asm.js所带来的的性能提升，微软发布了一个叫做"Chess Battle"的demo。Chess Battle让两个版本的开源象棋AI对战，其中一个用C实现然后转成asm.js，另外一个用原生JS实现。如下图所示，每个走棋回合限制为200毫秒，其中asm.js版本的AI因为可以在每个回合进行更多的评估运算(用于决定走棋策略)，胜率获得了极大提升。

asm.js对战原生JavaScript

asm.js运行的快慢取决于不同的测试用例、运行硬件、浏览器引擎优化程度等，一般来说我们可认为asm.js能达到原生C/C++运行速度的50%，有些场景下甚至能持平Clang编译的C/C++用例。asm.js运行比原生js快，那么它如此高效的原因是什么呢？阮一峰在他的一篇博客里写到的结论是：

一旦 JavaScript 引擎发现运行的是 asm.js，就知道这是经过优化的代码，可以跳过语法分析这一步，直接转成汇编语言。另外，浏览器还会调用 WebGL 通过 GPU 执行 asm.js，即 asm.js 的执行引擎与普通的 JavaScript 脚本不同。这些都是 asm.js 运行较快的原因。

这篇博客应该是对很多人造成了误导，具体错误在于：

首先，"跳过语法分析,直接生成汇编"是不存在的，语法分析是编译中不可缺少的一环节，asm.js跟原生JS的编译运行过程是一致的。
其次，WebGL作为一个图形api和asm.js技术可以说是没有任何直接关系，原生JS也调用WebGL来实现GPU硬件加速。
最后，也是最离谱的一点，WebGL 通过 GPU 执行 asm.js ？不管是asm.js、原生JavaScript还是WebAssembly其编译产物都是CPU机器码而不是GPU机器码。而且WebGL只是一个图形渲染api，就算是把JS编译到GPU也需要类似CUDA/OpenCL这些通用计算api来支持。最新的WebGPU同时支持了图形和通用计算，这倒是目前web端在GPU里"执行JS"的可行方法。

先抛开JavaScript 不谈，我们可以思考一下，对于任何一门编程语言来说决定其运行快慢的根源是什么呢？我认为用一句话来总结就是：代码运行的快慢，从硬件层面上看，直接取决于生成的机器码所需时钟周期的总和。从编程语言层面上看，取决于编译后的产物在运行时有多少"动态决议"。

例如，弱类型语言比强类型语言慢，是因为编译时类型是不确定的，需要运行时进行额外的型别推导，这就是"动态决议"；

例如，C++里虚函数比普通函数开销大，是因为编译时函数地址是不确定的。普通函数编译后生成的跳转目的地是一串固定的地址，而虚函数的跳转地址是在运行时从CPU的寄存器里读取的，这也是"动态决议"，编译后的机器码多了一条寄存器取值指令；

类似的场景还有GC机制、模板编程、JIT优化等等，归根结底就是如果在编译时候能完成更多事情，那么生成的机器码运行周期就越短，代码也就运行地越快。asm.js在减少运行时的"动态决议”这里所做的工作，wiki原文如下：

Much of this performance gain over normal JavaScript is due to 100% type consistency and virtually no garbage collection.

可翻译为：

与原生JavaScript相比，这里性能提升的主要原因是100%的类型一致性以及几乎没有(自动的)垃圾回收机制。

简而言之就是，asm.js的实现去掉大部分的自动GC机制，然后改成了强类型语言，编译器能够更大程度地进行优化，这才是asm.js能比普通JS运行更快的原因。在asm.js里不再支持除了浮点和整形之外的类型，内存的开辟和释放也需要代码手动进行处理。部分引擎甚至还可以以AOT或者JIT的形式运行asm.js。关于asm.js的原理，在微软的文档里也有一段更加详细的描述：

Asm.js is a strict subset of JavaScript that can be used as a low-level, efficient target language for compilers. As a sublanguage, asm.js effectively describes a sandboxed virtual machine for memory-unsafe languages like C or C++. A combination of static and dynamic validation allows JavaScript engines to employ techniques like type specialized compilation without bailouts and ahead-of-time (AOT) compilation for valid asm.js code. Such compilation techniques help JavaScript execute at “predictable” and “near-native” performance, both of which are non-trivial in the world of compiler optimizations for dynamic languages like JavaScript.

这段话从编译器优化的角度对asm.js原理描述地非常贴切了，比较难准确翻译，大概释义如下：

asm.js是JavaScript的一个严格子集，是一种面向编译器的底层且高效的目标语言。作为一种子语言，asm.js高效地为类似C/C++这样的内存不安全语言描述了一个沙盒虚拟机。静态验证和动态验证的结合允许JavaScript引擎对有效的asm.js代码使用型别特化编译和提前(AOT)编译等技术。这样的编译技术可以帮助JavaScript具有"可预见性"和“接近原生”的性能表现，这两种特性在JavaScript这样的动态语言编译器优化中是非常重要的。

其中"bailouts"应该是微软这个JS编译器里的专用名词，没有特别合适的翻译。"predictable"可理解为“更少的动态决议”。asm.js目前看已经是过时的技术，并非本文的重点也不再展开继续讨论，如果想继续了解JavaScript编译优化的实现细节，读者可参阅文献的内容自行研读。

第三章 WebAssembly：绕过JS 直接生成机器码

Asm.js的思路是将一种编程语言转换成另外一种编程语言，输出的还是JS代码。那么这里你肯定也想到了，我们为什么不能绕过JavaScript ，将C/C++代码直接转成浏览器可以识别的更底层的语言呢？这就是由Asm.js衍生出的WebAssembly技术。

3.1 WebAssembly是什么？

如上图所示，为了能便于程序员阅读和编辑 WebAssembly，源码除了被编译成二进制外还会生成一份文本文件。左边红色部分是C++源码，中间紫色部分是文本格式的.Wat文件的内容，右边蓝色部分是.wasm文件的内容。

多数情况下，人们把Wasm定义成web上的编程语言，认为这是一个前端编程技术。其实这里有一些的误解，首先Wasm并不是一个新的"编程语言"，没有人会手写.wasm文件来进行编程。WebAssembly 有一套完整的语义，但作为开发者并不需要去了解它，开发者依然可以继续使用自己熟悉的编程语言，由各个语言的编译器将其编译成Wasm格式后运行在浏览器内置的Wasm虚拟机中，我认为Wasm更倾向于是一个应用在web场景中的编译领域新技术。其次，Wasm也并非只能运行在浏览器内，设计者对其抱有更加远大的宏图大业，这部分我们将在后面Wasm容器化这里继续展开讨论。

Mozzila官方对WebAssembly的描述为：

WebAssembly is a new type of code that can be run in modern web browsers — it is a low-level assembly-like language with a compact binary format that runs with near-native performance and provides languages such as C/C++, C# and Rust with a compilation target so that they can run on the web. It is also designed to run alongside JavaScript, allowing both to work together.

可翻译为：

WebAssembly是一种可以在现代浏览器中运行的新型代码——它是一种低级的类似汇编的语言，具有紧凑的二进制格式，运行起来具有接近原生的性能，其为C/C++、C#和Rust等语言提供了一个编译目标，以便它们可以在web上运行。它还被设计为与JavaScript一起运行，允许两者一起工作。

通过这段描述已经可以对WebAssembly有一个初步认识，我们再进一步给它拆开来看：

首先，WebAssembly是一门新的编程语言，它于2019年12月5日正式成为与HTML、CSS以及JavaScript 并列的web领域第四类编程语言。
其次，WebAssembly是"汇编语言"而不是高级语言，程序员不直接编写WebAssembly代码，而是通过特殊的编译器将高级语言转换成WebAssembly代码。
再次，WebAssembly是预处理过后的二进制格式，它实际是一个IR(Intermediate Representation)！类似Java的ByteCode或者.Net的MSIL/CIL。
最后，WebAssembly是web上的语言，这意味着主流的浏览器可以读取并且执行它。

最后简单总结，程序员依然还是编写高级语言，然后通过“特殊的编译器”生成WebAssembly二进制代码，最终WebAssembly代码再被一个嵌入在浏览器里的"特殊的虚拟机"执行。这就是WebAssembly的全部工作过程。

3.2 为什么需要WebAssembly？

在web领域，我们已经有了JavaScript这样利器，但美中不足的是JavaScript的性能不佳，即使可以通过第二章里提到的各种编译优化来解决一部分问题，但在类似图形图像处理、3D游戏、AR、VR这些高性能应用的场景下，我们似乎任然需要一个更好的选择。




“快”是相对的，目前我们可以认为在运行速度上：原生C/C++代码 > WebAssembly > asm.js > 原生JavaScript。其中WebAssembly比asm.js要快的原因在于：

WebAssembly 体积更小，JavaScript 通过gzip压缩后已经可以节约很大一部分空间，但WebAssembly 的二进制格式在被精心设计之后可以比gzip压缩后的JavaScript 代码小10-20%左右。
WebAssembly 解析更快，WebAssembly 解析速度比 JavaScript 快了一个数量级，这也是得益于其二进制的格式。除此之外，WebAssembly还可以在多核CPU上进行并行解析。
WebAssembly 可以更好利用CPU特性，之前我们说到asm.js可以通过各种“奇技淫巧”来编译优化，但其还是受限于JavaScript的实现。而WebAssembly可以完全自由发挥，使得其可以利用更多CPU特性，其中例如：64位整数、加载/存储偏移量以及各种CPU指令。在这一部分，WebAssembly能比asm.js平均提速5%左右。
编译工具链的优化，WebAssembly的运行效率同时取决于两部分，第一个是生成代码的编译器，第二个是运行它的虚拟机。WebAssembly对其编译器进行了更多的优化，使用Binaryen编译器代替了Emscripten，这部分所带来的的速度提升大约在5%-7%。

当然，速度上的提升并不是全部。WebAssembly的意义在于开辟了一个新的标准，不再拘泥于JavaScript而是直接面向跟底层的机器码。用任何语言都可以开发WebAssembly，而WebAssembly又可以高效运行在任何环境下，这也是Mozilla的程序员对WebAssembly抱有的最远大的宏图大业。文章将在第六章对WebAssembly在非web端的应用继续展开讨论。

3.3 WebAssembly与JavaScript运行性能详细对比

关于WebAssembly的性能，整体上我认为可以描述为“很快，但是不够快”。也就是说，我们期望它比JavaScript快非常多，快个10倍或者8倍，但实际上只能快一点点，大概也就是不到2倍左右，而且在不同的测试场景下差异可能会很大。也许你会说100%的性能提升已经很高了，但实际上这也许不能说服大量开发人员完全转向一个崭新的有学习成本的技术。

Zaplib(一个高性能web框架)的工程师从最大性能和标准性能两方面对WebAssembly与JavaScript性能进行更详细的对比，结论如下：

3.3.1 最大性能(尽可能"奇技淫巧"地使用JS)

在最大性能上，特殊编写的原生JS是可以跟Wasm大致持平的。其原因在于JS可以通过ArrayBuffer来模拟成一个"memory managed language"：

可以尽可能避免掉自动GC的额外开销。
可以对数据的局部性(cache locality)进行优化来提升缓存命中，从而提升数据读写的效率。(缓存局部性对数组的性能很重要！)
当你尽可能避免掉其它开销，只使用循环、局部变量、算术、函数调用的时候，原生JS会非常快。

举个例子如下，这是一个计算多个2维向量平均长度的TS函数

// Unoptimized Typescript
type Vec2 = { x: number, y: number };

function avgLen(vecs: Vec2[]): number {
    let total = 0;
    for (const vec of vecs) {
        total += Math.sqrt(vec.x*vec.x + vec.y*vec.y);
    }
    return total / vecs.length;a
}
 


这是使用了ArrayBuffer替换数组了实现：

// Optimized Typescript, using ArrayBuffers
function avgLen(vecs: ArrayBuffer): number {
    let total = 0;
    const float64 = new Float64Array(vecs);
    for (let i=0; i<float64.length; i += 2) {
        const x = float64[i];
        const y = float64[i+1];
        total += Math.sqrt(x*x + y*y);
    }
    return total / (float64.length / 2);
}

在示例中，ArrayBuffer每16字节存储一个二维向量，前8字节是向量x，后8字节是向量y。后者代码的性能会远高于前者，具体细节有兴趣可以参考( https://zaplib.com/docs/blog_ts ++.html)。总而言之就是，可以通过JS的ArrayBuffer来手动管理JS内存，尽量避免掉性能开销大的地方，剩下的普通指令的执行跟Wasm并无本质差异。除此之外，浏览器里的JS相比Wasm在某些方面甚至还具有优势：

JS可以访问一些零拷贝(zero-copy)的方法。例如TextEncoder和FileReader.readAsArrayBuffer，而Wasm还需要额外再进行一次内存拷贝。

而Wasm相比JS的优势在于：

SIMD加速。SIMD.js的API已经被弃用，取而代之的是Wasm的SIMD实现。
前置的编译优化。

3.3.2 标准性能(正常使用编程语言)

对于实际情况而言，用标准的JS的进行性能对比才是有意义的，原因在于：

代码的编写复杂度和可维护性也是很重要的，"奇技淫巧"并不适合生产工作中使用。
代码工程会依赖大量第三方库，这些库大概率都是标准JS来编写的。




如上图，这个3D人物动画是一个经典的CPU计算密集的测试用例，且可以直观感受到性能在帧数上的表现( http://aws-website-webassemblyskeletalanimation-ffaza.s3-website-us-east-1.amazonaws.com/)。感兴趣的同学可以在自己浏览器里尝试一下，当3D人物数量为100时JS版本会有明显卡顿，切换到Wasm则不会有卡顿感。

这是在17年Wasm诞生之初的测试，可以看到在不同的环境下Wasm比标准JS快了8-15倍。随着JS的不断优化，现在再去测试可能就不会有这么大的差异了。更重要的是，这个测试用例不一定能代表真实的web应用，真正的web应用可能不会命中这么多"优化项"，8倍以上的性能差异往往只存在于测试用例中。这里我必须再重复一下就是，Wasm快10%到1000%都有可能，不同的测试环境下不可一概而论。

3.4 如何正确使用WebAssembly？

首先需要再次强调的是，WebAssembly的诞生并不是要取代JavaScript，web端整个主框架还是HTML+JS+CSS这一套。web应用的大部分基础功能也依然是靠JavaScript来实现，我们只是将web应用中对性能有较高要求的模块替换为wasm实现。在这样的场景下，正确使用WebAssembly的步骤为：

整理web应用中所有模块，梳理出有性能瓶颈的地方。例如你的web应用里有视频上传、文件对比、视频编解码、游戏等模块，这些都是很适合用WebAssembly来实现的。相反，基础的网页交互功能并不适合用WebAssembly来实现。
进行简单的demo性能测试，看是否能达到预期的加速效果。如果加速效果并不明显，那么就不适合切换到Wasm。
确定用来编译成WebAssembly的源语言，目前主流的语言基本都是支持WebAssembly的，唯一不同的区别是其编译器的优化程度。如果你使用过C++、RUST，最好还是用这两种语言来编写，其编译优化程度会更高。当然了如果你想使用PHP/GO/JS/Python这些你更加熟悉的语言的话，也是不错的选择，毕竟有时候开发效率会比运行效率要更加重要。
编码实现，然后导出.wasm文件。这一步基本没什么难度，确定了语言之后使用对应的编译器即可，需要注意的是记得尽量多打开debug选项，不然有运行时报错的话你就只能对着一堆二进制代码懵逼了。
编写JavaScript胶水代码，加载.wasm模块。在最小可行版本的实现中，在 Web 上访问 WebAssembly 的唯一方法是通过显式的JavaScript API调用，而在ES6标准中，WebAssembly 也可以直接从<script type='module'>的HTML标签加载和运行。
3.5 使用示例

3.5.1 快速运行试验

看了刚才运行WebAssembly的步骤，是否觉得还是有些繁琐呢？没关系，这里教你一个快速体验运行WebAssembly的方法：

打开任意的浏览器，例如Chrome。
按F12，启动开发者工具。
找到Console页签，复制这一段代码，回车运行。
 WebAssembly.compile(new Uint8Array(`
  00 61 73 6d  01 00 00 00  01 0c 02 60  02 7f 7f 01
  7f 60 01 7f  01 7f 03 03  02 00 01 07  10 02 03 61
  64 64 00 00  06 73 71 75  61 72 65 00  01 0a 13 02
  08 00 20 00  20 01 6a 0f  0b 08 00 20  00 20 00 6c
  0f 0b`.trim().split(/[\s\r\n]+/g).map(str => parseInt(str, 16))
)).then(module => {
  const instance = new WebAssembly.Instance(module)
  const { add, square } = instance.exports

  console.log('2 + 4 =', add(2, 4))
  console.log('3^2 =', square(3))
  console.log('(2 + 5)^2 =', square(add(2 + 5)))

})
 


这里我们是通过直接手写二进制机器码的方式生成了一段wasm代码，并使用了WebAssembly.compile接口来进行编译，最后调用了wasm实现的add和square函数。如果顺利的话，你的浏览器会编译这段WebAssembly代码并调用执行，输出对应的计算结果，具体如下图所示：

当然，如果如果没有按预期输出的话，那就说明你当前的浏览器版本是不支持WebAssembly的。

第四章 WebAssembly在Web端的应用

一家名为"Scott Logic"的软件开发商在2022年6月发布了2022年WebAssembly现状报告(这个统计允许开发者选择多个选项，所以总和是大于100%的)，在关于WebAssembly应用的统计中有几个信息值得关注：




首先，WebAssembly最多的应用场景依然是在Web站点开发上，大约占65%。

其次，WebAssembly 在Serverless和容器化方面的应用大幅增加，由去年的20%提升到了35%。

最后，增长幅度最大的是在"作为插件环境"应用场景，WebAssembly的沙盒化安全环境很适合用于托管不受信任的第三方代码。

本章会介绍一些公司内外的web端应用场景，关于服务端的应用会在第五章继续介绍。

4.1 常见web端应用概览

4.1.1.【Google-可视化】谷歌地球3D地图 https://www.google.com/intl/zh-CN/earth/

在最早的版本Google Earth是只能跑在Chorom浏览器的，因为其底层用的是跟WebAssembly类似的Native Client技术。目前的Google Earth已经可以运行在Firefox、Edge、Opera浏览器，其关键的一点就是用WebAssembly代替了原来Native Client。

4.1.2 【Bilibili-编解码】哔哩哔哩视频网站 https://member.bilibili.com/platform/upload/video/frame

B站视频上传的功能里有大量的Wasm模块，类似视频上传、封面图处理这些都是计算比较密集的场景。如上图所示，B站用到了Wasm版FFmpeg来加速视频编解码，这应该是WebAssembly最常见的应用了。除此之外还用到了Wasm版Tensorflow，这里应该是用来实现 "AI智能生成封面" 的功能。

4.1.3 【Figma-设计工具】Figma在线UI设计 https://www.figma.com/

Figma是近年来少有的可以称得上拥有“硅谷速度”的创新型公司。2018年初，Figma的估值才刚刚过1亿美元，还仅仅是一个小众设计工具，到了2021年，Figma估值暴涨100倍来到了100亿美元，其在设计圈的地位已经足以跟此前几乎处于垄断地位的Adobe产品抗衡，成为了产品圈、设计圈内人人必用的工具。

Figma可以说是典型的WebAssembly应用了，使用了zaplib(一款基于wasm和Rust的高性能web应用框架)来进行开发。外围的交互操作还是用原生的JS+CSS+HTML来实现的，中间核心绘图区域是一个由wasm+webGL来驱动的的canvas模块。

4.1.4 【Adobe-设计工具】Photoshop Web版 https://www.adobe.com/express/feature/image/editor

就在几年前，直接在浏览器中运行像 Photoshop 这样复杂的软件的想法还很难想象。然而，通过使用各种新的网络技术，Adobe 现在已经将 Photoshop 的公开测试版带到了网络上。

Adobe工程师这里所说的新技术，其中很重要一部分就是WebAssembly。除了解决性能问题，更重要的是Photeshop的web端和PC端应用可以由同一份源码编译生成。Adobe使用Emscripten将Photeshop的完整C++工程直接移植到了web端，而无需用JS重写。Emscripten 是一个功能齐全的工具链，它不仅可以帮你将 C++ 编译为 Wasm，还提供了一个转换层，可以将 POSIX API 调用转换为 Web API 调用，将 OpenGL 转换为 WebGL。

4.1.5 【Zoom-在线会议】Zoom Web版 https://support.zoom.us/hc/en-us/articles/214629443-Getting-started-with-the-Zoom-web-client

将Zoom移植到Web端，其复杂程度绝对不低于前面所说的几个应用。除了视频流的处理，Zoom还提供了自动字幕、虚拟背景等功能，这些都是典型的CPU计算密集应用。ZoomWeb的核心是WebRTC，在WebAssembly诞生后，Zoom的工程师将WebAssembly SIMD的能力引入了ZoomWeb。WebAssembly SIMD提供了可移植、高性能的SIMD命令集，可用于目前绝大多数主流CPU架构。音视频编解码、图像处理这些都是SIMD的典型应用场景，ZoomWeb中虚拟背景的底层计算就是利用WebAssembly SIMD来实现的。

4.1.6 【Google-机器学习】TensorFlow.js https://www.tensorflow.org/js?hl=zh-cn

TensorFlow.js 是一个 JavaScript 库，用于在浏览器和 Node.js 训练和部署机器学习模型。在2020年，TensorFlow.js 引入了一个新的 WebAssembly 加速后端。从 TensorFlow.js 2.3.0 版开始，Wasm后端通过XNNPACK 利用SIMD指令和多线程，速度提高了10 倍，其中XNNPACK 是一个高度优化的神经网络运算符库。

TensorFlow.js 从2.1.0 开始支持 SIMD，从 TensorFlow.js 2.3.0 开始支持多线程。Wasm SIMD是wasm标准第3阶段的提案，Wasm threads是wasm标准第2阶段的提案，目前绝大多数浏览器环境都可支持该两种能力。SIMD和多线程的性能增益彼此独立。TensorFlow的基准测试表明，SIMD 为普通 Wasm 带来了 1.7-4.5 倍的性能提升，而多线程在此之上又带来了 1.8-2.9 倍的加速。

4.1.7 【FFmpeg-音视频处理】ffmpeg.wasm https://github.com/ffmpegwasm/ffmpeg.wasm

FFmpeg就不用多介绍了吧，20多年前Fabrice Bellard发起的FFmpeg项目不知道养活了多少公司和音视频开发者。XX播放器，XX格式工厂基本都是在FFmpeg上面套了个UI。ffmpeg.wasm的意义就在于可以不再完全依赖浏览器的音视频能力，强大的几乎支持所有格式的音视频处理能力可以被移植到web端。根据目前了解到的信息，FFmpeg在操作系统、硬件、驱动等环境支持的情况下，是可以利用GPU或者其它硬件来加速解码的。大多数浏览器也都支持硬件编解码加速，但运行在浏览器内的ffmpeg.wasm应该是只能纯CPU软解的，这里可能会存在一定的性能问题。

4.1.8 【Unity/Unreal-游戏引擎】H5游戏开发、web端游戏运营工具

https://beta.unity3d.com/jonas/AngryBots/，这是Unity在4年前发布的一个demo，使用Unity开发并发布到Web端。其游戏效果已经很好了，且可在浏览器里流程运行。使用原始HTML5技术如果想达到跟这个demo一样的体验和性能，投入的成本将会非常大。目前所有版本的unity以及Unreal4.18之前版本的UE，都支持将游戏内容打包发布到web端。在Unity里的平台名叫"webGL"，在UE里则是"HTML5"。将游戏内容发布到web端，主要需要解决3个问题，首先是将引擎的底层代码和脚本代码编译成wasm的方式来执行，其次引擎的"平台无关层"需要适配webGL这个图形api，最后则是适配浏览器的系统接口。在wasm未诞生之前，引擎则是将代码转成asm.js来执行。至于UE为什么在后面的主线版本不再支持web端，官方给个说法是"未达到预期效果且不好维护"。

第五章 WebAssembly在服务端的应用

看到这里你也许会觉得疑惑，WebAssembly不是跑在浏览器里的前端技术？为什么能跟服务端的docker、K8S、容器化这些概念扯上关系？就像之前文章说到的，这绝对不是一个仅限于前端的新技术，WebAssembly有着更远大的的宏图大业。

Docker的创始人Solomon Hykes在 2019 年 3 月份发布了一条Twitter引起了众多讨论，译文如下：

如果2008年的时候，WASM和 WASI(WebAssembly System Interface, WASM系统接口)这两个东西已经存在了的话，我们就没有必要创立 Docker了。在服务器上运行WebAssembly是计算的未来，目前缺少的就是一个标准的系统接口，希望WASI能够弥补上这块缺失的拼图。
5.1 WASI：解决跨平台运行操作系统的差异

如下图所示：WebAssembly运行在浏览器内，与系统交互靠的是JS胶水语言的能力，JS通过浏览器内核再到操作系统内核。而WebAssembly脱离了浏览器后，运行在各个操作系统中也需要抹平系统api的差异性，这就是WASI需要解决的问题。

WASI(WebAssembly System Interface, WASM系统接口),这里的系统接口指的就是例如文件操作、网络连接、系统时钟、随机数之类的操作系统调用，开发WASI的唯一目的就是将WebAssembly向浏览器之外推进，最终能够真正做到一份wasm代码运行在所有不同环境不同操作系统的机器中。

例如C这样的语言可以跨平台运行，这实际上是源码级的跨平台，一次编写多次编译，编译器根据目标平台选择对应的系统api。如下图所示：C源码被clang编译了3次，生成了三份对应不同目标平台的机器码。




wasm是二进制级别的跨平台，这种可移植性让用户分发代码更容易。wasm只需要被提前编译一次，就能在不同操作系统上运行。在编译的时候并不确定其目标平台，wasi这里实际需要的是一个跨平台的runtime！如下图所示：C源码只被编译了1次，.wasm通过WebAssembly runtime运行在不同系统中。

看到这是不是有种熟悉的感觉了？因为JAVA也就是这么干的，WebAssembly runtime对应的就是JVM，.wasm则对应java bytecode。所不同的是，WebAssembly支持了更多的语言，而且运行在浏览器里支持更加完备。




5.2 WASI现在的进展？

WASI实际上是一个标准，目前最主流的实现方案是Bytecode Alliance使用Rust开发的Wasmtime。截止到我写这篇文章的时候已经有11.3K的star。看了最新git记录，整个开发应该是仍然处于"疯狂打码中"的状态。




在2022年9月，Bytecode Alliance发布了Wasmtime1.0：




快！安全！能够用于生产环境！这就是开发团队对1.0版本最直接的介绍，如果说以前WASI还处于探索阶段，这个版本的推出已经意味着WASI可以在生产环境进行更多的尝试了，整个社区目前也是非常的活跃。

5.3 WebAssembly在服务端的应用

在云计算的概念里，服务端的容器虚拟化大概可以划分为三个不同的抽象层：

Hypervisor VM，或者又称microVMs，其是最底层的虚拟方案，能够直接与硬件进行交互。常见案例有：AWS Firecracker、VMware。
在往上一层是Application containers，所熟知的Docker就在这一层，依然是比较"重"的虚拟方案。
最上层是High level language VMs，JVM、Python runtimes以及WebAssembly都属于这一层。

那么在服务端，WebAssembly到底可以应用在哪些方面？其优势是什么呢？官方给出的建议是有以下五个场景是比较适合的：

(1)微服务/无服务平台，WebAssembly是非常适合用作微服务和无服务平台的。后端即服务(Backend as a Service，BaaS)，函数即服务(Function as a Service，FaaS)都可以归属到severless无服务模型。WebAssembly的启动时间相比docker或者其它VM要快很多，WebAssembly的运行时是非常"轻"的，启动一个WebAssembly实例只需要5微秒。除此之外，轻量级所带来的另外一个优势就是可以在一台机器上搭载更多实例。

(2)第三方插件系统，当平台需要运行第三方开发者的代码，安全性就是不可避免的问题。而WebAssembly是沙盒化的，并且第三方程序无法访问未明确交给给它的任何系统资源。除此之外，平台和第三方插件之间的通信也是很快的。

(3)为数据库实现UDF功能，UserDefineFunction(用户自定义函数，UDF)是数据库应用程序加速的一种方法。指的是将逻辑代码放到数据库中运行，通过降低应用程序和数据库之间的交互开销来提升整个程序的运行效率。例如 Google BigQuery 允许用户从SQL 查询调用以 JavaScript 编写的代码，阿里云的 MaxCompute 可以直接将 Java 或 Python 代码作为 UDF 嵌入 SQL。数据库可以基于WebAssembly runtime来实现UDF能力，其优势在于：支持更多语言、安全隔离、跨平台、性能好、冷启动快等。

(4)搭建可信执行环境，Trusted execution environments (TEEs)指的是为不想或者不能信任底层系统的应用程序单独开辟一个在CPU上安全运行环境，此时的TEE应用程序独立于其它操作系统、虚拟环境、内核以及其它系统软件。TEE技术常用于移动支付、隐私计算等安全性要求较高的场景。使用WebAssembly搭建TEE的优势在于：支持更多语言、WebAssembly运行时支持大多数主流CPU架构。




(5)开发可移植的应用程序，借助跨平台的WebAssembly runtime，WebAssembly应用程序可以运行在不同CPU架构、不同操作系统的计算机上。开发者只需要专注于程序的逻辑功能，而不需要过多担心平台的差异性、性能、安全等问题。

第六章 总结
第一个问题：你的Web应用性能瓶颈在哪里？先想清楚这个问题再做优化

本文的主角并非webGL，但是文章里多次不可避免的提到，其根本原因就在于Wasm解决的是Web端CPU计算密集的性能问题，而性能瓶颈可能压根就不在这里。Figma就是最典型的例子，他们使用wasm将应用移植到web端，并对web端的性能进行了大量优化，但最后复盘发现性能提升的真正来源其实是webGL渲染器的改进，也就是GPU硬件加速的收益，显然这跟WebAssembly并没有任何关系。

在之前我们有提到，Unreal在4.23版本之后将web端的支持从主线分支移除。但近期有一家叫做"Wonder Interactive"创业公司又将这部分能力弥补了回来，并且将在5月份的洛杉矶GamesBeat进行宣讲。在他们计划的工作里，UE5的web端支持将对接最新的WebGPU来实现，游戏AI、寻路等场景也可以用WebGPU新增的通用计算(GPGPU)接口来加速。除此之外，游戏资源的压缩、下载和加载也都需要被考虑，WebAssembly提供的能力也只是整个流程中的一个环节。

第二个问题：现在已经是2023年了，WebAssembly到底算成功了吗？

如果要从技术的成熟度上来说，我认为是成功的，WebAssembly已经投入到了大量的生产应用中。

如果要从推广应用的角度来说，我认为目前是不成功的，因为90%以上的场景其实不需要WebAssembly。

另外一个角度来说就是，WebAssembly很快，但是还不够快，不足以让开发者完全转向拥抱一个崭新的技术。

第三个问题：我们如何拥抱WebAssembly这个新技术？

有3个地方我觉得还是很有应用价值的：

第一个是PC端的应用移植到web端，无需二次开发，保持多端代码一致性。

第二个是音视频处理这些高性能应用的场景，切换到WebAssembly确实能带来很大的性能提升。

第三个就是后端微服务/无服务这一块，可以实现支持多语言的云函数之类的平台。

简单来说就WebAssembly并不是什么神奇的技术，更像是当年JVM未完成理想的开源plus版本，作为开发者没必要跟风追捧或者诋毁。不同的测试用例、硬件环境、编译器优化程度、浏览器引擎优化程度都会对Wasm的运行产生影响，不在具体场景下空谈性能都是没有意义的，根据自己的应用场景+性能测试结果+改造工作量综合评估是否使用即可。




延伸阅读：

1.Google NaCl技术 09年顶会文章

https://storage.googleapis.com/pub-tools-public-publication-data/pdf/34913.pdf

2.NaCl asm.js wasm 区别

https://www.reddit.com/r/html5/comments/4hww2n/what_is_the_difference_between_webassembly_nacl/

3.为什么局部性对数据的性能这么重要？

https://stackoverflow.com/questions/12065774/why-does-cache-locality-matter-for-array-performance

4.IR和字节码有什么区别？

https://www.quora.com/What-is-the-difference-between-intermediate-language-and-bytecode

5.机器码和字节码有什么区别？

https://www.geeksforgeeks.org/difference-between-byte-code-and-machine-code/

6.云原生的 WebAssembly 能取代 Docker 吗？

https://kubesphere.io/zh/blogs/can-webassembly-replace-docker/

7.如何使用Wasm为数据库实现UDF功能？

https://www.secondstate.io/articles/udf-saas-extension/

8.基于WebAssembly实现H.265格式的解封装、解码和播放

https://fed.taobao.org/blog/2019/03/19/web-player-h265/

9.NVDIA官方教程 GPU工作原理(知乎有翻译版，必看)

https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31151/
