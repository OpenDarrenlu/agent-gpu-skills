# Hensel Lifting & Newton-Raphson method

**作者**: 等风​Cryptography

**原文链接**: https://zhuanlan.zhihu.com/p/457343520

---

The proof of Hensel's lemma is constructive, and leads to an efficient algorithm for Hensel lifting, which is fundamental for factoring polynomials, and gives the most efficient known algorithm for exact linear algebra over the rational numbers.

——Wikipedia

引言

Hensel's Lifting Lemma是解析数论的分支p-adic analysis的基础，它为有限域下解高次多项式同余方程提供了一种可行有效的方法，具有非常重要的意义。下面我们将以此为基础，深入去讨论高次同余方程的解结构，包括它的解数和解的形式。

高次多项式同余方程的约简

通常情况下，我们并不需要讨论所有形式的高次多项式同余方程。事实上，我们可进行如下约简：

模素数幂

假设模数为n，n的标准素因子分解为 n=\prod_{i=1}^{k}p_i^{\alpha_i} n=\prod_{i=1}^{k}p_i^{\alpha_i} ，由于CRT（Chinese Remainder Theorem，中国剩余定理）的存在，我们只需要讨论n的素因子分解中的素数幂即可，然后将模各个素数幂同余方程得到的解用CRT组合起来即可得到模n意义下高次同余方程的解。

首一多项式

设 f(x)=a_dx^d+a_{d-1}x^{d-1}+...+a_1x_1+a_0 ，我们可以乘 a_d 的逆元，将其化为首一的多项式。但是在后面的讨论中，我们发现其实也不用化为首一多项式。

高次多项式的降阶

多项式的最高次幂称为多项式的阶，群中也有阶这一概念。根据欧拉定理（相当于模乘法群的阶），我们可以对多项式的阶降下来。对于素数p，其欧拉函数为 \varphi(p)=p-1 ；对于素数幂 p^k ，其欧拉函数为 \varphi(p^k)=p^k-p^{k-1} 。由欧拉定理知：

x^{\varphi(p^k)}\equiv1 \ (mod \ p^k)

特别的，对于素数p，有：

x^{p-1}\equiv1 \ (mod\ p)

由此我们可以把模每个素数幂 p^k 的多项式的阶约束在 (0,\varphi(p^k)-1] (Z_+) 范围内。

一些引理
Lemma 1 高次多项式同余方程的解数

设 m=\prod_{i=1}^{k}m_i ， 满足gcd(m_i,m_j)=1,1\leq i,j\leq k,i\ne j ， f(x)为整系数多项式 ，则

（1） f(x)\equiv0 \ (mod \ m)有解\Leftrightarrow 方程组f(x)\equiv0 \ (mod \ m_i)，1\leq i\leq k均有解

（2） 设对于1\leq i\leq k，都有T_i为f(x)\equiv0 \ (mod \ m_i)的解数，则对于f(x)\equiv0 \ (mod \ m)，其解数T=\prod_{i=1}^{k}T_i

证明也非常好证，对于模素数幂下的每一个解系，我们用CRT去组合每一个解系中的解元素，根据乘法原理可知解数 T=\prod_{i=1}^{k}T_i 。

Lemma 2 Newton-Raphson method

牛顿-拉弗森方法，或者叫牛顿迭代法，是一种利用了收敛性迭代求方程近似解的有效方法，具体原理网上有很多，我们这里只使用它的迭代形式：

x_{n+1}=x_n-\frac{f(x_n)}{f'(x_n)}

Lemma 3 Hensel Lemma

若我们已知 f(x) 在模 p^{k-1}(k\geq 2) 意义下的一个解 f(r)\equiv0 \ (mod \ p^{k-1}) ，那么如果：

f(x)\equiv0 \ (mod \ p^k)有解\Leftrightarrow \frac{f(r)}{p^{k-1}}+tf'(r)\equiv0 \ (mod \ p)有解(*)
若f'(r)\not\equiv0\ (mod \ p)，那么在模p的意义下存在唯一的t满足f(r+tp^{k-1})\equiv0 \ (mod \ p^k)，并且t由如下形式给出：t\equiv-\frac{f(r)}{f'(r)p^{k-1}} \ (mod \ p)(**)
若f'(r)\equiv0 \ (mod \ p)，\forall t\in Z_p，若非f(r+tp^{k-1})\equiv0 \ (mod \ p^k)，则必f(r+tp^{k-1})\not\equiv0 \ (mod \ p^k)(***)
Proof of Lemma 3

(*)式证明见文末链接，我们这里只证明(**)式和(***)式。

因为要满足 f(x)\equiv0 \ (mod \ p^k) ，必先满足 f(x)\equiv0 \ (mod \ p^{k-1}) ，所以在得到 f(r)\equiv0 \ (mod \ p^{k-1}) 后，我们考虑 r'=r+tp^{k-1}(t\in Z_p) ，并讨论t所需满足的条件。

对f(r+tp^{k-1})作Taylor展开：

f(r+tp^{k-1})=f(r)+f'(r)tp^{k-1}+\frac{f''(r)}{2!}(tp^{k-1})^2+...

注意到从第三项开始，后面的所有项模 p^k 都是0，因此可化为:

f(r+tp^{k-1})\equiv f(r)+f'(r)tp^{k-1} \ (mod \ p^k)

所以:

若 f'(r)\equiv0 \ (mod \ p) ，即 f'(r)=kp ，则

f(r+tp^{k-1})\equiv f(r)+kf'(r)p^k\equiv f(r) \ (mod \ p^k)

即:

若 f(r)\equiv0 \ (mod \ p^k) ，那么 \forall t\in Z_p，f(r+tp^{k-1})\equiv0 \ (mod \ p^k) ;
若 f(r)\not\equiv0 \ (mod \ p^k) ，那么 \forall t\in Z_p，f(r+tp^{k-1})\not\equiv0 \ (mod \ p^k) 。

2.若 f'(r)\not\equiv0 \ (mod \ p) ，为了寻找解，我们令 f(r+tp^{k-1})\equiv0 \ (mod \ p^k) ，可得

t\equiv-\frac{f(r)}{f'(r)p^{k-1}} \ (mod \ p^k)

即

r'=r+tp^{k-1}\equiv r-\frac{f(r)}{f'(r)} \ (mod \ p^k)

值得注意的是，这里的除法和（**）的除法均指模意义下的逆元而不是通常意义下的除法。

Taylor系数的补充说明

在上一步作Taylor展开的时候，尽管 (tp^{k-1})^i\equiv0 \ (mod \ p^k)，i\geq 2 ，但是有同学可能会问：Taylor系数 \frac{f^{(i)}(r)}{i!} 一定是个整数吗（对于f(x)为多项式的情况）？答案是肯定的，证明如下：

设f(x)=\sum_{i=0}^{n}{a_ix^i} ，则有 f^{(d)}(x)=a_nn(n-1)...(n-d+1)x^{n-d}+a_{n-1}(n-1)(n-2)...(n-d)x^{n-d-1}+...a_dd!

除d!后都变成了二项式系数!

二项式系数表达式

而我们熟知二项式系数必为整数。因此，我们证明了对于整系数多项式f(x)，其泰勒展开系数必为整数。我们完成了Hensel's Lifting Lemma的必要证明。

结论

在Hensel's lifting Lemma的proof中，我们有

r'=r-\frac{f(r)}{f'(r)} \ (mod \ p^{k})

写为递推形式：

r_1\equiv r \ (mod \ p)

r_k\equiv r_{k-1}-\frac{f(r_{k-1})}{f'(r_{k-1})} \ (mod \ p^k)

正好与牛顿迭代法的形式相近！

然后根据Hensel's Lemma，从k=1开始，对于每一个k，去计算对应的 f'(r_k) \ mod \ p 和 f(r_k) \ mod \ p^k 来确定k+1所对应的解需要如何选择（本来准备把计算代码写了放上来的，但想了想知乎还是放理论吧，毕竟理论带师），将所得各个素数幂的解用CRT组合起来即可。

思考

Hensel's lifting Lemma的形式和Newton-Raphson method的形式如此相近，真的只是巧合吗？Brilliant上提到

Brilliant
参考资料
高次同余方程的解数和解法
Brilliant网站
Hensel lifting
Wikipedia
