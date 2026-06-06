# Matplotlib 中英文及公式字体设置

**作者**: cherichy宁做我

**原文链接**: https://zhuanlan.zhihu.com/p/118601703

---

网上铺天盖地matplotlib字体教程，但是为啥我还要再写一个呢？

因为广大的科研人员的需求一般都是要求如下的格式：

中文：宋体
英文：Times New Roman
公式：与英文配合

然而现有的问题如 用Python的matplotlib画图，怎么保证xlabel中中文用宋体，英文用新罗马？ 也没有令人满意的方案。在探索了一段时间后，得到了两种解决方案，可以在同一行内采用不同的中英文以及公式字体。

现有的方法及问题
如果直接设置中文字体，如在matplotlibrc里面修改font.serif: SimSun，则英文字体会被覆盖，因为中文字体中往往都自带英文字体，但大多比较丑。
如果设置usetex=Ture，即调用外部tex程序来渲染，则会因为调用的是pdflatex而显示不了中文。

所以针对这两种方法的问题，分别有对应的解决方案：

使用内置tex

Matplotlib（下称mpl）自带tex引擎，可以解析并显示Latex字符串。虽然中文会覆盖英文字体，但是mpl的数学字体和一般的text字体是独立的，所以我们可以把文字放到Latex字符串里面，并使用\mathrm{text}来使其显示为正体。

mpl自带有几种数学字体，最常用的有cm系列和stix系列，前者是Latex默认的数学字体，而stix的正体和Times New Roman差别很小，一般用来和Times New Roman搭配。所以这里我们采用stix，即可达到相同的视觉效果。

代码如下：

import matplotlib.pyplot as plt
from matplotlib import rcParams

config = {
    "font.family":'serif',
    "font.size": 20,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)

plt.title(r'宋体 $\mathrm{Times \; New \; Roman}\/\/ \alpha_i > \beta_i$')
plt.axis('off')
plt.savefig("usestix.png")

结果如图：

使用外部tex

因为pdflatex对unicode的支持不太好，用不了中文，但是可以使用xelatex。mpl中使用xelatex的方法为改用pgf后端，pgf后端还支持lualatex。在pgf后端中，可以通过设置pgf.preamble来在latex中设置所需要的字体。

在xelatex中可以分别对中英文以及公式字体进行修改，其中中文需要使用xeCJK宏包，公式则使用unicode-math宏包，但是使用了默认的公式字体。这里的英文是真Times New Roman，并非如前一种方法中是用了stix字体，当然这里也可以使用相同的设置。

update：由于之前有知友问如何在公式里面支持中文，这个需求其实是Latex的问题，和本文主题有点偏。不过也在这里说一下吧，这种需求需要在使用xeCJK宏包后开启CJKmath的选项，即\xeCJKsetup{CJKmath=true}。

代码如下：

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams

matplotlib.use("pgf")
pgf_config = {
    "font.family":'serif',
    "font.size": 20,
    "pgf.rcfonts": False,
    "text.usetex": True,
    "pgf.preamble": [
        r"\usepackage{unicode-math}",
        #r"\setmathfont{XITS Math}", 
        # 这里注释掉了公式的XITS字体，可以自行修改
        r"\setmainfont{Times New Roman}",
        r"\usepackage{xeCJK}",
        r"\xeCJKsetup{CJKmath=true}",
        r"\setCJKmainfont{SimSun}",
    ],
}
rcParams.update(pgf_config)

plt.title(r"宋体 Times New Roman $\alpha_{一} > \beta_{二}$")
plt.axis('off')
plt.savefig("usetex.png")

结果如图:
