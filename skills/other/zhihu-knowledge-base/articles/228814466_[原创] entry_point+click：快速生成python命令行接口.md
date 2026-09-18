# [原创] entry_point+click：快速生成python命令行接口

**作者**: 小志哥​​上海引望智能技术有限公司 员工

**原文链接**: https://zhuanlan.zhihu.com/p/228814466

---

Introduction

如果你是bash熟练技工，那你肯定知道bash命令行工具有多趁手： sed、 grep、 awk等；还有花式的重定向、管道，一条命令解决问题，简单明了心情舒畅。有没有办法让我们的脚本，也拥有这些原生命令行工具的使用体验呢？

当然有，entry point + Click，将python脚本打包成命令行工具的好帮手！本文能帮助你快速将你的python脚本包装成功能强大、调用方便的命令行工具，从此以后成为街上最靓的数据民工！

本文介绍的内容总结如下：

通过entrypoint把你的python脚本工具安装到bin路径，在bash中可以直接复用，不用敲长长的路径，奥力给！
通过click，一个强大的python第三方库，实现多级嵌套命令。一个入口，多种功能，就算你的脚本数量再多，功能再杂，也能给你安排得明明白白。还有强大的管道功能！
前言

一套得心应手的工具有多重要，经常跟数据处理打交道的 数据民工 工程师可能深有体会。据说，数据处理工程的一天，是从不知道哪里来的祖传脚本、屎山数据开始，再以人模狗样的最终数据结束，中间留下来了一堆打满补丁的脚本和乱七八糟、随便命名的中间文件。别人以为我们生产的是数据，实际上大部分时间我们都在生产脚本。一个又一个的临时脚本，在代码复用上，鸡肋且为难。

而日常工作被乱七八糟、没有版本控制的脚本占据，不但非常影响工作体验，也非常直接地影响了你的 游戏时间 下班时间。如何在代码复用和满足千奇百怪需求中取得平衡呢？笔者在这个问题上也想了很久，下面给大家抛砖引玉一个。

关于脚本仓库

很多时候，我们写了很多数据处理的脚本，杂乱无章地堆在各种各样的角落。这样做有什么坏处呢？

不好找
其次，即使脑子知道大概在那里，但是记不住路径，还是得敲一大段路径，慢慢地tab、tab、tab补全，烦死个人
最后，不好管理，面临着多个版本的问题，容易出错。

最大的坏处：程序如果很复杂很难搞，人就很容易烦躁；一烦躁，数据就做错了，又得全部重新来过；如果连做错了都不知道，下游算法出了问题，那就要被3.25，要背锅了。当数据工具人是最难的，正所谓做对了没功劳，做错了就背锅，没有功劳却也没有苦劳；咳咳，好像偏题了，回来回来。

所以，我们推荐使用一个脚本仓库，把自己的脚本给组织起来：

当你写脚本的时候，要考虑一下哪些东西可以复用。把最通用的部分，提取出来，组织到脚本仓库中去。
其次，当多次处理同一类型的事务时，要总结一下流程，标准化成通用事务脚本，组织到脚本仓库中去。
不要忘记在每次数据处理时，把commit hash记下来，确保可复现性。
给数据工具人的一个路径参考
首先，生命苦短，我用python。第三方库丰富，代码可读性高，debug方便。
其次，将数据处理操作分为三个档次。这三个档次的定制程度依次上升，标准程度依次下降。
最常用的、最标准化的脚本工具，用本文的方法制作成命令行工具。
通用的流程、事务，按逻辑结构组织在脚本仓库里。
非常细致、定制化的脚本以及具体操作脚本，跟着数据走，可以放在与数据文件相同的地方。
最后，数据成品才是最重要的。不要在代码抽象、组织上花费太多时间！
一个最终效果的例子

回到正题，今天讲的是python命令行接口。在具体介绍前，先给大家看看，利用entry point+Click，我们能达成什么效果。

# 我有这么一个长长的数据文件
>> head what_a_mess.jsonl
{"name":"小明", "duration": 3434, "date": "03/14", ...}
{"name":"小红", "duration": 9119, "date": "04/24", ...}
{"name":"小明", "duration": 34, "date": "03/13", ...}
{"name":"小明", "duration": 116, "date": "03/11", ...}
{"name":"小红", "duration": 99, "date": "03/14", ...}

# jsonl是我们自己用click搞的命令行工具
>> jsonl --help
Usage: jsonl [OPTIONS] COMMAND [ARGS]...

  An exciting wheel! Young and Simple!

Options:
  -h, --help  Show this message and exit.

Commands:
  advanced   高级功能!
  build      提供多种构建数据集的方法
  check      检查正确性
  ops        数学操作
  to         简单多样的转格式工具
  visualize  提供花式可视化方法

# 选取特定范围的数据，对各键值进行采样，得到数据后，把jsonl数据转换成tsv数据。只要一行，不用打路径，给力吗?
>> grep "小明" what_a_mess.jsonl | jsonl ops sample by_key "date" --ratio 0.5 - - | jsonl to csv - "a_day_of_hard_work.tsv"

# 记得把操作步骤给记下来，确保可复现性哦。
>> cat 'grep "小明" what_a_mess.jsonl | jsonl ops sample by_key "date" --ratio 0.5 - - | jsonl to csv - "a_day_of_hard_work.tsv"' > run.sh

有了多级命令，再多的功能，也能塞得下；再杂乱的功能，也能有逻辑性地组织起来。搞数据的姿势对了，原来可以这么快活^.^

HOW-TO

那么，现在唯一的问题就是：要怎么搞？下面分为两个章节，给大家介绍实现方法。

entry_point

entry_point是python在安装模块（module）时提供的一个功能，其能把模块中给定的函数包装成可执行文件，并部署到系统路径（即Linux中的 PATH路径）。具体路径视你的python环境不同而不同。举例子，如果你是anaconda环境，那么可执行文件就会安装在 anaconda3/env/$环境名字/bin下。这里有个更全面的英文介绍。下面简要介绍一下流程：

首先，你要有个仓库。如果同学对python module不熟悉，那请先搜索引擎查一查学一学噢。

然后，编辑仓库根目录下的 setup.py文件。下面给一个python3例子：我们的module下面有一个代码文件夹和一个setup.py文件。

>> ls $MY_MODULE_PATH
my_tools/    setup.py

编辑setup文件，增加entry_points选项

# setup.py文件
from os.path import join, dirname

from setuptools import setup
import setuptools

setup(
    name="my_project_name",
    version='1.0',
    description='这是一个示例',
    url='my_github_repos_url，也可以不填',
    packages=setuptools.find_packages(),
    keywords=['keyword'],
    install_requires=[
        "contexttimer",
    ],
    python_requires=">=3",
    entry_points="""
        [console_scripts]
        jsonl=my_tools.cli.command.jsonl:cli
        labeling=my_tools.projects.labeling.cli.command:run
    """
)

解释一下，jsonl=my_tools.cli.command.jsonl:cli 指的是，在 my_tools/cli/command/jsonl.py中有一个 cli()函数，我们以其为接口，包装成一个命令行工具，名字为 jsonl。

最后，记得执行安装模块操作 pip install -e .，python才会真正把工具部署在可执行路径中：

>> cd $MY_MODULE_PATH; ls
my_tools/    setup.py
>> pip install -e .
>> jsonl --help
Usage: jsonl [OPTIONS] COMMAND [ARGS]...

  An exciting wheel! Young and Simple!

Options:
  -h, --help  Show this message and exit.

Commands:
  advanced   高级功能!
  build      提供多种构建数据集的方法
  check      检查正确性
  ops        数学操作
  to         简单多样的转格式工具
  visualize  提供花式可视化方法
Click

Click是一个开源的python第三方库，专门用来生成命令行接口。其源代码在https://github.com/pallets/click ，文档是 文档。它是为了解决什么问题而诞生的呢？这里引用一下官方介绍：

Click is a Python package for creating beautiful command line interfaces in a composable way with as little code as necessary. It's the "Command Line Interface Creation Kit". It's highly configurable but comes with sensible defaults out of the box.
It aims to make the process of writing command line tools quick and fun while also preventing any frustration caused by the inability to implement an intended CLI API.
Click in three points:
* Arbitrary nesting of commands
* Automatic help page generation
* Supports lazy loading of subcommands at runtime

翻译一下：Click能让你用尽可能少的代码，以可组合的形式生成漂亮的命令行接口，有以下三个优点：

任意的命令嵌套
自动生成帮助页面
懒惰式的子命令加载（可提高加载速度）

Click的官方文档很充足、易懂，这里就不对其进行二次介绍，希望同学能够自行阅读。这里有个其他好心人的中文学习笔记，英文捉急的同学可以参考下。

下面，仅给出，实现我们开头的例子所需要的三个关键魔法。

命令嵌套

命令嵌套，能帮助我们把不同的脚本文件按照作用、类型、功能组织起来，达到更好的抽象性。有了嵌套，命令再多，也不担心乱掉。

这里给一个三级命令的命令嵌套例子。记住，命令的嵌套层数是任意的；但是，不要弄得过深，把握好度。

# my_tools/cli/command/jsonl.py 文件
import click

# 根入口，也就是嵌套命令的最高一层。如果以上面介绍的entrypoint方式安装，则可以通过jsonl命令直接在bash中调用
@click.group()
def cli():
    pass

# 子命令，可用jsonl check调用
@cli.command()
def check():
    click.echo('Initialized the database')

# 子命令，可用jsonl to调用
@cli.group()
def to():
    pass

# 三级子命令，可用jsonl to csv调用
@to.command()
def csv():
    pass

# 也可以这么添加子命令。二级命令，可用jsonl ops调用。
from my_tools.data.jsonl.ops.cli import ops
cli.add_command(ops)

if __name__ == '__main__':
    cli()
管道传输数据

click提供了一个很方便的功能，能让你的工具让Linux原生工具一样，方便地使用各种管道功能。管道的使用，对工作效率的提升是肉眼可见的，举个例子：

grep "xiaoming" my_data.jsonl > tmp
jsonl to csv tmp final.csv
rm tmp
# 但如果有了管道，一行就搞定。程序运行速度也提升了，特别是在巨大文件的情况下。
grep "xiaoming" my_data.jsonl | jsonl to csv - final.csv

当使用type为click.File的click.argument作为接口的参数时，就可以启用管道功能。其会自动根据输入打开输入流、或者输出流，以供程序使用。在bash中使用时，使用 - 代替 输入文件路径或者输出文件路径，便将输入输出重定向到管道中。

注意：当你像下面的程序一样使用 click.File时，得到的input、output，不再是一个文件路径，而是一个已经打开的文件对象。也就是说，就算你在bash中使用时用了真实的文件路径而不是 -，你的input也不会是一个string字符串，而相当于是一个open($input_string, “r”)的文件对象。

下面是一个实现的例子：

############## Utils ###############
@jsonl.command(help="显示这个jsonl的一些统计信息")
@click.argument("input", type=click.File("r"))
@click.argument("output", type=click.File("w"))
@click.option("--mode", default="0", type=click.Choice(["0", "1"]))
def csv(input, output, mode=0):
    import json
    from my_tools.data.jsonl.to import to_csv
    jsons = [json.loads(line) for line in input.readlines()]
    output.write(to_csv(jsons))
命令的简略输入

有时候命令太长，很容易敲错；或者，单纯就是懒得敲，记不住，任性。这种情况下，我们就会想，要是有简略命令就好了。幸运的是，可以通过设置，让click支持简略命令。

通过以下代码，你就可以在二级命令、或者更高层的命令中使用命令的任意前缀，只要这个前缀不要短到引起歧义。

实现效果：

# 假如我们有个这样老长的三级命令：jsonl advanced_technology get_this_motherfucker_done
# 通过AliasedGroup设置后，这样简短的几个单词，就可以跑起来啦！
>> jsonl ad get_this   
RUNNING!

实现例子：

import click

class AliasedGroup(click.Group):
    def get_command(self, ctx, cmd_name):
        rv = click.Group.get_command(self, ctx, cmd_name)
        if rv is not None:
            return rv
        matches = [x for x in self.list_commands(ctx) if x.startswith(cmd_name)]
        if not matches:
            return None
        elif len(matches) == 1:
            return click.Group.get_command(self, ctx, matches[0])
        ctx.fail('Too many matches: %s' % ', '.join(sorted(matches)))
# 这句是用来开启-h缩写的，很方便哦。
context_settings = dict(help_option_names=['-h', '--help'])

# CLI entry
@click.command(cls=AliasedGroup, context_settings=context_settings,
               help='Tools for jsonl.')
def jsonl():
    pass
    
@jsonl.command(cls=AliasedGroup, context_settings=context_settings,
               help='Black magic top technology very danger!')
def advanced_technology():
    pass

@advanced_technology.command(help='booming!')
def get_this_motherfucker_done():
    pass
    
Click和argparse：谁更好？

argparse是python内置的官方库，相信大家或多或少都用过。那么，Click相对于argparse有什么优劣呢？笔者从实用的角度总结了一下：

Click的优点
包装命令行工具更加直观，代码量更少
支持嵌套命令
支持管道I/O
支持命令简写
argparse的优点
原生库，官方支持
使用的人多，出现的场景多，学习成本更小。
支持一个参数一次使用有不定长的输入
可以支持 python run.py --inputs *.list --output result.txt场景，与bash通配符（wildcards）结合起来非常方便；而Click无法支持。

总的来说，argparse是不能不会的，但是Click是更好的那一个。两种姿势，同学们都得学习一番！

总结

在前言里，我们给出了一种 数据民工 数据工程师的数据处理的脚本管理的可能技术路径。在正文里，我们先是展示了最终效果，然后介绍了entry_point的作用和实现方法，以及Click的作用和三个实用技巧，最后对比了一下Click和argparse的优劣。看完本文，你应该对python命令行接口的相关内容有一定的了解了，接下来请在实践中好好体会，理论联系实践。

最后请记住，结果是最重要的，但是磨刀不误砍柴工，我们的刀，要得心应手，才能随心所欲。
