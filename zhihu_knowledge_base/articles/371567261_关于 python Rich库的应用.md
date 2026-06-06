# 关于 python Rich库的应用

**作者**: 离散PKU → UCLA → Ai Infra调包侠

**原文链接**: https://zhuanlan.zhihu.com/p/371567261

---

这几天在做 MIT6.824的 lab，翻到这么一个debug by pretty printing 的网站感觉帮了很多忙。如果只是想做 MIT 那个 lab，直接用 dtest 这个代码就好了。我这里就是试着解析一下其中的 python 代码片段，希望之后要是遇到其他场景可以举一反三。

进度条——rich.process

关于 rich.process 为什么比 tqdm 好用可以参考 这篇文章 ，一个直观的感觉就是 rich 显示出来的进度条确实更好看。

用法一：可以直接 rich.process.track 替代 tqdm.tqdm，下面这段代码就可以体现出 rich 这个库对于 tqdm 的优势了

import time

# from tqdm import tqdm
# for step in tqdm(range(10)):
#     print("step %d" % step)
#     time.sleep(0.1)
    
from rich.progress import track
from rich import print
for step in track(range(10)):
    print("step %d" % step)
    time.sleep(0.1)




用法二：可以使用 rich.progress.Progress 来完成多个进度条的刷新。我理解一个 Progress 代表 展示一种特定风格的进度条模板，在这个模板之下，可以加各种 task，他们代表真正被显示出来的内容。Progress，Task 都是Object，我们只用负责 init 和 update 就好，展示出来的东西库会处理好。

例如。在 MIT6.824的 lab 中，我们希望同时并发的去测试 lab 2A, lab 2B和 lab 2C并显示各自的测试进度。下面代码基本就是多进程测试 MIT6.824 lab代码的核心。基本的流程为

每次选择指定数量的进程提交给executor
使用 wait 等待其中有部分进程执行完毕
分析执行好的函数的返回值，如果是失败应该保存失败对应日志，如果成功则删除对应日志(这个代码在下面没有体现，但是在 dtest 可以看到，是用 pathlib 和 os 这两个库完成的，比较简单)

下面代码中，Process 对象传入的参数就是进度条会显示的内容，具体各个 Column 可以参考process 官网。基本上需要展示的信息也不需要那么多，我感觉就"当前耗时"和"预期剩下的时间"这两个比较重要。下面那个 SpinnerColumn 就是单纯来美化界面的。

另外 [color]text[\color] 这个如果用 rich.print 打出来就会打出有颜色的字体。这里如果有 fail 就显示红色，没有就显示黄色，可以明显区分

# [...] Some boring imports

def run_test(test: str): # 预期是每次执行一个命令，表示测试一次 lab
    time.sleep(0.1) # supposed to use subprocess.run to run test command here
    if random.uniform(0, 1) < 0.1:
        return test, False
    return test, True

def main():
    num_workers, iterations, tests = 4, 100, ['2A', '2B', '2C']
    test_instances = itertools.chain.from_iterable(itertools.repeat(tests, iterations))
    test_instances = iter(test_instances)
    total_iter = iterations*len(tests)
    
    total_progress = Progress(
        TextColumn("[progress.description]{task.description}", justify="right"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•", TimeElapsedColumn(),
        "•", TimeRemainingColumn(),
    )
    total_task = total_progress.add_task("[yellow]Tests[/yellow]", total=total_iter)
    task_progress = Progress(
        "[progress.description]{task.description}",
        SpinnerColumn(),
        BarColumn(),
        "{task.completed}/{task.total}",
    )
    tasks = {test: task_progress.add_task(f"[yellow]{test}[/yellow]", total=iterations) for test in tests}
    progress_table = Table.grid()
    progress_table.add_row(total_progress)
    progress_table.add_row(Panel.fit(task_progress))
    
    # 这里可以用 with task_progress: 替代，不过这也的话一次只能展示一个 progress，
    # 使用 table 的话，就也可以展示多种不同类型的进度条，关于 table 我在之后讲
    with Live(progress_table, refresh_per_second=1) as live:
        def handle_sigint(signum, frame):
            live.stop()
            sys.exit(1)
        signal.signal(signal.SIGINT, handle_sigint)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures, completed = [], 0
            while completed < total_iter:
                n = len(futures)
                if n < num_workers:
                    for test in itertools.islice(test_instances, num_workers-n):
                        futures.append(executor.submit(run_test, test))
                # 一旦有一个进程执行好，就返回
                done, not_done = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    test, success = future.result()
                    if not success:
                        print(f"Fail test {test} - {completed}")
                        task_progress.update(tasks[test], description=f"[red]{test}[/red]")
                    completed += 1
                    task_progress.update(tasks[test], advance=1)
                    total_progress.update(total_task, advance=1)
                futures = list(not_done)

关于使用场景的思考： 爬虫里就可以套这个。有一个例子可以关注 "Python实用宝典"公众号, 后台回复 rich示例, 里面的 downloader.py 就是。基本用法就是设定 total 为 content length，然后每次 advance 获取的 data 的 length；另外我觉得机器学习模型的训练也可以用，log 可以直接放到文件里，没必要展示到 stdout 里，具体的进度用 rich 的进度条来展示就行。

表格——rich.table

和进度条彼此独立的一个东西是表格。

用法一： 通过 add_column 定义列的名字；通过 add_row 定义每一行具体的值。下面就是一个很简单的 print 一张表格的函数

# first_col, col_name_list = "Test", ["Failed", "Total"]
def print_results(results: Dict[str, Dict[str, int]], first_col:str, col_name_list: List[str]):
    table = Table(show_header=True, header_style="bold")
    table.add_column(first_col)
    for col_name in col_name_list:
        table.add_column(col_name, justify="right")
    
    for row_name, col2value in results.items():
        color = "green" if col2value["Failed"] else "red"  
        value = f"[{color}]{row_name}[/{color}]"
        row = [value]
        for col in col_name_list:
            row.add(str(col2value[col]))
        table.add_row(*row)
    print(table)

用法二： 和 Progress 相结合，这样就可以展示多种类型的进度条；同时需要使用 with Live

Rich还有其他的一些用法，比如 Console，还可以显示 markdown，但是我感觉这些功能很少能用到。所以就懒得看了 orz。
