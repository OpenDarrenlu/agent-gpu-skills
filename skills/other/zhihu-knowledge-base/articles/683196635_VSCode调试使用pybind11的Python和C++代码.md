# VSCode调试使用pybind11的Python和C++代码

**作者**: KevinMaster Student at ZJU, MLSys, AI Infra

**原文链接**: https://zhuanlan.zhihu.com/p/683196635

---

demo代码开源在：KevinZeng08/pybind-debug: Debug pybind11-mixed Python and C++ program in VSCode

vscode商店中下载 Python C++ Debugger 插件

2. 编写launch.json

{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python C++ Debugger",
            "type": "pythoncpp",
            "request": "launch",
            "pythonLaunchName": "Python: Current File",
            "cppAttachName": "(gdb) Attach"
        },
        {
            "name": "(gdb) Attach",
            "type": "cppdbg",
            "request": "attach",
            "program": "/home/zbw/miniconda3/bin/python",
            "processId": "",
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ]
        },
        {
            "name": "Python: Current File",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/test_add.py",
            "console": "integratedTerminal"
        }
    ]
}

3. 设置 ptrace_scope 为 0

echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope

4. 在python和cpp文件设置断点，vscode启动 Python C++ Debugger 调试

一些解释：

在我的vscode中，如果跳过第三步，运行gdb attach的permission验证过程中，无法输入密码直接退出

Authenticating as: admin,,, (admin1) Password: [1] + Stopped (tty output) /usr/bin/pkexec "/usr/bin/gdb" --interpreter=mi --tty=${DbgTerm} 0<"/tmp/Microsoft-MIEngine-In-cdlwi1ey.d0f" 1>"/tmp/Microsoft-MIEngine-Out-yeoazv5q.ght" You have stopped jobs

参考 Debugging mixed Python C++ in VS Code. Can't enter sudo password 方案，设置 ptrace_scope 为 0，允许任何进程（无论是否为父进程）都能够对其他进程进行跟踪。ptrace_scope为1时，只有父进程（即启动了子进程的进程）才能对子进程进行跟踪。
