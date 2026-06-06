# 自己编译的opencv python binding 报ImportError

**作者**: cherichy宁做我

**原文链接**: https://zhuanlan.zhihu.com/p/423343732

---

最近从python3.7升级到了3.9，然后重新又编译了一下opencv。结果，诶嘿，二话不说就给您报一个ImporError: DLL load failed while importing cv2: 找不到指定的模块。

>>> import cv2
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\myprogram\python\lib\site-packages\cv2\__init__.py", line 180, in <module>
    bootstrap()
  File "C:\myprogram\python\lib\site-packages\cv2\__init__.py", line 152, in bootstrap
    native_module = importlib.import_module("cv2")
  File "C:\myprogram\python\lib\importlib\__init__.py", line 127, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
ImportError: DLL load failed while importing cv2: 找不到指定的模块。

按照以前的经验，以为是PATH没调好，经过漫长的调试（2天），怎么搞都依然报错。翻了半天的Github issue 看到了这个，里面提到了Python在3.8以后就不再自动Load PATH中的路径了，所以我们需要将编译时用到的dll的路径在import cv2之前能加到系统中去。

Python Bindings importing error · Issue #17632 · opencv/opencv (github.com)

在cv2的__init__.py 文件的第135行也写到了，需用os.add_dll_dirctory()把“BINARIES_PATHS”里面的dll路径加进去。

if sys.version_info[:2] >= (3, 8):  # https://github.com/python/cpython/pull/12302
    for p in l_vars['BINARIES_PATHS']:
         try:
            os.add_dll_directory(p)

所以只需要在config.py文件里面加入编译时依赖的dll文件路径即可。因为我编译的时候link了CUDA，以及OpenBlas，所以加入就好啦！

BINARIES_PATHS = [
    os.path.join('E:/opencvbuild/opencv454-gpu', 'x64/vc16/bin'),
    os.path.join(os.getenv('CUDA_PATH', 'D:/CppLib/CUDA/v11.1'), 'bin'),
    'D:/CppLib/bin',
    'D:/Qt/5.15.2/msvc2019_64/bin',
    'F:/codes/vcpkg/installed/x64-windows/bin'
] + BINARIES_PATHS
