# Google 的木马：Google Play Service

**作者**: Max Lv芯片架构师，开源软件开发者

**原文链接**: https://zhuanlan.zhihu.com/p/22591067

---

很难想象 Google 打算基于 Google Play Service 提供这样一项服务：用户可以使用某一款短信类应用向通讯录内任意联系人的 Android 手机（需要预装 Google Play Service）发送信息，并推介安装相应的应用，即便别人并没有安装此应用。

这个功能听上去有点类似 iMessage，只是 Google 会开放给更多的开发者。虽然初衷是用于通信类软件，但我估计未来很有可能一发不可收拾，接着 Google Play Service 成了名副其实的 Trojan 和 AdWare。


至于为什么 Google 打算推广这样一个“邪恶”的功能呢？用脚趾想也猜的出是为了推广 Allo 和 Duo 呗。

更多请参考：https://developers.google.com/android/guides/app-preview-messaging
