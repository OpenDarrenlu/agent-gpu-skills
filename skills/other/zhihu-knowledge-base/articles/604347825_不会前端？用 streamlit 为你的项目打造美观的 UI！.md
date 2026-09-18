# 不会前端？用 streamlit 为你的项目打造美观的 UI！

**作者**: Uranus​清华大学 计算机系博士在读

**原文链接**: https://zhuanlan.zhihu.com/p/604347825

---

照例先放 github 链接：

https://github.com/UranusSeven/qing_bureau_of_construction
github.com/UranusSeven/qing_bureau_of_construction

上一篇文章我用 OCR + whoosh 做了一个古籍搜索系统，并使用 PyInquirer 做了一个命令行的搜索交互工具，效果是这样的：

Command line tool powered by PyInquirer

不过一方面 PyInquirer 已经许久没有维护，和 iPython 等常用的 libs 开始产生冲突，另一方面命令行对学者们来说实在不够友好，因此我决定换掉 PyInquirer。

然而，作为 infrastructure engineer，写前端/UI完全不在我的技术栈里啊。这时候，我想到了 streamlit。streamlit 号称几分钟就能把一个脚本（准确来说是 data script）变成一个 web 应用。在去年 2 月份，streamlit 被 snowflake 以 8 亿美元的价格收购，想必被业界大佬 snowflake 看好的产品不会差吧 :P

于是乎，我尝试用 streamlit 重写了 UI 部分。这个过程出乎意料的顺利，只用了 50 行左右就完成了预期的效果：

Web UI powered by streamlit

代码如下：

def app():
    st.title("清宮造辦處電子檔案搜索系統")
    keywords = st.text_input('請以 **繁體中文** 輸入待查詢內容:')
    keywords = keywords.split()

    if keywords:
        query = content_t_cn_parser.parse(" AND ".join(keywords))
        results = searcher.search(query, limit=None, optimize=False)

        if not results:
            st.write("未找到匹配結果.")

        choices = []
        for hit in results:
            vol = hit["vol"]
            page = hit["page"]
            side = "上半" if hit["side"] == "0" else "下半"
            content: str = hit["content_raw"]
            content = " ".join(list(jieba.cut(content, cut_all=False)))
            content = highlight(keywords, content)

            choice = (vol, page, side, content)
            choices.append(choice)

        for vol, page, side, content in choices:
            location = f"{vol} 卷 {int(page) - 1} 頁{side}部分"
            if CHROME_EXISTS:
                pdf_file_path = f"file://{PDF_FILES_DIR}/{vol}.pdf#page={int(page) + 1}"
                location = f"[{location}]({pdf_file_path})"
                st.markdown(location)
            else:
                st.write(location)
            st.caption(content)

能达到这么好的效果，我觉得主要归因于 streamlit 提供了非常好用的 I/O 方法和非常符合直觉的编程模式：

st.title 以标题形式打印文本，类似于 markdown 中的 #
st.text_input 在页面上展示文本输入框，并返回输入的文本
st.write 打印文字，并且可以嵌入 html 玩出更多花样
st.markdown 以 markdown 形式打印文字，可以方便地打印各种格式
st.caption 用来打印解释性文字

当 st.text_input 发生变化后，streamlit 会自动刷新页面，这样代码和最终渲染出的页面一一对应，非常符合直觉，编程体验满分！然而我用到的实际上还只是 streamlit 的冰山一角，除了上面的接口外，streamlit 还提供了很多可交互的组件，包括各种按钮，表格等等等等。

现在，streamlit cloud 现在还提供免费的应用托管，能快速把你的 github 项目托管到云上。欢迎大家来试试我的项目 :D
