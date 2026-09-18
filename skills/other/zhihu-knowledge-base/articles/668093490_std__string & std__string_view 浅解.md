# std::string & std::string_view 浅解

**作者**: JiLi-QA百无一用

**原文链接**: https://zhuanlan.zhihu.com/p/668093490

---

本文内容主要源自 C++之旅 第十章 和 http://learncpp.com

1.string

https://en.cppreference.com/w/cpp/string/basic_string

https://www.learncpp.com/cpp-tutorial/introduction-to-stdstring/

1.1 string 的实现

https://gcc.gnu.org/onlinedocs/gcc-4.6.2/libstdc++/api/a01074_source.html

标准库定义了一个通用的字符串模版basic_string, string 实际上是此模版用字符类型 char 实例化的一个别名
template<typename Char>
class basic_string {
    // ...
}
using string = basic_string<char>;


小字符串优化（SSO）：

在 std::string 实现中，短字符串（长度小于某个阈值“大约为14”）可以直接存储在 std::string 对象的内存空间中，而不需要额外的堆分配。这种做法可以提高性能，因为避免了堆内存的分配和释放开销。

长字符串存储：

对于超过 SSO 阈值的长字符串，它们的内容则存储在自由存储区（即堆内存）。这是因为字符串对象自身的固定大小内存空间不足以容纳长字符串。
在这种情况下，std::string 对象会包含一个指向动态分配内存区域的指针，这个区域存储实际的字符串数据。
1.2 std::getline() & std::cin
std::getline()不会自动忽略前导空白字符, 它会读取并存储字符串的完整行，包括前导空白和字符串中间的空白，直到遇到换行符。
std::getline(std::cin, stringVariables)

std::cin读取数据会忽略前导任何空白字符，遇到非前导空白字符则会自动停止
" hello world "
std::cin 只会读取到 "hello"

使用 std::ws可以让std::getline 忽略前导空白
std::string name{};
std::getline(std::cin >> std::ws, name); // note: added std::ws here

1.3 Do not pass std::stringby value
不要把std::string 为函数参数传入，这会造成昂贵的拷贝
使用std::tring_view
1.4 returning std::string

返回 std::string 应采用传值的方式（依赖于移动语义和拷贝消除）

返回语句中出现以下情况即可

类型为 std::string 的局部变量
通过函数调用或者操作符返回 std::string
在返回语句创建 std::string

std::string 和std::vector 支持 move semantics

返回值优化 (RVO) 和命名返回值优化 (NRVO)：这些优化技术允许编译器在返回局部对象时省略一些复制操作。在这种情况下，函数中的局部对象直接在调用方的上下文中构建，而不是在函数内部构建然后复制到调用方。这减少了不必要的构造和析构调用。
移动语义 (Move Semantics)：当从函数返回一个局部对象时，C++11 引入的移动构造函数和移动赋值运算符会被自动使用（如果它们被定义）。这是因为返回局部对象本质上涉及到临时对象的创建，而临时对象是右值（rvalue）。移动语义允许资源（如动态分配的内存）从这些临时右值对象“移动”到新对象，而不是复制。
#include <cstring>
#include <iostream>

class String {
private:
    char* data;
    size_t size;

public:
    // 构造函数
    String(const char* p) {
        size = strlen(p) + 1;
        data = new char[size];
        memcpy(data, p, size);
    }

    // 析构函数
    ~String() {
        delete[] data;
    }

    // 拷贝构造函数
    String(const String& other) {
        size = other.size;
        data = new char[size];
        memcpy(data, other.data, size);
        std::cout << "Copy constructor called\n";
    }

    // 移动构造函数
    String(String&& other) noexcept 
        : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
        std::cout << "Move constructor called\n";
    }

    // ... 其他成员函数 ...
};


String createString() {
    String temp("Hello, World!");
    return temp;
}


当 createString 返回时，temp 是一个局部对象，它会被视为右值。因此，如果 String 类定义了移动构造函数，它将被调用来构造 main 函数中的 myString 对象。这避免了深层复制，只是简单地转移了资源所有权，这是更高效的。

2.string_view (C++17)

https://www.learncpp.com/cpp-tutorial/introduction-to-stdstring_view/

https://en.cppreference.com/w/cpp/string/basic_string_view

string_view就像一个不拥有其指向的内容的指针或者引用。（需要注意引用内容的生命周期）

2.1 解决了 std::string 拷贝和初始化昂贵的问题
std::string_view 具有显著的显著就是 read only
#include <iostream>
#include <string_view>

// str provides read-only access to whatever argument is passed in
void printSV(std::string_view str) // now a std::string_view
{
    std::cout << str << '\n';
}

int main()
{
    std::string_view s{ "Hello, world!" }; // now a std::string_view
    printSV(s);

    return 0;
}

字面量后缀
using namespace std::literals::string_view_literals;
auto s1 = "Stephen"sv;

2.2 string_view 和 const string&作为函数参数的比较

使用string_view作为参数有三个优点

可以用于以多种不同方式管理的字符串序列
可以轻松的传递子串
传递C风格的字符串无需创建string 对象

string_view可以被 c-style string , string, string_view 初始化，C-style string和 std::string 会隐式的转化为string_view。

#include <iostream>
#include <string>
#include <string_view>

void printSV(std::string_view str)
{
    std::cout << str << '\n';
}

int main()
{
    printSV("Hello, world!"); // call with C-style string literal

    std::string s2{ "Hello, world!" };
    printSV(s2); // call with std::string

    std::string_view s3 { s2 };
    printSV(s3); // call with std::string_view

    return 0;
}


但是 string_view 不会被隐式的转化为 std::string

#include <iostream>
#include <string>
#include <string_view>

void printString(std::string str)
{
        std::cout << str << '\n';
}

int main()
{
        std::string_view sv{ "Hello, world!" };

        // printString(sv);   // compile error: won't implicitly convert std::string_view to a std::string

        std::string s{ sv }; // okay: we can create std::string using std::string_view initializer
        printString(s);      // and call the function with the std::string

        printString(static_cast<std::string>(sv)); // okay: we can explicitly cast a std::string_view to a std::string

        return 0;
}

2.4 支持 constexpr
std::string 对于 constexpr 的支持没有 std::string_view 这么好
#include <iostream>
#include <string_view>

int main()
{
    constexpr std::string_view s{ "Hello, world!" }; // s is a string symbolic constant
    std::cout << s << '\n'; // s will be replaced with "Hello, world!" at compile-time

    return 0;
}

