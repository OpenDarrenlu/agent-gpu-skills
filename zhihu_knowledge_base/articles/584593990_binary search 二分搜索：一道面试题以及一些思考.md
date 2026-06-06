# binary search 二分搜索：一道面试题以及一些思考

**作者**: 离散PKU → UCLA → Ai Infra调包侠

**原文链接**: https://zhuanlan.zhihu.com/p/584593990

---

今天面试被面了一道二分搜索相关的题。感觉自己在二分搜索上还是有一些问题。因此打算以这道题目为例复习一下二分搜索相关的知识。感觉这样写写文章还是很必要的。很多时候我思路好像很清楚，但是说出来就说的稀里糊涂的；但是这种稀里糊涂会让我在面试的时候和面试官说不清楚，也可能让我在写题的时候因为一些 corner case 犯错。所以看的感觉有问题欢迎评论，我一定解答清楚/知错就改 orz。

Problem Statement

给定 k_list = [k1, k2, ..., km], y_list = [y1, y2, ..., yn]. 其中 k_list 中的元素 ki 代表一条直线 y = ki，y_list 中的元素 yi 代表一个点(i, yi)，整个 y_list 代表一条由 n-1个线段组成的折线，从左到右依次把每个点连起来(题目保证 yi != yi+1)。要求返回一个 dict，key 是 ki，value 是对应直线和折线的交点的个数。

Solution
要通过二分查找快速判断哪些直线和一条线段相交
假设每一段折线代表一个开区间 (y_min, y_max), k_list 中在这个区间中的 k 都需要加一。由于 k_list 是排序的，所以我们要找的是一个范围 [i_min, i_max)，满足all(y_min < k_list[i_min:i_max] < y_max)。这里可以利用 i_min 可以用upper_bound 找，i_max 可以用 lower_bound 找。 想到这一步的时候，我们只知道我们需要找一个区间就行了，只能说前闭后开可以很方便的表示一个区间。其真正的方便之处在于更加简洁地写出区间更新的代码

在O(1)时间内完成对[i, j]范围内的数据都实现加一
这里利用前缀和的思想。我们假设ki 对应的个数 k_count[i] = sum(k_count_prev[:i+1])，那么每次如果我们要对区间 [i_min, i_max)进行+1，就只需要对 k_count_prev[i_min] += 1, k_count_prev[i_max]-=1。经过这样子更新之后，最后算出来的 k_count 里面只有[i_min, i_max)这个区间的元素的值出现了变化。我们只需要最后用k_count[i] = sum(k_count_prev[:i+1]) 就可以了。
# 这段代码没有真正跑起来过
import bisect
def find_intersection_count(k_list, y_list):
    m, n = len(k_list), len(y_list)
    k_list = sorted(k_list)
    k_count_prev = [0] * (m+1)
    for i in range(n-1):
        y_min, y_max = y_list[i], y_list[i+1]
        if y_min > y_max:
            y_min, y_max = y_max, y_min
        # 找到有多少个 k 在开区间(y_min, y_min)之中
        # k_list[i_min] > y_min, k_list[i_max] >= y_max
        # i_min是需要被算进去的，i_max 是不需要被算进去的
        i_min = bisect.bisect_right(k_list, y_min)
        i_max = bisect.bisect_left(k_list, y_min)

        # 快速对 [i_min, i_max)中所有数+1
        k_count_prev[i_min] += 1
        k_count_prev[i_max] -= 1
    prev_count = 0
    k_count = {}
    for i in range(m):
        prev_count += k_count_prev[i]
        k_count[k_list[i]] = prev_count
    for i in range(n):
        if y_list[i] in k_count:
            k_count[y_list[i]] += 1
补充

总结一下，这道题目其实并不难。但是我对于二分搜索库函数的定义不太熟悉，所以当时写的其实算是伪码。感觉算是一个小瑕疵。另外一开始没有快速想到那个常数时间进行区间更新的办法，差一点点面试官就要提示我了，还是感觉心有余悸吧 orz。

关于python/c++ 二分搜索的函数
value a a a b b b c c c
index 0 1 2 3 4 5 6 7 8
bound l u
import bisect
# a[i] >= x, upper_bound
i = bisect.bisect_left(a, x, lo=0, hi=len(a))
# a[i] > x, upper_bound
i = bisect.bisect_right(a, x, lo=0, hi=len(a))

#include<algorithm>
// *it >= val 
const auto& it = std::lower_bound(first, last, val);
// *it > val
const auto& it = std::upper_bound(first, last, val);
其他类型的二分搜索

leetcode 第四题 median-of-two-sorted-arrays 是一道很经典的二分搜索 hard 题。这道题与今天被面试的这道题代表着两种不同的困难的方向：

上面这道题是，不一定能想到要用二分，想到之后依然需要接住其他的知识点来完成整道题
leetcode 这题就属于很明显就要二分，但是问题是怎么二分呢？

我理解二分的本质在于：找到区间的中点，判断这个中点是不是满足条件，如果不满足条件则选择某一半区间。例如在这道 leetcode 中，巧妙的构造了一个大区间[0, N2*2)，其实 (mid2-1)/2和mid2/2才是 nums2的一个划分，当我们确定了nums2的划分之后，nums1的划分也被确定了。我们要做的就是判断这个划分是不是合理的。也就是找到这个划分下，L1 <= R2 以及 L2 <= R1是否成立。因为 L1和 L2的数量就是 (N1+N2)/2。

这个解法的详情参考这个 discussion，很难读懂，我可能读懂了一遍又忘了。因为这道题目最麻烦的就是要针对奇数和偶数进行分类讨论。这个解法用一个很玄妙的方式规避了这个讨论，就是不管这个数组的长度是奇数还是偶数，我都按照 (nums[(mid-1)/2] + nums[mid/2])/2来解决这个问题。看起来就像，假如是奇数，我强行插入一个数让他变成偶数。

而最后更新mid2的方式则很简单，如果 mid2还不够大，那么 lo = mid2+1；如果 mid2太大了，那么 hi = mid2 - 1。

// 这么优雅的代码自然不是我写出来的，但是是我目前找到的最简介明了的了
class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
    int N1 = nums1.size();
    int N2 = nums2.size();
    if (N1 < N2) return findMedianSortedArrays(nums2, nums1);        // Make sure A2 is the shorter one.

    int lo = 0, hi = N2 * 2;
    while (lo <= hi) {
        int mid2 = (lo + hi) / 2;   // Try Cut 2 
        int mid1 = N1 + N2 - mid2;  // Calculate Cut 1 accordingly

        double L1 = (mid1 == 0) ? INT_MIN : nums1[(mid1-1)/2];        // Get L1, R1, L2, R2 respectively
        double L2 = (mid2 == 0) ? INT_MIN : nums2[(mid2-1)/2];
        double R1 = (mid1 == N1 * 2) ? INT_MAX : nums1[(mid1)/2];
        double R2 = (mid2 == N2 * 2) ? INT_MAX : nums2[(mid2)/2];

        if (L1 > R2) lo = mid2 + 1;                // A1's lower half is too big; need to move C1 left (C2 right)
        else if (L2 > R1) hi = mid2 - 1;        // A2's lower half too big; need to move C2 left.
        else return (max(L1,L2) + min(R1, R2)) / 2;        // Otherwise, that's the right cut.
    }
    return -1;
}
};

