### 得物，快速排序, 快排
## 剑指offer版partition
#%%
nums = [0, 0, 2, 1, 0, 2,1]
def qs(nums):
    n = len(nums)

    def partition(l ,r):
        if l >= r:
            return r
        small = l-1
        for i in range(l, r):
            if nums[i] <= nums[r]:
                small += 1
                nums[small], nums[i] = nums[i], nums[small]
        small +=1
        nums[r], nums[small] = nums[small], nums[r]
        return small

    def quicksort(l, r):
        if l > r:
            return
        index = partition(l, r)
        # print(index)
        quicksort(l, index-1)
        quicksort(index+1, r)
        return

    quicksort(0, n-1)

    return
qs(nums)
print(nums)

### 微视
## 梯度下降、牛顿法求根号n
## 牛顿法
'''
x = x - f(x)/f'(x)
'''
#%%
def sqrt(n, prec):
    x = n
    y = 0.0
    while (abs(x-y)) > prec:
        y = x
        x = 0.5*(x + n / x)
    return x
#e.g.
print(sqrt(9, 0.001))
## 梯度下降
'''
x = x - alpha*loss'(x)
'''
# %%
def gradient_descent(n, prec):
    x = n
    alpha = 0.001  # 注意学习率不能太大，否则会震荡甚至发散，啥，不信 ？！陷入死循环之后你会改回来的
    deta = 1
    count = 1
    while abs(deta)>prec:
        deta = 4*x*(x**2-y)
        x -= alpha * deta
        count += 1
    return x,count
# ————————————————
# 版权声明：本文为CSDN博主「海晨威」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/songyunli1111/article/details/90474836
#e.g.
print(gradient_descent(9, 0.001))

#### 小红书2面
### 413. 等差数列划分
## dp很直观，也很优雅
# https://www.cnblogs.com/grandyang/p/5968340.html
class Solution {
public:
    int numberOfArithmeticSlices(vector<int>& A) {
        int res = 0, n = A.size();
        vector<int> dp(n, 0);
        for (int i = 2; i < n; ++i) {
            if (A[i] - A[i - 1] == A[i - 1] - A[i - 2]) {
                dp[i] = dp[i - 1] + 1;
            }
            res += dp[i];
        }
        return res;
    }
};


#### 小红书三面
### 543. 二叉树的直径

#### 与上面相似的一道题
### 124. 二叉树中的最大路径和
class Solution:
    def __init__(self):
        self.maxSum = float("-inf")

    def maxPathSum(self, root: TreeNode) -> int:
        def maxGain(node):
            if not node:
                return 0

            # 递归计算左右子节点的最大贡献值
            # 只有在最大贡献值大于 0 时，才会选取对应子节点
            leftGain = max(maxGain(node.left), 0)
            rightGain = max(maxGain(node.right), 0)
            
            # 节点的最大路径和取决于该节点的值与该节点的左右子节点的最大贡献值
            priceNewpath = node.val + leftGain + rightGain
            
            # 更新答案
            self.maxSum = max(self.maxSum, priceNewpath)
        
            # 返回节点的最大贡献值
            return node.val + max(leftGain, rightGain)
   
        maxGain(root)
        return self.maxSum
