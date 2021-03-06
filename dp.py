#%%
### 44. 通配符匹配


#%%
### 91. 解码方法


#%% 97. 交错字符串
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        m = len(s1)
        n = len(s2)
        l = len(s3)
        if m+n != l: return False
        dp = [[ False for _a in range(n+1)] for _b in range(m+1) ]
        for i in range(0, m+1):
            for j in range(0, n+1):
                if i == 0 and j == 0: 
                    dp[i][j] = True
                    continue
                flag1 = False
                flag2 = False
                if i >= 1:
                    flag1 = dp[i-1][j] and s1[i-1] == s3[i+j-1]
                if j >= 1:
                    flag2 = dp[i][j-1] and s2[j-1] == s3[i+j-1]
                dp[i][j] = flag1 or flag2
        return dp[m][n]

## dfs解法
# https://leetcode-cn.com/problems/interleaving-string/solution/shou-hua-tu-jie-dfshui-su-dfsji-yi-hua-by-hyj8/

#%%
### 877. 石子游戏
## dp[i][j]实际上表示[i,j]范围内,先手比后手多拿的石子数  
class Solution:
    def stoneGame(self, piles: List[int]) -> bool:
        length = len(piles)
        dp = [[0] * length for _ in range(length)]
        for i, pile in enumerate(piles):
            dp[i][i] = pile
        for i in range(length - 2, -1, -1):
            for j in range(i + 1, length):
                dp[i][j] = max(piles[i] - dp[i + 1][j], piles[j] - dp[i][j - 1])
        return dp[0][length - 1] > 0

#%%
### 486. 预测赢家
## 动态规划，解法和877. 石子游戏一样，唯一不同在于可以平局，此时1胜利
class Solution:
    def PredictTheWinner(self, nums: List[int]) -> bool:
        length = len(nums)
        dp = [[0] * length for _ in range(length)]
        for i, num in enumerate(nums):
            dp[i][i] = num
        for i in range(length - 2, -1, -1):
            for j in range(i + 1, length):
                dp[i][j] = max(nums[i] - dp[i + 1][j], nums[j] - dp[i][j - 1])
        return dp[0][length - 1] >= 0
## dfs
# dfs仍然表示作为先手比对方多拿的分，不指代某个人
# 整个过程其实是模拟博弈过程
# 时间O(2^n) 空间O(n) 递归栈开销
class Solution:
    def PredictTheWinner(self, nums: List[int]) -> bool:
        def dfs(nums, l, r):
            if l > r:
                return 0
            choose_left = nums[l] - dfs(nums, l+1, r)
            choose_right = nums[r] - dfs(nums, l, r-1)
            return max(choose_left, choose_right)
        return dfs(nums, 0, len(nums) - 1) >= 0

### 375. 猜数字大小 II
## 英文题干：Given a particular n,
#  return the minimum amount of money you need to guarantee a win regardless of what number I pick.
# 采用最差的策略，也能至少赢一次的代价

## dfs, 超时
# dfs(l,r)表示我们能求得从l开始猜到r，猜到答案的最小代价
from abc import ABCMeta
import sys
class Solution:
    def getMoneyAmount(self, n: int) -> int:
        def dfs(l, r):
            if l >= r:
                return 0
            mn = sys.maxsize
            for i in range(l, r+1):
                cost = i + max(dfs(l, i-1), dfs(i+1, r))
                mn = min(mn, cost)
            return mn
        return dfs(1, n)
## 记忆化搜索, 勉强能过
import sys
class Solution:
    def getMoneyAmount(self, n: int) -> int:
        # python可以元组作为key，其实也可以用二阶n x n矩阵
        memo = {}
        def dfs(l, r):
            if l >= r:
                return 0
            if (l, r) in memo:
                return memo[(l,r)]
            mn = sys.maxsize
            for i in range(l, r+1):
                cost = i + max(dfs(l, i-1), dfs(i+1, r))
                mn = min(mn, cost)
            memo[(l,r)] = mn
            return mn
        return dfs(1, n)
## dp
public class Solution {
    public int getMoneyAmount(int n) {
        int[][] dp = new int[n + 1][n + 1];
        for (int len = 2; len <= n; len++) {
            for (int start = 1; start <= n - len + 1; start++) {
                int minres = Integer.MAX_VALUE;
                for (int piv = start; piv < start + len - 1; piv++) {
                    int res = piv + Math.max(dp[start][piv - 1], dp[piv + 1][start + len - 1]);
                    minres = Math.min(res, minres);
                }
                dp[start][start + len - 1] = minres;
            }
        }
        return dp[1][n];
    }
}
### 最长不重复子数组

### 718. 最长重复子数组
## 二维dp
class Solution:
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        m = len(nums1)
        n = len(nums2)
        dp = [[0 for j in range(n)] for i in range(m)]
        res = 0
        for i in range(m):
            for j in range(n):
                tmp = 0 if i < 1 or j < 1 else dp[i-1][j-1]
                if nums1[i] == nums2[j]:
                    dp[i][j] = tmp + 1
                res = max(res, dp[i][j])
        return res

### 滑动窗口

## 二分查找 + 哈希
### 难度比较大


### 334. 递增的三元子序列
## 三指针法O(n^3)
## 优化版暴力 O(n^2)
class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        n = len(nums)
        for i in range(n):
            left_min = False
            for j in range(i):
                if nums[j] < nums[i]:
                    left_min = True
                    break
            right_max = False
            for k in range(i+1, n):
                if nums[k] > nums[i]:
                    right_max = True
                    break
            if left_min and right_max:
                return True
        return False
## 记忆数组O(n), 在暴力法基础上优化
class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        n = len(nums)
        left = [nums[0]]*n
        right = [nums[-1]]*n
        for i in range(n):
            if i >= 1: 
                left[i] = min(left[i-1], nums[i])
        for j in range(n-1, -1, -1):
            if j < n-1: 
                right[j] = max(right[j+1], nums[j])
        for k in range(n):
            if left[k] < nums[k] and right[k] > nums[k]:
                return True
        return False
## dp
# dp[i] 表示 < 第i个数的的个数，初始化的时候包括了本身
class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        n = len(nums)
        dp = [1] * n
        for i in range(n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[j]+1, dp[i])
                if dp[i] >= 3:
                    return True
        return False


## 贪心算法
class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        n = len(nums)
        mn = 2147483648 + 1
        mid = 2147483648 + 1
        for i in range(n):
            if nums[i] < mn:
                mn = nums[i]
            elif nums[i] > mn and nums[i] < mid:
                mid = nums[i]
            elif nums[i] > mid:
                return True
        return False

##########背包问题集合
### 322. 零钱兑换
## leetcode上背包问题总结
## https://leetcode-cn.com/problems/coin-change/solution/yi-pian-wen-zhang-chi-tou-bei-bao-wen-ti-sq9n/
## 本题属于完全背包问题（每个），且每个元素可以重复选择

## 一维dp
# dp表示拼成（正好拼成）目标金额，至少需要的硬币数（恰装满背包目标承重，最少需要的物品数量）
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [sys.maxsize] * (amount+1)
        dp[0] = 0
        for i in range(1, amount+1):
            for c in coins:
                if i - c >= 0:
                    dp[i] = min(dp[i], dp[i-c] + 1)
        return dp[amount] if dp[amount] < sys.maxsize else -1

### 518. 零钱兑换 II
## 二维dp
# 注意边界条件，且遍历不要覆盖边界条件
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        n = len(coins)
        dp = [[0 for _a in range(amount+1)] for _b in range(n+1)]
        for i in range(1, n + 1):
            dp[i][0] = 1
            for j in range(1, amount +1):
                if i >= 1 and j - coins[i-1] >= 0:
                    dp[i][j] = dp[i][j - coins[i-1]] + dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j]
        return dp[n][amount]

## 二维dp，压缩空间
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        n = len(coins)
        dp = [0] * (amount+1)
        dp[0] = 1
        for i in range(1, n + 1):
            for j in range(1, amount +1):
                if i >= 1 and j - coins[i-1] >= 0:
                    dp[j] = dp[j] + dp[j - coins[i-1]]
                else:
                    dp[j] = dp[j]
        return dp[amount]
### 416. 分割等和子集
## 还是先写二维，状态压缩不用着急
'''
dp[i][j] = dp[i-1][j] || dp[i-1][j-nums[i]]
'''
##根据状态转移判断压缩空间的时候，内层应该倒序
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        n = len(nums)
        sumation = sum(nums)
        if sumation & 1 != 0:
            return False
        target = sumation // 2
        dp = [False] * (target+1)
        dp[0] =True
        for i in range(1, n+1):
            for j in range(target, 0, -1):
                if j - nums[i-1] >= 0:
                    dp[j] = dp[j] or dp[j-nums[i-1]]
        return dp[target]
### 698. 划分为k个相等的子集

## dfs, 排序剪枝 （不这样做会慢很多）
# 注意找出一个满足条件的组合以后，start也要归零，visit数组
class Solution:
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        sumation = sum(nums)
        if sumation % k != 0:
            return False
        visited = [0] * n
        target = sumation // k
        nums.sort(reverse=True)
        def dfs(start, out, k):
            if k == 1:
                return True
            if out > target:
                return False
            if out == target:
                return dfs(0, 0, k-1)
            for i in range(start, n):
                if visited[i]:
                    continue
                visited[i] = 1
                flag = dfs(i+1, out + nums[i], k)
                if flag: return flag
                visited[i] = 0
            return False
        return dfs(0, 0, k)

### 279. 完全平方数
class Solution {
public:
    int numSquares(int n) {
        vector<int> dp(n+1,INT_MAX);
        dp[0] = 0;
        for(int i = 1; i <= n; i++){
            for(int j = 1; i - j*j >= 0; j++){
                dp[i] = min(dp[i], dp[i-j*j] + 1);
            }
        }
        return dp[n];
    }
};

##########股票买卖问题集合
## https://labuladong.gitbook.io/algo/mu-lu-ye/tuan-mie-gu-piao-wen-ti 
## 按这篇文章的思路，不难
## 注意：1.边界条件 2.优化空间的方式


##########子串子序列问题集合


### 337. 打家劫舍 III
class Solution:
    def rob(self, root: TreeNode) -> int:
        def _rob(root):
            if not root: return 0, 0
            
            ls, ln = _rob(root.left)
            rs, rn = _rob(root.right)
            
            return root.val + ln + rn, max(ls, ln) + max(rs, rn)

        return max(_rob(root))




########## 最大矩形、最大正方形集合
### 85. 最大矩形
## 单调栈  TODO
## 暴力，三层遍历，有点类似最大正方形

### 84. 柱状图中最大的矩形

