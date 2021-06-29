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
