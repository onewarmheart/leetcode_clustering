
#%%
### 67. 二进制求和
## 依赖于python的高精度计算，因为还原的10进制可能会很大
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        x = int(a, 2); y = int(b, 2)
        # x 存放无进位相加结果
        # y 存放进位
        while y:
            answer = x ^ y
            carry = (x & y) << 1
            x, y = answer, carry
        return bin(x)[2:]

## 模拟加法的解法
## 和大整数加法类似，模拟进位
## 首部补0，或者两次反转

### 318. 最大单词长度乘积
## 最多O(N^2)
## 位掩码+预计算
class Solution:
    def maxProduct(self, words: List[str]) -> int:
        n = len(words)
        lens = [0] * n # idx -> lens
        masked = [0] * n # idx->masked numb
        for i, word in enumerate(words):
            tmp = 0
            for ch in word:
                pos = ord(ch) - ord('a')
                tmp |= (1 << pos)
            lens[i] = len(word)
            masked[i] = tmp

        # 好像没啥用
        # words.sort(key=lambda x: len(x), reverse=True)
        mx = 0
        for i in range(n):
            for j in range(i+1, n):
                if not (masked[i] & masked[j]):
                    mx = max(mx, lens[i] * lens[j])
        return mx

## 位掩码+哈希+预计算
class Solution:
    def maxProduct(self, words: List[str]) -> int:
        n = len(words)
        mp = collections.defaultdict(int) # masked numb -> lens
        for i, word in enumerate(words):
            tmp = 0
            for ch in word:
                pos = ord(ch) - ord('a')
                tmp |= (1 << pos)
            mp[tmp] = max(mp[tmp], len(word))

        # 好像没啥用
        # words.sort(key=lambda x: len(x), reverse=True)
        mx = 0
        for x in mp:
            for y in mp:
                if x == y:
                    continue
                if not x & y:
                    mx = max(mx, mp[x] * mp[y]) 
        return mx



#%%
### 338. 比特位计数
## 动态规划
# 数字n的一比特数，分情况，偶数就等于2//n的，奇数要+1，比如bits[1] = bits[0] + 1
class Solution:
    def countBits(self, n: int) -> List[int]:
        bits = [0]
        for i in range(1, n + 1):
            bits.append(bits[i >> 1] + (i & 1))
        return bits



### 461. 汉明距离
## Brian Kernighan 算法，相比于移位法快，只遍历1
## 时间复杂度：O(logC)，其中 C 是元素的数据范围，在本题中 log C=log 2^{31} = 31
class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        tmp = x ^ y
        res = 0
        while tmp > 0:
            tmp = tmp & (tmp -1)
            res += 1
        return res