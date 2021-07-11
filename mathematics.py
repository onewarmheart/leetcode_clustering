
### 172. 阶乘后的零
## 计算有多少个5, 25, 125等
class Solution:
    def trailingZeroes(self, n: int) -> int:
        i = 1
        res = 0
        while n >= pow(5,i):
            res += n // (pow(5,i))
            i+=1
        return res


### 292. Nim 游戏
class Solution:
    def canWinNim(self, n: int) -> bool:
        return n%4!=0
        
### 319. 灯泡开关
# 只有因数的个数是奇数的时候才会被打开，
# 每个数，即使是质数也有1和本身两个因数，只有完全平凡数会是奇数个因数
# n里有多少个完全平方数？开个根号向下取整，比它小的就是了
import math
class Solution:
    def bulbSwitch(self, n: int) -> int:
        return int(math.sqrt(n))

### 50. Pow(x, n)
### 递归 快速幂
class Solution:
    def myPow(self, x: float, n: int) -> float:
        def dfs(x, n):
            if n == 0: return 1
            tmp = dfs(x, n >> 1)
            return tmp*tmp*x if n & 1 else tmp*tmp
        return 1/dfs(x, -n) if n < 0 else dfs(x, n)


### 2的幂
## 二进制中只有一位为1，具体做的时候找最后一个1
## 抹去最后一个1看是否为0
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        return n > 0 and (n & (n-1)) == 0
## 通过负数，即补码（反码+1），仅保留最后一个1，看是否和原数相等
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        return n > 0 and (n & -n) == n

### 3的幂
## 暴力法
## 注意n是整数，取值范围应该是[1,∞)
class Solution:
    def isPowerOfThree(self, n: int) -> bool:
        if n <= 0: return False
        resi = 0
        while n > 1:
            t = n // 3
            resi = n % 3
            if resi: return False
            n = t
        return True
## 某质数的幂的除数，只能是质数的幂（更低阶）
class Solution:
    def isPowerOfThree(self, n: int) -> bool:
        return n > 0 and 1162261467 % n == 0

### 4的幂
class Solution:
    def isPowerOfFour(self, n: int) -> bool:
        return n > 0 and (n & (n-1) == 0) and (n & 0xaaaaaaaa  == 0)


