
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