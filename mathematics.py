
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

