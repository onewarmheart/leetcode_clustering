
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

