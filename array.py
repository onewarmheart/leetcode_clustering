#%%
### 496. 下一个更大元素 I
## 单调栈
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # 遍历原数组，建栈，顺便存储索引
        c2idx = {}
        s = []
        res = [-1] * len(nums2)
        for i in range(len(nums2)-1, -1, -1):
            while len(s) and nums2[i] > s[-1]:
                s.pop()
            if len(s):
                res[i] = s[-1]
            s.append(nums2[i])
            c2idx[nums2[i]] = i
        # 遍历查找数组，添加解
        ans = []
        for num in nums1:
            ans.append(res[c2idx[num]])
        return ans


#%% 
### 503. Next Greater Element II
## 由上一题过来比较容易想到
class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        mx_idx = nums.index(max(nums))
        res = [-1] * len(nums)
        s = []
        for i in range(mx_idx, -1, -1):
            while len(s) and nums[i] >= s[-1]:
                s.pop()
            if len(s):
                res[i] = s[-1]
            s.append(nums[i])
        for i in range(len(nums)-1, mx_idx, -1):
            while len(s) and nums[i] >= s[-1]:
                s.pop()
            if len(s):
                res[i] = s[-1]
            s.append(nums[i])
        return res

#%%
### 31. 下一个排列

### 60. 排列序列


#%%
### 73. 矩阵置零
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        def cross(x, y):
            ## 遍历列
            for c in range(n):
                if matrix[x][c]: matrix[x][c] = '#'
            ## 遍历行
            for r in range(m):
                if matrix[r][y]: matrix[r][y] = '#'            
                
        m = len(matrix)
        if not m: return
        n = len(matrix[0])
        for i in range(m):
            for j in range(n):
                if not matrix[i][j]:
                    cross(i, j)
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '#':
                    matrix[i][j] = 0


#%%
### 134. 加油站

## 暴力
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        n = len(gas)
        for i in range(n):
            remain = gas[i]
            j = i
            while remain >= cost[j]:
                remain += gas[(j+1)%n] - cost[j]
                j = (j + 1) % n
                if j == i:
                    return i
        return -1

## 稍微改进(讲道理应该和下面一样)
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        n = len(gas)
        i = 0
        for i in range(n):
            remain = gas[i]
            j = i
            while remain >= cost[j]:
                remain += gas[(j+1)%n] - cost[j]
                j = (j + 1) % n
                if j == i:
                    return i
            ## 这句删了竟然不影响
            if j < i: break
            i = j
        return -1   

## 很快
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        n = len(gas)
        i = 0
        stop_idx = [-1] * n
        while True:
            if stop_idx[i] != -1: break
            remain = gas[i]
            j = i
            while remain >= cost[j]:
                remain += gas[(j+1)%n] - cost[j]
                j = (j + 1) % n
                if j == i:
                    return i
            stop_idx[i] = j
            ## 这一段可能可以不用
            # if j > i:
            #     stop_idx[i:j+1] = len(stop_idx[i:j+1]) * [0]
            # elif j < i:
            #     stop_idx[i:] = len(stop_idx[i:]) * [0]
            #     stop_idx[:j+1] = len(stop_idx[:j+1]) * [0]
            i = (j + 1) % n
        return -1     

#%%
### 150. 逆波兰表达式求值
## 遇到数字压栈，遇到字符出栈计算，算完再压栈
## python中 // 表示整除，向下取整
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stk = []
        nums = tokens
        symb_set = {'+', '-', '*', '/'}
        for c in nums:
            if c not in symb_set:
                stk.append(int(c))
            elif c == '+':
                nums1 = int(stk.pop())
                nums2 = int(stk.pop())
                stk.append(nums1 + nums2)
            elif c == '-':
                nums1 = int(stk.pop())
                nums2 = int(stk.pop())
                stk.append(nums2 - nums1)
            elif c == '*':
                nums1 = int(stk.pop())
                nums2 = int(stk.pop())
                stk.append(nums1 * nums2)
            elif c == '/':
                nums1 = int(stk.pop())
                nums2 = int(stk.pop())
                stk.append(int(nums2 / nums1))
        return stk[-1]

### 224. 基本计算器
## 击穿括号法
class Solution(object):
    def calculate(self, s):
        res = 0
        stk = [1]
        numb = 0
        sign = 1
        i = 0
        for c in s:
            if c == ' ':
                i+=1
            elif c == '(':
                stk.append(sign)
                i+=1
            elif c == ')':
                stk.pop()
                i+=1
            elif c == '+':
                sign = stk[-1]
                i+=1
            elif c == '-':
                sign = -stk[-1]
                i+=1
            else:
                # 数字的情况，可能有多位
                numb = 0
                while i < len(s) and s[i].isdigit():
                    numb = numb*10 + int(s[i])
                    i+=1
                res += numb * sign
        return res

## 模拟计算法
class Solution(object):
    def calculate(self, s):
        res, num, sign = 0, 0, 1
        stack = []
        for c in s:
            if c.isdigit():
                num = 10 * num + int(c)
            elif c == "+" or c == "-":
                res += sign * num
                num = 0
                sign = 1 if c == "+" else -1
            elif c == "(":
                stack.append(res)
                stack.append(sign)
                res = 0
                sign = 1
            elif c == ")":
                res += sign * num
                num = 0
                res *= stack.pop()
                res += stack.pop()
        res += sign * num
        return res



#%%
### 227. 基本计算器 II
## 乘除法后的数字计算好再压栈
class Solution(object):
    def calculate(self, s):
        res = 0
        stk = []
        numb = 0
        sign = 1
        i = 0
        while i < len(s):
            c = s[i]
            if c == ' ':
                i+=1
            elif c in ('*', '/', '+', '-'):
                stk.append(c)
                i+=1
            else:
                # 数字的情况，可能有多位
                numb = 0
                while i < len(s) and s[i].isdigit():
                    numb = numb*10 + int(s[i])
                    i+=1
                if not stk:
                    stk.append(numb)
                    continue
                if stk[-1] == '*':
                    stk.pop()
                    numb2 = stk.pop()
                    numb *= numb2
                elif stk[-1] == '/':
                    stk.pop()
                    numb2 = stk.pop()
                    numb = numb2 // numb
                stk.append(numb)
        sign = 1
        for c in stk:
            if c == '+':
                sign = 1
            elif c == '-':
                sign = -1
            else:
                res = res + sign * c
        return res

#%%
### 229. 求众数 II
## 摩尔投票法
## 当且仅当第三个候选人出现的时候，才进行票数抵消，否则都只是计票
class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        n = len(nums)
        if n < 2:
            return nums
        if n < 3:
            return [nums[1]] if nums[1] == nums[0] else nums
        # 初始化候选人及其票数
        cand1 = nums[0]; cnt1=0
        cand2 = nums[1]; cnt2=0
        # 计数
        for num in nums:
            if num == cand1:
                cnt1+=1
                continue
            if num == cand2:
                cnt2+=1
                continue
            if cnt1 <=0:
                cand1 = num
                cnt1 = 1
                continue
            if cnt2 <= 0:
                cand2 = num
                cnt2 = 1
                continue
            cnt1-=1
            cnt2-=1
        # 统计实际票数
        cnt1 = 0; cnt2 = 0
        for num in nums:
            if num == cand1:
                cnt1+=1
            elif num == cand2:
                cnt2 += 1
        # 确认是否大于1/3
        res = []
        if cnt1 > n//3: res.append(cand1)
        if cnt2 > n//3: res.append(cand2)
        return res

#%%
### 283. 移动零
### 双指针
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # 首0位置
        i = 0
        n = len(nums)
        while i < n:
            while i < n and nums[i] != 0:
                i+=1
            if i >= n:
                return
            j = i+1
            # i之后首个不为0的位置
            while j < n and nums[j] == 0:
                j+=1
            if j >= n:
                return
            nums[i] = nums[j]
            nums[j] = 0
            i+=1
## 更清晰的双指针, 
# 同样一个指针指向处理好的序列后的第一个
# 另一个指针指向待处理序列第一个
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        i = j = 0
        n = len(nums)
        while j < n:
            if nums[j] != 0:
                if nums[i] != nums[j]:
                    nums[i], nums[j] = nums[j], nums[i]
                i+=1
            j += 1

### 如果题目要求不用保留顺序，逆向考虑
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        i = n - 1
        while i > 0:
            # print(nums[i])
            if nums[i] == 0:
                i-=1
            for j in range(i - 1, -1, -1):
                if nums[j] == 0:
                    nums[j] = nums[i]
                    nums[i] = 0
                    break
                if j == 0:
                    return
#%%
### 560. 和为K的子数组
## 前缀和
# 为什么能拿到暴力n^2才能拿的数据？
# pre_sum表示以此位结尾的子数组之和
# 所有结尾在之前的子数组, 只记录在前缀和的频次里, 不做其他区分
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        # 记录某个前缀和出现的次数
        mp = {0: 1} # 初始值，从第一位开始的区间
        res = 0
        pre_sum = 0
        for num in nums:
            pre_sum += num
            pre = mp.get(pre_sum - k, 0)
            if pre:
                res+=pre
            if pre_sum in mp: 
                mp[pre_sum] += 1
            else: 
                mp[pre_sum] = 1
        return res

#%%
### 263. 丑数
### ugly_numb = 2^a*3^b*5^c, 且题目中说了必为正整数
class Solution:
    def isUgly(self, n: int) -> bool:
        if n <= 0: return False
        factors = [2,3,5]
        for factor in factors:
            while n % factor == 0:
                n //= factor
        return n == 1
#%%
### 264. 丑数 II
## 堆+哈希
from heapq import *
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        hp = [1]
        i = 0
        factors = [2,3,5]
        visited = {1}
        num = 1
        while i < n:
            num = heappop(hp)
            for factor in factors:
                if (new_num := num*factor) not in visited:
                    heappush(hp, new_num)
                    visited.add(new_num)
            i+=1
        return num
## 动态规划
# 下一个丑数是由前面的丑数转移得到的
# 这种迭代方式确保任何一个比目前最大数大的第一个丑数(下一个丑数)
# 因为每次每个丑数，都至少乘过一遍2,3,5, 不会遗漏
# 直接知道下一个最小的丑数，所以省去了堆排序的复杂度
# 注意有多个最小值的情况(只可能同时出现)，指针应该都+1，目的是去重
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        p2 = p3 = p5 = 0
        dp = [0] * n
        dp[0] = 1
        i = 1
        while i < n:
            # 下一个丑数是由前面的丑数转移得到的
            dp[i] = min(dp[p2] * 2, dp[p3] * 3, dp[p5] * 5)
            if dp[i] == dp[p2] * 2:
                p2 += 1
            if dp[i] == dp[p3] * 3:
                p3 += 1
            if dp[i] == dp[p5] * 5:
                p5 +=1
            i+=1
        return dp[n-1]
#%%
### 313. 超级丑数
## 堆+哈希，代码完全同上
class Solution:
    def nthSuperUglyNumber(self, n: int, primes: List[int]) -> int:
        hp = [1]
        i = 0
        factors = primes
        visited = {1}
        num = 1
        while i < n:
            num = heappop(hp)
            for factor in factors:
                if (new_num := num*factor) not in visited:
                    heappush(hp, new_num)
                    visited.add(new_num)
            i+=1
        return num

### 350. 两个数组的交集 II
## 方法1: 哈希表 O(m+n)
## 方法2: 排序过以后, 用双指针O(min(n,m))
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1.sort()
        nums2.sort()
        n = len(nums1)
        m = len(nums2)
        i = 0; j = 0
        res = []
        while i < n and j < m:
            if nums1[i] == nums2[j]:
                res.append(nums1[i])
                i+=1; j+=1
            elif nums1[i] < nums2[j]:
                i+=1
            elif nums1[i] > nums2[j]:
                j+=1
        return res
## 方法3: 如果较大数组的元素存储在磁盘上，内存是有限的，并且你不能一次加载所有的元素到内存中，应该采用方法1，因为可以并行分块读


### 386. 字典序排数
class Solution:
    def lexicalOrder(self, n: int) -> List[int]:
        l = [str(i) for i in range(1,n+1)]
        l.sort()
        l = [int(i) for i in l]
        return l

# TODO
### 406. 根据身高重建队列
## 自定义排序（身高从大到小，位置小的优先）+插入


### 315. 计算右侧小于当前元素的个数
## 离散化树状数组、前缀和
## https://leetcode-cn.com/problems/count-of-smaller-numbers-after-self/solution/ji-suan-you-ce-xiao-yu-dang-qian-yuan-su-de-ge-s-7/



### 448. 找到所有数组中消失的数字
## 利用数组本身作为哈希表，加上一个大于n的数--编码，取余还原--解码


### 581. 最短无序连续子数组
## 排序 o(nlogn)

## 找待排序子数组
## 这个算法背后的思想是无序子数组中最小元素的正确位置可以决定左边界，最大元素的正确位置可以决定右边界。
class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        maxn, right = float("-inf"), -1
        minn, left = float("inf"), -1

        for i in range(n):
            if maxn > nums[i]:
                right = i
            else:
                maxn = nums[i]
            
            if minn < nums[n - i - 1]:
                left = n - i - 1
            else:
                minn = nums[n - i - 1]
        
        return 0 if right == -1 else right - left + 1
