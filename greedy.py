#%%
### 135. 分发糖果

## 暴力，超时
class Solution:
    def candy(self, ratings: List[int]) -> int:
        n = len(ratings)
        res = [1] * n
        for i in range(n):
            if i == 0:
                continue
            if ratings[i] > ratings[i-1]:
                res[i] = res[i-1] + 1
                continue
            j = i
            while j > 0 and ratings[j] < ratings[j-1]:
                if res[j] < res[j-1]:
                    break
                res[j-1] += 1
                j -= 1
        return sum(res)
## 两次遍历，减少重复计算
class Solution:
    def candy(self, ratings: List[int]) -> int:
        n = len(ratings)
        left = [1] * n
        res = [1] * n
        for i in range(n):
            if i > 0 and ratings[i] > ratings[i-1]:
                left[i] = left[i-1] + 1
        for j in range(n-1, -1, -1):
            if j < n -1 and ratings[j] > ratings[j+1]:
                res[j] = res[j+1] + 1
            res[j] = max(res[j], left[j])
        return sum(res)

#%%
### 在满足条件的范围内，merge最大的，
### 但是有相同的最大值时，不好处理
'''
[8, 9]
[3, 9]
3
'''
class Solution:
    def maxNumber(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        res = []
        idx = []
        i = 0; j = 0
        n = len(nums1)
        m = len(nums2)
        while k > 0 and (i < n and j < m):
            mn_left_1 = (m - j) - k + 1
            if mn_left_1 < 0:
                wait_nums1 = nums1[i:mn_left_1]
            else:
                wait_nums1 = nums1[i:]

            mn_left_2 = (n - i) - k + 1
            if mn_left_2 < 0:
                wait_nums2 = nums2[j:mn_left_2]
            else:
                wait_nums2 = nums2[j:]
            mx1 = -1; idx1 = -1
            mx2 = -1; idx2 = -1
            if wait_nums1:
                mx1 = max(wait_nums1)
                idx1 = i + wait_nums1.index(mx1)
            if wait_nums2:
                mx2 = max(wait_nums2)
                idx2 = j + wait_nums2.index(mx2)
            # print(mx1, "|", idx1, "|", k)
            # print(mx2, "|", idx2, "|", k)
            if mx1 >= mx2:
                res.append(mx1)
                i = idx1+1
            else:
                res.append(mx2)
                j = idx2+1
            k -= 1
        return res

## 遍历所有种长度可能性的解法, 合并思路和上面类似; 取n个最大的数，并按字典序返回用单调栈
## 我认为的复杂度：O(k∗2(M+N))， 博主O(k^2(M+N))
## https://leetcode-cn.com/problems/create-maximum-number/solution/yi-zhao-chi-bian-li-kou-si-dao-ti-ma-ma-zai-ye-b-7/
class Solution:
    def maxNumber(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        def pick_max(nums, k):
            stack = []
            drop = len(nums) - k
            for num in nums:
                while drop and stack and stack[-1] < num:
                    stack.pop()
                    drop -= 1
                stack.append(num)
            return stack[:k]

        def merge(A, B):
            ans = []
            while A or B:
                bigger = A if A > B else B
                ans.append(bigger.pop(0))
            return ans

        res = []
        for i in range(k+1):
            if i <= len(nums1) and k-i <= len(nums2):
                res.append(merge(pick_max(nums1, i), pick_max(nums2, k-i)))
        return max(res)
#%%
## 316. 去除重复字母
## 单调栈+哈希表；单调栈保证取出的子序列顺序
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        stack = []
        n = len(s)
        last_m = {}
        visited = set()
        # 记录最后一个结点的坐标
        for i in range(n):
            last_m[s[i]] = i
        
        for i in range(n):
            if s[i] in visited:
                continue
            # 单调栈，从栈底到栈顶，字母序增大
            while stack and s[i] < stack[-1]:
                top = stack[-1]
                # 栈顶元素如果后面还有的话，需要让位给遇到的小字母序的元素
                if last_m[top] > i:
                    stack.pop()
                    visited.remove(top)
                # 如果后面不再出现，为了保证顺序，该元素应该被保留，新遇到的小字母序的元素也要被加进来
                else: 
                    break
            visited.add(s[i])
            stack.append(s[i])
        return ''.join(stack)

## 贪心算法


#%%
### 330. 按要求补齐数组
### 贪心算法
class Solution:
    def minPatches(self, nums: List[int], n: int) -> int:
        patches = 0; x = 1
        i = 0
        while x <= n:
            # 边界条件注意下, 后者是<=，主要考虑x=1的情况
            if i < len(nums) and nums[i] <= x:
                x += nums[i]
                i += 1
                continue
            else:
                # 贪心，补充当前覆盖不到最小的数x
                x *= 2
                patches += 1
        return patches


#%%
### 376. 摆动序列
## 贪心算法，交错选山峰和谷底
## 画图，举例子，找规律
class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return n
        
        prediff = nums[1] - nums[0]
        res = 2 if prediff != 0 else 1
        for i in range(2, n):
            diff = nums[i] - nums[i-1]
            if (diff > 0 and prediff <= 0) or (diff < 0 and prediff >= 0):
                prediff = diff
                res += 1
        return res

## 动态规划
'''
up[i]表示前i个数字中，上升数组的最大长度
down[i]表示前i个数字中，下降数组的最大长度
最终结果res = max(up[n-1], down[n-1])

状态转移
up[i] = max(up[i-1], down[i-1]+1), nums[i] > nums[i-1]
up[i] = up[i-1], nums[i] <= nums[i-1]

down[i] = max(up[i-1]+1, down[i-1]), nums[i] < nums[i-1]
down[i] = down[i-1], nums[i] >= nums[i-1]
'''

class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return n
        firstdiff = nums[1] - nums[0]
        up = [0] * n; up[0] = 1; up[1] = (2 if firstdiff > 0 else 1)
        down = [0] * n; down[0] = 1; down[1] = (2 if firstdiff < 0 else 1)

        for i in range(2, n):
            if nums[i] > nums[i-1]:
                up[i] = max(up[i-1], down[i-1]+1)
                down[i] = down[i-1]
            elif nums[i] < nums[i-1]:
                down[i] = max(up[i-1]+1, down[i-1])
                up[i] = up[i-1]
            else:
                up[i] = up[i-1]
                down[i] = down[i-1]
        return max(up[n-1], down[n-1])

## 动态规划，缩减空间
#略





            
