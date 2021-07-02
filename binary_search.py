##74. 搜索二维矩阵
## 二分，复杂度O(n+m)
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m = len(matrix)
        if m == 0: return False
        n = len(matrix[0])
        i = 0; j = n-1
        while(i <= m-1 and j >= 0):
            # print(matrix[i][j])
            if target == matrix[i][j]:
                return True
            elif target > matrix[i][j]:
                i+=1
            else: j-=1
        return False

##240. 搜索二维矩阵 II
# 解法1:与上题二分查找矩阵方法一样


##167. 两数之和 II - 输入有序数组
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        n = len(numbers)
        res = [-1,-1]
        for i in range(n):
            res[0] = i+1
            # print(i,'|',target-numbers[i])
            flag = self.binary_search(i+1, n-1, numbers, target-numbers[i], res)
            if flag: return res
        return res
    def binary_search(self, i, j, numbers, sup, res):
        while(i <= j):
            mid = int((i + j)/2)
            if sup == numbers[mid]:
                res[1] = mid+1
                return True
            elif sup > numbers[mid]:
                i = mid + 1
            else: j = mid - 1
        return False

##167. 两数之和 II - 输入有序数组
#解法1:二分，o(nlogn)
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        n = len(numbers)
        for i in range(n):
            low, high = i + 1, n - 1
            while low <= high:
                mid = (low + high) // 2
                if numbers[mid] == target - numbers[i]:
                    return [i + 1, mid + 1]
                elif numbers[mid] > target - numbers[i]:
                    high = mid - 1
                else:
                    low = mid + 1

        return [-1, -1]
#解法2:双指针


## 350. 两个数组的交集 II
# 解法1: O(m+n)
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        if len(nums1) > len(nums2):
            return self.intersect(nums2, nums1) # 优化空间复杂度
        
        m = collections.Counter()
        for num in nums1:
            m[num] += 1
        
        intersection = list()
        for num in nums2:
            if (count := m.get(num, 0)) > 0:
                intersection.append(num)
                m[num] -= 1
                if m[num] == 0:
                    m.pop(num)
        
        return intersection
# 解法2: 排序+双指针 O(mlogm+nlogn)
# 解法3: 排序+二分查找 O(mlogm+nlogn)

#%%
### 4. 寻找两个正序数组的中位数


#%%
### 374. 猜数字大小
# The guess API is already defined for you.
# @param num, your guess
# @return -1 if my number is lower, 1 if my number is higher, otherwise return 0
# def guess(num: int) -> int:
class Solution:
    def guessNumber(self, n: int) -> int:
        l = 1; r = n
        while l < n:
            mid = (l+r)//2
            flag = guess(mid)
            if flag == 0:
                return mid
            elif flag == -1:
                r = mid - 1
            elif flag == 1:
                l = mid + 1
        return l