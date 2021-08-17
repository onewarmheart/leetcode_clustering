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

### 二分搜索总结
## https://www.cnblogs.com/grandyang/p/6854825.html



### 378. 有序矩阵中第 K 小的元素
## TODO 怎么保证求出来的数在其中？
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        res = 0
        m = len(matrix)
        n = len(matrix[0])
        mn = matrix[0][0]
        mx = matrix[m-1][n-1]
        while True:
            i = 0; j = n - 1
            cnt = 0
            mid = (mn + mx)//2
            while 0 <= i < m and 0 <= j < n:
                if matrix[i][j] <= mid:
                    cnt += j+1
                    i += 1
                elif matrix[i][j] > mid:
                    j -= 1
            if cnt > k:
                mx = mid - 1
            elif cnt < k:
                mn = mid + 1
            else:
                res = mid
                break
        return res

## 刘冲改了一版，72 / 85通过，但还是有点问题
# 锯齿形切割，本质上是类似于排序数组中双指针的方式，利用单调特性，跳过一些无需遍历的东西
# 这样cnt得到的是包括第k小在内的，小于等于它的数，那排在前面的就是k-1，满足题意
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        res = 0
        m = len(matrix)
        n = len(matrix[0])
        mn = matrix[0][0]
        mx = matrix[m-1][n-1]
        while mn <= mx:
            i, j = m - 1, 0
            cnt = 0
            mid = (mn + mx)//2
            is_in = False
            while 0 <= i and j < n:
                if matrix[i][j] == mid:
                    is_in = True
                    cnt += i + 1
                    i -= 1
                    j += 1
                elif matrix[i][j] > mid:
                    i -= 1
                else:
                    cnt += i + 1
                    j += 1
            # print(cnt, mid)
            if cnt > k:
                mx = mid - 1
            elif cnt < k:
                mn = mid + 1
            elif cnt == k:
                if is_in:
                    return mid
                else:
                    mx = mid - 1
        return -1
## https://leetcode-cn.com/problems/kth-smallest-element-in-a-sorted-matrix/solution/you-xu-ju-zhen-zhong-di-kxiao-de-yuan-su-by-leetco/
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        n = len(matrix)

        def check(mid):
            i, j = n - 1, 0
            num = 0
            while i >= 0 and j < n:
                if matrix[i][j] <= mid:
                    num += i + 1
                    j += 1
                else:
                    i -= 1
            return num >= k

        left, right = matrix[0][0], matrix[-1][-1]
        while left < right:
            mid = (left + right) // 2
            if check(mid):
                right = mid
            else:
                left = mid + 1
        
        return left


##########二分查找消除边界疑惑
### 34. 在排序数组中查找元素的第一个和最后一个位置


## c++的内置函数
lower_bound(): 指向首个不小于 value 的元素的迭代器，或若找不到这种元素则为 last
upper_bound(): 指向首个大于 value 的元素的迭代器，或若找不到这种元素则为 last
https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/solution/shi-yong-lower_bound-he-upper_bound-han-shu-by-ste/



int lower_bound(vector<int>& nums, int target)
{
    int low = 0, high = nums.size();
    while(low < high)
    {
        int mid = low + (high - low >> 1);
        if(nums[mid] < target)
            low = mid + 1;
        else
            high = mid;
    }
    return low;
}

int upper_bound(vector<int>& nums, int target)
{
    int low = 0, high = nums.size();
    while(low < high)
    {
        int mid = low + (high - low >> 1);
        if(nums[mid] <= target) //其实就一个等号的区别
            low = mid + 1;
        else
            high = mid;
    }
    return low;
}

### 33. 搜索旋转排序数组
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int n = nums.size();
        int left = 0, right = n - 1;
        while(left <= right){
            int mid = left + (right - left) / 2;
            if(nums[mid] == target) return mid;
            if(nums[mid] >= nums[left]){ // =
                if(target >= nums[left] && target <= nums[mid]) right = mid - 1;
                else left = mid + 1;
            }
            else{
                if(target >= nums[mid] && target <= nums[right]) left = mid + 1;
                else right = mid - 1;                
            }
        }
        return -1;
    }
};
