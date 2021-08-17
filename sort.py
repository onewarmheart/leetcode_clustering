#%%
### 164. 最大间距
## 桶排序
class Solution:
    def maximumGap(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2: return 0
        mx = max(nums)
        mn = min(nums)
        d = max(1,(mx - mn)//(n-1))
        # 灵魂+1 保证了即使最极端的情况，两个数，也可以落在两个桶，左闭右开
        bucket_size = (mx - mn)//d + 1

        # 记录每个桶内的最大、最小值
        out = [ [-1, -1] for _ in range(bucket_size)]
        j = 1
        for i in range(n):
            idx = (nums[i] - mn) // d
            if out[idx][0] == -1 or nums[i] < out[idx][0]:
                out[idx][0] = nums[i]
            if out[idx][1] == -1 or nums[i] > out[idx][1]:
                out[idx][1] = nums[i]
        res = 0
        pre_i = 0
        for i in range(bucket_size):
            # 空桶跳过
            if out[i][0] == -1:
                continue
            if i >= 1: 
                res = max(res, out[i][0] - out[pre_i][1])
            pre_i = i
        return res

## 基数排序
#%%
### 179. 最大数
class Solution:
    def largestNumber(self, nums: List[int]) -> str:
        nums_str = []
        for num in nums:
            nums_str.append(str(num))
        nums_str = sorted(nums_str, reverse = True)
        return "".join(nums_str)

#%%
### 220. 存在重复元素 III
## 桶排序+滑动窗口（哈希表存桶id，不是所有的桶都要存，i.e. 无空桶）
'''
[8,7,15,1,6,1,9,15]
1
3
'''
## 如果不算右侧的桶，上面case过不了
class Solution:
    def containsNearbyAlmostDuplicate(self, nums: List[int], k: int, t: int) -> bool:
        def getID(val, step):
            return (val+1)//step-1 if val < 0 else val//step

        n = len(nums)
        bin_map = {}
        for i in range(n):
            idx = getID(nums[i], t+1)
            if idx in bin_map:
                return True
            if idx+1 in bin_map and abs(nums[i] - bin_map[idx+1]) < t + 1:
                return True
            if idx-1 in bin_map and abs(nums[i] - bin_map[idx-1]) < t + 1:
                return True
            bin_map[idx] = nums[i]
            # 维护滑窗
            if i - k >= 0: bin_map.pop(getID(nums[i-k], t+1))
            
        return False
## 有序字典+二分查找
## 有bug
import collections
class Solution:
    def containsNearbyAlmostDuplicate(self, nums: List[int], k: int, t: int) -> bool:
        order_map = collections.OrderedDict()
        n = len(nums)
        # 不小于目标值的第一个元素
        def lower_bound(nums, target):
            '''
            return the target lower bound index in nums
            '''
            nums = list(nums)
            first, last = 0, len(nums)
            while first < last:
                mid = first + (last - first) // 2
                # 注意此处是小于号
                if nums[mid] < target:
                    first = mid + 1
                else:
                    last = mid
            return first

        for i in range(n):
            idx = lower_bound(order_map, nums[i] - t)
            if idx < len(order_map) and list(order_map)[idx] <= nums[i] + t:
                return True
            order_map[nums[i]] = i
            # 维护滑窗
            if i - k >= 0: order_map.pop(nums[i-k])
        return False

#%%
### 274. H 指数
## 排序(内置排序一般默认是堆排序, 空间o(1))
class Solution:
    def hIndex(self, citations: List[int]) -> int:
        citations.sort()
        h = -1
        n = len(citations)
        for i in range(n):
            tmp = min(citations[i], n-i)
            h = max(tmp, h)
        return h
## 计数排序
class Solution:
    def hIndex(self, citations: List[int]) -> int:
        n = len(citations)
        papers = [0] * (n+1)
        for i in range(n):
            papers[min(citations[i],n)] += 1
        for i in range(n, -1, -1):
            if i < n:
                 papers[i] += papers[i+1]
            if papers[i] >= i:
                return i
        return 0
#%%
### 295. 数据流的中位数
### 先假设规定正确，计算过程很简单，难点在于理解为什么如此规定
from heapq  import *
class MedianFinder:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.A = [] #小顶堆
        self.B = [] #大顶堆

    def addNum(self, num: int) -> None:
        if len(self.A) != len(self.B):
            heappush(self.A, num)
            heappush(self.B, -heappop(self.A))
        else:
            heappush(self.B, -num)
            heappush(self.A, -heappop(self.B))
        # print(self.A, self.B)
    def findMedian(self) -> float:
        if len(self.A) != len(self.B):
            return self.A[0]
        else:
            return (self.A[0] - self.B[0]) / 2.0

#%%
### 324. 摆动排序 II
## 排序，反序穿插
## 理解？
class Solution:
    def wiggleSort(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        nums.sort()
        cp = nums.copy()
        # (n + 1) // 2得到的是分割线右边的元素，属于右侧数组最后一个数
        l = (n + 1) // 2 - 1; r = n -1
        for i in range(n):
            # print(cp[l], cp[r])
            if i & 1:
                nums[i] = cp[r]
                r -= 1
            else:
                nums[i] = cp[l]
                l -= 1 
        return


##########堆排序问题集合
## 建堆 O(n) n为堆大小
# 从1开始编号，左子结点2*i，右子结点2*i+1，倒数第一个节点是n/2
#%%
def buildMaxHeap(nums):
    n = len(nums)
    def left(x):
        return 2*x
    def right(x):
        return 2*x+1
    def maxHeapify(nums, i):
        n = len(nums)
        largest = i
        if left(i) < n and nums[left(i)] > nums[i]:
            largest = left(i)
        if right(i) < n and nums[right[i]] > nums[largest]:
            largest = right(i)
        
        if largest != i:
            nums[largest], nums[i] = nums[i], nums[largest]
            maxHeapify(nums, largest)
        return

    for i in range(n//2, 0, -1):
        maxHeapify(nums, i)
nums = [4, 1, 3, 2, 16, 9, 10, 14, 8, 7]
buildMaxHeap(nums)
print(buildMaxHeap)
res = [16, 14, 10, 8, 7, 9, 3, 2, 4, 1]

## 以下两个建堆操作都是O(log(n))，完全二叉树的树高，因为都只需要滑滑梯一次
## 删除节点
def extract(nums):
    n = len(nums)
    mx = nums[1]
    nums[1] = nums[-1]
    nums.pop()
    maxHeapify(nums, 1)
    return mx
## 添加节点
def insert(nums, key):
    nums.append(key)
    n = len(nums)
    i = n
    while i > 1 and nums[i] > nums[i//2]:
        nums[i//2], nums[i] = nums[i], nums[i//2]
        i = i//2
    return 

### 75. 颜色分类
## 双指针，O(n) 一趟遍历， 空间O(1)
# p0和p1分别表示已排序的0后的第一个数，p1表示已排序后1的第一个数
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        n = len(nums)
        p0 = p1 = 0
        for i in range(n):
            if nums[i] == 0:
                nums[i], nums[p0] = nums[p0], nums[i]
                if p0 < p1:
                    nums[i], nums[p1] = nums[p1], nums[i]
                p1 += 1
                p0 += 1
            elif nums[i] == 1:
                nums[i], nums[p1] = nums[p1], nums[i]
                p1 += 1
        return

## 两趟的话，完全可以用字典了
# 0, 1, 2用vec，否则就用有序字典，或者真实的key存一个对应的数组





