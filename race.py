
### 5798. 循环轮转矩阵
class Solution:
    def rotateGrid(self, grid: List[List[int]], k: int) -> List[List[int]]:

        def rotate(nums, k):
            lens = len(nums)
            k = k % lens
            nums.extend(nums[:k])
            nums = nums[k:]
            return nums

        m = len(grid)
        n = len(grid[0])
        mn = min(m, n)
        max_layer = mn // 2 
        for i in range(max_layer):
            out = []
            # 顺时针
            for t in range(i, n-i):
                out.append(grid[i][t])

            for t in range(i+1, m-i):
                out.append(grid[t][n-i-1])  

            for t in range(n-i-2, i-1, -1):
                out.append(grid[m-i-1][t])     

            for t in range(m-i-2, i, -1):
                out.append(grid[t][i])
            # 旋转
            new_out = rotate(out, k)
            # 放回
            iter = 0
            for t in range(i, n-i):
                grid[i][t] = new_out[iter]
                iter+=1

            for t in range(i+1, m-i):
                grid[t][n-i-1] = new_out[iter]
                iter+=1

            for t in range(n-i-2, i-1, -1):
                grid[m-i-1][t] = new_out[iter]
                iter+=1

            for t in range(m-i-2, i, -1):
                grid[t][i] = new_out[iter]
                iter+=1
        return grid

### 5799. 最美子字符串的数目
## 暴力，超时
class Solution:
    def wonderfulSubstrings(self, word: str) -> int:
        n = len(word)
        res = 0
        for i in range(n):
            cur_sum = 0
            flag_m = [0] * 10
            for j in range(i, n):
                key = ord(word[j]) - ord('a')
                if flag_m[key] == 1:
                    flag_m[key] = 0
                    cur_sum -= 1 
                elif flag_m[key] == 0:
                    flag_m[key] = 1
                    cur_sum += 1                     
                if cur_sum > 1: continue 
                else: res+=1
        return res

## 前缀和（前缀哈希）
class Solution:
    def wonderfulSubstrings(self, word: str) -> int:
        n = len(word)
        mp = Counter([0]) # 初始值要注意
        pre_state = 0
        res = 0
        for c in word:
            pre_state ^= (1 << (ord(c) - ord('a')))
            if pre_state in mp:
                res += mp[pre_state]
            for i in range(0, 10, 1):
                if (pre_state_ := pre_state ^ (1 << i)) in mp:
                    res += mp[pre_state_]
            # print(bin(pre_state))
            mp[pre_state]+=1
        return res


### 5801. 消灭怪物的最大数量
# 错误做法，只考虑下一次的位置，不考虑到达时间
# import sys
class Solution:
    def eliminateMaximum(self, dist: List[int], speed: List[int]) -> int:
        n = len(dist)
        res = 0
        killed = set()
        while len(killed) < n:
            invaid = 0
            nearest = pow(10,5) + 1
            nearest_pos = -1
            for i in range(n):
                if i in killed: continue
                dist[i] = dist[i] - speed[i]
                if dist[i] < nearest:
                    nearest = dist[i]
                    nearest_pos = i
                if dist[i] <= 0:
                    invaid += 1
                if invaid >= 2:
                    return res + 1
            res += 1
            killed.add(nearest_pos)
        return res

# import sys
class Solution:
    def eliminateMaximum(self, dist: List[int], speed: List[int]) -> int:
        n = len(dist)
        res = 0
        time = [0] * n
        for i in range(n):
            time[i] = dist[i] / speed[i]
        time.sort()
        for i in range(n):
            if time[i] < i or abs(time[i] - i) <= 0.00000000001:
                return i
        return n

### 统计好数字的数目
## 快速幂
class Solution:
    def countGoodNumbers(self, n: int) -> int:
        def quickPower(x, n):
            if n <= 0:
                return 1
            tmp = quickPower(x, n >> 1)
            return tmp * tmp * x if n & 1 else tmp * tmp
        return quickPower(5, (n+1)//2) * quickPower(4, n//2)

### 1923. 最长公共子路径
# 作者：LeetCode-Solution
# 链接：https://leetcode-cn.com/problems/longest-common-subpath/solution/zui-chang-gong-gong-zi-lu-jing-by-leetco-ypip/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
class Solution:
    def longestCommonSubpath(self, n: int, paths: List[List[int]]) -> int:
        # 根据正文部分的分析，我们选择的 mod 需要远大于 10^10
        # Python 直接选择 10^9+7 * 10^9+9 作为模数即可
        # 乘积约为 10^18，远大于 10^10
        mod = (10**9 + 7) * (10**9 + 9)

        # 本题中数组元素的范围为 [0, 10^5]
        # 因此我们在 [10^6, 10^7] 的范围内随机选取进制 base
        base = random.randint(10**6, 10**7)
        
        m = len(paths)
        # 确定二分查找的上下界
        left, right, ans = 1, len(min(paths, key=lambda p: len(p))), 0
        while left <= right:
            length = (left + right) // 2
            mult = pow(base, length, mod)
            s = set()
            check = True

            for i in range(m):
                hashvalue = 0
                # 计算首个长度为 len 的子数组的哈希值
                for j in range(length):
                    hashvalue = (hashvalue * base + paths[i][j]) % mod

                t = set()
                # 如果我们遍历的是第 0 个数组，或者上一个数组的哈希表中包含该哈希值
                # 我们才会将哈希值加入当前数组的哈希表中
                if i == 0 or hashvalue in s:
                    t.add(hashvalue)
                # 递推计算后续子数组的哈希值
                for j in range(length, len(paths[i])):
                    hashvalue = (hashvalue * base - paths[i][j - length] * mult + paths[i][j]) % mod
                    if i == 0 or hashvalue in s:
                        t.add(hashvalue)
                if not t:
                    check = False
                    break
                s = t
            
            if check:
                ans = length
                left = length + 1
            else:
                right = length - 1
        
        return ans
