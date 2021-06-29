
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

