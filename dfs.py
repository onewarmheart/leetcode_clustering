### 31. 下一个排列
## dfs


### 842. 将数组拆分成斐波那契序列
#%%
## 递归
from typing import Collection


def dfs(start, s, res):
    if start >= len(s) and len(res) >= 3:
        return True
    flag = False
    for i in range(start, len(s)):
        cur_num = convert2int(s[start:i+1])

        if cur_num == -1 or cur_num > pow(2,31)-1:
             break
        if len(res) >= 2 and res[-2] + res[-1] < cur_num:
            break

        if  len(res) < 2 or res[-2] + res[-1] == cur_num:
            res.append(cur_num)
            flag = dfs(i+1, s, res)
            if flag: break
            res.pop(-1)
    return flag

def convert2int(s):
    tmp = 0
    if len(s)>1 and s[0] == '0':
        return -1

    for i in range(len(s)):
        # print(ord(s[i]) - ord('0'))
        tmp = tmp*10 + (ord(s[i]) - ord('0'))
    return tmp

class Solution:
    def splitIntoFibonacci(self, num: str) -> List[int]:
        res = []
        # print(convert2int('123'))
        flag = dfs(0, num, res)
        return res if flag else []

#%%
## 遍历解法
class Solution:
    def splitIntoFibonacci(self, S: str) -> List[int]:
        ans = list()

        def backtrack(index: int):
            if index == len(S):
                return len(ans) >= 3
            
            curr = 0
            for i in range(index, len(S)):
                if i > index and S[index] == "0":
                    break
                curr = curr * 10 + ord(S[i]) - ord("0")
                if curr > 2**31 - 1:
                    break
                
                if len(ans) < 2 or curr == ans[-2] + ans[-1]:
                    ans.append(curr)
                    if backtrack(i + 1):
                        return True
                    ans.pop()
                elif len(ans) > 2 and curr > ans[-2] + ans[-1]:
                    break
        
            return False
        
        backtrack(0)
        return ans

#%%
### 306. 累加数
## 题解同842



#%%
### 37数独
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        def dfs(pos: int):
            nonlocal valid
            if pos == len(spaces):
                valid = True
                return
            
            i, j = spaces[pos]
            for digit in range(9):
                if line[i][digit] == column[j][digit] == block[i // 3][j // 3][digit] == False:
                    line[i][digit] = column[j][digit] = block[i // 3][j // 3][digit] = True
                    board[i][j] = str(digit + 1)
                    dfs(pos + 1)
                    line[i][digit] = column[j][digit] = block[i // 3][j // 3][digit] = False
                if valid:
                    return
            
        line = [[False] * 9 for _ in range(9)]
        column = [[False] * 9 for _ in range(9)]
        block = [[[False] * 9 for _a in range(3)] for _b in range(3)]
        valid = False
        spaces = list()

        for i in range(9):
            for j in range(9):
                if board[i][j] == ".":
                    spaces.append((i, j))
                else:
                    digit = int(board[i][j]) - 1
                    line[i][digit] = column[j][digit] = block[i // 3][j // 3][digit] = True

        dfs(0)

## debug
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        # 递归填充
        def dfs(pos) -> bool:
            if pos >= len(spaces):
                print('bug')
                return True
            i, j = spaces[pos]
            print('pos:',pos,'value:',spaces[pos])
            # print(len(rows))
            # print(len(rows[i]))
            flag = False
            for x in range(9):
                # print(i,'|',j,'|',x)
                if rows[i][x] == cols[j][x] == blocks[i//3][j//3][x] == False:
                    rows[i][x] = cols[j][x] = blocks[i//3][j//3][x] = True
                    board[i][j] = str(x+1)
                    flag = dfs(pos+1)
                    # 剪枝
                    if flag: return flag
                    rows[i][x] = cols[j][x] = blocks[i//3][j//3][x] = False
            return flag

        # 初始化状态
        rows = [[False] * 9 for _ in range(9)]
        cols = [[False] * 9 for _ in range(9)]
        blocks =  [[[False] * 9 for _a in range(3)] for _b in range(3)]
        spaces = []
        # 记录已被填充和待填充的位置
        for i in range(9):
            for j in range(9):
                s = board[i][j]
                if s == '.':
                    spaces.append((i,j))
                else:
                    numb = int(s)-1
                    rows[i][numb] = cols[i][numb] = blocks[i//3][j//3][numb] = True
        # print(spaces)
        # print(len(spaces))
        dummy = dfs(0)

#%%
### 79. 单词搜索
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        m = len(board)
        n = len(board[0])
        visited = [[0] * n for _ in range(m)]
        # print(visited)

        def dfs(i, j, visited, depth, out):
            if visited[i][j] or depth > len(word):
                return False
            if word[depth-1] != out[depth-1]:
                return False
            if len(out) == len(word) and out[-1] == word[-1]:
                return True

            visited[i][j] = 1

            flag = False
            locations = ((0, 1),(0, -1),(1, 0),(-1,0))
            for p, q in (locations):
                x, y = i+p, j+q
                if x >= m or y >= n or x < 0 or y < 0:
                    continue
                flag = dfs(x, y, visited, depth+1, out+board[x][y])
                if flag: break

            visited[i][j] = 0
            return flag

        for i in range(m):
            res = False
            for j in range(n):
                res = dfs(i, j, visited, 1, board[i][j])
                if res: return res 
        return res

## 简化代码

# %%
### 212. 单词搜索 II
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
## %%  
        WORD_KEY = '$'
      
        # words = ["oath","pea","eat","rain","oaths"]
        trie = {}
        for word in words:
            node = trie
            for letter in word:
                # retrieve the next node; If not found, create a empty node.
                node = node.setdefault(letter, {})
            # mark the existence of a word in trie node
            node[WORD_KEY] = word
        # print(trie)
        # {'o': {'a': {'t': {'h': {'$': 'oath', 's': {'$': 'oaths'}}}}}, 'p': {'e': {'a': {'$': 'pea'}}}, 'e': {'a': {'t': {'$': 'eat'}}}, 'r': {'a': {'i': {'n': {'$': 'rain'}}}}}
           
        rowNum = len(board)
        colNum = len(board[0])
        
        matchedWords = []
        
        def backtracking(row, col, parent):    
            
            letter = board[row][col]
            if letter == '#': return
            currNode = parent[letter]
            
            # check if we find a match of word
            # 优化点3: 匹配到了单词就把该结尾单词节点删除
            word_match = currNode.pop(WORD_KEY, False)
            if word_match:
                # also we removed the matched word to avoid duplicates,
                #   as well as avoiding using set() for results.
                matchedWords.append(word_match)
            
            # Before the EXPLORATION, mark the cell as visited 
            board[row][col] = '#'
            
            # Explore the neighbors in 4 directions, i.e. up, right, down, left
            for (rowOffset, colOffset) in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                newRow, newCol = row + rowOffset, col + colOffset     
                if newRow < 0 or newRow >= rowNum or newCol < 0 or newCol >= colNum:
                    continue
                if not board[newRow][newCol] in currNode:
                    continue
                backtracking(newRow, newCol, currNode)
        
            # End of EXPLORATION, we restore the cell
            board[row][col] = letter


            # 优化点2: 如果是叶子节点，回溯的过程将其剪枝
            # Optimization: incrementally remove the matched leaf node in Trie.
            if not currNode: 
                parent.pop(letter)

        for row in range(rowNum):
            for col in range(colNum):
                # starting from each of the cells
                if board[row][col] in trie:
                    backtracking(row, col, trie)
        
        return matchedWords
#%%
### 357. 计算各个位数不同的数字个数
class Solution:
    def countNumbersWithUniqueDigits(self, n: int) -> int:
        if n == 0: return 1
        res = 0
        used = [ 0 for _ in range(10)]
        size = min(n, 10)
        def dfs(pos, used):
            nonlocal res
            if pos >= size:
                return
            for i in range(10):
                if pos == 0 and i == 0:
                    res+=1
                    continue
                if used[i] > 0:
                    continue
                else:
                    res += 1
                used[i] = 1
                dfs(pos+1, used)
                used[i] = 0
            return
        dfs(0, used)
        return res

## 数学解法

#%%
### 301. 删除无效的括号
class Solution:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        leftRemove, rightRemove = 0, 0
        for char in s:
            if char == '(':
                leftRemove += 1
            elif char == ')':
                if leftRemove > 0:
                    leftRemove -= 1
                else:
                    rightRemove += 1
        
        res = set()
        def dfs(idx, leftCount, rightCount, leftRemove, rightRemove, strs):
            if idx == len(s):
                if not leftRemove and not rightRemove:
                    res.add(strs)
                return
            # 添加字符
            if s[idx] == '(' and leftRemove:
                dfs(idx+1, leftCount, rightCount, leftRemove-1, rightRemove, strs)
            if s[idx] == ')' and rightRemove:
                dfs(idx+1, leftCount, rightCount, leftRemove, rightRemove-1, strs)
                
            # 不添加字符
            if s[idx] not in '()':
                dfs(idx+1, leftCount, rightCount, leftRemove, rightRemove, strs+s[idx])
            elif s[idx] == '(':
                dfs(idx+1, leftCount+1, rightCount, leftRemove, rightRemove, strs+s[idx])
            elif rightCount < leftCount:
                dfs(idx+1, leftCount, rightCount+1, leftRemove, rightRemove, strs+s[idx])
            return
        dfs(0, 0, 0, leftRemove, rightRemove, '')
        return list(res)


#%%
### 417. 太平洋大西洋水流问题

## 逆流题解
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        m = len(heights)
        n = len(heights[0])
        
        # 初始化两个中间结果
        taiping = [[ 0 for _a in range(n)] for _b in range(m)]
        taixi = [[ 0 for _a in range(n)] for _b in range(m)]

        
        def dfs(i , j, visited, sea):
            if visited[i][j]:
                return
            
            sea_matrix = taiping if sea else taixi
            sea_matrix[i][j] = 1

            visited[i][j] = 1

            directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            for di, dj in directions:
                newi, newj = i+di, j+dj
                if newi < 0 or newi > m -1 or newj > n - 1 or newj < 0:
                    continue
                if heights[newi][newj] >= heights[i][j]:
                    dfs(newi, newj, visited, sea)
            
            # 遍历过的不再遍历，不需要重置状态
            # visited[i][j] = 0
            return

        # 逆流   
        # 上方, 第一行
        visited = [[ 0 for _a in range(n)] for _b in range(m)]
        i = 0
        for j in range(n):
            dfs(i, j, visited, 1)
        # 左侧, 第一列
        visited = [[ 0 for _a in range(n)] for _b in range(m)]
        j = 0
        for i in range(m):
            dfs(i, j, visited, 1)
        # 下方, 最后一行
        visited = [[ 0 for _a in range(n)] for _b in range(m)]
        i = m-1
        for j in range(n):
            dfs(i, j, visited, 0)
        # 右侧, 最后一列
        visited = [[ 0 for _a in range(n)] for _b in range(m)]
        j = n-1
        for i in range(m):
            dfs(i, j, visited, 0)        
        
        # print('visited\n',visited)
        # print('taiping\n',taiping)
        # print('taixi\n',taixi)

        res = []
        for i in range(m):
            for j in range(n):
                if taiping[i][j] == 1 and taixi[i][j] == 1:
                    res.append([i,j])
        return res
## 省去visited数组
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        m,n=len(heights),len(heights[0])
        pos=[(0,1),(0,-1),(1,0),(-1,0)]

        def dfs(x,y,arr):
            arr[x][y]=1
            for dx,dy in pos:
                xx,yy=x+dx,y+dy
                if 0<=xx<m and 0<=yy<n and heights[xx][yy]>=heights[x][y] and not arr[xx][yy]:
                    dfs(xx,yy,arr)

        Pacific=[[0]*n for _ in range(m)]
        Atlantic=[[0]*n for _ in range(m)]

        for i in range(n):
            dfs(0,i,Pacific)
            dfs(m-1,i,Atlantic)
        for j in range(m):
            dfs(j,0,Pacific)
            dfs(j,n-1,Atlantic)

        return [[i,j] for i in range(m) for j in range(n) if Pacific[i][j] and Atlantic[i][j]]

作者：yim-6
链接：https://leetcode-cn.com/problems/pacific-atlantic-water-flow/solution/python3-ni-liu-er-shang-shun-liu-er-xia-yb7u0/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

# 顺流题解
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        m,n=len(heights),len(heights[0])
        pos=[(0,1),(0,-1),(1,0),(-1,0)]

        def dfs(x,y):
            if (x,y) in memo:
                return memo[(x,y)]
            visited[x][y]=1

            ans=0
            if x==0 or y==0:
                ans |= 2

            if x==m-1 or y==n-1:
                ans |= 1     
            
            for dx,dy in pos:
                xx,yy=x+dx,y+dy
                if 0<=xx<m and 0<=yy<n and heights[xx][yy]<=heights[x][y] and not visited[xx][yy]:
                    ans |= dfs(xx,yy)
            return ans            

        memo={}
        res=[]
        visited=[[0]*n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                visited=[[0]*n for _ in range(m)]
                ans = dfs(i,j)
                memo[(i,j)]=ans
                if ans==3:
                    res.append((i,j))                

        return res

作者：yim-6
链接：https://leetcode-cn.com/problems/pacific-atlantic-water-flow/solution/python3-ni-liu-er-shang-shun-liu-er-xia-yb7u0/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
## BFS题解



#%%
### 207. 课程表
### 210. 课程表II

## dfs解法:
# 两个关键点: 设计状态0,1,2，确定如何找环；终止条件，所有1阶邻居完成后该节点完成，并存入stack
import collections
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        # if  not prerequisites: return [0]
        actions = collections.defaultdict(list)
        for pres in prerequisites:
            actions[pres[1]].append(pres[0])
        stack = []
        visited = [0] * numCourses
        def dfs(u, visited):
            visited[u] = 1
            for v in actions[u]:
                if visited[v] == 0:
                    valid = dfs(v, visited)
                    if not valid: return valid
                elif visited[v] == 1:
                    return False
            visited[u] = 2
            stack.append(u)
            return True
        
        for u in range(numCourses):
            if visited[u] == 0:
                valid = dfs(u, visited)
                if not valid: return []

        return stack[::-1]
        
## bfs解法
# 终止条件，不存在没有入边的节点
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        # 存储有向图
        edges = collections.defaultdict(list)
        # 存储每个节点的入度
        indeg = [0] * numCourses
        # 存储答案
        result = list()

        for info in prerequisites:
            edges[info[1]].append(info[0])
            indeg[info[0]] += 1
        
        # 将所有入度为 0 的节点放入队列中
        q = collections.deque([u for u in range(numCourses) if indeg[u] == 0])

        while q:
            # 从队首取出一个节点
            u = q.popleft()
            # 放入答案中
            result.append(u)
            for v in edges[u]:
                indeg[v] -= 1
                # 如果相邻节点 v 的入度为 0，就可以选 v 对应的课程了
                if indeg[v] == 0:
                    q.append(v)

        if len(result) != numCourses:
            result = list()
        return result

#%%
## 解法1，trie树上的dfs，超时，不清楚为什么
class Solution:
    def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
        # 建前缀树
        trie = {}
        for word in words:
            node = trie
            for w in word:
                node = node.setdefault(w, {})
            node['#'] = '#'

        def dfs(word, idx, cnt, node):
            if len(word) == idx:
                if cnt >= 1 and "#" in node:
                    return True
                else:
                    return False
            # 遇到动作“#”
            if "#" in node:
                if dfs(word, idx, cnt+1, trie):
                    return True
            # 剪枝
            if word[idx] not in node:
                return False
            # 遇到动作结点
            if dfs(word, idx+1, cnt, node[word[idx]]):
                return True
            return False

        res = []
        for word in words:
            if dfs(word, 0, 0, trie):
                res.append(word)
        return res
## 同上，边建字典树边查询，超时，不清楚为什么
class Solution:
    def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
        trie = {}
        def dfs(word, idx, cnt, node):
            if len(word) == idx:
                if cnt >= 1 and "#" in node:
                    return True
                else:
                    return False
            # 遇到动作“#”
            if "#" in node:
                if dfs(word, idx, cnt+1, trie):
                    return True
            # 剪枝
            if word[idx] not in node:
                return False
            # 遇到动作结点
            if dfs(word, idx+1, cnt, node[word[idx]]):
                return True
            return False
            
        # 建前缀树
        words.sort(key=len)
        res = []
        for word in words:
            if dfs(word, 0, 0, trie):
                res.append(word)
            else:
                node = trie
                for w in word:
                    node = node.setdefault(w, {})
                node['#'] = '#'
        return res
        
## trie树dfs, 没有超时
class Solution:
    def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
        def check_word(word, pre_dict):
            if len(word) == 0:
                return True
            cur_dict = pre_dict
            for index, c in enumerate(word):
                cur_dict = cur_dict.get(c, None)
                if cur_dict is None:
                    return False
                if cur_dict.get('end', 0) == 1:
                    # 当前字符串前缀与树中单词匹配，递归搜索
                    if check_word(word[index+1:], pre_dict):
                        return True
            return False
        
        words.sort(key=lambda x: len(x))
        ans = []
        pre_dict = {}
        for item in words:
            if len(item) == 0:
                continue
            if check_word(item, pre_dict):
                ans.append(item)
            else:
                # insert word
                cur_dict = pre_dict
                for c in item:
                    if cur_dict.get(c, None) is None:
                        cur_dict[c] = {}
                    cur_dict = cur_dict.get(c)
                cur_dict['end'] = 1
        return ans
## 解法二 哈希表
class Solution:
    def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
        words.sort(key=len)
        min_len = max(1, len(words[0]))
        prev = set()
        res = []
 
        """
        方法1 动态规划方法判断
        def check(word, prev):
            if not prev: return False
            n = len(word)
            dp = [False] * (n + 1)
            dp[0] = True
            for i in range(1, n + 1):
                for j in range(i):
                    if not dp[j]: continue
                    if word[j:i] in prev:
                        dp[i] = True
                        break
            return dp[-1]
        """
        
        """
        # 方法2, DFS吧
        # def check(word):
        #     if not prev: return False
        #     if not word: return True
        #     for i in range(1, len(word) + 1):
        #         if word[:i] in prev and check(word[i:]):
        #             return True
        #     return False
        """
        # 方法3, 加了一个长度限制, 速度加快很多
        def check(word):
            if  word in prev: return True
            for i in range(min_len, len(word) - min_len + 1):
                if word[:i] in prev and check(word[i:]):
                    return True
            return False

        for word in words:
            if check(word):
                res.append(word)
            prev.add(word)
        return res
## 解法3 参考word break
##https://www.cnblogs.com/grandyang/p/6254527.html

#%%
### 282. 给表达式添加运算符
## dfs, 回溯
class Solution:
    def addOperators(self, num: str, target: int) -> List[str]:
        res = []
        n = len(num)
        self.out = ''
        def dfs(idx, cur_res, pre_add):
            if idx >= len(num) and cur_res == target:
                res.append(self.out)
            path_n = len(self.out)
            for i in range(idx, n):
                cur_str = num[idx:i+1]
                cur_num = int(cur_str)
                if idx == 0:
                    self.out = cur_str
                    dfs(i+1, cur_num, cur_num)
                    self.out = self.out[:path_n]
                else:
                    self.out = self.out + '+' + cur_str
                    dfs(i+1, cur_res + cur_num, cur_num)
                    self.out = self.out[:path_n]

                    self.out = self.out + '-' + cur_str
                    dfs(i+1, cur_res - cur_num, -cur_num)
                    self.out = self.out[:path_n]

                    self.out = self.out + '*' + cur_str
                    dfs(i+1, cur_res - pre_add + cur_num*pre_add, cur_num*pre_add)
                    self.out = self.out[:path_n]
                # 只允许第一位是0
                if cur_num == 0:
                    return
        dfs(0, 0, 0)
        return res

### 390. 消除游戏
## 编号映射 + 递归
## https://leetcode-cn.com/problems/elimination-game/solution/mei-ri-suan-fa-day-85-tu-jie-suan-fa-yi-xing-dai-m/
class Solution:
    def lastRemaining(self, n: int) -> int:
        if n == 1:
            return 1
        return 2 * (n // 2 + 1 - self.lastRemaining(n//2))


### 365. 水壶问题
## dfs, 递归实现 （递归层数可能超过python允许最大层数，届时使用栈来实现）
# 状态定义和终止条件、visit记录 很重要
class Solution:
    def canMeasureWater(self, x: int, y: int, z: int) -> bool:
        if x + y < z:
            return False
        visited = set()
        def dfs(resi_x, resi_y):
            if resi_x == z or resi_y == z or (resi_x + resi_y == z):
                return True
            if (resi_x, resi_y) in visited:
                return False
            visited.add((resi_x, resi_y))
            # 装满x
            if dfs(x, resi_y):
                return True
            # 装满y
            if dfs(resi_x, y):
                return True
            # 倒空x
            if dfs(0, resi_y):
                return True
            # 倒空y
            if dfs(resi_x, 0):
                return True
            # 从x向y直到倒空或倒满
            if dfs(resi_x - min(resi_x, y-resi_y), resi_y + min(resi_x, y-resi_y)):
                return True
            if dfs(resi_x + min(x-resi_x, resi_y), resi_y - min(x-resi_x, resi_y)):
                return True
            return False
        return dfs(0, 0)
## dfs, 栈实现
# 作者：LeetCode-Solution
# 链接：https://leetcode-cn.com/problems/water-and-jug-problem/solution/shui-hu-wen-ti-by-leetcode-solution/
class Solution:
    def canMeasureWater(self, x: int, y: int, z: int) -> bool:
        if x + y < z:
            return False
        stk = [[0,0]]
        visited = set()
        while stk:
            resi_x, resi_y = stk.pop()
            if (resi_x, resi_y) in visited:
                continue
            if resi_x == z or resi_y == z or (resi_x + resi_y == z):
                return True
            visited.add((resi_x, resi_y))
            # 装满x
            stk.append([x, resi_y])
            # 装满y 
            stk.append([resi_x, y])
            # 倒空x
            stk.append([0, resi_y])
            # 倒空y
            stk.append([resi_x, 0])
            # 从x向y直到倒空或倒满
            stk.append([resi_x - min(resi_x, y-resi_y), resi_y + min(resi_x, y-resi_y)])
            # 从y向x直到倒空或倒满
            stk.append([resi_x + min(x-resi_x, resi_y), resi_y - min(x-resi_x, resi_y)])
        return False

## 最大公约数
class Solution:
    def canMeasureWater(self, x: int, y: int, z: int) -> bool:
        if x + y < z:
            return False
        if x == 0 or y == 0:
            return z == 0 or x + y == z
        return z % math.gcd(x, y) == 0

