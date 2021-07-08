#%% 
### 103. 二叉树的锯齿形层序遍历

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

## 类似bfs + 两个栈 
from typing import Deque


class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        s1 = [root]
        s2 = []
        res = []
        floor = 0

        while len(s1) != 0 or len(s2) != 0:
            s = s1 if len(s1) != 0 else s2
            s_ = s2 if s == s1 else s1
            out = []
            while len(s) > 0:
                tmp = s.pop(-1)
                if not tmp:
                    continue
                out.append(tmp.val)
                first, second = (tmp.right,tmp.left)  if floor % 2 != 0 else (tmp.left, tmp.right)
                if first: s_.append(first)
                if second: s_.append(second)
            floor += 1
            if out: res.append(out)
        return res

## bfs层次遍历+双端队列存储每层结果（实现奇数行结果逆序存储）
import collections
class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        res = []
        q =  [root] if root else []
        floor = 0
        lens = 1
        while len(q) != 0:
            deq = collections.deque()
            lens_ = 0
            while lens > 0:
                # print(q)
                tmp = q.pop(0)
                lens-=1
                # print(floor)
                if floor % 2 == 1:
                    deq.appendleft(tmp.val)
                else:
                    deq.append(tmp.val)
                    
                if tmp.left: 
                    q.append(tmp.left)
                    lens_+=1
                if tmp.right: 
                    q.append(tmp.right)
                    lens_+=1
            lens = lens_
            floor += 1
            res.append(list(deq))
        return res
            
#%%
### 107. 二叉树的层序遍历 II

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
import collections
class Solution:
    def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
        q = collections.deque()
        if root: q.append(root)
        res = []
        while len(q) != 0:
            size = len(q)
            out = []
            while size > 0:
                tmp = q.popleft()
                size-=1
                out.append(tmp.val)
                if tmp.left:
                    q.append(tmp.left)
                if tmp.right:
                    q.append(tmp.right)
            res.append(out)
        return res[::-1]


#%%
### 310. 最小高度树

## BFS解法，由外向内:借助记录度的map找到最外层，借助邻接表进行层次遍历
## 也称为拓扑排序解法，同 207. 课程表
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        if n == 1: return [0]
        res = []
        neighbor_map = collections.defaultdict(list)
        degree_map = collections.defaultdict(int)
        for edge in edges:
            degree_map[edge[0]] += 1
            degree_map[edge[1]] += 1
            neighbor_map[edge[0]].append(edge[1])
            neighbor_map[edge[1]].append(edge[0])
        
        q = collections.deque()
        for key in degree_map.keys():
            if degree_map[key] == 1:
                q.append(key)
        while len(q):
            size = len(q)
            res = []
            while size > 0:
                tmp = q.popleft()
                res.append(tmp)
                for neighbor in neighbor_map[tmp]:
                    degree_map[neighbor] -= 1
                    if degree_map[neighbor] == 1:
                        q.append(neighbor)
                size-=1
        return res

## 记忆化dfs
## 要保存两个节点拼起来的key，标识往这个方向搜索，之前想的单个结点的思路不对
## https://zhuanlan.zhihu.com/p/65366581
class Solution(object):
    def findMinHeightTrees(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: List[int]
        """
        
        if n == 0:
            return [0]
        if n < 0:
            return []
        
        treeNodes = [[] for _ in xrange(n)]
        treeHeights = [0 for _ in xrange(n)]
        minHeights = sys.maxint
        visited = {}
        
        for x, y in edges:
            treeNodes[x].append(y)
            treeNodes[y].append(x)
        
        
        def dfs(idx, treeNodes, preIdx):
            if str(preIdx) + '_' + str(idx) in visited:
                return visited[str(preIdx) + '_' + str(idx)]
            height = 0
            for nextIdx in treeNodes[idx]:
                if nextIdx != preIdx:
                    height = max([height, dfs(nextIdx, treeNodes, idx)])
            visited[str(preIdx) + '_' + str(idx)] = height + 1
            return height + 1
              
        for idx in range(n):
            treeHeights[idx] = dfs(idx, treeNodes, -1)
            minHeights = min([minHeights, treeHeights[idx]])

        rtn = []
        for idx in range(n):
            if treeHeights[idx] == minHeights:
                rtn.append(idx)
        
        return rtn

## 由终点开始的dfs (超时了)
## 类似417. 太平洋大西洋水流问题
## 每次选择根节点遍历会重置visited
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        from collections import defaultdict
        if not edges: return [0]
        graph = defaultdict(list)
        # 记录每个节点最高的高度
        lookup = [0] * n
        for x, y in edges:
            graph[x].append(y)
            graph[y].append(x)

        # print(graph)
        def dfs(i, visited, depth):
            lookup[i] = max(lookup[i], depth)
            for j in graph[i]:
                if j not in visited:
                    dfs(j, visited | {j}, depth + 1)

        leaves = [i for i in graph if len(graph[i]) == 1]
        for i in leaves:
            dfs(i, {i}, 1)
        min_num = min(lookup)
        return [i for i in range(n) if lookup[i] == min_num]


#%%
### 407. 接雨水 II
from heapq import *
class Solution:
    def trapRainWater(self, heightMap: List[List[int]]) -> int:
        """
        水从高出往低处流，某个位置储水量取决于四周最低高度，从最外层向里层包抄，用小顶堆动态找到未访问位置最小的高度
        """
        if not heightMap:return 0
        # imax = float('-inf')
        ans = 0
        heap = []
        visited = set()
        row = len(heightMap)
        col = len(heightMap[0])
        # 将最外层放入小顶堆
        # 第一行和最后一行
        for j in range(col):
            # 将该位置的高度、横纵坐标插入堆
            heappush(heap, [heightMap[0][j], 0, j])  
            heappush(heap, [heightMap[row - 1][j], row - 1, j])
            visited.add((0, j))
            visited.add((row - 1, j))
        # 第一列和最后一列
        for i in range(1, row-1):
            heappush(heap, [heightMap[i][0], i, 0])
            heappush(heap, [heightMap[i][col - 1], i, col - 1])
            visited.add((i, 0))
            visited.add((i, col - 1))
        while heap:
            h, i, j = heappop(heap)
            # 之前最低高度的四周已经探索过了，所以要更新为次低高度开始探索
            # 从堆顶元素出发，探索四周储水位置
            for x, y in [[0, 1], [1, 0], [0, -1], [-1, 0]]:
                tmp_x = x + i 
                tmp_y = y + j
                # 是否到达边界
                if tmp_x < 0 or tmp_y < 0 or tmp_x >= row or tmp_y >= col or (tmp_x, tmp_y) in visited:
                    continue
                visited.add((tmp_x, tmp_y))
                if heightMap[tmp_x][tmp_y] < h:
                    ans += h - heightMap[tmp_x][tmp_y]
                heappush(heap, [max(heightMap[tmp_x][tmp_y],h), tmp_x, tmp_y])
        return ans

### 42. 接雨水
## 暴力解法要清楚

## 简单解法，类似于动态规划
class Solution:
    def trap(self, height: List[int]) -> int:
        n = len(height)
        if n <= 2: return 0
        left = [0] * n
        right = [0] * n
        left[0] = height[0]
        right[n-1] = height[n-1]

        for i in range(1, n):
            left[i] = max(left[i-1], height[i-1])
        for i in range(n-2, -1,-1):
            right[i] = max(right[i+1], height[i+1])
        res = 0
        for i in range(1, n-1):
            res += max(min(left[i],right[i]) - height[i], 0)
        return res
## 简单解法升级版
class Solution {
public:
    int trap(vector<int>& height) {
        int l = 0, r = height.size() - 1, level = 0, res = 0;
        while (l < r) {
            int lower = height[(height[l] < height[r]) ? l++ : r--];
            level = max(level, lower);
            res += level - lower;
        }
        return res;
    }
};

## 单调栈
class Solution:
    def trap(self, height: List[int]) -> int:
        n = len(height)
        if n < 2: return 0
        stack = []
        res = 0
        for i, h in enumerate(height):
            while stack and h > height[stack[-1]]:
                top_idx = stack.pop()
                if not stack:
                    break
                last_idx = stack[-1]
                res += (min(height[last_idx], height[i]) - height[top_idx]) * (i - last_idx - 1)
            stack.append(i)
        return res


### 127. 单词接龙
## bfs
# 层数自增，遇到答案即返回
# 遇到在wordList的词，就从中删除，避免处理重复单词
from collections import deque
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        wordSet = set(wordList)
        q = deque([beginWord])
        res = 0
        alphabet = [chr(ord('a') + i) for i in range(26)]
        while len(q):
            layer_lens = len(q)
            for k in range(layer_lens, 0, -1):
                tmp_str = q.popleft()
                if tmp_str == endWord: return res + 1
                for i in range(len(tmp_str)):
                    cp_tmp_str = tmp_str 
                    for ch in alphabet:
                        cp_tmp_str = tmp_str[:i] + ch + tmp_str[i+1:]
                        if cp_tmp_str in wordSet and cp_tmp_str != tmp_str:
                            q.append(cp_tmp_str)
                            wordSet.remove(cp_tmp_str)
            res += 1
        return 0

