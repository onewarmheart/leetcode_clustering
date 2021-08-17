### 剑指 Offer 03. 数组中重复的数字
class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
        n = len(nums)
        i = 0
        for i in range(n):
            while i != nums[i]:
                if nums[i] == nums[nums[i]]:
                    return nums[i]
                nums[nums[i]], nums[i] = nums[i], nums[nums[i]]
        return -1

### 剑指 Offer 04. 二维数组中的查找
class Solution:
    def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
        m = len(matrix)
        if m == 0: return False
        n = len(matrix[0])
        i = 0; j = n-1
        while i < m and j >= 0:
            if matrix[i][j] == target:
                return True
            elif matrix[i][j] > target:
                j -= 1
            else:
                i += 1
        return False
### 剑指 Offer 05. 替换空格
class Solution:
    def replaceSpace(self, s: str) -> str:
        ss = ""
        for c in reversed(s):
            if c != ' ':
                ss += c
            else:
                ss += '02%'
        ss = list(reversed(ss))
        return ''.join(ss)

### 剑指 Offer 06. 从尾到头打印链表
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

## 从头到尾打印再反转
# 链接：https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/solution/python3-c-by-z1m/
class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        res = []
        while head:
            res.append(head.val)
            head = head.next
        return res[::-1]  # 或者 reverse(res)

## 递归1
class Solution:
    def __int__(self):
        self.new_head = None

    def reverseList(self, node):
        if not node:
            self.new_head = ListNode(-1)
            return self.new_head
        p = self.reverseList(node.next)
        p.next = node
        return node

    def reversePrint(self, head: ListNode) -> List[int]:
        res = []
        tail = self.reverseList(head)
        tail.next = None
        p = self.new_head.next
        while p:
            res.append(p.val)
            p = p.next
        return res
## 递归2
# 链接：https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/solution/python3-c-by-z1m/
class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        if not head: return []
        return self.reversePrint(head.next) + [head.val]

### 剑指 Offer 07. 重建二叉树
### 105. 从前序与中序遍历序列构造二叉树
# 以下pre_idx改为root_idx_in_pre比较好理解
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def __init__(self):
        self.preorder = []
        self.inorder = []
        self.lens = 0
    def dfs(self, l, r, pre_idx):
        if pre_idx > self.lens - 1 or l >= r:
            return None
        root = TreeNode(self.preorder[pre_idx])
        for i, x in enumerate(self.inorder[l:r]):
            if x == self.preorder[pre_idx]:
                root.left = self.dfs(l, l+i, pre_idx+1)
                root.right = self.dfs(l+i+1, r, pre_idx+i+1)
                break
        return root

    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        self.preorder = preorder
        self.inorder = inorder
        self.lens = len(preorder)
        return self.dfs(0, self.lens, 0)

### 剑指 Offer 09. 用两个栈实现队列
class CQueue:

    def __init__(self):
        self.s1 = []
        self.s2 = []


    def appendTail(self, value: int) -> None:
        self.s1.append(value)

    def deleteHead(self) -> int:
        if self.s2:
            return self.s2.pop()
        while self.s1:
            self.s2.append(self.s1.pop())
        if self.s2:
            return self.s2.pop()
        return -1

### 剑指 Offer 10- I. 斐波那契数列
class Solution:
    def fib(self, n: int) -> int:
        if n == 0: return 0
        f = [0] * (n+1)
        f[1] = 1
        for i in range(2, n+1):
            f[i] = f[i-1] + f[i-2]
        return f[n] % 1000000007

### 剑指 Offer 10- II. 青蛙跳台阶问题
class Solution:
    def numWays(self, n: int) -> int:
        if n == 0: return 1
        f = [1] * (n+1)
        if n >= 2:
            f[2] = 2
        for i in range(2, n+1):
            f[i] = f[i-1] + f[i-2]
        return f[n] % 1000000007

### 剑指 Offer 11. 旋转数组的最小数字
## 区间闭合由j决定；停止条件和返回值相关；mid+1或-1, 看是否要保留mid
## 上位中位数：first + (length)//2 下位中位数：first + (length+1)//2
class Solution:
    def minArray(self, numbers: List[int]) -> int:
        n = len(numbers)
        i = 0, j = n -1
        while i < j:
            mid = i+(j-i)//2
            if numbers[mid] < numbers[j]:
                j = mid
            elif numbers[mid] > numbers[j]:
                i = mid + 1
            else:
                j -= 1
        return numbers[j]

### 剑指 Offer 12. 矩阵中的路径
## TODO 有4个case没有通过
## 时间复杂度
[["A","B","C","E"],["S","F","E","S"],["A","D","E","E"]]
"ABCESEEEFS"
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        def dfs(i, j, pos):
            if i >= m or i < 0 or j >= n or j < 0:
                return False
            if word[pos] != board[i][j] or visited[i][j] == 1:
                return False
            if pos >= len(word) - 1:
                print("pos:", pos, "word:", len(word))
                return True
            visited[i][j] = 1
            directions = ((1,0), (0,1), (-1,0), (0,-1))
            for di, dj in directions:
                new_i, new_j = di + i, dj + j
                if dfs(new_i, new_j, pos+1):
                    return True
            return False
        
        m = len(board)
        n = len(board[0])
        for i in range(m):
            for j in range(n):
                visited = [[0 for _a in range(n)] for _b in range(m)]
                if dfs(i, j, 0):
                    return True
        return False

### 剑指 Offer 13. 机器人的运动范围
class Solution:
    def movingCount(self, m: int, n: int, k: int) -> int:
        def calcu_digit(i, j):
            res_i = 0
            while i:
                res_i += i % 10
                i //= 10
            res_j = 0
            while j:
                res_j += j % 10
                j //= 10
            return res_i + res_j

        def dfs(i, j):
            nonlocal res
            if i >= m or i < 0 or j >= n or j < 0:
                return
            if visited[i][j] == 1 or calcu_digit(i,j) > k:
                return
            res += 1
            visited[i][j] = 1
            directions = ((1,0), (0,1), (-1,0), (0,-1))
            for di, dj in directions:
                new_i, new_j = di + i, dj + j
                dfs(new_i, new_j)
            return
        res = 0
        visited = [[0 for _a in range(n)] for _b in range(m)]
        dfs(0, 0)
        return res


### 剑指 Offer 15. 二进制中1的个数
class Solution:
    def hammingWeight(self, n: int) -> int:
        k = 1
        res = 0
        for i in range(32):
            if n & k:
                res += 1
            k <<= 1
        return res


### 剑指 Offer 16. 数值的整数次方
## 快速幂，注意指数为负数的情况
## python中整除//表示向下取整， 和int()强制转换不一样
##取余，遵循尽可能让商向0靠近的原则
## 取模，遵循尽可能让商向负无穷靠近的原则(python中%是取模长，和整除一样，记忆成向下就行)
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n == 0:
            return 1
        if n < 0:
            return 1/self.myPow(x, -n)
        tmp = self.myPow(x, n//2)
        return tmp*tmp*x if n & 1 else tmp*tmp

### 剑指 Offer 14- I. 剪绳子
## 注意当lens作为更长绳子的组成部分，可以取本身
class Solution:
    def cuttingRope(self, n: int) -> int:
        dp = [0] * (n+1)
        dp[1] = 1
        res = 0
        for lens in range(1, n+1):
            for j in range(1, lens//2+1):
                dp[lens] = max(dp[lens], max(dp[j], j)*max(dp[lens-j], lens-j))
        return dp[n]

### 剑指 Offer 14- II. 剪绳子 II
## 动态规划的过程中进行求模不可以，影响后续迭代结果
## 只能用贪心（归纳法）
class Solution:
    def cuttingRope(self, n: int) -> int:
        if n <= 3:
            return n-1
        res = 1
        base = 1000000007
        while n > 4:
            n -= 3
            res *= 3
            res %= base
        return (res * n) % base

### 剑指 Offer 18. 删除链表的节点
## 双指针
# 善用辅助作用的前置头指针，注意边界情况（初始化细致）
class Solution:
    def deleteNode(self, head: ListNode, val: int) -> ListNode:
        if not head: return None
        pre_head = ListNode(-1)
        pre_head.next = head
        p2 = pre_head
        p1 = head
        while p1:
            if p1.val == val:
                p2.next = p1.next
                p1.next = None
                break
            p1 = p1.next
            p2 = p2.next
        return pre_head.next

### 剑指 Offer 21. 调整数组顺序使奇数位于偶数前面
# 双指针j永远维护在已调整的最后一个
class Solution:
    def exchange(self, nums: List[int]) -> List[int]:
        n = len(nums)
        i = 0; j =-1
        for i in range(n):
            if nums[i] & 1:
                nums[j+1], nums[i] = nums[i], nums[j+1]
                j+=1
        return nums

### 剑指 Offer 22. 链表中倒数第k个节点
class Solution:
    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
        p1 = p2 = head
        for i in range(k):
            p1 = p1.next
        while p1:
            p1 = p1.next
            p2 = p2.next
        return p2

### 剑指 Offer 24. 反转链表
class Solution:
    def __init__(self):
        self.new_head = None
        
    def reverseList(self, head: ListNode) -> ListNode:
        def helper(head):
            if not head:
                self.new_head = ListNode(-1)
                return self.new_head
            p = helper(head.next)
            p.next = head
            head.next = None
            return head
        p = helper(head)
        return self.new_head.next

### 剑指 Offer 25. 合并两个排序的链表
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        pre_head = ListNode(-1)
        p = pre_head
        while l1 and l2:
            if l1.val < l2.val:
                tmp = l1
                l1 = l1.next
                p.next = tmp
                p = p.next
            else:
                tmp = l2
                l2 = l2.next
                p.next = tmp
                p = p.next
        if l1:
            p.next = l1
        else:
            p.next = l2
        return  pre_head.next

### 剑指 Offer 26. 树的子结构
## helper用于确定了顶点后，同步遍历，比较子结构
## 主函数用于做前序遍历，确定是否有相同顶点
class Solution:
    def helper(self, root, target):
        if not target:
            return True
        if not root:
            return False
        cur = (root.val == target.val)
        left = self.helper(root.left, target.left)
        right = self.helper(root.right, target.right)
        return cur and left and right

    def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:
        if not B or not A:
            return False
        res = False
        if A.val == B.val:
            res = res or self.helper(A, B)
        return res or self.isSubStructure(A.left, B) or self.isSubStructure(A.right, B)


### 剑指 Offer 27. 二叉树的镜像
class Solution:
    def mirrorTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return root
        new_root = TreeNode(root.val)
        new_root.right = self.mirrorTree(root.left)
        new_root.left = self.mirrorTree(root.right)
        return new_root

### 剑指 Offer 28. 对称的二叉树
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        def helper(a, b):
            if not a and not b:
                return True
            if not a or not b:
                return False
            mid = False
            if a.val == b.val:
                mid = True
            left = helper(a.left, b.right)
            right = helper(a.right, b.left)
            return mid and left and right
        return helper(root, root)

### 剑指 Offer 29. 顺时针打印矩阵
## 记录上下左右的边界(统一闭区间)+收缩边界
# 边界交叉为空的时候弹出，举例：遍历完最上一行以后，上边界向下扩；下一步要靠上边界和下边界确定遍历的行数，
# 此时break，避免了遍历底部这行，再走一遍回头路
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        res = []
        m = len(matrix)
        if m == 0:
            return res
        n = len(matrix[0])
        l = 0
        r = n-1
        t = 0
        b = m -1
        while True:
            # 从左到右
            for i in range(l, r+1):
                res.append(matrix[t][i])
            t += 1
            if t > b:
                break
            # 从上到下
            for i in range(t, b+1):
                res.append(matrix[i][r])
            r -= 1
            if l > r:
                break            
            # 从右到左
            for i in range(r, l-1, -1):
                res.append(matrix[b][i])
            b -= 1
            if t > b:
                break
            # 从下到上
            for i in range(b, t-1, -1):
                res.append(matrix[i][l])
            l += 1
            if l > r:
                break        
        return res

## 以下自己在"循环轮转矩阵"用过的模拟法，过不了单行和单列的，加上注释后还有些过不了，因为会重复遍历
# 改进方法见上面
'''
[[7],[9],[6]]
'''
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        res = []
        m = len(matrix)
        if m == 0:
            return res
        n = len(matrix[0])
        r = 0; c = 0
        mn = min(m, n)//2
        while r < mn and c < mn:
            for j in range(c, n-c):
                res.append(matrix[r][j])
            for i in range(r+1, m-r):
                res.append(matrix[i][n-c-1])
            # if m-r-1 == r:
            #     r+=1
            #     c+=1
            #     continue
            # if n-c-1 == c:
            #     r+=1
            #     c+=1
            #     continue
            for j in range(n-c-1-1, c-1, -1):
                res.append(matrix[m-r-1][j])
            for i in range(m-r-1-1, r, -1):
                res.append(matrix[i][c])
            r+=1
            c+=1
        return res     


### 剑指 Offer 32 - I. 从上到下打印二叉树
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def levelOrder(self, root: TreeNode) -> List[int]:
        res = []
        if not root: return res
        q = collections.deque()
        q.append(root)
        while q:
            node = q.popleft()
            res.append(node.val)
            if node.left: q.append(node.left)
            if node.right: q.append(node.right)
        return res

### 剑指 Offer 32 - II. 从上到下打印二叉树 II
## 交替记录层数
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        res = []
        if not root:
            return res
        cur = 1
        next = 0
        q= collections.deque()
        q.append(root)
        out = []
        while q:
            if cur == 0:
                cur = next
                next = 0
                res.append(out)
                out = []
            node = q.popleft()
            cur -= 1
            out.append(node.val)
            if node.left:
                next += 1
                q.append(node.left)
            if node.right:
                next += 1
                q.append(node.right) 
        res.append(out) 
        return res 

## 不需要如上的两个常数记录，通过q的长度计算层宽度
class Solution:
    def levelOrder(self, root: TreeNode) -> List[int]:
        res = []
        if not root: return res
        q = collections.deque()
        q.append(root)
        while q:
            layer_lens = len(q)
            out = []
            for i in range(layer_lens, 0, -1):
                node = q.popleft()
                out.append(node.val)
                if node.left: q.append(node.left)
                if node.right: q.append(node.right)
            res.append(out)
        return res



## 奇偶行反转
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        res = []
        if not root: return res
        q = collections.deque()
        q.append(root)
        n = 1
        while q:
            layer_lens = len(q)
            out = []
            for i in range(layer_lens, 0, -1):
                node = q.popleft()
                out.append(node.val)
                if node.left: q.append(node.left)
                if node.right: q.append(node.right)
            if n & 1 == 0:
                out = out[::-1]
            res.append(out)
            n+=1
        return res

## 双端队列存储每行结果，奇偶行区分左右
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        res = []
        if not root: return res
        q = collections.deque()
        q.append(root)
        n = 1
        while q:
            next = 0
            cur = len(q)
            out = collections.deque()
            while cur:
                node = q.popleft()
                if n & 1 == 0:
                    out.appendleft(node.val)
                else:
                    out.append(node.val)
                if node.left: 
                    q.append(node.left)
                    next += 1
                if node.right: 
                    q.append(node.right)
                    next += 1
                cur -= 1
            cur = next
            res.append(list(out))
            n+=1
        return res

### 剑指 Offer 42. 连续子数组的最大和
## 先想清楚用序列型还是坐标型，以及如何获取答案；然后找子问题和状态转移规律
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        dp = [0] * n
        res = dp[0] = nums[0]
        for i in range(1, n):
            if dp[i-1] < 0:
                dp[i] = nums[i]
            else:
                dp[i] = dp[i-1] + nums[i]
            res = max(dp[i], res)
        return res

### 剑指 Offer 39. 数组中出现次数超过一半的数字
## 摩尔投票法，可以通过想哈希表法怎么缩减空间开销想出来
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        vote = 0
        candi = 0
        for num in nums:
            if vote == 0:
                candi = num
                vote += 1
                continue
            vote += 1 if num == candi else -1
        return candi

## 以上摩尔投票法，简单理解
# https://leetcode-cn.com/problems/shu-zu-zhong-chu-xian-ci-shu-chao-guo-yi-ban-de-shu-zi-lcof/solution/mian-shi-ti-39-shu-zu-zhong-chu-xian-ci-shu-chao-3/
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        votes = 0
        for num in nums: #每一个人都要出来挑战
            if votes == 0: #擂台上没人 选一个出来当擂主 x就是擂主  votes就是人数
                x = num
            votes += 1 if num == x else -1 #如果是自己人就站着呗 如果不是 就同归于尽
        return x
### 以上同 169. 多数元素
## partition应该也可以实现时间复杂度为 O(n)、空间复杂度为 O(1) 


### 剑指 Offer 33. 二叉搜索树的后序遍历序列
## 分治；递归（子问题）
# 后序遍历根节点是最后一个
# 边界问题，举例子就能准确解答
class Solution:
    def verifyPostorder(self, postorder: List[int]) -> bool:
        def dfs(i, j):
            # if i >= j - 1:
            if i >= j:
                return True
            p = i
            while postorder[p] < postorder[j]: p+=1
            q = p
            while postorder[q] > postorder[j]: q+=1
            return q == j and dfs(i, p-1) and dfs(p, q-1)
        return dfs(0, len(postorder) - 1)
## 单调栈解法 TODO 

### 剑指 Offer 34. 二叉树中和为某一值的路径
# python中的list作为参数，是会传址的，所以要.copy()
# 路径是从根到叶子，看题目的要求
class Solution:
    def pathSum(self, root: TreeNode, target: int) -> List[List[int]]:
        res = []
        if not root:
            return res
        def dfs(root, out, sumation):
            if not root:
                return 
            out.append(root.val)
            sumation+=root.val
            if sumation == target and not root.left and not root.right:
                res.append(out.copy())
            dfs(root.left, out, sumation)
            dfs(root.right, out, sumation)
            out.pop()
            return
        dfs(root, [], 0)
        return res


### 剑指 Offer 36. 二叉搜索树与双向链表

## 中序遍历
# head和pre是全局变量，设置成类内成员变量self.xx也可以
class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        pre = None
        head = None
        def dfs(root):
            nonlocal head
            nonlocal pre
            if not root:
                return
            dfs(root.left)
            if not pre:
                head = root
            else:
                root.left = pre
                pre.right = root
            pre=root
            dfs(root.right)
            return
        if not root: return None
        dfs(root)
        head.left, pre.right = pre, head
        return head

## 后序遍历，连接左右子（双项）链表
# 自己的解法，不够简单；
class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        def dfs(root):
            if not root:
                return None, None
            lmn, lmx = dfs(root.left)
            if lmx:
                lmx.right = root
                root.left = lmx
            rmn, rmx = dfs(root.right)
            if rmn:
                root.right = rmn
                rmn.left = root
            if not lmn:
                lmn = root
            if not rmx:
                rmx = root
            return lmn, rmx
        if not root: return None
        a, b = dfs(root)
        b.right = a
        a.left =b
        return a


### 剑指 Offer 37. 序列化二叉树
## 层次遍历序列化
class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        data = ""
        if not root: return data
        queue = collections.deque()
        queue.append(root)
        # print(root.val)
        while queue:
            node = queue.popleft()
            # print(node.val)
            if node:
                data += (',' + str(node.val)) if data else str(node.val)
                queue.append(node.left)
                queue.append(node.right)
            else:
                data += ',' + 'null'
        return data

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        if not data: 
            return None
        data = data.split(',')
        n = len(data)
        if n == 0:
            return None
        root = TreeNode(int(data[0]))
        queue = collections.deque()
        queue.append(root)
        i = 1; j = 2
        while queue:
            node = queue.popleft()
            left = right = None
            if i < n and data[i] != 'null':
                left = TreeNode(int(data[i]))
                node.left = left
                queue.append(left)
            if j < n and data[j] != 'null':
                right = TreeNode(int(data[j]))
                node.right = right 
                queue.append(right)
            i += 2
            j += 2
        return root

### 剑指 Offer 55 - I. 二叉树的深度
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1

### 剑指 Offer 55 - II. 平衡二叉树
## 返回深度，顺便剪枝
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        res = True
        def helper(root):
            nonlocal res
            if not root:
                return 0
            l = helper(root.left)
            if l < 0:
                return l 
            r = helper(root.right) 
            if r < 0:
                return r
            res = (res and abs(l - r) <= 1)
            if not res:
                return -1
            return max(l, r) + 1
        helper(root)
        return res

### 剑指 Offer 54. 二叉搜索树的第k大节点
## 顺便剪枝
class Solution:
    def kthLargest(self, root: TreeNode, k: int) -> int:
        res = 0
        i = 0
        def dfs(root):
            nonlocal res, i
            if not root:
                return
            dfs(root.right)
            i += 1
            if i > k:
                return
            if i == k:
                res = root.val
                return
            dfs(root.left)
            return
        dfs(root)
        return res

### 剑指 Offer 68 - I. 二叉搜索树的最近公共祖先
## 利用BST特性
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if q.val < p.val:
            q, p = p, q
        while root:
            if q.val < root.val:
                root = root.left
            elif p.val > root.val:
                root = root.right
            else:
                break
        return root
## 一般方法，不利用BST的特性
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        res = None
        def dfs(root):
            nonlocal res
            if not root:
                return 0
            cnt = 0
            if root == p or root == q:
                cnt += 1
            l = dfs(root.left)
            if l < 0:
                return l
            r = dfs(root.right)
            if r < 0:
                return r
            if l == 1 and r == 1:
                res = root
                return -1
            if cnt == 1 and l+r == 1:
                res = root
                return -1
            return cnt + l + r
        dfs(root)
        return res

### 剑指 Offer 68 - II. 二叉树的最近公共祖先
## 一般方法，同上


### 剑指 Offer 35. 复杂链表的复制
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        p = head
        mp = {}
        new_pre_head = Node(-1)
        q = new_pre_head
        while p:
            node = Node(p.val)
            q.next = node
            mp[p] = node
            p = p.next
            q = q.next

        p = head
        q = new_pre_head.next
        while p:
            q.random = None if not p.random else mp[p.random]
            p = p.next
            q = q.next
        return new_pre_head.next

### 剑指 Offer 40. 最小的k个数
## krahets的partition
# 为什么要加k=size的单独处理的情况？因为这种方法下i不可能取到k
# 如
'''
[0,0,2,3,2,1,1,2,0,4]
10
'''
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        if k >= len(arr): return arr
        def quick_sort(l, r):
            i, j = l, r
            while i < j:
                while i < j and arr[j] >= arr[l]:
                    j -= 1
                while i < j and arr[i] <= arr[l]:
                    i += 1
                arr[i], arr[j] = arr[j], arr[i]
            arr[l], arr[i] = arr[i], arr[l]
            if i < k:
                return quick_sort(i+1, r)
            elif i > k:
                return quick_sort(l, i-1)
            return arr[:k]
        return quick_sort(0, len(arr) - 1)

## krahets的快排
# 不需要判断k是否等于size这种情况
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        # if k >= len(arr): return arr
        def quick_sort(l, r):
            i, j = l, r
            if l >= r: return
            while i < j:
                while i < j and arr[j] >= arr[l]:
                    j -= 1
                while i < j and arr[i] <= arr[l]:
                    i += 1
                arr[i], arr[j] = arr[j], arr[i]
            arr[l], arr[i] = arr[i], arr[l]
            quick_sort(i+1, r)
            quick_sort(l, i-1)
            return
        quick_sort(0, len(arr) - 1)
        return arr[:k]

## 剑指offer书上的解法
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        if k >= len(arr): return arr
        if k == 0: return []
        def partition(arr, start, end):
            small = start - 1
            mark = end
            for i in range(start, mark):
                if arr[i] < arr[mark]:
                    small += 1
                    arr[small], arr[i] = arr[i], arr[small]
            small+=1
            arr[small], arr[mark] = arr[mark], arr[small]
            return small
        
        index = -1
        l = 0; r = len(arr) - 1
        k = k - 1
        index = partition(arr, l, r)
        while index != k:
            if index < k:
                l = index+1
                index = partition(arr, l, r)
            elif index > k:
                r = index-1
                index = partition(arr, l, r)
            else:
                break
        return arr[:k+1]
## 快速排序，用剑指offer版partition

## 堆排序



### 剑指 Offer 38. 字符串的排列
class Solution:
    def permutation(self, s: str) -> List[str]:
        s = list(s)
        n = len(s)
        res = []
        def dfs(start, out):
            if start == n - 1:
                res.append("".join(out.copy()))
                return 
            st = set()
            for i in range(start, n):
                if out[i] in st:
                    continue
                st.add(out[i])
                out[i], out[start] = out[start], out[i]
                dfs(start+1, out)
                out[i], out[start] = out[start], out[i]
            return
        dfs(0, s)
        return res

### 剑指 Offer 45. 把数组排成最小的数
## 自定义比较函数;注意python中的用法
class Solution:
    def minNumber(self, nums: List[int]) -> str:
        def cmp(x, y):
            s1, s2 = x+y, y+x
            if s1 > s2:
                return 1
            elif s2 > s1:
                return -1
            return 0
        nums = [str(num) for num in nums]
        nums.sort(key=functools.cmp_to_key(cmp))
        return "".join(nums)
## 可以代入任意比较型排序，如快排



### 剑指 Offer 44. 数字序列中某一位的数字
class Solution:
    def findNthDigit(self, n: int) -> int:
        digit = 1
        start = 1
        while n - 9 * digit * start > 0:
            n -= 9 * digit * start
            digit += 1
            start *= 10
        num = start + ceil(n/digit) - 1
        num = str(num)
        return ord(num[(n % digit) - 1]) - ord('0')

### 剑指 Offer 43. 1～n 整数中 1 出现的次数
class Solution:
    def countDigitOne(self, n: int) -> int:
        res = 0
        i = 1
        while i <= n:
            divider = i * 10
            res += n // divider * i + min(max(n % divider - i + 1, 0), i)
            i *= 10
        return res
## c++ grandyang
class Solution {
public:
    int countDigitOne(int n) {
        int res = 0, a = 1, b = 1;
        while (n > 0) {
            res += (n + 8) / 10 * a + (n % 10 == 1) * b;
            b += n % 10 * a;
            a *= 10;
            n /= 10;
        }
        return res;
    }
};

### 剑指 Offer 50. 第一个只出现一次的字符
class Solution:
    def firstUniqChar(self, s: str) -> str:
        mp = collections.Counter()
        res = " "
        for c in s:
            mp[c] += 1
        for c in s:
            if mp[c] == 1:
                res = c
                break
        return res


### 剑指 Offer 48. 最长不含重复字符的子字符串
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        res = 0
        mp = {}
        start = 0
        for i, c in enumerate(s):
            if c not in mp:
                mp[c] = i
            elif mp[c] >= start:
                start = mp[c] + 1
            mp[c] = i
            res = max(res, i - start + 1)
        return res

### 剑指 Offer 56 - I. 数组中数字出现的次数
class Solution:
    def singleNumbers(self, nums: List[int]) -> List[int]:
        nums1 = []
        nums2 = []
        res = nums[0]
        n = len(nums)
        for i in range(1, n):
            res ^= nums[i]
        def get_first_one(res):
            tmp = 1
            for i in range(n):
                if res & (tmp):
                    break
                tmp <<= 1
            return tmp
        bound = get_first_one(res)
        for num in nums:
            if num & bound:
                nums1.append(num)
            else:
                nums2.append(num)
        res1 = nums1[0]
        res2 = nums2[0]
        for i in range(1, len(nums1)):
            res1 ^= nums1[i]
        for i in range(1, len(nums2)):
            res2 ^= nums2[i]
        return [res1, res2]


### 剑指 Offer 56 - II. 数组中数字出现的次数 II
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        array = []
        idx = 1
        n = len(nums)
        mx = max(nums)
        while idx <= mx:
            tmp = 0
            for i in range(n):
                if nums[i] & idx:
                    tmp += 1
            array.append(tmp % 3)
            idx <<= 1
        # reverse(array)
        res = 0
        for i, a in enumerate(array):
            res += a * pow(2, i)
        return res

### 剑指 Offer 66. 构建乘积数组
class Solution:
    def constructArr(self, a: List[int]) -> List[int]:
        n = len(a)
        res = [1] * n
        tmp = [1] * n
        for i in range(1, n):
            res[i] = res[i-1] * a[i-1]
        for i in range(n-2, -1, -1):
            tmp[i] = tmp[i+1] * a[i+1]
            res[i] *= tmp[i]
        return res

### 剑指 Offer 65. 不用加减乘除做加法
## TODO 符号位是个是么概念，为什么转成无符号整形以后，移位还能和原来效果一样？
## python 负数会出错
class Solution:
    def add(self, a: int, b: int) -> int:
        while b:
            ele_sum = a ^ b
            carry = (a & b) << 1
            a = ele_sum
            b = carry
        return a
## c++
# c++不支持负数的移位，需要加上转换为非负数后操作
class Solution {
public:
    int add(int a, int b) {
        while (b) {
            int ele_sum = a ^ b;
            int carry = (unsigned int)(a & b) << 1;
            a = ele_sum;
            b = carry;
        }
        return a;
    }
};


### 剑指 Offer 62. 圆圈中最后剩下的数字
# f(n) = (f(n-1) + m)%n
class Solution:
    def lastRemaining(self, n: int, m: int) -> int:
        dp = [0] * (n+1)
        # dp[1] = 0
        for i in range(2, n+1):
            dp[i] = (dp[i-1] + m)%i
        return dp[n]
