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