
### 145. 二叉树的后序遍历
## 迭代解法
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        stk = []
        res = []
        prev = None
        while stk or root:
            while root:
                stk.append(root)
                root = root.left
            root = stk.pop()
            if not root.right or root.right == prev:
                res.append(root.val)
                prev = root
                root = None
            else:
                stk.append(root)
                root = root.right
        return res

### 173. 二叉搜索树迭代器
## 扁平化解法
class BSTIterator:
    def __init__(self, root: TreeNode):
        self._res = []
        self._idx = -1
        self._inorder(root)
        
    def _inorder(self, root):
        if not root:
            return
        self._inorder(root.left)
        self._res.append(root.val)
        self._inorder(root.right)
        return

    def next(self) -> int:
        self._idx += 1
        return self._res[self._idx]


    def hasNext(self) -> bool:
        return self._idx+1 < len(self._res)

## 栈解法
class BSTIterator:
    def __init__(self, root: TreeNode):
        self.root = root
        self.stk = []

    def next(self) -> int:
        while self.root:
            self.stk.append(self.root)
            self.root = self.root.left
        last = self.stk[-1]
        self.stk.pop()
        self.root = last.right
        return last.val

    def hasNext(self) -> bool:
        return (self.root is not None) or (len(self.stk) != 0)


##538. 把二叉搜索树转换为累加树

# 解法1:中序遍历；全局变量
# 时间 o(n), 空间 o(n), 最差o(n), 最好o(logn) 树高, 栈开销
# 递归子问题的结果，用静态变量传递或者传入参数又传出

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def convertBST(self, root: TreeNode) -> TreeNode:
        def dfs(root: TreeNode):
            nonlocal total
            if root:
                dfs(root.right)
                total += root.val
                root.val = total
                dfs(root.left)
        
        total = 0
        dfs(root)
        return root
# 解法2:中序遍历；传参
class Solution:
    def convertBST(self, root: TreeNode) -> TreeNode:
        sm = 0
        self.dfs(root, sm)
        return root

    def dfs(self, root, sm):
        if not root: return sm
        sm = self.dfs(root.right, sm)
        sm += root.val
        sm = self.dfs(root.left, root.val)
        return sm
# 解法3:morris遍历；