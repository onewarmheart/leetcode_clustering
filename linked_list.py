#%%
### 92. 反转链表 II
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseBetween(self, head: ListNode, left: int, right: int) -> ListNode:
        i = 0
        p = head
        stack = []
        left_tmp = right_tmp = None

        def reverseLinkedList(p1, p2):
            while stack:
                tmp = stack.pop()
                if stack: tmp.next = stack[-1]
            p1.next = right_tmp
            return p2

        while p:
            i += 1
            if i == left-1:
                left_tmp = p
            if i >= left and i <= right:
                if i == left:
                    p1 = p
                if i == right:
                    p2 = p
                stack.append(p)
            p = p.next
        right_tmp = p2.next
        between = reverseLinkedList(p1, p2)
        if left_tmp:
            left_tmp.next = between
        else:
            return between
        return head