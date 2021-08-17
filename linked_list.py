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

### 23. 合并K个升序链表
## 优先队列
# 作者：powcai
# 链接：https://leetcode-cn.com/problems/merge-k-sorted-lists/solution/leetcode-23-he-bing-kge-pai-xu-lian-biao-by-powcai/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        import heapq
        dummy = ListNode(0)
        p = dummy
        head = []
        for i in range(len(lists)):
            if lists[i] :
                heapq.heappush(head, (lists[i].val, i))
                lists[i] = lists[i].next
        while head:
            val, idx = heapq.heappop(head)
            p.next = ListNode(val)
            p = p.next
            if lists[idx]:
                heapq.heappush(head, (lists[idx].val, idx))
                lists[idx] = lists[idx].next
        return dummy.next

