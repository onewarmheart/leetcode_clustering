### 有效数独
## 用数组记录出现过的数字
## 数组最后一个维度本质是简易哈希表
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        rows = [[0 for a in range(10)] for b in range(9)]
        cols = [[0 for a in range(10)] for b in range(9)]
        blocks = [[[0 for a in range(10)] for b in range(3)] for c in range(3)]

        for i in range(9):
            for j in range(9):
                numb_s = board[i][j]
                if numb_s == '.':
                    continue 
                numb = int(numb_s) 
                if rows[i][numb] \
                    or cols[j][numb] or blocks[i//3][j//3][numb]:
                    return False
                else:
                    rows[i][numb]=cols[j][numb]=blocks[i//3][j//3][numb]=1
        return True
