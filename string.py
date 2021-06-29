###6. Z 字形变换




###58. 最后一个单词的长度


#%%
### 71. 简化路径
class Solution:
    def simplifyPath(self, path: str) -> str:
        path = path.split('/')
        res = ['/']
        for i, item in enumerate(path):
            if not item or item == '.':
                continue
            if item == '..':
                if res[-1] == '/':
                    continue
                if len(res) > 1:
                    res.pop()
                    continue
            res.append(item)
        res="/".join(res[1:])
        res = "/" + res
        return res
#%%
## 125. 验证回文串
class Solution:
    def isPalindrome(self, s: str) -> bool:
        i = 0; j = len(s) -1
        while i < j:
            if not s[i].isalpha() and not s[i].isdigit():
                i+=1
                continue
            if not s[j].isalpha() and not s[j].isdigit():
                j-=1
                continue
            if s[i].lower() != s[j].lower():
                return False
            i+=1
            j-=1
        return True
#%%
### 205. 同构字符串
## 对称关系，双边判断
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        def isOneWay(s, t):
            hashmap = {}
            n = len(s)
            for i in range(n):
                tmp = hashmap.get(s[i], '')
                if not tmp:
                    hashmap[s[i]] = t[i]
                    continue
                if tmp != t[i]:
                    return False
            return True
        return isOneWay(s, t) and isOneWay(t, s)

## hashmap确保不能一对多，
## set看对面的字符是否有使用过，确保不能多对一
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        hashmap = {}
        used_val = set()
        n = len(s)
        for i in range(n):
            val = hashmap.get(s[i], '')
            if not val:
                if t[i] in used_val:
                    return False
                hashmap[s[i]] = t[i]
                used_val.add(t[i])
                continue
            if val != t[i]:
                return False
        return True

## 273. 整数转换英文表示
## 转成3位整数的表示
## 按位封装成子问题/子函数
# 作者：LeetCode
# 链接：https://leetcode-cn.com/problems/integer-to-english-words/solution/zheng-shu-zhuan-huan-ying-wen-biao-shi-by-leetcode/
class Solution:
    def numberToWords(self, num):
        """
        :type num: int
        :rtype: str
        """
        def one(num):
            switcher = {
                1: 'One',
                2: 'Two',
                3: 'Three',
                4: 'Four',
                5: 'Five',
                6: 'Six',
                7: 'Seven',
                8: 'Eight',
                9: 'Nine'
            }
            return switcher.get(num)

        def two_less_20(num):
            switcher = {
                10: 'Ten',
                11: 'Eleven',
                12: 'Twelve',
                13: 'Thirteen',
                14: 'Fourteen',
                15: 'Fifteen',
                16: 'Sixteen',
                17: 'Seventeen',
                18: 'Eighteen',
                19: 'Nineteen'
            }
            return switcher.get(num)

        def ten(num):
            switcher = {
                2: 'Twenty',
                3: 'Thirty',
                4: 'Forty',
                5: 'Fifty',
                6: 'Sixty',
                7: 'Seventy',
                8: 'Eighty',
                9: 'Ninety'
            }
            return switcher.get(num)

        def two(num):
            if not num: return ''
            if num < 10:
                return one(num)
            elif num < 20:
                return two_less_20(num)
            else:
                tenner = num // 10
                rest = num - tenner * 10
                return ten(tenner) + ' ' + one(rest) if rest else ten(tenner)

        def three(num):
            hundreder = num // 100
            rest = num - hundreder*100
            if hundreder and rest:
                return one(hundreder) + ' Hundred ' + two(rest)
            elif not hundreder and rest:
                return two(rest)
            elif hundreder and not rest:
                return one(hundreder) + ' Hundred'


        billion = num // 1000000000
        million = (num - billion * 1000000000) // 1000000
        thousand = (num - billion * 1000000000 - million * 1000000) // 1000
        rest = num - billion * 1000000000 - million * 1000000 - thousand * 1000

        if not num:
            return 'Zero'

        result = ''
        if billion:
            result = three(billion) + ' Billion'
        if million:
            result += ' ' if result else ''
            result += three(million) + ' Million'
        if thousand:
            result += ' ' if result else ''
            result += three(thousand) + ' Thousand'
        if rest:
            result += ' ' if result else ''
            result += three(rest)
        return result