import random
from collections import deque
from heapq import *


#https://leetcode.com/problems/number-of-good-pairs/
def numIdenticalPairs(nums):
    count = 0
    for i in range(len(nums) - 1):
        for j in range(i + 1, len(nums)):
            if nums[i] == nums[j]:
                count += 1
    return count


print('1. No. of Good pairs: ', numIdenticalPairs([1,2,3,1,1,3]))


# https://leetcode.com/problems/decompress-run-length-encoded-list/
def decompressRLElist(nums):
    result = []
    for i in range(0, len(nums), 2):
        freq, val = nums[i], nums[i + 1]
        result = result + [val] * freq
    return result


print('2. Decompress Run-Length Encoded List: ', decompressRLElist([1,2,3,1,1,3]))


# https://leetcode.com/problems/reverse-words-in-a-string-iii/
def reverseWords(s) -> str:
    s = s.strip()
    if not s:
        return ''
    s = s.split()
    result = []
    for word in s:
        temp = []
        for ch in word:
            temp.append(ch)
        while temp:
            result.append(temp.pop())
        result.append(' ')
    result = ''.join(result[:-1])
    return result


print('3. Reverse words in String: ', reverseWords("Let's take LeetCode contest"))


# https://leetcode.com/problems/find-the-highest-altitude/
def largestAltitude(gain):
    gain = [0] + gain
    for i in range(1, len(gain)):
        gain[i] += gain[i - 1]
    return max(gain)


print('4. Largest Altitude Cyclist: ', largestAltitude([-5,1,5,0,-7]))
print('4. Largest Altitude Cyclist: ', largestAltitude([-4,-3,-2,-1,4,3,2]))


#https://leetcode.com/problems/happy-number/
def isHappy(n):
    slow = fast = n
    while True:
        slow = getNextNum(slow)
        fast = getNextNum(getNextNum(fast))
        if slow == fast:
            break
    if slow == 1:
        return True
    return False


def getNextNum( num):
    s = 0
    while num > 0:
        s += (num % 10) ** 2
        num //= 10
    return s


print('5. Is Happy Number 19: ', isHappy(19))
print('5. Is Happy Number 2 : ', isHappy(2))


# https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
def maxProfit(prices):
    maxProfit = 0
    minPrice = prices[0]
    for i in range(1, len(prices)):
        profit = prices[i] - minPrice
        maxProfit = profit if maxProfit < profit else maxProfit
        minPrice = prices[i] if minPrice > prices[i] else minPrice
    return maxProfit


print('6: Buy and Sell Stocks I: ', maxProfit([7,1,5,3,6,4]))
print('6: Buy and Sell Stocks I: ', maxProfit([7,6,4,3,1]))


# https://leetcode.com/problems/design-hashmap/
class Bucket:
    def __init__(self):
        self.bucket = []

    def put(self, key, value):
        for ind, val in enumerate(self.bucket):
            if val[0] == key:
                self.bucket[ind] = (key, value)
                return
        self.bucket.append((key, value))

    def get(self, key):
        for k, v in self.bucket:
            if k == key:
                return v
        return -1

    def delete(self, key):
        for ind, val in enumerate(self.bucket):
            if val[0] == key:
                del self.bucket[ind]


class MyHashMap:
    def __init__(self):
        self.max = 999
        self.hashmap = [Bucket() for i in range(self.max)]

    def get_hash(self, key):
        return key % 999

    def put(self, key, value):
        h = self.get_hash(key)
        self.hashmap[h].put(key, value)

    def get(self, key):
        h = self.get_hash(key)
        return self.hashmap[h].get(key)

    def remove(self, key):
        h = self.get_hash(key)
        self.hashmap[h].delete(key)


myHashMap = MyHashMap();
myHashMap.put(1, 1)
myHashMap.put(2, 2)
print('7: Get 1- Present ', myHashMap.get(1))
print('7: Get 3- Absent ', myHashMap.get(3))
myHashMap.put(2, 1)
print('7: Get 2- updated ', myHashMap.get(2))
myHashMap.remove(2)
print('7: Get 2- Absent ', myHashMap.get(2))


# https://leetcode.com/problems/longest-common-prefix/
def longestCommonPrefix(strs):
    if not strs:
        return ''
    if len(strs) == 1:
        return strs[0]
    result = []
    strs.sort()
    for x, y in zip(strs[0], strs[-1]):
        if x == y:
            result.append(x)
        else:
            break
    return ''.join(result)


print('8: Longest Common Prefix in list: ', longestCommonPrefix(["flower","flow","flight"]))
print('8: Longest Common Prefix in list: ', longestCommonPrefix(["dog","racecar","car"]))


# https://leetcode.com/problems/maximum-subarray/
def maxSubArray(nums):
    curSum = maxSum = nums[0]
    for i in range(1, len(nums)):
        curSum = nums[i] if nums[i] > curSum + nums[i] else curSum + nums[i]
        maxSum = maxSum if maxSum > curSum else curSum
    return maxSum


print('9: Max Subarray Sum: ', maxSubArray([-2,1,-3,4,-1,2,1,-5,4]))
print('9: Max Subarray Sum: ', maxSubArray([5,4,-1,7,8]))



# https://leetcode.com/problems/valid-sudoku/
def isValidSudoku(board):
    seen = set()
    for i in range(len(board)):
        for j in range(len(board[0])):
            num = board[i][j]
            if num != '.':
                if (num, i) in seen or (j, num) in seen or (i // 3, j // 3, num) in seen:
                    return False
                seen.add((num, i))
                seen.add((j, num))
                seen.add((i // 3, j // 3, num))
    return True


board=[["5","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
print('10. Is Valid Sudoku: ', isValidSudoku(board))
board=[["8","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
print('10. Is Valid Sudoku: ', isValidSudoku(board))


# https://leetcode.com/problems/sudoku-solver/
def use_in_row(board,num,row):
    for i in range(9):
        if board[row][i]!='.' and int(board[row][i])==num:
            return False
    return True


def use_in_col(board,num,col):
    for i in range(9):
        if board[i][col]!='.' and int(board[i][col])==num:
            return False
    return True


def use_in_box(board,num,row,col):
    for i in range(3):
        for j in range(3):
            if board[row+i][col+j]!='.' and int(board[row+i][col+j])==num:
                return False
    return True


def can_place_num(board,num,row,col):
    return use_in_row(board,num,row) and use_in_col(board,num,col) and use_in_box(board,num,row-row%3,col-col%3)


def find_empty_location(board,l):
    for i in range(9):
        for j in range(9):
            if board[i][j]=='.':
                l[0],l[1]= i,j
                return True
    return False


def sudokuSolver(board):
    l=[0,0]
    if not find_empty_location(board,l):
        return True
    row,col=l[0],l[1]
    for num in range(1,10):
        if can_place_num(board,num,row,col):
            board[row][col]=str(num)
            if sudokuSolver(board):
                return True
            board[row][col]='.'
    return False


board = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]
print('11. Sudoku Solver: ', sudokuSolver(board))
print(board)


# https://leetcode.com/problems/length-of-longest-fibonacci-subsequence/
def longest_fibonacci_subsequence(arr):
    ar=set(arr)
    ans=0
    for i in range(len(arr)):
        for j in range(i+1,len(arr)):
            x,y = arr[j],arr[i]+arr[j]
            length=2
            while y in ar:
                x,y = y, x+y
                length+=1
            ans= ans if length<=ans else length
    return ans if ans>=3 else 0


print('12. Longest Fibonacci Subsequence: ', longest_fibonacci_subsequence([1,2,3,4,5,6,7,8]))
print('12. Longest Fibonacci Subsequence: ', longest_fibonacci_subsequence([1,3,7,11,12,14,18]))


# https://leetcode.com/problems/find-distance-in-a-binary-tree/
def find_distance_binary_tree(root,p,q):
    def helper(node):
        if not node:
            return None
        if node.val in [p,q]:
            return node
        left,right = helper(node.left),helper(node.right)
        if not left:
            return right
        if not right:
            return left
        return node
    root = helper(root)
    que = deque()
    p_ind, q_ind = 0,0
    que.append([root,0])
    while que:
        node,path = que.popleft()
        if node.val==p:
            p_ind=path
            if q_ind:
                break
        if node.val==q:
            q_ind=path
            if p_ind:
                break
        if node.left:
            que.append([node.left,path+1])
        if node.right:
            que.append([node.right,path+1])
    return p_ind+q_ind


class TreeNode:
     def __init__(self, val=0, left=None, right=None):
         self.val = val
         self.left = left
         self.right = right


root = TreeNode()
root.left=TreeNode(5)
root.right= TreeNode(1)
root.left.left= TreeNode(6)
root.left.right= TreeNode(2)
root.right.left= TreeNode(0)
root.right.right= TreeNode(8)
root.left.right.left= TreeNode(7)
root.left.right.right = TreeNode(4)
print('13. Distance between 2 nodes in Binary Tree: ', find_distance_binary_tree(root,5,0))
print('13. Distance between 2 nodes in Binary Tree: ', find_distance_binary_tree(root,5,7))



# https://leetcode.com/problems/find-the-celebrity/
def find_celebrity(n):
    celeb=0
    for i in range(n):
        if knows(celeb,i):
            celeb=i
    for i in range(n):
        if celeb==i:
            continue
        if knows(celeb,i) or not knows(i,celeb):
            return -1
    return celeb


def knows(i, j):
    grid = [[1, 1, 0], [0, 1, 0], [1, 1, 1]]
    return grid[i][j]


print('14. find_celebrity: ', find_celebrity(2))


# https://leetcode.com/problems/remove-duplicate-letters/
def remove_duplicate_letters(s):
    stack=['!']
    visited=set()
    last_ind= {c:i for i,c in enumerate(s)}
    for i,c in enumerate(s):
        if c in visited:
            continue
        while stack[-1]>c and last_ind[stack[-1]]>i:
            visited.remove(stack.pop())
        visited.add(c)
        stack.append(c)
    return ''.join(stack[1:])


print('15. Remove duplicate letters: ', remove_duplicate_letters('cbacdcbc'))


# https://leetcode.com/problems/number-of-subarrays-with-bounded-maximum/
def numSubarrayBoundMax(nums,left,right):
    def count(bound):
        ans=cur=0
        for x in nums:
            cur=cur+1 if x<=bound else 0
            ans+=cur
        return ans
    return count(right)- count(left-1)


print('16. Number of Subarrays with Bounded Maximum: ', numSubarrayBoundMax([2,9,2,5,6],2,8))


# https://leetcode.com/problems/h-index/
def h_index_1(citations):
    citations.sort(reverse=True)
    for i in range(len(citations)):
        if i>=citations[i]:
            return i
    return len(citations)


print('17. H- Index I ', h_index_1([3,0,6,1,5]))


# https://leetcode.com/problems/design-circular-queue/
class MyCircularQueue:

    def __init__(self, k: int):
        self.list=[None]*k
        self.front =self.rear=0

    def enQueue(self, value: int) -> bool:
        if self.isFull():
            return False
        self.list[self.rear]= value
        self.rear=(self.rear+1)%len(self.list)
        return True

    def deQueue(self) -> bool:
        if self.isEmpty():
            return False
        self.list[self.front]=None
        self.front= (self.front+1)%len(self.list)
        return True

    def Front(self) -> int:
        if self.isEmpty():
            return -1
        return self.list[self.front]

    def Rear(self) -> int:
        if self.isEmpty():
            return -1
        return self.list[self.rear-1]

    def isEmpty(self) -> bool:
        return self.front==self.rear and self.list[self.front] is None

    def isFull(self) -> bool:
        return self.front==self.rear and self.list[self.front] is not None


myCircularQueue =  MyCircularQueue(3)
print('18: Circular Queue: true ',myCircularQueue.enQueue(1))
print('18: Circular Queue: true ',myCircularQueue.enQueue(2))
print('18: Circular Queue: true ',myCircularQueue.enQueue(3))
print('18: Circular Queue: false ',myCircularQueue.enQueue(4))
print('18: Circular Queue: 3 ',myCircularQueue.Rear())
print('18: Circular Queue: true ',myCircularQueue.isFull())
print('18: Circular Queue: true ',myCircularQueue.deQueue())
print('18: Circular Queue: true ',myCircularQueue.enQueue(4))
print('18: Circular Queue: 4 ',myCircularQueue.Rear())


# https://leetcode.com/problems/maximum-nesting-depth-of-two-valid-parentheses-strings/
def maxDepthAfterSplit(seq):
    result, depth = [], 0
    for brace in seq:
        if brace == '(':
            depth += 1
        result.append(depth % 2)
        if brace == ')':
            depth -= 1
    return result


print('19. Max Depth of brackets after Split: ', maxDepthAfterSplit("()(())()"))
print('19. Max Depth of brackets after Split: ', maxDepthAfterSplit("(()())"))


# https://leetcode.com/problems/two-sum/
def two_sum(nums, target):
    helper = {}
    for ind, num in enumerate(nums):
        sec = target - num
        if sec in helper:
            return [helper[sec], ind]
        helper[num] = ind
    return [-1, -1]


print('20. Two Sum: ', two_sum([2, 7, 11, 15], 9))

# https://leetcode.com/problems/find-median-from-data-stream/

class MedianFinder:

    def __init__(self):
        self.minHeap = []
        self.maxHeap = []

    def addNum(self, num: int) -> None:
        if not self.maxHeap or -self.maxHeap[0] >= num:
            heappush(self.maxHeap, -num)
        else:
            heappush(self.minHeap, num)
        if len(self.maxHeap) > len(self.minHeap) + 1:
            heappush(self.minHeap, -heappop(self.maxHeap))
        elif len(self.minHeap) > len(self.maxHeap):
            heappush(self.maxHeap, -heappop(self.minHeap))

    def findMedian(self) -> float:
        if len(self.minHeap) == len(self.maxHeap):
            return -self.maxHeap[0] / 2.0 + self.minHeap[0] / 2.0
        return -self.maxHeap[0] / 1.0


medianFinder = MedianFinder()
medianFinder.addNum(1)
medianFinder.addNum(2)
print('21. Data Stream Median Finder: ', medianFinder.findMedian())
medianFinder.addNum(3)
print('21. Data Stream Median Finder: ', medianFinder.findMedian())


# https://leetcode.com/problems/word-ladder/
def ladderLength(beginWord, endWord, wordList):
    wordList = set(wordList)
    chSet = {ch for word in wordList for ch in word}
    if endWord not in wordList:
        return 0
    que = deque()
    que.append([beginWord, 1])
    while que:
        word, length = que.popleft()
        if word == endWord:
            return length
        for i in range(len(word)):
            for ch in chSet:
                nextWord = word[:i] + ch + word[i + 1:]
                if nextWord in wordList:
                    wordList.remove(nextWord)
                    que.append([nextWord, length + 1])
    return 0


print('22. Word Ladder : ', ladderLength('hit','cog',["hot","dot","dog","lot","log","cog"]))
print('22. Word Ladder : ', ladderLength('hit','cog',["hot","dot","dog","lot","log"]))


# https://leetcode.com/problems/meeting-scheduler/
def meetingScheduler(slot1,slot2,duration):
    slot1.sort(key=lambda x:x[0])
    slot2.sort(key=lambda x: x[0])
    i,j,l1,l2 = 0,0, len(slot1),len(slot2)
    while i<l1 and j<l2:
        s1,s2 = slot1[i][0],slot2[j][0]
        e1,e2=  slot1[i][1],slot2[j][1]
        start,end = max(s1,s2), min(e1,e2)
        if start<end:
            if end-start>=duration:
                return [start,start+duration]
        if e1<e2:
            i+=1
        else:
            j+=1
    return []


print('23. Meeting Scheduler, min. available duration: ', meetingScheduler([[10,50],[60,120],[140,210]], [[0,15],[60,70]],8))
print('23. Meeting Scheduler, min. available duration: ', meetingScheduler([[10,50],[60,120],[140,210]], [[0,15],[60,70]],  12))


# https://leetcode.com/problems/find-servers-that-handled-most-number-of-requests/
def busiestServer(k,arrival,load):
    available= list(range(k))
    busy= []
    res=[0]*k
    for i, a in enumerate(arrival):
        while busy and busy[0][0]<=a:
            l,x = heappop(busy)
            heappush(available, i+(x-i)%k)
        if available:
            j = heappop(available)%k
            heappush(busy,[a+load[i],j])
            res[j]+=1
    ans=max(res)
    return [a for a in range(k) if res[a]==ans]


print('24. Busiest Server: ', busiestServer(3,[1,2,3,4,5], [5,2,3,3,3] ))
print('24. Busiest Server: ', busiestServer(3, [1,2,3,4], [1,2,1,2]))


# https://leetcode.com/problems/decode-string/
def decode_string_with_brackets(st):
    stack, ans, n = [],'',''
    for ch in st:
        if ch.isalpha():
            ans+=ch
        elif ch.isdigit():
            n+=ch
        elif ch == '[':
            stack.append((n,ans))
            n=ans=''
        else:
            count,prev= stack.pop()
            ans = prev + ans* int(count)
    return ans


print('25. Decode String with Brackets, aaabcbc: ', decode_string_with_brackets("3[a]2[bc]"))
print('25. Decode String with Brackets, abcabccdcdcdef: ', decode_string_with_brackets("2[abc]3[cd]ef"))


# https://leetcode.com/problems/random-pick-with-weight/
class RandomPickWithWeights:
    def __init__(self,w):
        self.prefix_sum =[]
        self.total_sum =0
        for wei in w:
            self.total_sum+=wei
            self.prefix_sum.append(self.total_sum)

    def pick_index(self):
        target = self.total_sum*random.random()
        low,high = 0, len(self.prefix_sum)
        while low<high:
            mid = low +(high-low)//2
            if target>self.prefix_sum[mid]:
                low=mid+1
            else:
                high=mid
        return low


solution = RandomPickWithWeights([1,3])
print('26. Random Weights: ',solution.pick_index())
print('26. Random Weights: ',solution.pick_index())
print('26. Random Weights: ',solution.pick_index())
print('26. Random Weights: ',solution.pick_index())
print('26. Random Weights: ',solution.pick_index())


# https://leetcode.com/problems/smallest-string-starting-from-leaf/
def smallest_string_from_leaf_lexico(root):
    def dfs(node,path):
        nonlocal ans
        if not node:
            return None
        path = chr(97+ node.val) + path
        dfs(node.left,path)
        dfs(node.right,path)
        if not node.left and not node.right:
            if not ans:
                ans=path
            else:
                ans = min(ans,path)
    ans=''
    dfs(root,'')
    return ans


root = TreeNode(25)
root.left=TreeNode(1)
root.right= TreeNode(3)
root.left.left= TreeNode(1)
root.left.right= TreeNode(3)
root.right.left= TreeNode(0)
root.right.right= TreeNode(2)
print('27. Smallest string from leaf, adz: ', smallest_string_from_leaf_lexico(root))


# https://leetcode.com/problems/spiral-matrix/submissions/
def spiral_matrix(mat):
    ver,hor= len(mat),len(mat[0])
    i,j=0,-1
    direction=1
    res=[]
    while ver*hor>0:
        for _ in range(hor):
            j+=direction
            res.append(mat[i][j])
        ver-=1
        for _ in range(ver):
            i+=direction
            res.append(mat[i][j])
        hor-=1
        direction*=-1
    return res


print('28. Spiral Matrix: ', spiral_matrix([[1,2,3],[4,5,6],[7,8,9]]))


# https://leetcode.com/problems/rectangle-area/
def area_covered_by_2rectangles(ax1 = -3, ay1 = 0, ax2 = 3, ay2 = 4, bx1 = 0, by1 = -1, bx2 = 9, by2 = 2):
    area1 = abs(ax1-ax2)*abs(ay1-ay2)
    area2 = abs(bx1-bx2)*abs(by1-by2)
    x = min(ax2,bx2) - max(ax1,bx1)
    y = min(ay2,by2) - max(ay1,by1)
    if x<=0 or y<=0:
        return area1+area2
    return area1+area2-x*y


print('29. Area Covered by 2 Rectangles, 45: ', area_covered_by_2rectangles())


# https://leetcode.com/problems/word-break/submissions/
def word_break_segments(s,words):
    words = set(words)
    visited= set()
    que=deque()
    que.append(0)
    while que:
        start = que.popleft()
        if start in visited:
            continue
        for end in range(start+1, len(s)+1):
            if s[start:end] in words:
                que.append(end)
                if end==len(s):
                    return True
        visited.add(start)
    return False


print('30. Word Break Segments: ', word_break_segments('applepenapple',["apple","pen"]))
print('30. Word Break Segments: ', word_break_segments('catsandog', ["cats","dog","sand","and","cat"]))


# https://leetcode.com/problems/minimum-domino-rotations-for-equal-row/
def minimum_domino_rotations_for_equal_row(A,B):
    def count(x):
        rot_a,rot_b=0,0
        for i in range(len(A)):
            if A[i]!=x and B[i]!=x:
                return -1
            elif A[i]!=x:
                rot_a+=1
            elif B[i]!=x:
                rot_b+=1
        return min(rot_a,rot_b)
    rotations= count(A[0])
    if rotations!=-1 or A[0]==B[0]:
        return rotations
    return count(B[0])


print('31. Minimum Domino Rotations For Equal Row: ', minimum_domino_rotations_for_equal_row([2,1,2,4,2,2],[5,2,6,2,3,2]))
print('31. Minimum Domino Rotations For Equal Row: ', minimum_domino_rotations_for_equal_row([3,5,1,2,3], [3,6,3,3,4]))


# https://leetcode.com/problems/next-greater-element-ii/
def next_greater_element_in_list(nums):
    stack=[]
    res=[-1]*len(nums)
    for _ in range(2):
        for ind,val in enumerate(nums):
            while stack and nums[stack[-1]]<val:
                res[stack.pop()]=val
            stack.append(ind)
    return res


print('32. Next Greater Element in List: ', next_greater_element_in_list([3,8,4,1,2]))


# https://leetcode.com/problems/serialize-and-deserialize-n-ary-tree
# Definition for a Node.
class Node(object):
    def __init__(self, val=None, children=[]):
        self.val = val
        self.children = children


def serialize(root):
    if not root:
        return ''

    def helper(x):
        if x:
            return [[i.val, helper(i)] for i in x.children]

    return [root.val, helper(root)]


def deserialize(data):
    if not data:
        return None

    def helper(vals):
        if not vals:
            return
        k = vals[0]
        for i in vals[1:]:
            node = Node(k)
            node.children = [helper(v) for v in i]
            return node

    return helper(data)


root= Node(1)
a,b,c = Node(3),Node(2),Node(4)
a.children=[Node(5),Node(6)]
root.children=[a,b,c]
data = serialize(root)
print('33. Serialize: ', data)
print('33. Deserialize: ',deserialize(data).val)


# https://leetcode.com/problems/find-servers-that-handled-most-number-of-requests/
def busiestServer(k, arrival, load):
    serverList = {i: [] for i in range(k)}
    intervals = [(arrival[i], arrival[i] + load[i]) for i in range(len(arrival))]
    maxLoad = 0
    ans = {}
    for start, end in intervals:
        for j in range(k):
            if len(serverList[j]) == 0 or -serverList[j][0] <= start:
                heappush(serverList[j], -end)
                if len(serverList[j]) >= maxLoad:
                    ans[j] = maxLoad = len(serverList[j])
                break
    res = []
    for k, v in ans.items():
        if v == maxLoad:
            res.append(k)
    return res


arrival = [1, 2, 3]
load = [10, 12, 11]
print('34. Busiest Server: ',busiestServer(3, arrival, load))


# https://leetcode.com/problems/3sum/
def threeSum(nums):
    nums.sort()
    result = []
    for i in range(len(nums)):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            s = nums[i] + nums[left] + nums[right]
            if s == 0:
                result.append([nums[i], nums[left], nums[right]])
                left += 1
                right -= 1
                while left < right and nums[left] == nums[left - 1]:
                    left += 1
                while right > left and nums[right] == nums[right + 1]:
                    right -= 1
            elif s < 0:
                left += 1
            else:
                right -= 1
    return result


print('35. Three Sum: ', threeSum([-1,0,1,2,-1,-4]))


# https://leetcode.com/problems/design-tic-tac-toe/
class TicTacToe:
    def __init__(self, n):
        self.n = n
        self.board = [[0 for i in range(self.n)] for j in range(self.n)]

    def move(self, row, col, player):
        self.board[row][col] = player
        if self.rowWin(row, player) or self.colWin(col, player):
            return player
        if (row == col or row + col == self.n - 1) and self.diahWin(player):
            return player
        return 0

    def rowWin(self, row, player):
        rowset = set([self.board[row][x] for x in range(self.n)])
        return len(rowset) == 1 and self.board[row][0] == player

    def colWin(self, col, player):
        colset = set([self.board[x][col] for x in range(self.n)])
        return len(colset) == 1 and self.board[0][col] == player

    def diahWin(self, player):
        return self.leftDiagWin(player) or self.rightDiagWin(player)

    def leftDiagWin(self, player):
        diagset = set([self.board[x][x] for x in range(self.n)])
        return len(diagset) == 1 and self.board[0][0] == player

    def rightDiagWin(self, player):
        diagset = set([self.board[x][self.n - x - 1] for x in range(self.n)])
        return len(diagset) == 1 and self.board[0][self.n - 1] == player


ticTacToe = TicTacToe(3)
ticTacToe.move(0, 0, 1)
ticTacToe.move(0, 2, 2)
ticTacToe.move(2, 2, 1)
ticTacToe.move(1, 1, 2)
ticTacToe.move(2, 0, 1)
ticTacToe.move(1, 0, 2)
print('36. TicTacToe winner 1: ', ticTacToe.move(2, 1, 1))


# https://leetcode.com/problems/rotate-string/
def rotate_string(st,goal):
    if not st or not goal:
        return True
    if len(goal)!=len(st):
        return False
    st=st+st
    if goal in st:
        return True
    return False


print('37. Rotate String: ', rotate_string('abcde','cdeab'))
print('37. Rotate String: ', rotate_string('abcde','cdeba'))


# https://leetcode.com/problems/minimum-time-difference/
def findMinDifference(timePoints):
    res = []
    full = 24 * 60
    for time in timePoints:
        hr, min = time.split(':')
        total = int(hr) * 60 + int(min)
        if total == 0:
            total = full
        res.append(int(total))
    res.sort()
    #curMin = min(res[-1] - res[0], full - (res[-1] - res[0]))
    curMin = res[-1] - res[0] if res[-1] - res[0]<full-(res[-1] - res[0]) else full - (res[-1] - res[0])
    for i in range(1, len(res)):
        diff = abs(res[i - 1] - res[i])
        diff = diff if full-diff>diff else full - diff
        curMin = curMin if curMin<diff else diff
    return curMin


print('38. Find Min Time Difference in Mins: ', findMinDifference(["00:00","23:59","00:00"]))


# https://leetcode.com/problems/maximum-sum-bst-in-binary-tree/
def maxSumBST(root) -> int:
    res = 0

    def traverse(node):
        if not node:
            return 1, 0, None, None
        nonlocal res

        ls, l, ll, lr = traverse(node.left)
        rs, r, rl, rr = traverse(node.right)

        if ((ls == 2 and lr < node.val) or ls == 1) and ((rs == 2 and rl > node.val) or rs == 1):
            size = node.val + l + r
            res = max(res, size)
            return 2, size, ll if ll is not None else node.val, rr if rr is not None else node.val
        return 0, None, None, None

    traverse(root)
    return res


root = TreeNode(1)
root.left=TreeNode(4)
root.right= TreeNode(3)
root.left.left= TreeNode(2)
root.left.right= TreeNode(4)
root.right.left= TreeNode(2)
root.right.right= TreeNode(5)
root.right.right.left= TreeNode(4)
root.right.right.right = TreeNode(6)
print('39. Maximum Size BST branch: ', maxSumBST(root))