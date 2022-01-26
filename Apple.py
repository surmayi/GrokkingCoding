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


'''
from heapq import *


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
print('1. Serialize: ', data)
print('1. Deserialize: ',deserialize(data).val)


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
        print(serverList)
    print(maxLoad, ans)
    res = []
    for k, v in ans.items():
        if v == maxLoad:
            res.append(k)
    return res


arrival = [1, 2, 3]
load = [10, 12, 11]
print(busiestServer(3, arrival, load))
'''
