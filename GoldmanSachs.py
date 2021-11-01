import bisect
import math
from collections import deque


# https://leetcode.com/problems/robot-bounded-in-circle/
def isRobotBounded(instructions):
    directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
    idx = 0
    x, y = 0, 0

    for inst in instructions:
        if inst == 'L':
            idx = (idx + 3) % 4
        elif inst == 'R':
            idx = (idx + 1) % 4
        else:
            x += directions[idx][0]
            y += directions[idx][1]

    if (x == 0 and y == 0) or idx != 0:
        return True
    return False


print('1. isRobotBounded: ', str(isRobotBounded('GGLLGG')))
print('1. isRobotBounded: ', str(isRobotBounded('GGGLGLGLGG')))
print('1. isRobotBounded: ', str(isRobotBounded('GGGRGLGL')))
print('1. isRobotBounded: ', str(isRobotBounded('GG')))


# 2. https://leetcode.com/problems/find-pivot-index/
def find_pivot_index(nums):
    sumL, sumR = 0, sum(nums)
    for i in range(len(nums)):
        if sumL == sumR - nums[i]:
            return i
        sumL += nums[i]
        sumR -= nums[i]
    return -1


print('2. find_pivot_index: ', str((find_pivot_index([2, 1, -1]))))


# 3. Given an array of integers, print the array in such a way that the first element is first maximum and second element is first minimum and so on.
# https://www.geeksforgeeks.org/alternative-sorting/

def print_alternate_sorting(nums):
    nums.sort()
    n = len(nums) - 1
    i = 0
    while i < n:
        print(nums[n], end=' ')
        print(nums[i], end=' ')
        n -= 1
        i += 1


print('3. print_alternate_sorting: ', end='')
print_alternate_sorting([3, 1, 5, 3, 7, 9, 2, 5])
print('\n3. Alternate Sorting: ', end='')
print_alternate_sorting([7, 1, 2, 3, 4, 5, 6])
print('\n3. Alternate sorting: ', end='')
print_alternate_sorting([1, 6, 9, 4, 3, 7, 8, 2])


# 4. Remove duplicates from linkedlist
# O(N)
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None


class LinkedList:
    def __init__(self, head=None):
        self.head = head

    def append(self, val):
        if not self.head:
            self.head = Node(val)
            return
        cur = self.head
        while cur.next:
            cur = cur.next
        cur.next = Node(val)

    def remove_duplicates(self):
        duplicates = []
        cur = self.head
        prev = None
        while cur:
            if cur.val not in duplicates:
                duplicates.append(cur.val)
                prev = cur
            else:
                prev.next = cur.next
            cur = prev.next

    def print_list(self):
        cur = self.head
        while cur:
            print(cur.val, end='->')
            cur = cur.next
        print()

    # https://leetcode.com/problems/reverse-linked-list/
    def reverse_list(self):
        prev = None
        while self.head:
            nxt = self.head.next
            self.head.next = prev
            prev = self.head
            self.head = nxt
        self.head=prev


ll = LinkedList()
ll.append(1)
ll.append(2)
ll.append(1)
ll.append(3)
ll.append(3)
ll.append(4)
ll.append(3)
print('\n4.  Remove duplicates from linked list: ')
ll.remove_duplicates()
ll.print_list()
print('Reverse linked list: ')
ll.reverse_list()
ll.print_list()


# 5. https://leetcode.com/problems/minimum-size-subarray-sum/
# O(N)
def minSubArrayLen(nums, target):
    winStart, res, winSum = 0, math.inf, 0
    for winEnd in range(len(nums)):
        winSum += nums[winEnd]
        while winSum >= target:
            res = min(res, winEnd - winStart + 1)
            winSum -= nums[winStart]
            winStart += 1
    return 0 if res == math.inf else res


print('5. minSubArrayLen', str(minSubArrayLen([2, 3, 1, 2, 4, 3], 7)))
print('5. minSubArrayLen', str(minSubArrayLen([1, 4, 4], 4)))
print('5. minSubArrayLen', str(minSubArrayLen([1, 1, 1, 1, 1, 1, 1, 1], 11)))


# 8. https://leetcode.com/problems/string-compression/
# time - O(N)
def string_compression(chars):
    res = ''
    prev = chars[0]
    count = 0
    for i in range(len(chars)):
        if chars[i] == prev:
            count += 1
        else:
            if count == 1:
                res += prev
            else:
                res += prev + str(count)
            count = 1
            prev = chars[i]
    if count == 1:
        res += prev
    else:
        res += prev + str(count)
    for i in range(len(res)):
        chars[i] = res[i]

    return len(res)


nums = ["a", "a", "b", "b", "c", "c", "c"]
res = string_compression(nums)
print('6. string_compression, length- ', res, ' result- ', ''.join(nums[:res]))

nums = ["a", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b"]
res = string_compression(nums)
print('6. string_compression, length- ', res, ' result- ', ''.join(nums[:res]))


# Input:  [{"Bob","87"], {"Mike", "35"],{"Bob", "52"], {"Jason","35"], {"Mike", "55"], {"Jessica", "99"]]
# Output: 99
# Explanation: Since Jessica's average is greater than Bob's, Mike's and Jason's average.
# O(N)
def find_max_average(input):
    avgMap = {}
    for name, avg in input:
        if name in avgMap:
            avgMap[name][0] += int(avg)
            avgMap[name][1] += 1
        else:
            avgMap[name] = [int(avg), 1]
    maxAvg = 0
    for name, avgs in avgMap.items():
        avg = avgs[0] / avgs[1]
        maxAvg = max(maxAvg, avg)
    return maxAvg


avgs = [("Bob", "87"), ("Mike", "35"), ("Bob", "52"), ("Jason", "35"), ("Mike", "55"), ("Jessica", "99")]
print('7. find_max_average: ', find_max_average(avgs))


# https://leetcode.com/problems/first-unique-character-in-a-string/
# leetcode -> l
# O(N)
def find_first_unique_char(string):
    charMap = {}
    ordered = []
    for ch in string:
        charMap[ch] = charMap.get(ch, 0) + 1
        if ch not in ordered:
            ordered.append(ch)
    for val in ordered:
        if charMap[val] == 1:
            return val


string = 'leetcodelove'
print('8. find_first_unique_char', find_first_unique_char(string))


#  https://www.geeksforgeeks.org/return-maximum-occurring-character-in-the-input-string/
def find_maximum_occuring_char_instring(string):
    strMap = {}
    maxFreq, res = 0, ''
    for ch in string:
        strMap[ch] = strMap.get(ch, 0) + 1
        if strMap[ch] > maxFreq:
            maxFreq = strMap[ch]
            res = ch
    return res


string = 'loeooeetcodelove'
print('9. find_maximum_occuring_char_instring', find_maximum_occuring_char_instring(string))


# https://leetcode.com/problems/median-of-two-sorted-arrays/
# https://leetcode.com/problems/median-of-two-sorted-arrays/discuss/1532231/Binary-Search-Explained
def findMedianSortedArrays(nums1, nums2):
    nums = nums1 + nums2
    nums.sort()
    n = len(nums)
    if n % 2 == 1:
        return nums[n // 2]
    return (nums[(n - 1) // 2] + nums[n // 2]) / 2


print('10. findMedianSortedArrays', findMedianSortedArrays([1, 2, 4, 4, 5], [1, 2, 5, 9, 10]))


# #Problem #31 : Longest Uniform Substring
def longestUniformSubString(str):
    winStart, maxLen, vals, res = 0, 0, {}, [-1, -1]
    for winEnd in range(len(str)):
        right = str[winEnd]
        vals[right] = vals.get(right, 0) + 1
        while len(vals) > 1:
            left = str[winStart]
            vals[left] -= 1
            if vals[left] == 0:
                del vals[left]
            winStart += 1
        if winEnd - winStart + 1 > maxLen:
            maxLen = winEnd - winStart + 1
            res = [winStart, winEnd + 1]
    return str[res[0]:res[1]]


print('11. longestUniformSubString: ', longestUniformSubString('aaabccccbbbbba'))


# https://www.geeksforgeeks.org/find-starting-indices-substrings-string-s-made-concatenating-words-listl/
def subStringIndexWithConcatationOfWordList(arr, words):
    word_len = len(words[0])
    word_count = len(words)
    wordMap = {}
    result = []
    for word in words:
        wordMap[word] = wordMap.get(word, 0) + 1
    for i in range(len(arr) - word_len * word_count + 1):
        word_seen = {}
        for j in range(word_count):
            next_word_ind = i + j * word_len
            next_word = arr[next_word_ind:next_word_ind + word_len]
            if next_word not in wordMap:
                break
            word_seen[next_word] = word_seen.get(next_word, 0) + 1
            if word_seen[next_word] > wordMap[next_word]:
                break
            if j + 1 == word_count:
                result.append(i)
    return result


print('12. subStringIndexWithConcatationOfWordList: ',
      subStringIndexWithConcatationOfWordList('catfoxcat', ['cat', 'fox']))


# https://leetcode.com/problems/string-to-integer-atoi/

def myAtoi(s):
    if len(s) == 0:
        return 0
    sign = 1
    i, l = 0, len(s)
    while i < l and s[i] == ' ':
        i += 1
    if i < l and s[i] == '-':
        sign = -1
        i += 1
    elif i < l and s[i] == '+':
        i += 1
    num = []
    while i < l and s[i].isnumeric():
        num.append(s[i])
        i += 1
    if len(num) == 0:
        return 0
    num = int(''.join(num))
    num *= sign
    if num > 2 ** 31 - 1:
        return 2 ** 31 - 1
    elif num < -2 ** 31:
        return -2 ** 31
    else:
        return num


print('13. stringToInt', myAtoi('12312312312323'))
print('13. stringToInt', myAtoi('-123123123'))
print('13. stringToInt', myAtoi('  -23123'))


# https://leetcode.com/problems/group-anagrams/
def groupAnagrams(strs):
    dic = dict([])
    for word in strs:
        temp_list = [0 for i in range(26)]
        for ch in word:
            temp_list[ord(ch) - ord('a')] += 1
        dic[tuple(temp_list)] = dic.get(tuple(temp_list), []) + [word]
    return dic.values()


print('14. groupAnagrams: ', groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))


# https://www.geeksforgeeks.org/to-find-smallest-and-second-smallest-element-in-an-array/
def print2Smallest(arr):
    if len(arr) <= 2:
        return arr
    first = second = math.inf
    for i in range(len(arr)):
        if arr[i] < first:
            second = first
            first = arr[i]
        elif arr[i] < second and arr[i] != first:
            second = arr[i]
    return [first, second]


print('15. print2Smallest: ', print2Smallest([3, 5, 1, 6, 12, 34, 8, 0, 333]))


# 4. https://leetcode.com/problems/reaching-points/
# log(max(ty,tx))
def reachingPoints(sx, sy, tx, ty):
    while tx >= sx and ty >= sy:
        if tx == ty:
            return sx == tx and sy == ty
        elif tx > ty:
            if ty > sy:
                tx = tx % ty
            else:
                return (tx - sx) % sy == 0
        else:
            if tx > sx:
                ty = ty % tx
            else:
                return (ty - sy) % sx == 0
    return False


print('16.reaching points: ', str(reachingPoints(2, 1, 5, 3)))
print('16.reaching points: ', str(reachingPoints(2, 1, 5, 4)))
print('16.reaching points: ', str(reachingPoints(3, 3, 12, 9)))


# 7. Graph - https://leetcode.com/problems/number-of-provinces/
# O(n^2)
def findProvinceCount(isConnected):
    n = len(isConnected)
    visited = [False for j in range(n)]
    q = deque()
    count = 0
    for i in range(n):
        if not visited[i]:
            visited[i] = True
            q.append(i)
            count += 1
            while q:
                node = q.popleft()
                for j in range(n):
                    if isConnected[node][j] == 1 and not visited[j]:
                        visited[j] = True
                        q.append(j)
    return count


isConnected = [[1, 1, 0], [1, 1, 0], [0, 0, 1]]
print('17. Province count ', str(findProvinceCount(isConnected)))
isConnected = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
print('17. Province count ', str(findProvinceCount(isConnected)))


# https://www.geeksforgeeks.org/size-of-the-largest-trees-in-a-forest-formed-by-the-given-graph/
# O(V+E)

def largestTreeSize(nodes, edges):
    if nodes <= 0 or not edges:
        return 0
    graph = {i: [] for i in range(nodes)}
    visited = [False for i in range(nodes)]

    for edge in edges:
        v1, v2 = edge[0], edge[1]
        graph[v1].append(v2)
        graph[v2].append(v1)
    count = 0
    for i in range(nodes):
        if not visited[i]:
            count = max(count, largestTreeSize_helper(graph, visited, i))
    return count


def largestTreeSize_helper(graph, visited, i):
    visited[i] = True
    size = 1
    for j in range(len(graph[i])):
        if not visited[graph[i][j]]:
            size += largestTreeSize_helper(graph, visited, graph[i][j])
    return size


V = 7
edges = [[0, 1], [0, 2], [3, 4], [4, 6], [3, 5]]

print('18. largestTreeSize: ', str(largestTreeSize(V, edges)))


# https://leetcode.com/problems/high-five/
# NlogN
def highFive(items):
    vals = {}
    result = []
    for item in items:
        if item[0] not in vals:
            vals[item[0]] = [item[1]]
        else:
            vals[item[0]].append(item[1])
    for id, score in vals.items():
        score.sort()
        result.append([id, sum(score[-5:]) // 5])
    result.sort(key=lambda x: x[0])
    return result


scores = [[1, 91], [1, 92], [2, 93], [2, 97], [1, 60], [2, 77], [1, 65], [1, 87], [1, 100], [2, 100], [2, 76]]
print('19. high Five: ', str(highFive(scores)))


# https://leetcode.com/problems/height-checker/submissions/
def heightChecker(heights):
    real = list(heights)
    real.sort()
    count = 0
    for i in range(len(heights)):
        if heights[i] != real[i]:
            count += 1
    return count


print('20. height wise stand', str(heightChecker([5, 2, 3, 4, 1])))


# https://leetcode.com/problems/coin-change/
# 2^n - Time Limit Exceeded

def coinChange_recursion(coins, amount):
    if amount == 0:
        return 0
    n = amount + 1
    for coin in coins:
        if coin <= amount:
            nxt = coinChange_recursion(coins, amount - coin)
            if nxt >= 0:
                n = min(n, 1 + nxt)
    return -1 if n == amount + 1 else n


print('21. coin change - Exceed time lmit: ', coinChange_recursion([1, 3, 5], 11))


def coinChange(coins, amount):
    if not coins or amount <= 0:
        return 0
    dp = [math.inf for i in range(amount + 1)]
    dp[0] = 0
    for coin in coins:
        for j in range(coin, amount + 1):
            dp[j] = min(dp[j], dp[j - coin] + 1)
    return dp[amount] if dp[amount] < math.inf else -1


print('21. coin change: ', coinChange([1, 3, 5], 11))


# https://leetcode.com/problems/trapping-rain-water/
def trapRainwater(height):
    if not height:
        return 0
    res = 0
    left, right = 0, len(height) - 1
    leftMax, rightMax = height[left], height[right]
    while left < right:
        if leftMax < rightMax:
            left += 1
            leftMax = max(leftMax, height[left])
            res += leftMax - height[left]
        else:
            right -= 1
            rightMax = max(rightMax, height[right])
            res += rightMax - height[right]
    return res


print('22. trap Rainwater: ', str(trapRainwater([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1])))


# https://www.geeksforgeeks.org/count-inversions-of-size-three-in-a-give-array/
# O(N^2)
def countInversions(arr):
    if not arr:
        return 0
    count = 0
    n = len(arr)
    for i in range(n):
        largeLeft = 0
        smallRight = 0
        for j in range(i):
            if arr[j] > arr[i]:
                largeLeft += 1
        for j in range(i + 1, n):
            if arr[j] < arr[i]:
                smallRight += 1
        count += smallRight * largeLeft
    return count


print('23. count inversions-4: ', countInversions([8, 4, 2, 1]))
print('23. count inversions-2: ', countInversions([9, 6, 4, 5, 8]))


# https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/
def shortestSubArratWithSum_atleastK(nums, k):
    if not nums:
        return -1
    sums = [0 for i in range(len(nums) + 1)]
    for i in range(1, len(sums)):
        sums[i] = sums[i - 1] + nums[i - 1]

    cur = len(nums) + 1
    que = deque()
    for i in range(len(sums)):
        while que and sums[i] - sums[que[0]] >= k:
            cur = min(cur, i - que.popleft())
        while que and sums[i] < sums[que[-1]]:
            que.pop()
        que.append(i)
    return -1 if cur > len(nums) else cur


print('24. shortestSubArrayWithSum_atleastK: ', shortestSubArratWithSum_atleastK([2, -1, 2], 3))


# https://leetcode.com/problems/longest-word-in-dictionary-through-deleting/
def findLongestWord(string, dictionary):
    ans = ''
    for word in dictionary:
        if len(word) < len(ans) or (len(word) == len(ans) and word > ans):
            continue
        pos = -1
        for ch in word:
            if ch not in string[pos + 1:]:
                pos = -1
                break
            pos = string.index(ch, pos + 1)
        if pos != -1:
            ans = word
    return ans


print('25. findLongestWord: ', findLongestWord('abpcplea', ["ale", "apple", "monkey", "plea"]))


# https://www.geeksforgeeks.org/lexicographical-maximum-substring-string/
def LexicographicalMaxString(str):
    maxStr = ''
    for i in range(len(str)):
        maxStr = max(maxStr, str[i:])
    return maxStr


print('26. LexicographicalMaxString: ', LexicographicalMaxString('acbacbc'))


# https://www.geeksforgeeks.org/josephus-problem-set-1-a-on-solution/
def josephusProblem(n, k):
    result = [0 for i in range(len(n))]
    for i in range(len(result)):
        result[i] = n[i]
    return helper_josephus(result, 0, k)


def helper_josephus(result, start, k):
    if len(result) == 1:
        return result[0]
    start = (start + k) % len(result)
    del result[start]
    return helper_josephus(result, start, k)


# O(N) - Best one
def josephus(nums, k):
    start = 0
    while len(nums) > 1:
        start = (start + k) % len(nums)
        del nums[start]
    return nums[-1]


def josephus2(n, k):
    if n == 1:
        return 1
    else:
        return (josephus2(n - 1, k) + k - 1) % n + 1


print('27. Josephus problem: ', josephusProblem([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 2))
print('27. Josephus problem- Best: ', josephus([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 2))
print('27. Choosen place for Josephus to spare person: ', josephus2(14, 2))


# https://leetcode.com/problems/diagonal-traverse/submissions/
def diagonal_traversal(mat):
    result = []
    row, col = 0, 0
    rows, cols = len(mat), len(mat[0])
    upward = True
    for _ in range(rows * cols):
        result.append(mat[row][col])
        if upward:
            if col + 1 == cols:
                row += 1
                upward = False
            elif row == 0:
                col += 1
                upward = False
            else:
                row -= 1
                col += 1
        else:
            if row + 1 == rows:
                col += 1
                upward = True
            elif col == 0:
                row += 1
                upward = True
            else:
                row += 1
                col -= 1
    return result


print('28. Diagonal Traversal of matrix: ', str(diagonal_traversal([[1, 2, 3], [4, 5, 6], [7, 8, 9]])))


# O(L1*L2) - length of both strings
def longestCommonSubstring(string1, string2):
    l1, l2 = len(string1), len(string2)
    dp = [[0 for j in range(l2 + 1)] for i in range(l1 + 1)]
    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            if string1[i - 1] == string2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[l1][l2]


def longest_common_substring(str1, str2):
    vals1, vals2 = {}, {}
    for ch in str1:
        vals1[ch] = vals1.get(ch, 0) + 1
    for ch in str2:
        vals2[ch] = vals2.get(ch, 0) + 1
    count = 0
    for key, freq in vals1.items():
        if key in vals2:
            count += min(vals2[key], freq)
    return count


X = "AGGTAB"
Y = "GXTXGAYB"
print("29. Length of LCS is ", longestCommonSubstring(X, Y))
print("29. Length of LCS2 is ", longest_common_substring(X, Y))


# https://www.geeksforgeeks.org/print-all-prime-factors-of-a-given-number/
# Time - O(sqrt(n))
def primeFactorizers(n):
    if n <= 2:
        return 0
    result = []
    while n % 2 == 0:
        n //= 2
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        while n % i == 0:
            result.append(i)
            n //= i
    if n > 1:
        result.append(n)
    return result


print('30. Prime Factorizers: ', str(primeFactorizers(315)))


# House robber - https://leetcode.com/problems/house-robber/
def rob(nums):
    if not nums:
        return 0
    prev_prev, prev = 0, 0,
    for i in range(len(nums)):
        cur = max(prev_prev + nums[i], prev)
        prev_prev = prev
        prev = cur
    return prev


print('31. House Robber max Amount:', rob([1, 2, 3, 1]))


# Delete and earn - https://leetcode.com/problems/delete-and-earn/
# Convert this problem into house robber
# We first transform the nums array into a points array that sums up the total number of points for that particular value. A value of x will be assigned to index x in points.
# nums: [2, 2, 3, 3, 3, 4] (2 appears 2 times, 3 appears 3 times, 4 appears once)
# points: [0, 0, 4, 9, 4] <- This is the gold in each house!
def deleteAndEarn(nums):
    if not nums:
        return 0
    sums = [0 for i in range(max(nums) + 1)]
    for num in nums:
        sums[num] += num
    prev_prev, prev = 0, 0
    for i in range(len(sums)):
        cur = max(prev_prev + sums[i], prev)
        prev_prev = prev
        prev = cur
    return prev


print('32. Delete and Earn (Reduce to house robber)', deleteAndEarn([1, 2, 3, 4]))
print('32. Delete and Earn (Reduce to house robber)', deleteAndEarn([2, 1, 2, 3, 2, 2]))


# Consecutive numbers sum - https://leetcode.com/problems/consecutive-numbers-sum/
# O(sqrt(n))
def consecutiveNumbersSum(n):
    if n <= 1:
        return n
    k = 1
    ans = 0
    while k < n:
        kx = n - k * (k - 1) / 2
        if kx <= 0:
            break
        if kx % k == 0:
            ans += 1
        k += 1
    return ans


print('33. Count of sets of Sum of consecutive number equal to n', consecutiveNumbersSum(15))
print('33. Count of sets of Sum of consecutive number equal to n', consecutiveNumbersSum(9))


# https://leetcode.com/problems/count-number-of-teams/
def CountNumberOfTeams(ratings):
    def cal(input):
        count = 0
        for i in range(len(input)):
            smallBefore, largeAfter = 0, 0
            for j in range(i - 1, -1, -1):
                if input[j] < input[i]:
                    smallBefore += 1
            for j in range(i + 1, len(input)):
                if input[j] > input[i]:
                    largeAfter += 1
            count += largeAfter * smallBefore
        return count

    result = 0
    result += cal(ratings)
    result += cal(ratings[::-1])
    return result


print('34. Count Number of Teams: ', CountNumberOfTeams([2, 5, 3, 4, 1]))
print('34. Count Number of Teams: ', CountNumberOfTeams([2, 1, 3]))


# https://leetcode.com/problems/fraction-addition-and-subtraction/
# o(N)
def fractionAdditionAndSubtraction(expression):
    if not expression:
        return

    groupedVals = getFractions_helper(expression)
    darr, narr = [], []

    for val in groupedVals:
        val = val.split('/')
        narr.append(int(val[0]))
        darr.append(int(val[1]))

    lcm = darr[0]
    for i in range(1, len(darr)):
        lcm = lcm_helper(lcm, darr[i])

    for i in range(len(narr)):
        narr[i] = int(lcm / darr[i]) * narr[i]

    totalSum = sum(narr)
    gcd = gcd_helper(totalSum, lcm)
    return str(totalSum // gcd) + '/' + str(lcm // gcd)


def gcd_helper(x, y):
    while y:
        x, y = y, x % y
    return x


def lcm_helper(x, y):
    greater = max(x, y)
    while greater % x != 0 or greater % y != 0:
        greater += 1
    return greater


def getFractions_helper(expression):
    result = []
    frac = ''
    for ex in expression:
        if ex in ['+', '-'] and '/' in frac:
            result.append(frac)
            frac = ''
        frac += ex
    result.append(frac)
    return result


print('35. fractionAdditionAndSubtraction: ', fractionAdditionAndSubtraction('-1/2+1/2'))


# https://leetcode.com/problems/fraction-to-recurring-decimal/
# O(D) - non repeating digits after decimal
def fractionTODecimal(numerator, denominator):
    isNeg = (numerator > 0 and denominator < 0) or (numerator < 0 and denominator > 0)
    n, d = abs(numerator), abs(denominator)
    res = [str(n // d)]
    index = 1
    n = n % d
    if n != 0:
        res.append('.')
        index += 1
    dic = {}
    while n > 0:
        if n in dic:
            res = res[:dic[n]] + ['('] + res[dic[n]:] + [')']
            break
        else:
            dic[n] = index
        n = n * 10
        res.append(str(n // d))
        index += 1
        n = n % d
    return '-' + ''.join(res) if isNeg else ''.join(res)


print('36. fraction to Decimal: ', fractionTODecimal(2, 3))
print('36. fraction to Decimal: ', fractionTODecimal(4, 333))
print('36. fraction to Decimal: ', fractionTODecimal(2, 4))


# https://leetcode.com/problems/remove-duplicates-from-an-unsorted-linked-list/
# O(N)
def removeAllDuplicatesUnsorted(head):
    if not head:
        return []
    prev, cur = None, head
    dupList = {}
    while cur:
        dupList[cur.val] = dupList.get(cur.val, 0) + 1
        cur = cur.next
    cur = head
    while cur:
        if dupList[cur.val] > 1:
            if cur == head:
                head = head.next
                cur = head
            else:
                prev.next = cur.next
                cur = prev.next
        else:
            prev = cur
            cur = cur.next
    return head


ll = Node(6)
ll.next = Node(1)
ll.next.next = Node(12)
ll.next.next.next = Node(1)
ll.next.next.next.next = Node(2)
ll.next.next.next.next.next = Node(2)
ll.next.next.next.next.next.next = Node(3)
ll.next.next.next.next.next.next.next = Node(1)
head = removeAllDuplicatesUnsorted(ll)
print('37.  Remove duplicates from linked list: ', )
while head:
    print(head.val, end='->')
    head = head.next


# https://leetcode.com/problems/h-index/
# O(N)
def find_H_Index(citations):
    citations.sort(reverse=True)
    n = len(citations)
    for i in range(n):
        if citations[i] <= i:
            return i
    return n


print('38. Find H-Index of scientist: ', find_H_Index([3, 0, 6, 1, 5]))


# https://leetcode.com/problems/longest-word-in-dictionary/
def longestWordInDictionary(words):
    res = ''
    for word in words:
        isValid = True
        for i in range(1, len(word)):
            if word[:i] not in words:
                isValid = False
                break
        if isValid and len(word) >= len(res):
            if len(word) == len(res):
                res = word if word < res else res
            else:
                res = word
    return res


def longestWordInDictionaryOptimised(words):
    res, word_set = '', set([''])
    for word in words:
        if word[:-1] in word_set:
            word_set.add(word)
            if len(word) > len(res):
                res = word
            if len(word) == len(res):
                res = word if word < res else res
    return res


print('39. Longest World in Dictionary: ', longestWordInDictionary(["w", "wo", "wor", "worl", "world"]))
print('39. Longest World in Dictionary Optimised: ',
      longestWordInDictionaryOptimised(["w", "wo", "wor", "worl", "world"]))


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
print('40: Get 1- Present ', myHashMap.get(1))
print('40: Get 3- Absent ', myHashMap.get(3))
myHashMap.put(2, 1)
print('40: Get 2- updated ', myHashMap.get(2))
myHashMap.remove(2)
print('40: Get 2- Absent ', myHashMap.get(2))


# https://leetcode.com/problems/lru-cache/
class DLLNode:
    def __init__(self):
        self.key = 0
        self.value = 0
        self.prev = None
        self.next = None


class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.size = 0
        self.head, self.tail = DLLNode(), DLLNode()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_node(self, node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node):
        prv = node.prev
        nxt = node.next
        prv.next = nxt
        nxt.prev = prv

    def _move_to_head(self, node):
        self._remove_node(node)
        self._add_node(node)

    def _pop_tail(self):
        node = self.tail.prev
        self._remove_node(node)
        return node

    def get(self, key):
        node = self.cache.get(key, None)
        if not node:
            return -1
        self._move_to_head(node)
        return node.value

    def put(self, key, value):
        node = self.cache.get(key, None)
        if not node:
            newNode = DLLNode()
            newNode.key, newNode.value = key, value
            self.cache[key] = newNode
            self._add_node(newNode)
            self.size += 1
            if self.size > self.capacity:
                tail = self._pop_tail()
                del self.cache[tail.key]
                self.size -= 1
        else:
            node.value = value
            self._move_to_head(node)


lRUCache = LRUCache(2)
print('41. Put 1,1: ', lRUCache.put(1, 1))
print('41. Put 2,2: ', lRUCache.put(2, 2))
print('41. Get 1: ', lRUCache.get(1))
print('41. Put 3,3- 2 deleted: ', lRUCache.put(3, 3))
print('41. Get 2: -Absent ', lRUCache.get(2))
print('41. Put 4,4: - 1 deleted ', lRUCache.put(4, 4))
print('41. Get 1 - Absent: ', lRUCache.get(1))
print('41. Get 3 : ', lRUCache.get(3))
print('41. Get 4 : ', lRUCache.get(4))


# Array Game - https://www.geeksforgeeks.org/minimum-number-increment-operations-make-array-elements-equal/
def array_game_brute_force(nums):
    count = 0
    equal = False
    while True:
        for i in range(1, len(nums)):
            if nums[i - 1] != nums[i]:
                equal = False
                break
            equal = True
        if equal:
            break
        maxEle = max(nums)
        incr = False
        for i in range(len(nums)):
            if not incr and nums[i] == maxEle:
                incr = True
            elif incr or nums[i] != maxEle:
                nums[i] += 1
        count += 1
    print(nums)
    return count


print('42. array_game_brute_force: ', array_game_brute_force([3, 4, 6, 6, 3]))
print('42. array_game_brute_force: ', array_game_brute_force([4, 3, 4]))
print('42. array_game_brute_force: ', array_game_brute_force([1, 2, 3]))


def array_game_optimized(nums):
    smallest = min(nums)
    return sum(nums) - smallest * len(nums)


print('42. array_game_optimized: ', array_game_optimized([3, 4, 6, 6, 3]))
print('42. array_game_optimized: ', array_game_optimized([4, 3, 4]))
print('42. array_game_optimized: ', array_game_optimized([1, 2, 3]))


# Problem 61 - Like 2 sum, but get all pairs
def profit_targets(nums, k):
    nums.sort()
    count = 0
    left, right = 0, len(nums) - 1
    while left < right:
        s = nums[left] + nums[right]
        if s == k:
            count += 1
            left += 1
            right -= 1
            while left < right and nums[left] == nums[left - 1]:
                left += 1
            while right >= 0 and nums[right] == nums[right - 1]:
                right -= 1
        elif s < k:
            left += 1
        else:
            right -= 1
    return count


print('43. profit targets: ', profit_targets([5, 7, 9, 9, 13, 5, 11, 6, 6, 3, 3], 12))
print('43. profit targets: ', profit_targets([1, 3, 46, 1, 3, 9], 47))


# https://leetcode.com/problems/longest-increasing-subsequence/
# O(n2)
def longest_subsequence(nums):
    sub = [nums[0]]
    for num in nums:
        if num > sub[-1]:
            sub.append(num)
        else:
            j = 0
            while num > sub[j]:
                j += 1
            sub[j] = num
    return len(sub)


print('44. Longest Increasing Subsequence: ', longest_subsequence([0, 3, 1, 6, 2, 2, 7]))
print('44. Longest Increasing Subsequence: ', longest_subsequence([1, 10, 9, 2, 5, 3, 7, 101, 18]))


# O(nlogn)
def longest_subsequence_optimized_binary_left(nums):
    sub = []
    for num in nums:
        ind = bisect.bisect_left(sub, num)
        if ind == len(sub):
            sub.append(num)
        else:
            sub[ind] = num
    return len(sub)


print('44. Longest Increasing Subsequence: ', longest_subsequence_optimized_binary_left([0, 3, 1, 6, 2, 2, 7]))
print('44. Longest Increasing Subsequence: ',
      longest_subsequence_optimized_binary_left([1, 10, 9, 2, 5, 3, 7, 101, 18]))


# https://leetcode.com/discuss/interview-question/571497/hackkerank-online-test-software-developer
# O(N)
def Even_or_odd_multiplication_array(nums):
    n = len(nums)
    Reven, Rodd = nums[0], nums[1]
    isAdd = False
    for i in range(2, n - 1, 2):
        if isAdd:
            Reven += nums[i]
            Rodd += nums[i + 1]
            isAdd = False
        else:
            Reven *= nums[i]
            Rodd *= nums[i]
            isAdd = True
    Reven, Rodd = Reven % 2, Rodd % 2
    if Reven > Rodd:
        return 'EVEN'
    elif Reven < Rodd:
        return 'ODD'
    else:
        return 'NEUTRAL'


print('45: Even_or_odd_multiplication_array: ', Even_or_odd_multiplication_array([12, 3, 5, 7, 13, 12]))
print('45: Even_or_odd_multiplication_array: ', Even_or_odd_multiplication_array([1, 2, 1]))
print('45: Even_or_odd_multiplication_array: ', Even_or_odd_multiplication_array([2, 3, 8]))


# https://leetcode.com/problems/rotate-array/
# O(N)
def rotateArray_from_right_optimised(nums, k):
    n = len(nums)
    k = k % n

    def rotate_array(nums, start, end):
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1

    rotate_array(nums, 0, n - 1)
    rotate_array(nums, 0, k - 1)
    rotate_array(nums, k, n - 1)
    return nums


print('46. rotateArray_from_right: ', rotateArray_from_right_optimised([1, 2, 3, 4, 5, 6, 7, 8], 3))


# https://leetcode.com/problems/implement-trie-prefix-tree/
# O(N)
class TrieNode:
    def __init__(self):
        self.children = {}
        self.isEndWord = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert_word(self, word):
        cur = self.root
        for ch in word:
            if ch not in cur.children:
                cur.children[ch] = TrieNode()
            cur = cur.children[ch]
        cur.isEndWord = True

    def search_word(self, word):
        cur = self.root
        for ch in word:
            if ch not in cur.children:
                return False
            cur = cur.children[ch]
        return cur.isEndWord

    def starts_with(self, prefix):
        cur = self.root
        for ch in prefix:
            if ch not in cur.children:
                return False
            cur = cur.children[ch]
        return True


trie = Trie();
trie.insert_word("apple")
print('47. Search word in Trie: ', trie.search_word("apple"))
print('47. Search word in Trie: ', trie.search_word("app"))
print('47. Search prefic in Trie: ', trie.starts_with("app"))
trie.insert_word("app")
print('47. Search word in Trie: ', trie.search_word("app"))


# https://leetcode.com/problems/longest-palindromic-substring/
# O(N^2)
def longest_substring_palindrome(string):
    maxLen, maxPal = 0, ''
    n = len(string)
    for i in range(n * 2 - 1):
        left, right = i // 2, (i + 1) // 2
        while left >= 0 and right < n and string[left] == string[right]:
            if right - left + 1 > maxLen:
                maxLen = right - left + 1
                maxPal = string[left:right + 1]
            right += 1
            left -= 1
    return maxPal


print('48. longest_substring_palindrome: ', longest_substring_palindrome('babad'))
print('48. longest_substring_palindrome: ', longest_substring_palindrome('bbca'))


# https://leetcode.com/problems/palindromic-substrings/
# O(N)
def count_palindromic_substrings(string):
    count = 0
    n = len(string)
    for i in range(2 * n - 1):
        left, right = i // 2, (i + 1) // 2
        while left >= 0 and right < n and string[left] == string[right]:
            left -= 1
            right += 1
            count += 1
    return count


print('49. count_palindromic_substrings: ', count_palindromic_substrings('aaa'))
print('49. count_palindromic_substrings: ', count_palindromic_substrings('abc'))


# https://leetcode.com/discuss/interview-question/1321204/efficient-harvest-faang-oa-question-2021
def efficient_harvest(arr, k):
    maxProfit = 0
    n = len(arr)
    for i in range(n // 2):
        profit = 0
        for j in range(i, i + k):
            j2 = (j + n // 2) % n
            profit += arr[j] + arr[j2]
        maxProfit = max(maxProfit, profit)
    return maxProfit


print('50. efficient_harvest: ', efficient_harvest([-3, 7, 3, 1, 5, 1], 2))
print('50. efficient_harvest: ', efficient_harvest([-3, 3, 6, 1], 1))


# https://leetcode.com/problems/path-with-maximum-gold/
# O(N*M)
def get_maximum_gold(grid):
    rows = len(grid)
    cols = len(grid[0])

    def dfs(i, j):
        if 0 <= i < rows and 0 <= j < cols and grid[i][j] != 0:
            money = grid[i][j]
            grid[i][j] = 0
            p1 = dfs(i + 1, j)
            p2 = dfs(i - 1, j)
            p3 = dfs(i, j + 1)
            p4 = dfs(i, j - 1)
            grid[i][j] = money
            return max(p1, p2, p3, p4) + money
        else:
            return 0

    money = 0
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != 0:
                money = max(money, dfs(i, j))
    return money


print('51. get_maximum_gold: ', get_maximum_gold([[1, 0, 7], [2, 0, 6], [3, 4, 5], [0, 3, 0], [9, 0, 20]]))
print('51. get_maximum_gold: ', get_maximum_gold([[0, 6, 0], [5, 8, 7], [0, 9, 0]]))


# https://leetcode.com/discuss/interview-question/949160/goldman-sachs-phone-most-frequent-ip-address-from-the-logs
# both time and space - O(N)
def most_frequent_ip(logs):
    ipMap = {}
    maxFreq = 0
    for log in logs:
        ip = log.split()[0]
        ipMap[ip] = ipMap.get(ip, 0) + 1
        maxFreq = max(maxFreq, ipMap[ip])
    result = []
    for key, val in ipMap.items():
        if val == maxFreq:
            result.append(key)
    return result


print('52. most_frequent_ip: ', most_frequent_ip(
    ["10.0.0.1 - GET 2020-08-24", "10.0.0.1 - GET 2020-08-24", "10.0.0.2 - GET 2020-08-20",
     "10.0.0.2 - GET 2020-08-20" "10.0.0.3 - GET 2020-08-20"]))


# https://newbedev.com/find-the-number-of-unordered-pair-in-an-array
# Merge sort O(nlogn)
# Implement merge sort, place a counter and increase it whenever right<left while merging.
def find_number_of_unordered_pair(arr):
    result, merged = merge_sort(arr)
    return result


def merge_sort(arr):
    if len(arr) <= 1:
        return 0, arr
    mid = len(arr) // 2
    leftcount, left = merge_sort(arr[:mid])
    rightcount, right = merge_sort(arr[mid:])
    mergecount, merged = merge(left, right)
    total = mergecount + leftcount + rightcount
    return total, merged


def merge(left, right):
    merged = []
    count = 0
    ll, lr = len(left), len(right)
    i, j = 0, 0
    while i < ll and j < lr:
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            count += 1
            merged.append(right[j])
            j += 1
    if i < ll:
        merged.extend(left[i:])
    if j < lr:
        merged.extend(right[j:])
    return count, merged


print('53. find_number_of_unordered_pair: ', find_number_of_unordered_pair([7, 2, 0, 4, 5, 4, 6, 7]))


# https://leetcode.com/problems/power-of-three/
def is_power_of_three(n):
    while n > 1:
        if n % 3 != 0:
            return False
        n //= 3
    return n == 1


print('54. is_power_of_three: ', is_power_of_three(27))
print('54. is_power_of_three: ', is_power_of_three(45))


# https://leetcode.com/problems/longest-substring-without-repeating-characters/
def longest_substring_without_repeating_chars(string):
    winStart, vals = 0, {}
    maxLen = 0
    for winEnd in range(len(string)):
        right = string[winEnd]
        if right in vals:
            winStart = max(winStart, vals[right] + 1)
        vals[right] = winEnd
        maxLen = max(maxLen, winEnd - winStart + 1)
    return maxLen


print('55. longest_substring_without_repeating_chars: ', longest_substring_without_repeating_chars("abcabcbb"))


# https://leetcode.com/problems/k-diff-pairs-in-an-array/
def findPairs(nums, k):
    if not nums:
        return 0
    count = 0
    vals = {}
    for num in nums:
        vals[num] = vals.get(num, 0) + 1
    for key, val in vals.items():
        if k > 0 and key + k in vals:
            count += 1
        if k == 0 and val >= 2:
            count += 1
    return count


nums = [3, 1, 4, 1, 5]
k = 2
print('56. k-diff-pairs-in-an-array: ', findPairs(nums, k))


# https://leetcode.com/problems/search-a-2d-matrix-ii/
def search_in2D_matrix(matrix, target):
    row, rows = 0, len(matrix)
    cols = len(matrix[row]) - 1
    while row < rows:
        if matrix[row][0] <= target <= matrix[row][cols]:
            result = binary_search(matrix[row], target)
            if result != -1:
                return True
        row += 1
    return False


def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = low + (high - low) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1


matrix = [[1, 4, 7, 11, 15], [2, 5, 8, 12, 19], [3, 6, 9, 16, 22], [10, 13, 14, 17, 24], [18, 21, 23, 26, 30]]
target = 5
print('57. Search in 2D matrix: ', search_in2D_matrix(matrix, target))
print('57. Search in 2D matrix: ', search_in2D_matrix(matrix, 100))


def search_in_2d_matrix_optimized(matrix, target):
    rows, cols = len(matrix), len(matrix[0])
    row = rows - 1
    col = 0
    while row >= 0 and col < cols:
        if target < matrix[row][col]:
            row -= 1
        elif target > matrix[row][col]:
            col += 1
        else:
            return True
    return False


print('57. Search in 2D matrix optimized: ', search_in_2d_matrix_optimized(matrix, target))
print('57. Search in 2D matrix optimized: ', search_in_2d_matrix_optimized(matrix, 100))


# https://leetcode.com/problems/robot-return-to-origin/
def judgeCircleRobot(moves):
    direction = [[0, 1], [1, 0], [0, -1], [-1, 0]]
    x = y = 0
    ind = 0
    for move in moves:
        if move == 'U':
            ind = 0
        if move == 'R':
            ind = 1
        if move == 'D':
            ind = 2
        if move == 'L':
            ind = 3
        x += direction[ind][0]
        y += direction[ind][1]
    if x == y == 0:
        return True
    return False


print(' 58. Check if robot-return-to-origin: ', judgeCircleRobot('LDRRLRUULR'))
print(' 58. Check if robot-return-to-origin: ', judgeCircleRobot('UD'))


# https://leetcode.com/problems/3sum-smaller/
def threeSumSmaller(nums, target):
    nums.sort()
    count = 0
    for i in range(len(nums)):
        left, right = i + 1, len(nums) - 1
        while left < right:
            s = nums[i] + nums[left] + nums[right]
            if s < target:
                count += right - left
                left += 1
            else:
                right -= 1
    return count


print('59. three Sum smaller than target: ', str(threeSumSmaller([-2, 0, 1, 3], 2)))


# https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string-ii/
def removeAdjacentDuplicates(s, k):
    stack = []
    for ch in s:
        if not stack or stack[-1][0] != ch:
            stack.append([ch, 1])
        else:
            stack[-1][1] += 1
        if stack[-1][1] == k:
            stack.pop()
    result = []
    for val in stack:
        for _ in range(val[1]):
            result.append(val[0])
    return ''.join(result)


print('60. removeAdjacentDuplicates: ', str(removeAdjacentDuplicates('abcd', 2)))
print('60. removeAdjacentDuplicates: ', str(removeAdjacentDuplicates('deeedbbcccbdaa', 3)))


# https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/
class TreeNode:
    def __init__(self,val):
        self.val = val
        self.left,self.right = None, None
        self.next =None


    def print_preorder(self):
        if not self:
            return
        print(self.val, end=' ')
        if self.left:
            self.left.print_preorder()
        if self.right:
            self.right.print_preorder()

def zigzagTraversal(root):
    if not root:
        return
    que= deque()
    que.append(root)
    zigzag=False
    result=[]
    while que:
        ls = len(que)
        temp=deque()
        while ls>0:
            node = que.popleft()
            ls-=1
            if zigzag:
                temp.appendleft(node.val)
            else:
                temp.append(node.val)
            if node.left:
                que.append(node.left)
            if node.right:
                que.append(node.right)
        result.append(list(temp))
        zigzag= not zigzag
    return result


root = TreeNode(12)
root.left = TreeNode(7)
root.right = TreeNode(1)
root.left.left = TreeNode(9)
root.right.left = TreeNode(10)
root.right.right = TreeNode(5)
root.right.left.left = TreeNode(20)
root.right.left.right = TreeNode(17)
print("61. Zigzag traversal: " + str(zigzagTraversal(root)))

root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)
root.right.left = TreeNode(6)
root.right.right = TreeNode(7)
print("61. Zigzag traversal: " + str(zigzagTraversal(root)))


# https://leetcode.com/problems/linked-list-cycle/
class Node:
    def __init__(self, value, next=None):
        self.value = value
        self.next = next


    def print_list(self):
        temp = self
        while temp is not None:
          print(str(temp.value) + " ", end='')
          temp = temp.next
        print()


def has_cycle(head):
    slow=fast=head
    while fast and fast.next:
        slow = slow.next
        fast=fast.next.next
        if slow==fast:
            return True
    return False



head = Node(1)
head.next = Node(2)
head.next.next = Node(3)
head.next.next.next = Node(4)
head.next.next.next.next = Node(5)
head.next.next.next.next.next = Node(6)
print("62. LinkedList has cycle: " + str(has_cycle(head)))

head.next.next.next.next.next.next = head.next.next
print("62. LinkedList has cycle: " + str(has_cycle(head)))

head.next.next.next.next.next.next = head.next.next.next
print("62. LinkedList has cycle: " + str(has_cycle(head)))


# https://leetcode.com/problems/find-median-from-data-stream/
from heapq import *
class MedianFinder(object):

    def __init__(self):
        self.maxHeap=[]
        self.minHeap=[]

    def addNum(self, num):
        if not self.maxHeap or -self.maxHeap[0]>=num:
            heappush(self.maxHeap,-num)
        else:
            heappush(self.minHeap,num)
        if len(self.maxHeap)>len(self.minHeap)+1:
            heappush(self.minHeap,-heappop(self.maxHeap))
        elif len(self.minHeap)>len(self.maxHeap):
            heappush(self.maxHeap,-heappop(self.minHeap))

    def findMedian(self):
        if len(self.maxHeap)==len(self.minHeap):
            return -self.maxHeap[0]/2.0 + self.minHeap[0]/2.0
        return -self.maxHeap[0]/1.0


medianFinder =  MedianFinder();
medianFinder.addNum(1)
medianFinder.addNum(2)
print('63. Find median', medianFinder.findMedian())
medianFinder.addNum(3)
print('63. Find median', medianFinder.findMedian())


# https://leetcode.com/problems/valid-sudoku/
def validSudoku(board):
    game = set()
    for i in range(9):
        for j in range(9):
            num = board[i][j]
            if num != '.':
                if (i, num) in board or (num, j) in board or (i / 3, j / 3, num) in board:
                    return False
                game.add((i, num))
                game.add((num, j))
                game.add((i / 3, j / 3, num))
    return True


board = [["5", "3", ".", ".", "7", ".", ".", ".", "."]
    , ["6", ".", ".", "1", "9", "5", ".", ".", "."]
    , [".", "9", "8", ".", ".", ".", ".", "6", "."]
    , ["8", ".", ".", ".", "6", ".", ".", ".", "3"]
    , ["4", ".", ".", "8", ".", "3", ".", ".", "1"]
    , ["7", ".", ".", ".", "2", ".", ".", ".", "6"]
    , [".", "6", ".", ".", ".", ".", "2", "8", "."]
    , [".", ".", ".", "4", "1", "9", ".", ".", "5"]
    , [".", ".", ".", ".", "8", ".", ".", "7", "9"]]
print('64. is Valid Sudoku: ', str(validSudoku(board)))


# https://leetcode.com/problems/snakes-and-ladders/
def snakeAndLadders(board):
    n = len(board)
    flatten = get_flatten_board(board)
    seen = set()
    que = deque()
    que.append((1, 0))
    while que:
        label, step = que.popleft()
        if flatten[label] != -1:
            label = flatten[label]
        if label == n * n:
            return step
        for x in range(1, 7):
            newlabel = label + x
            if newlabel <= n * n and newlabel not in seen:
                seen.add(newlabel)
                que.append((newlabel, step + 1))
    return -1


def get_flatten_board(board):
    flatten = [-1]
    i = 0
    n = len(board)
    for row in range(n - 1, -1, -1):
        if i % 2:
            for col in range(n - 1, -1, -1):
                flatten.append(board[row][col])
        else:
            for col in range(n):
                flatten.append(board[row][col])
        i += 1
    return flatten


board = [[-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1], [-1, 35, -1, -1, 13, -1],
         [-1, -1, -1, -1, -1, -1], [-1, 15, -1, -1, -1, -1]]
print('65. Snakes and ladder: ', snakeAndLadders(board))


# hhttps://leetcode.com/problems/median-of-two-sorted-arrays/
def medianOfSortedArrays(nums1, nums2):
    A = nums1
    B = nums2
    n = len(A) + len(B)
    half = n // 2
    if len(A) > len(B):
        A, B = B, A
    l, r = 0, len(A) - 1
    while True:
        i = (l + r) // 2
        j = half - i - 2
        Aleft = A[i] if i >= 0 else float('-inf')
        Aright = A[i + 1] if i < len(A) - 1 else float('inf')
        Bleft = B[j] if j >= 0 else float('-inf')
        Bright = B[j + 1] if j < len(B) - 1 else float('inf')

        if Aleft <= Bright and Bleft <= Aright:
            if n % 2:
                return min(Aright, Bright)
            return (max(Aleft, Bleft) + min(Aright, Bright)) / 2.0
        elif Aleft > Bright:
            r = i - 1
        else:
            l = i + 1


# https://leetcode.com/problems/container-with-most-water/
def containerWithMostWater(heights):
    left, right = 0, len(heights) - 1
    maxVol = 0
    while left < right:
        vol = min(heights[left], heights[right]) * (right - left)
        maxVol = max(maxVol, vol)
        if heights[left] > heights[right]:
            right -= 1
        else:
            left += 1
    return maxVol


print('66. containerWithMostWater: ', containerWithMostWater([1, 8, 6, 2, 5, 4, 8, 3, 7]))


# https://leetcode.com/problems/sliding-window-maximum/
def slidingWindowMax(nums, k):
    n = len(nums)
    if n * k == 0:
        return []
    if k == 1:
        return nums
    que = deque()

    def clean(ind):
        if que and que[0] == ind - k:
            que.popleft()
        while que and nums[que[-1]] < nums[ind]:
            que.pop()

    maxInd = 0
    for i in range(k):
        clean(i)
        que.append(i)
        if nums[maxInd] < nums[i]:
            maxInd = i
    result = [nums[maxInd]]
    for i in range(k, n):
        clean(i)
        que.append(i)
        result.append(nums[que[0]])
    return result


print('67. Sliding Window Max: ', str(slidingWindowMax([1, 3, -1, -3, 5, 3, 6, 7], 3)))

# https://leetcode.com/problems/knight-probability-in-chessboard/
def knight_probability(n,k,row,col):
    moves= ((-1,-2),(-2,-1),(1,2),(2,1),(1,-2),(-2,1),(-1,2),(2,-1))
    mem={}
    def dfs(count,x,y,p):
        prob=0
        if 0<=x<n and 0<=y<n:
            if count<k:
                for mx,my in moves:
                    newx,newy = mx+x,my+y
                    if (newx,newy,count+1) not in mem:
                        mem[(newx,newy,count+1)] = dfs(count+1,newx,newy,p/8)
                    prob += mem[(newx,newy,count+1)]
            else:
                prob=p
        return prob
    return dfs(0,row,col,1.0)


print('68. knight_probability: ', knight_probability(3,2,0,0))


# https://leetcode.com/problems/elimination-game/
def lastRemaining(n):
    head=1
    step=1
    left=True
    while n>1:
        if left or (n&1):
            head+=step

        step*=2
        n//=2
        left=not left
    return head

print('69. Elimination Game: ', str(lastRemaining(9)))


# https://leetcode.com/problems/circular-array-loop/
def circular_array_loop(nums):
    for i in range(len(nums)):
        slow=fast=i
        direction = nums[i]>0
        while True:
            slow = getNextInd(slow,nums,direction)
            fast = getNextInd(fast,nums,direction)
            if fast!=-1:
                fast = getNextInd(fast,nums,direction)
            if slow==fast or slow==-1 or fast==-1:
                break
        if slow!=-1 and slow==fast:
            return True
    return False

def getNextInd(ind,nums,direction):
    if direction!=(nums[ind]>0):
        return -1
    nextInd = (ind+nums[ind])%len(nums)
    if nextInd==ind:
        return -1
    return nextInd

print('70. circular_array_loop: ', circular_array_loop([2,-1,1,2,2]))


# https://leetcode.com/problems/shortest-word-distance/
def shortest_word_distance(words,w1,w2):
    ind1, ind2= -1,-1
    result=float('inf')
    for i in range(len(words)):
        if words[i]==w1:
            ind1=i
        if words[i]==w2:
            ind2=i
        if ind1!=-1 and ind2!=-1:
            result= min(result,abs(ind1-ind2))
    return result


wordsDict = ["practice", "makes", "perfect", "coding", "makes"]
word1 = "coding"
word2 = "practice"
print('71. shortest_word_distance: ', shortest_word_distance(wordsDict,word1,word2))


# https://leetcode.com/problems/h-index-ii/
def h_index_ii(citations):
    n = len(citations)
    left, right = 0, n-1
    while left<=right:
        mid = (left+right)//2
        if citations[mid]==n-mid:
            return n-mid
        elif citations[mid]<n-mid:
            left= mid+1
        else:
            right = mid-1
    return n-left


print('72. H Index II, log(n) : ', str(h_index_ii([0,1,3,5,6])))


# https://leetcode.com/problems/high-five/
def high_five_heap(items):
    vals={}
    for item in items:
        if item[0] not in vals:
            vals[item[0]]= [item[1]]
        else:
            heappush(vals[item[0]],item[1])
            if len(vals[item[0]])>5:
                heappop(vals[item[0]])
    result=[]
    for id, heap in vals.items():
        score = sum(heap)//len(heap)
        heappush(result,(id,score))
    return result


print('73. High Five variation: ', str(high_five_heap([[1,91],[1,92],[2,93],[2,97],[1,60],[2,77],[1,65],[1,87],[1,100],[2,100],[2,76]])))


# https://leetcode.com/problems/min-stack/
class MinStack:
    def __init__(self):
        self.stack=[]
        self.minStack = []

    def push(self,val):
        self.stack.append(val)
        if not self.minStack or self.minStack[-1][0]>val:
            self.minStack.append([val,1])
        elif self.minStack[-1][0]==val:
            self.minStack[-1][1]+=1

    def pop(self):
        if self.minStack[-1][0]==self.stack[-1]:
            self.minStack[-1][1]-=1
            if self.minStack[-1][1]==0:
                self.minStack.pop()
        self.stack.pop()

    def top(self):
        return self.stack[-1]

    def getMin(self):
        return self.minStack[-1][0]


minStack =MinStack()
minStack.push(-2)
minStack.push(0)
minStack.push(-3)
print('74. Min Stack, get_min: ',minStack.getMin())
minStack.pop()
print('74. Min Stack get top: ',minStack.top())
print('74. Min Stack, get_min: ',minStack.getMin())


# https://leetcode.com/problems/find-the-winner-of-the-circular-game
def circular_game(n):
    n = [i for i in range(1,n+1)]
    start=0
    while len(n)>1:
        start = (start+k-1)%len(n)
        del n[start]
    return n[-1]


print('75. Circular game: ', str(circular_game(5)))


# https://leetcode.com/problems/minimum-moves-to-equal-array-elements
def min_moves(nums):
    minele= min(nums)
    n =len(nums)
    return sum(nums)- n*minele


print('76. min_moves: ', str(min_moves([1,2,3])))
print('76. min_moves: ', str(min_moves([1,1,1])))


# https://leetcode.com/problems/design-circular-deque
class MyCircularDeque(object):

    def __init__(self, k):
        self.size = 0
        self.k = k
        self.f = []
        self.r = []

    def insertFront(self, value):
        if self.size < self.k:
            self.f.append(value)
            self.size += 1
            return True
        return False

    def insertLast(self, value):
        if self.size < self.k:
            self.r.append(value)
            self.size += 1
            return True
        return False

    def deleteFront(self):
        if self.size > 0:
            if self.f:
                self.f.pop()
            else:
                self.r.pop(0)
            self.size -= 1
            return True
        return False

    def deleteLast(self):
        if self.size > 0:
            if self.r:
                self.r.pop()
            else:
                self.f.pop(0)
            self.size -= 1
            return True
        return False

    def getFront(self):
        if self.f:
            return self.f[-1]
        elif self.r:
            return self.r[0]
        else:
            return -1

    def getRear(self):
        if self.r:
            return self.r[-1]
        elif self.f:
            return self.f[0]
        else:
            return -1

    def isEmpty(self):
        return self.size == 0

    def isFull(self):
        return self.size == self.k


myCircularDeque = MyCircularDeque(3);
print('77. Circular Deque:', myCircularDeque.insertLast(1))  # return True
print('77. Circular Deque:', myCircularDeque.insertLast(2))  # return True
print('77. Circular Deque:', myCircularDeque.insertFront(3)) # return True
print('77. Circular Deque:', myCircularDeque.insertFront(4)) # return False, the queue is full.
print('77. Circular Deque:', myCircularDeque.getRear())      # return 2
print('77. Circular Deque:', myCircularDeque.isFull())       # return True
print('77. Circular Deque:', myCircularDeque.deleteLast())   # return True
print('77. Circular Deque:', myCircularDeque.insertFront(4)) # return True
print('77. Circular Deque:', myCircularDeque.getFront()) 


# https://leetcode.com/problems/minimum-path-sum
def min_path_sum(grid):
    n,m = len(grid), len(grid[0])
    for i in range(1,n):
        grid[i][0] += grid[i-1][0]
    for j in range(1,m):
        grid[0][j] += grid[0][j-1]
    for i in range(1,n):
        for j in range(1,m):
            grid[i][j] += min(grid[i-1][j],grid[i][j-1])
    return grid[-1][-1]


print('78. Min Path Sum: ', str(min_path_sum([[1,3,1],[1,5,1],[4,2,1]])))


# https://leetcode.com/problems/minimum-cost-for-tickets/
def minimum_cost_tickets(days,costs):
    lastDay = days[-1]
    days = set(days)
    total = [0 for i in range(lastDay+1)]
    for i in range(lastDay+1):
        if i not in days:
            total[i]= total[i-1]
        else:
            total[i]= min(
                costs[0]+ total[max(i-1,0)],
                costs[1]+ total[max(i-7,0)],
                costs[2]+ total[max(i-30,0)]
            )
    return total[-1]


print('79. Minimum cost tickets: ', minimum_cost_tickets([1,4,6,7,8,20],[2,7,15]))


# https://leetcode.com/problems/gas-station
def can_complete_gastation_circuit(gas,cost):
    n = len(gas)
    cummTotal, currTotal =0,0
    ans=0
    for i in range(n):
        cummTotal+= gas[i]-cost[i]
        currTotal += gas[i]-cost[i]
        if currTotal<0:
            ans=i+1
            currTotal=0
    return ans if cummTotal>=0 else -1


print('80. Can complete gas station circuit? start at what index? ', str(can_complete_gastation_circuit([1,2,3,4,5],[3,4,5,1,2])))


# https://leetcode.com/problems/pascals-triangle-ii/
def pascal_triangle(n):
    res=[1]
    if n<=0:
        return res
    for i in range(0,n):
        temp=[1]
        for j in range(1,len(res)):
            temp.append(res[j]+res[j-1])
        temp.append(1)
        res=list(temp)
    return res


print('81. Pascals Triangle: ', str(pascal_triangle(4)))


# https://leetcode.com/problems/balance-a-binary-search-tree
def balanceBST(root):
    nodes=[]
    def inorder(node):
        if not node:
            return
        inorder(node.left)
        nodes.append(node.val)
        inorder(node.right)

    def balance(nodes):
        if not nodes:
            return
        mid = len(nodes)//2
        n= TreeNode(nodes[mid])
        n.left= balance(nodes[:mid])
        n.right= balance(nodes[mid+1:])
        return n
    inorder(root)
    return balance(nodes)


root = TreeNode(1)
root.right= TreeNode(2)
root.right.right= TreeNode(3)
root.right.right.right = TreeNode(4)
root= balanceBST(root)
print('82. Balance BST: ', end='')
root.print_preorder()


# https://leetcode.com/problems/design-tic-tac-toe/
class TicTacToe:
    def __init__(self,n):
        self.n=n
        self.board= [[0 for i in range(self.n)] for j in range(self.n)]

    def move(self,row,col,player):
        self.board[row][col]=player
        if self.rowWin(row,player) or self.colWin(col,player):
            return player
        if (row==col or row+col==self.n+1) and self.diahWin(player):
            return player
        return 0

    def rowWin(self,row,player):
        rowset = set([self.board[row][x] for x in range(self.n)])
        return len(rowset)==1 and self.board[row][0]==player

    def colWin(self,col,player):
        colset = set([self.board[x][col] for x in range(self.n)])
        return len(colset)==1 and self.board[0][col]==player

    def diahWin(self,player):
        return self.leftDiagWin(player) or self.rightDiagWin(player)

    def leftDiagWin(self,player):
        diagset = set([self.board[x][x] for x in range(self.n)])
        return len(diagset)==1 and self.board[0][0]==player

    def rightDiagWin(self,player):
        diagset = set([self.board[x][self.n-x-1] for x in range(self.n)])
        return len(diagset)==1 and self.board[0][self.n-1]==player


ticTacToe = TicTacToe(3)
ticTacToe.move(0, 0, 1)
ticTacToe.move(0, 2, 2)
ticTacToe.move(2, 2, 1)
ticTacToe.move(1, 1, 2)
ticTacToe.move(2, 0, 1)
ticTacToe.move(1, 0, 2)
print('\n83. TicTacToe winner: ', ticTacToe.move(2, 1, 1))


# https://leetcode.com/problems/last-substring-in-lexicographical-order/
def last_substring_in_lexicographical_order(string):
    i,j,k=0,1,0
    n=len(string)
    while j+k<n:
        if string[i+k]==string[j+k]:
            k+=1
            continue
        elif string[i+k]>string[j+k]:
            j=j+k+1
        else:
            i=i+k+1
        if i==j:
            j+=1
        k=0
    return string[i:]


print('84. last_substring_in_lexicographical_order: ', last_substring_in_lexicographical_order('abab'))

# https://leetcode.com/problems/moving-average-from-data-stream/
class MovingAverage:
    def __init__(self,size):
        self.size=size
        self.len=0
        self.que=deque()
        self.winSum=0

    def next(self,num):
        self.len+=1
        self.que.append(num)
        left = self.que.popleft() if self.len>self.size else 0
        self.winSum= self.winSum-left+num
        return self.winSum/(float(min(len(self.que),self.size)))


movingAverage = MovingAverage(3)
print('85. Moving Average: ',movingAverage.next(1))
print('85. Moving Average: ',movingAverage.next(10))
print('85. Moving Average: ',movingAverage.next(3))
print('85. Moving Average: ',movingAverage.next(5))


# https://leetcode.com/problems/max-stack/
class MaxStack:
    def __init__(self):
        self.stack=[]

    def push(self,x):
        if not self.stack:
            self.stack.append((x,x))
        else:
            top = self.peekMax()
            top = None if top is None else max(top,x)
            self.stack.append((x,top))

    def pop(self):
        if not self.stack:
            return None
        return self.stack.pop()[0]

    def top(self):
        if not self.stack:
            return None
        return self.stack[-1][0]

    def peekMax(self):
        if not self.stack:
            return None
        return self.stack[-1][1]

    def popMax(self):
        top = self.top()
        m= self.peekMax()
        temp=[]
        while top!=m:
            temp.append(self.pop())
            top = self.top()
        self.pop()
        reversed(temp)
        while temp:
            self.push(temp.pop())
        return m


stk = MaxStack()
stk.push(5)
stk.push(1)
stk.push(5)
print('5 5 1 5 1 5')
print('86. Max Stack: ', stk.top())
print('86. Max Stack: ', stk.popMax())
print('86. Max Stack: ', stk.top())
print('86. Max Stack: ', stk.peekMax())
print('86. Max Stack: ', stk.pop())
print('86. Max Stack: ', stk.top())


# https://leetcode.com/problems/excel-sheet-column-title/
def excel_column_number(colNum):
    res=''
    while colNum>0:
        title = chr((colNum-1)%26+65)
        res=title+res
        colNum= (colNum-1)//26
    return res


print('87. Excel column Title: ', excel_column_number(28))
print('87. Excel column Title: ', excel_column_number(2147483647))


# https://leetcode.com/problems/closest-binary-search-tree-value/
def closest_to_target_binary_tree(root, target):
    closest=root.val
    while root:
        closest= closest if abs(target-closest)<abs(target-root.val) else root.val
        if target<root.val:
            root= root.left
        else:
            root = root.right
    return closest

root= TreeNode(4)
root.left= TreeNode(2)
root.right= TreeNode(5)
root.left.left = TreeNode(1)
root.left.right = TreeNode(3)
print('88. Node closest to target in BT: ', closest_to_target_binary_tree(root,3.47878))
print('88. Node closest to target in BT: ', closest_to_target_binary_tree(root,100))


# https://leetcode.com/problems/remove-duplicates-from-sorted-list/
def remove_duplicates_from_sorted_linklist(head):
    cur,prev=head,None
    while cur:
        if prev and prev.value==cur.value:
            prev.next=cur.next
            cur=cur.next
        else:
            prev=cur
            cur=cur.next
    return head

ll =Node(1)
ll.next=Node(1)
ll.next.next= Node(1)
ll.next.next.next= Node(1)
ll.next.next.next.next= Node(2)
ll.next.next.next.next.next= Node(3)
ll.next.next.next.next.next.next= Node(3)
ll.next.next.next.next.next.next.next= Node(3)
ll.next.next.next.next.next.next.next.next= Node(4)
ll.next.next.next.next.next.next.next.next.next= Node(4)
ll= remove_duplicates_from_sorted_linklist(ll)
ll.print_list()


# https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
def buy_and_sell_stock(prices):
    maxProfit=0
    minPrice= prices[0]
    for i in range(1, len(prices)):
        profit = prices[i]-minPrice
        maxProfit = maxProfit if maxProfit>profit else profit
        minPrice = minPrice if minPrice<prices[i] else prices[i]
    return maxProfit


print('89. Buy and Sell Stock: ', buy_and_sell_stock([7,1,5,3,6,4]))

# https://leetcode.com/problems/pascals-triangle/
def pascals_triangle2(rows):
    res=[[1]]
    for i in range(rows-1):
        temp=[1]
        for j in range(1,len(res[-1])):
            temp.append(res[-1][j-1]+res[-1][j])
        temp.append(1)
        res.append(temp)
    return res


print('90. Pascals triangle 2: ', pascals_triangle2(5))


# https://leetcode.com/problems/binary-tree-inorder-traversal/
def inorder_traversal(root):
    result=[]
    inorder(root,result)
    return result

def inorder(node,result):
    if not node:
        return
    inorder(node.left,result)
    result.append(node.val)
    inorder(node.right,result)


root = TreeNode(12)
root.left = TreeNode(7)
root.right = TreeNode(1)
root.left.left = TreeNode(9)
root.right.left = TreeNode(10)
root.right.right = TreeNode(5)
root.right.left.left = TreeNode(20)
root.right.left.right = TreeNode(17)
print('91. Inorder Traversal: ', inorder_traversal(root))

# https://leetcode.com/problems/move-zeroes/
def move_zero_toEnd(nums):
    cur,nxt = 0,1
    n= len(nums)
    while nxt<n:
        if nums[cur]==0 and nums[nxt]!=0:
            nums[cur],nums[nxt]= nums[nxt],nums[cur]
            cur+=1
        elif nums[cur]!=0:
            cur+=1
        nxt+=1
    return nums


print('92. Move all zeros to end of list: ', move_zero_toEnd([0,1,0,3,12]))


# https://leetcode.com/problems/binary-search/
def binary_searchI(nums,target):
    low,high=0,len(nums)-1
    while low<=high:
        mid = low + (high-low)//2
        if target==nums[mid]:
            return mid
        elif target<nums[mid]:
            high=mid-1
        else:
            low= mid+1
    return -1


print('93. Binary Search Ol Fashion: ', binary_searchI([-1,0,3,5,9,12],9))


# https://leetcode.com/problems/maximum-subarray/
def maximum_sub_array(nums):
    curSum=maxSum=nums[0]
    for i in range(1,len(nums)):
        curSum = max(nums[i], curSum+nums[i])
        maxSum = max(maxSum, curSum)
    return maxSum


print('94. Maximum Sum of Sub array: ', maximum_sub_array([-2,1,-3,4,-1,2,1,-5,4]))


# https://leetcode.com/problems/palindrome-permutation/
def can_string_permute_into_pallindrome(s):
    strMap={}
    for ch in s:
        strMap[ch]= strMap.get(ch,0)+1
    found=False
    for ch,freq in strMap.items():
        if freq%2 and not found:
            found=True
        elif freq%2 and found:
            return False
    return True


print('95. can_string_permute_into_pallindrome: ', can_string_permute_into_pallindrome('ababa'))
print('95. can_string_permute_into_pallindrome: ', can_string_permute_into_pallindrome('leetcode'))


# https://leetcode.com/problems/count-negative-numbers-in-a-sorted-matrix/
def count_negatives_in_list_of_lists(lists):
    count=0
    for i in range(len(lists)):
        for j in range(len(lists[0])-1,-1,-1):
            if lists[i][j]<0:
                count+=1
            else:
                break
    return count


print('96. count_negatives_in_list_of_lists: ', count_negatives_in_list_of_lists([[4,3,2,-1],[3,2,1,-1],[1,1,-1,-2],[-1,-1,-2,-3]]))


# https://leetcode.com/problems/kth-largest-element-in-an-array/
def find_kth_largest_number(nums,k):
    minHeap=[]
    for i in range(len(nums)):
        heappush(minHeap,nums[i])
        if len(minHeap)>k:
            heappop(minHeap)
    return minHeap[0]


print('97. Kth Largest Number: ', find_kth_largest_number([3,2,1,5,6,4],k))

# https://leetcode.com/problems/decode-ways/
def decode_ways_digit_string_to_alphabets(s):
    if s[0]=='0':
        return 0
    ones,twos= 1,1
    for i in range(1,len(s)):
        cur=0
        if s[i]!='0':
            cur=ones
        two_digit=int(s[i-1:i+1])
        if two_digit>=10 and two_digit<=26:
            cur+=twos
        twos=ones
        ones=cur
    return ones


print('98. decode_ways_digit_string_to_alphabets: ', decode_ways_digit_string_to_alphabets('226'))


# https://leetcode.com/problems/find-the-celebrity/
def find_celebrity(n):
    celeb=0
    for i in range(1,n):
        if knows(celeb,i):
            celeb=i
    for i in range(n):
        if celeb==i:
            continue
        if knows(celeb,i) or not knows(i,celeb):
            return -1
    return celeb


def knows(i,j):
    grid = [[1,1,0],[0,1,0],[1,1,1]]
    return grid[i][j]


print('99. find_celebrity: ', find_celebrity(2))


# https://leetcode.com/problems/count-and-say/
def count_and_say(n):
    res='1'
    if n==1:
        return res
    for i in range(1,n):
        temp=[]
        prev=res[0]
        count=0
        for j in range(len(res)):
            if prev==res[j]:
                count+=1
            else:
                temp.append(str(count))
                temp.append(str(prev))
                prev=res[j]
                count=1
        temp.append(str(count))
        temp.append(str(prev))
        res=''.join(temp)
    return res


print('100. Count and Say: ', count_and_say(5))


# https://leetcode.com/problems/number-of-islands/
def count_number_of_islands(grid):
    rows,cols = len(grid), len(grid[0])
    count=0
    for i in range(rows):
        for j in range(cols):
            if grid[i][j]!='0':
                count_number_of_islands_helper(grid,i,j)
                count+=1
    return count


def count_number_of_islands_helper(grid,i,j):
    stack= [(i,j)]
    while stack:
        row,col= stack.pop()
        grid[row][col]='0'
        if row>0 and grid[row-1][col]=='1':
            stack.append((row-1,col))
        if row<len(grid)-1 and grid[row+1][col]=='1':
            stack.append((row+1,col))
        if col>0 and grid[row][col-1]=='1':
            stack.append((row,col-1))
        if col<len(grid[0])-1 and grid[row][col+1]=='1':
            stack.append((row,col+1))


grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
print('101. count_number_of_islands: ', count_number_of_islands(grid))


# https://leetcode.com/problems/maximum-size-subarray-sum-equals-k/
def max_subarray_sum_equals_k(nums,k):
    sumMap={0:0}
    nums=[0]+nums
    for i in range(1,len(nums)):
        nums[i]+=nums[i-1]
        if nums[i] not in sumMap:
            sumMap[nums[i]]=i
    maxLen = 0
    for i in range(len(nums)-1,-1,-1):
        need = nums[i]-k
        if need in sumMap:
            maxLen = max(maxLen, i- sumMap[need])
    return maxLen


print('102. max_subarray_sum_equals_k: ', max_subarray_sum_equals_k([-2,-1,2,1],1))


# https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/
def best_time_to_buy_sell_stock_IV(prices,k):
    costs=[float('inf') for i in range(k)]
    profits= [0 for i in range(1+k)]
    costs[0]= prices[0]
    for i in range(1,len(prices)):
        for j in range(1,k+1):
            costs[j-1]= min(costs[j-1],prices[i]-profits[j-1])
            profits[j]= max(profits[j], prices[i]-costs[j-1])
    return profits[k]


print('103. best_time_to_buy_sell_stock_IV: ', best_time_to_buy_sell_stock_IV([3,2,6,5,0,3],k))


# https://leetcode.com/problems/serialize-and-deserialize-binary-tree/
def serialize(root):
    res=[]
    def helper(node):
        if not node:
            res.append('None')
            return
        res.append(str(node.val))
        helper(node.left)
        helper(node.right)
    helper(root)
    return ','.join(res)

def deserialize(data):
    def helper(data):
        if data[0]=='None':
            data.pop(0)
            return
        root= TreeNode(data[0])
        data.pop(0)
        root.left = helper(data)
        root.right = helper(data)
        return root

    data= data.split(',')
    return helper(data)


root= TreeNode(1)
root.left= TreeNode(2)
root.right= TreeNode(3)
root.right.left= TreeNode(4)
root.right.right= TreeNode(5)
data = serialize(root)
print('104: Serialize Binary Tree: ', data)
print('104: Deserialize Binary Tree: ', )
root = deserialize(data)
root.print_preorder()
print()


# https://leetcode.com/problems/next-permutation/
def next_permutation(nums):
    n = len(nums)
    for i in range(n-2,-1,-1):
        if nums[i]<nums[i+1]:
            for j in range(n-1,i,-1):
                if nums[j]>nums[i]:
                    nums[i], nums[j]= nums[j],nums[i]
                    nums[i+1:]= nums[i+1:][::-1]
                    return nums
    for i in range(n//2):
        nums[i], nums[n-i-1] = nums[n-i-1], nums[i]
    return nums


nums= [1,1,5]
print('105. Next permutation: ', next_permutation(nums))
nums= [1,2,3]
print('105. Next permutation: ', next_permutation(nums))
nums= [3,2,1]
print('105. Next permutation: ', next_permutation(nums))


# https://leetcode.com/problems/top-k-frequent-words/
def find_top_K_frequent_words(words,k):
    wordMap={}
    maxHeap =[]
    for word in words:
        wordMap[word]= wordMap.get(word,0)+1
    for word,freq in wordMap.items():
        heappush(maxHeap,(-freq,word))
    res=[]
    while k>0:
        res.append(heappop(maxHeap)[1])
        k-=1
    return res


print('106. find_top_K_frequent_words: ', find_top_K_frequent_words(["the","day","is","sunny","the","the","the","sunny","is","is"],4))


# https://leetcode.com/problems/letter-combinations-of-a-phone-number/
def letter_combinations_of_phone_numbers(digits):
    if not digits:
        return []
    digitMap = {'1':"","2": "abc", "3": "def", "4": "ghi", "5": "jkl",
                   "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
    res=[]
    def dfs(ind,curPath):
        if ind==len(digits):
            res.append(''.join(curPath))
            return
        for ch in digitMap[digits[ind]]:
            curPath.append(ch)
            dfs(ind+1,curPath)
            del curPath[-1]
    dfs(0,[])
    return res


print('107. letter_combinations_of_phone_numbers: ', letter_combinations_of_phone_numbers("23"))


# https://leetcode.com/problems/combination-sum/
def subsets_of_sum_equal_to_target(nums,target):
    nums.sort()
    res=[]
    def dfs(ind,curSum,curPath):
        if curSum==target:
            res.append(curPath)
            return
        for i in range(ind,len(nums)):
            if curSum+nums[i]<=target:
                dfs(i,curSum+nums[i],curPath+[nums[i]])
            else:
                break
    dfs(0,0,[])
    return res


print('108. subsets_of_sum_equal_to_target: ', subsets_of_sum_equal_to_target([2,3,6,7],7))


# https://leetcode.com/problems/permutations/
def permutation_sets(nums):
    allPaths=[]
    def dfs(ind,curPath):
        if ind==len(nums):
            allPaths.append(curPath)
            return
        for i in range(len(curPath)+1):
            newlist = list(curPath)
            newlist.insert(i,nums[ind])
            dfs(ind+1,newlist)
    dfs(0,[])
    return allPaths


print('109. permutation_sets: ', permutation_sets([1,2,3]))


# https://leetcode.com/problems/powx-n/
def pow_xton(x,n):
    if n==0:
        return 1
    if n%2==0:
        if n>1:
            return pow_xton(x*x,n//2)
        return 1.0/pow_xton(x*x,-n//2)
    else:
        if n>1:
            return x*pow_xton(x*x,n//2)
        return 1.0/(x*pow_xton(x*x,-n//2))


print('110. pow_xton: ', pow_xton(2.1,3))
print('110. pow_xton: ', pow_xton(2,-2))


# https://leetcode.com/problems/subsets/
def subsets(nums):
    subsets=[[]]
    for num in nums:
        for subset in subsets:
            subsets= subsets+ [subset+[num]]
    return subsets


print('111. Subsets: ', subsets([1,2,3]))


# https://leetcode.com/problems/word-search/
def word_search_in_grid(grid,word):
    rows,cols = len(grid), len(grid[0])
    seen=set()
    def dfs(ind,i,j):
        if ind==len(word):
            return True
        if i<0 or j<0 or i==rows or j==cols or grid[i][j]!=word[ind] or (i,j) in seen:
            return False
        seen.add((i,j))
        found = dfs(ind+1,i+1,j) or dfs(ind+1,i-1,j) or dfs(ind+1,i,j+1) or dfs(ind+1,i,j-1)
        seen.remove((i,j))
        return found

    for i in range(rows):
        for j in range(cols):
            if dfs(0,i,j):
                return True
    return False


print('112. word_search_in_grid: ', word_search_in_grid([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]],"ABCCED"))


# https://leetcode.com/problems/linked-list-cycle-ii/
def find_linked_list_cycle_start_node(head):
    slow=fast=head
    while fast and fast.next:
        slow=slow.next
        fast=fast.next.next
        if slow==fast:
            break
    if not fast or not fast.next:
        return -1
    fast=head
    while fast!=slow:
        slow=slow.next
        fast=fast.next
    return slow.value


head= Node(3)
head.next=Node(2)
head.next.next= Node(0)
head.next.next.next =Node(-4)
print('113. find_linked_list_cycle_start_node: ', find_linked_list_cycle_start_node(head))
head.next.next.next.next= head.next
print('113. find_linked_list_cycle_start_node: ', find_linked_list_cycle_start_node(head))


# https://leetcode.com/problems/binary-tree-right-side-view/
def right_view_of_tree(root):
    if not root:
        return
    que=deque()
    que.append(root)
    result=[]
    while que:
        ls = len(que)
        last=None
        while ls>0:
            node=que.popleft()
            ls-=1
            last=node.val
            if node.left:
                que.append(node.left)
            if node.right:
                que.append(node.right)
        result.append(last)
    return result


root = TreeNode(12)
root.left = TreeNode(7)
root.right = TreeNode(1)
root.left.left = TreeNode(9)
root.right.left = TreeNode(10)
root.right.right = TreeNode(5)
root.left.left.left = TreeNode(3)
print('114. right_view_of_tree: ', right_view_of_tree(root))


# https://leetcode.com/problems/count-primes/
def count_primes(n):
    if n<=2:
        return 0
    primes=[1]*n
    primes[0]=primes[1]=0
    for i in range(2,int(math.sqrt(n))+1):
        if primes[i]:
            for j in range(i*i,n,i):
                primes[j]=0
    return sum(primes)


print('115. count_primes: ', count_primes(10))


# https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/
def buy_and_sell_stockII(prices):
    maxProfit=0
    for i in range(1,len(prices)):
        if prices[i]>prices[i-1]:
            maxProfit+=(prices[i]-prices[i-1])
    return maxProfit


print('116. buy_and_sell_stockII: ', buy_and_sell_stockII([7,1,5,3,6,4]))


# https://leetcode.com/problems/jump-game/
# https://leetcode.com/problems/word-ladder/
# https://leetcode.com/problems/course-schedule-ii/
# https://leetcode.com/problems/maximal-square/
# https://leetcode.com/problems/count-complete-tree-nodes/


# 1year- Easy
# https://leetcode.com/problems/valid-anagram/
# https://leetcode.com/problems/implement-stack-using-queues/
# https://leetcode.com/problems/excel-sheet-column-number/
# https://leetcode.com/problems/minimum-depth-of-binary-tree/

# https://leetcode.com/problems/climbing-stairs/
# https://leetcode.com/problems/reverse-string/
# https://leetcode.com/problems/middle-of-the-linked-list/
# https://leetcode.com/problems/invert-binary-tree/

# Med/Hard
# https://leetcode.com/problems/spiral-matrix/
# https://leetcode.com/problems/maximum-subarray-sum-with-one-deletion/
# https://leetcode.com/problems/divide-array-in-sets-of-k-consecutive-numbers/
# https://leetcode.com/problems/last-stone-weight-ii/
# https://leetcode.com/problems/snapshot-array/
# https://leetcode.com/problems/lfu-cache/
# https://leetcode.com/problems/distinct-subsequences/
# https://leetcode.com/problems/design-search-autocomplete-system/
# https://leetcode.com/problems/game-of-life/

# https://leetcode.com/problems/reorder-data-in-log-files/
def reorder_log_file(logs):
    alpha, digit = [], []
    for log in logs:
        if log.split()[1].isalpha():
            alpha.append(log.split())
        else:
            digit.append(log)
    alpha.sort(key=lambda x: x[0])
    alpha.sort(key=lambda x: x[1:])
    for ind in range(len(alpha)):
        alpha[ind] = ' '.join(alpha[ind])
    alpha.extend(digit)
    return alpha


#print('Reorder log file: ', reorder_log_file(["dig1 8 1 5 1","let1 art can","dig2 3 6","let2 own kit dig","let3 art zero"]))


