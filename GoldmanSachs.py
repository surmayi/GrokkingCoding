# 1 https://leetcode.com/problems/robot-bounded-in-circle/
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


# 6. Given an array of integers, print the array in such a way that the first element is first maximum and second element is first minimum and so on.
# https://www.geeksforgeeks.org/alternative-sorting/

def print_alternate_sorting(nums):
    nums.sort()
    n = len(nums)-1
    i=0
    while i<n:
        print(nums[n],end=' ')
        print(nums[i],end =' ')
        n-=1
        i+=1


print('3. print_alternate_sorting: ', end='')
print_alternate_sorting([3, 1, 5, 3, 7, 9, 2, 5])
print('\n3. Alternate Sorting: ', end='')
print_alternate_sorting([7, 1, 2, 3, 4, 5, 6])
print('\n3. Alternate sorting: ', end='')
print_alternate_sorting([1, 6, 9, 4, 3, 7, 8, 2])


# 3. Remove duplicates from linkedlist
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
        prev= None
        while cur:
            if cur.val not in duplicates:
                duplicates.append(cur.val)
                prev=cur
            else:
                prev.next= cur.next
            cur=prev.next



    def print_list(self):
        cur = self.head
        while cur:
            print(cur.val, end='->')
            cur = cur.next
        print()


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


# 5. https://leetcode.com/problems/minimum-size-subarray-sum/
# O(N)
def minSubArrayLen(nums, target):
    winStart, res, winSum = 0, -math.inf, 0
    for winEnd in range(len(nums)):
        winSum += nums[winEnd]
        while winSum >= target:
            res = min(res, winEnd - winStart + 1)
            winSum -= nums[winStart]
            winStart += 1
        return 0 if res == -math.inf else res


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
    avgMap ={}
    for name,avg in input:
        if name in avgMap:
            avgMap[name][0]+=int(avg)
            avgMap[name][1]+=1
        else:
            avgMap[name]= [int(avg),1]
    maxAvg=0
    for name,avgs in avgMap.items():
        avg= avgs[0]/avgs[1]
        maxAvg=max(maxAvg,avg)
    return maxAvg


avgs = [("Bob", "87"), ("Mike", "35"), ("Bob", "52"), ("Jason", "35"), ("Mike", "55"), ("Jessica", "99")]
print('7. find_max_average: ', find_max_average(avgs))


# https://leetcode.com/problems/first-unique-character-in-a-string/
# leetcode -> l
# O(N)
def find_first_unique_char(string):
    charMap={}
    ordered=[]
    for ch in string:
        charMap[ch]= charMap.get(ch,0)+1
        if ch not in ordered:
            ordered.append(ch)
    for val in ordered:
        if charMap[val]==1:
            return val


string = 'leetcodelove'
print('8. find_first_unique_char', find_first_unique_char(string))


#  https://www.geeksforgeeks.org/return-maximum-occurring-character-in-the-input-string/
def find_maximum_occuring_char_instring(string):
    strMap={}
    maxFreq,res=0,''
    for ch in string:
        strMap[ch]= strMap.get(ch,0)+1
        if strMap[ch]>maxFreq:
            maxFreq=strMap[ch]
            res=ch
    return res


string = 'loeooeetcodelove'
print('9. find_maximum_occuring_char_instring', find_maximum_occuring_char_instring(string))


# https://leetcode.com/problems/median-of-two-sorted-arrays/
# https://leetcode.com/problems/median-of-two-sorted-arrays/discuss/1532231/Binary-Search-Explained
def findMedianSortedArrays(nums1, nums2):
    nums = nums1+nums2
    nums.sort()
    n = len(nums)
    if n%2==1:
        return nums[n//2]
    return (nums[(n-1)//2]+nums[n//2])/2


print('10. findMedianSortedArrays', findMedianSortedArrays([1, 2, 4, 4, 5], [1, 2, 5, 9, 10]))


# #Problem #31 : Longest Uniform Substring
def longestUniformSubString(str):
    winStart,maxLen,vals,res =0,0,{},[-1,-1]
    for winEnd in range(len(str)):
        right = str[winEnd]
        vals[right]= vals.get(right,0)+1
        while len(vals)>1:
            left=str[winStart]
            vals[left]-=1
            if vals[left]==0:
                del vals[left]
            winStart+=1
        if winEnd-winStart+1>maxLen:
            maxLen=winEnd-winStart+1
            res=[winStart,winEnd+1]
    return str[res[0]:res[1]]


print('11. longestUniformSubString: ', longestUniformSubString('aaabccccbbbbba'))


# https://www.geeksforgeeks.org/find-starting-indices-substrings-string-s-made-concatenating-words-listl/
def subStringIndexWithConcatationOfWordList(arr, words):
    word_len = len(words[0])
    word_count= len(words)
    wordMap ={}
    result=[]
    for word in words:
        wordMap[word]= wordMap.get(word,0)+1
    for i in range(len(arr)-word_len*word_count+1):
        word_seen={}
        for j in range(word_count):
            next_word_ind = i+j*word_len
            next_word = arr[next_word_ind:next_word_ind+word_len]
            if next_word not in wordMap:
                break
            word_seen[next_word]= word_seen.get(next_word,0)+1
            if word_seen[next_word]> wordMap[next_word]:
                break
            if j+1 == word_count:
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
    if len(arr)<=2:
        return arr
    first=second = math.inf
    for i in range(len(arr)):
        if arr[i]<first:
            second=first
            first=arr[i]
        elif arr[i]<second and arr[i]!=first:
            second=arr[i]
    return [first,second]


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
    if nodes<=0 or not edges:
        return 0
    graph = {i:[] for i in range(nodes)}
    visited= [False for i in range(nodes)]

    for edge in edges:
        v1,v2= edge[0],edge[1]
        graph[v1].append(v2)
        graph[v2].append(v1)
    count=0
    for i in range(nodes):
        if not visited[i]:
            count = max(count,largestTreeSize_helper(graph,visited,i))
    return count


def largestTreeSize_helper(graph,visited,i):
    visited[i]=True
    size=1
    for j in range(len(graph[i])):
        if not visited[graph[i][j]]:
            size+=largestTreeSize_helper(graph,visited,graph[i][j])
    return size


V = 7
edges = [[0, 1], [0, 2], [3, 4], [4,6], [3, 5]]

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

def coinChange_recursion(coins,amount):
    if amount==0:
        return 0
    n =amount+1
    for coin in coins:
        if coin<=amount:
            nxt = coinChange_recursion(coins,amount-coin)
            if nxt>=0:
                n = min(n,1+nxt)
    return -1 if n==amount+1 else n


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
    count=0
    n=len(arr)
    for i in range(n):
        largeLeft=0
        smallRight=0
        for j in range(i):
            if arr[j]>arr[i]:
                largeLeft+=1
        for j in range(i+1,n):
            if arr[j]<arr[i]:
                smallRight+=1
        count += smallRight*largeLeft
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
    maxStr=''
    for i in range(len(str)):
        maxStr= max(maxStr,str[i:])
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
def josephus(nums,k):
    start=0
    while len(nums)>1:
        start = (start+k)%len(nums)
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
    result=[]
    row,col =0,0
    rows,cols = len(mat),len(mat[0])
    upward=True
    for _ in range(rows*cols):
        result.append(mat[row][col])
        if upward:
            if col+1==cols:
                row+=1
                upward=False
            elif row==0:
                col+=1
                upward=False
            else:
                row-=1
                col+=1
        else:
            if row+1==rows:
                col+=1
                upward=True
            elif col==0:
                row+=1
                upward=True
            else:
                row+=1
                col-=1
    return result


print('28. Diagonal Traversal of matrix: ', str(diagonal_traversal([[1,2,3],[4,5,6],[7,8,9]])))


# O(L1*L2) - length of both strings
def longestCommonSubstring(string1, string2):
    l1,l2 = len(string1), len(string2)
    dp = [[0 for j in range(l2+1)] for i in range(l1+1)]
    for i in range(1,l1+1):
        for j in range(1,l2+1):
            if string1[i-1]==string2[j-1]:
                dp[i][j] = dp[i-1][j-1]+1
            else:
                dp[i][j] = max(dp[i-1][j],dp[i][j-1])
    return dp[l1][l2]


def longest_common_substring(str1,str2):
    vals1, vals2 = {},{}
    for ch in str1:
        vals1[ch]= vals1.get(ch,0)+1
    for ch in str2:
        vals2[ch]= vals2.get(ch,0)+1
    count=0
    for key,freq in vals1.items():
        if key in vals2:
            count += min(vals2[key],freq)
    return count


X = "AGGTAB"
Y = "GXTXGAYB"
print("29. Length of LCS is ", longestCommonSubstring(X, Y))
print("29. Length of LCS2 is ", longest_common_substring(X, Y))


# https://www.geeksforgeeks.org/print-all-prime-factors-of-a-given-number/
# Time - O(sqrt(n))
def primeFactorizers(n):
    if n<=2:
        return 0
    result=[]
    while n%2==0:
        n//=2
    for i in range(3, int(math.sqrt(n))+1,2):
        while n%i==0:
            result.append(i)
            n//=i
    if n>1:
        result.append(n)
    return result


print('30. Prime Factorizers: ', str(primeFactorizers(315)))


# House robber - https://leetcode.com/problems/house-robber/
def rob(nums):
    if not nums:
        return 0
    prev_prev, prev = 0,0,
    for i in range(len(nums)):
        cur = max(prev_prev+nums[i], prev)
        prev_prev= prev
        prev=cur
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
    sums = [0 for i in range(max(nums)+1)]
    for num in nums:
        sums[num] += num
    prev_prev, prev = 0,0
    for i in range(len(sums)):
        cur = max(prev_prev+sums[i], prev)
        prev_prev= prev
        prev=cur
    return prev


print('32. Delete and Earn (Reduce to house robber)', deleteAndEarn([1, 2, 3, 4]))
print('32. Delete and Earn (Reduce to house robber)', deleteAndEarn([2,1,2,3,2,2]))


# Consecutive numbers sum - https://leetcode.com/problems/consecutive-numbers-sum/
# O(sqrt(n))
def consecutiveNumbersSum(n):
    if n <=1:
        return n
    k=1
    ans=0
    while k<n:
        kx = n-k*(k-1)/2
        if kx<=0:
            break
        if kx%k==0:
            ans+=1
        k+=1
    return ans


print('33. Count of sets of Sum of consecutive number equal to n', consecutiveNumbersSum(15))
print('33. Count of sets of Sum of consecutive number equal to n', consecutiveNumbersSum(9))


# https://leetcode.com/problems/count-number-of-teams/
def CountNumberOfTeams(ratings):
    def cal(input):
        count=0
        for i in range(len(input)):
            smallBefore, largeAfter =0,0
            for j in range(i-1,-1,-1):
                if input[j]<input[i]:
                    smallBefore+=1
            for j in range(i+1,len(input)):
                if input[j]>input[i]:
                    largeAfter+=1
            count += largeAfter*smallBefore
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
#O(N)
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
            if len(word)==len(res):
                res=word if word<res else res
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
    return sum(nums)- smallest*len(nums)


print('42. array_game_optimized: ', array_game_optimized([3, 4, 6, 6, 3]))
print('42. array_game_optimized: ', array_game_optimized([4, 3, 4]))
print('42. array_game_optimized: ', array_game_optimized([1, 2, 3]))


# Problem 61 - Like 2 sum, but get all pairs
def profit_targets(nums, k):
    nums.sort()
    count=0
    left, right = 0, len(nums)-1
    while left<right:
        s= nums[left]+nums[right]
        if s==k:
            print(nums[left], nums[right])
            count+=1
            left+=1
            right -= 1
            while left<right and nums[left]==nums[left-1]:
                left+=1
            while right>=0 and nums[right]==nums[right-1]:
                right-=1
        elif s<k:
            left+=1
        else:
            right-=1
    return count


print('43. profit targets: ', profit_targets([5, 7, 9,9, 13,5, 11, 6, 6, 3, 3], 12))
print('43. profit targets: ', profit_targets([1, 3, 46, 1, 3, 9], 47))


# https://leetcode.com/problems/longest-increasing-subsequence/
# O(n2)
def longest_subsequence(nums):
    sub=[ nums[0]]
    for num in nums:
        if num>sub[-1]:
            sub.append(num)
        else:
            j=0
            while num>sub[j]:
                j+=1
            sub[j]=num
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
    n=len(nums)
    Reven, Rodd = nums[0],nums[1]
    isAdd=False
    for i in range(2,n-1,2):
        if isAdd:
            Reven+=nums[i]
            Rodd+=nums[i+1]
            isAdd=False
        else:
            Reven*=nums[i]
            Rodd*=nums[i]
            isAdd=True
    Reven, Rodd = Reven%2, Rodd%2
    if Reven>Rodd:
        return 'EVEN'
    elif Reven<Rodd:
        return 'ODD'
    else:
        return 'NEUTRAL'


print('45: Even_or_odd_multiplication_array: ', Even_or_odd_multiplication_array([12, 3, 5, 7, 13, 12]))
print('45: Even_or_odd_multiplication_array: ', Even_or_odd_multiplication_array([1, 2, 1]))
print('45: Even_or_odd_multiplication_array: ', Even_or_odd_multiplication_array([2, 3, 8]))



# https://leetcode.com/problems/rotate-array/
# O(N)
def rotateArray_from_right_optimised(nums,k):
    n= len(nums)
    k=k%n

    def rotate_array(nums,start,end):
        while start<end:
            nums[start],nums[end]= nums[end],nums[start]
            start+=1
            end-=1
    rotate_array(nums,0,n-1)
    rotate_array(nums,0,k-1)
    rotate_array(nums,k,n-1)
    return nums


print('46. rotateArray_from_right: ', rotateArray_from_right_optimised([1,2,3,4,5,6,7,8],3))


# https://leetcode.com/problems/implement-trie-prefix-tree/
# O(N)
class TrieNode:
    def __init__(self):
        self.children={}
        self.isEndWord=False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert_word(self,word):
        cur = self.root
        for ch in word:
            if ch not in cur.children:
                cur.children[ch]= TrieNode()
            cur=cur.children[ch]
        cur.isEndWord=True

    def search_word(self,word):
        cur = self.root
        for ch in word:
            if ch not in cur.children:
                return False
            cur = cur.children[ch]
        return cur.isEndWord

    def starts_with(self,prefix):
        cur= self.root
        for ch in prefix:
            if ch not in cur.children:
                return False
            cur = cur.children[ch]
        return True


trie = Trie();
trie.insert_word("apple")
print('47. Search word in Trie: ', trie.search_word("apple"))
print('47. Search word in Trie: ',trie.search_word("app"))
print('47. Search prefic in Trie: ',trie.starts_with("app"))
trie.insert_word("app")
print('47. Search word in Trie: ',trie.search_word("app"))


# https://leetcode.com/problems/longest-palindromic-substring/
# O(N^2)
def longest_substring_palindrome(string):
    maxLen, maxPal = 0,''
    n= len(string)
    for i in range(n*2-1):
        left,right = i//2, (i+1)//2
        while left>=0 and right<n and string[left]==string[right]:
            if right-left+1>maxLen:
                maxLen=right-left+1
                maxPal = string[left:right+1]
            right+=1
            left-=1
    return maxPal

print('48. longest_substring_palindrome: ', longest_substring_palindrome('babad'))
print('48. longest_substring_palindrome: ', longest_substring_palindrome('bbca'))


# https://leetcode.com/problems/palindromic-substrings/
# O(N)
def count_palindromic_substrings(string):
    count=0
    n= len(string)
    for i in range(2*n-1):
        left,right = i//2, (i+1)//2
        while left>=0 and right<n and string[left]==string[right]:
            left-=1
            right+=1
            count+=1
    return count


print('49. count_palindromic_substrings: ', count_palindromic_substrings('aaa'))
print('49. count_palindromic_substrings: ', count_palindromic_substrings('abc'))


# https://leetcode.com/discuss/interview-question/1321204/efficient-harvest-faang-oa-question-2021
def efficient_harvest(arr,k):
    maxProfit=0
    n=len(arr)
    for i in range(n//2):
        profit=0
        for j in range(i,i+k):
            j2 = (j+n//2)%n
            profit += arr[j]+arr[j2]
        maxProfit=max(maxProfit,profit)
    return maxProfit


print('50. efficient_harvest: ', efficient_harvest([-3,7,3,1,5,1],2))
print('50. efficient_harvest: ', efficient_harvest([-3,3,6,1],1))



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
    ipMap={}
    maxFreq=0
    for log in logs:
        ip = log.split()[0]
        ipMap[ip]= ipMap.get(ip,0)+1
        maxFreq= max(maxFreq,ipMap[ip])
    result=[]
    for key, val in ipMap.items():
        if val==maxFreq:
            result.append(key)
    return result


print('52. most_frequent_ip: ', most_frequent_ip(
    ["10.0.0.1 - GET 2020-08-24", "10.0.0.1 - GET 2020-08-24", "10.0.0.2 - GET 2020-08-20",
     "10.0.0.2 - GET 2020-08-20" "10.0.0.3 - GET 2020-08-20"]))


# https://newbedev.com/find-the-number-of-unordered-pair-in-an-array
# Merge sort O(nlogn)
# Implement merge sort, place a counter and increase it whenever right<left while merging.
def find_number_of_unordered_pair(arr):
    result,merged= merge_sort(arr)
    return result

def merge_sort(arr):
    if len(arr)<=1:
        return 0,arr
    mid = len(arr)//2
    leftcount, left = merge_sort(arr[:mid])
    rightcount, right = merge_sort(arr[mid:])
    mergecount, merged = merge(left,right)
    total = mergecount+ leftcount + rightcount
    return total, merged

def merge(left,right):
    merged=[]
    count=0
    ll,lr = len(left),len(right)
    i,j=0,0
    while i<ll and j<lr:
        if left[i]<=right[j]:
            merged.append(left[i])
            i+=1
        else:
            count+=1
            merged.append(right[j])
            j+=1
    if i<ll:
        merged.extend(left[i:])
    if j<lr:
        merged.extend(right[j:])
    return count, merged

print('53. find_number_of_unordered_pair: ', find_number_of_unordered_pair([7, 2, 0, 4, 5, 4, 6, 7]))


# https://leetcode.com/problems/power-of-three/
def is_power_of_three(n):
    while n>1:
        if n%3!=0:
            return False
        n//=3
    return n==1


print('54. is_power_of_three: ', is_power_of_three(27))
print('54. is_power_of_three: ', is_power_of_three(45))


# https://leetcode.com/problems/longest-substring-without-repeating-characters/
def longest_substring_with_repeating_chars(string):
    winStart,vals =0, {}
    maxLen=0
    for winEnd in range(len(string)):
        right = string[winEnd]
        if right in vals:
            winStart = max(winStart,vals[right]+1)
        vals[right]=winEnd
        maxLen= max(maxLen, winEnd-winStart+1)
    return maxLen

print('55. longest_substring_with_repeating_chars: ', longest_substring_with_repeating_chars("abcabcbb"))


# https://leetcode.com/problems/k-diff-pairs-in-an-array/
def findPairs(nums,k):
    if not nums:
        return 0
    count=0
    vals={}
    for num in nums:
        vals[num]= vals.get(num,0)+1
    for key,val in vals.items():
        if k>0 and key+k in vals:
            count+=1
        if k==0 and val>=2:
            count+=1
    return count

nums = [3,1,4,1,5]
k = 2
print('56. k-diff-pairs-in-an-array: ', findPairs(nums,k))


# https://leetcode.com/problems/search-a-2d-matrix-ii/
def search_in2D_matrix(matrix,target):
    row,rows= 0, len(matrix)
    cols=len(matrix[row])-1
    while row<rows:
        if matrix[row][0]<=target<=matrix[row][cols]:
            result = binary_search(matrix[row],target)
            if result!=-1:
                return True
        row+=1
    return False

def binary_search(arr,target):
    low,high = 0, len(arr)-1
    while low<=high:
        mid = low + (high-low)//2
        if arr[mid]==target:
            return mid
        elif arr[mid]<target:
            low=mid+1
        else:
            high=mid-1
    return -1


matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]]
target = 5
print('57. Search in 2D matrix: ', search_in2D_matrix(matrix,target))
print('57. Search in 2D matrix: ', search_in2D_matrix(matrix,100))


def search_in_2d_matrix_optimized(matrix,target):
    rows, cols = len(matrix), len(matrix[0])
    row= rows-1
    col= 0
    while row>=0 and col<cols:
        if target<matrix[row][col]:
            row-=1
        elif target>matrix[row][col]:
            col+=1
        else:
            return True
    return False

print('57. Search in 2D matrix optimized: ', search_in_2d_matrix_optimized(matrix,target))
print('57. Search in 2D matrix optimized: ', search_in_2d_matrix_optimized(matrix,100))


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
def threeSumSmaller(nums,target):
    nums.sort()
    count=0
    for i in range(len(nums)):
        left,right = i+1,len(nums)-1
        while left<right:
            s  = nums[i]+ nums[left]+nums[right]
            if s<target:
                count+=right-left
                left+=1
            else:
                right-=1
    return count


print(' three Sum smaller than target: ', str(threeSumSmaller([-2,0,1,3],2)))

# https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string-ii/
def removeAdjacentDuplicates(s,k):
    stack=[]
    for ch in s:
        if not stack or stack[-1][0]!=ch:
            stack.append([ch,1])
        else:
            stack[-1][1]+=1
        if stack[-1][1]==k:
            stack.pop()
    result=[]
    for val in stack:
        for _ in range(val[1]):
            result.append(val[0])
    return ''.join(result)


print('removeAdjacentDuplicates: ', str(removeAdjacentDuplicates('abcd',2)))
print('removeAdjacentDuplicates: ', str(removeAdjacentDuplicates('deeedbbcccbdaa',3)))



# https://leetcode.com/problems/snakes-and-ladders/
# https://leetcode.com/problems/reverse-linked-list/
# https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/
# https://leetcode.com/problems/valid-sudoku/
# https://leetcode.com/problems/linked-list-cycle/
# https://leetcode.com/problems/find-median-from-data-stream/

# https://www.youtube.com/results?search_query=median+of+two+sorted+arrays+leetcode


