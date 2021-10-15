# 1 https://leetcode.com/problems/robot-bounded-in-circle/
import math
from collections import deque


def isRobotBounded(instructions):
    directions = [[0,1],[1,0],[0,-1],[-1,0]]
    idx=0
    x,y=0,0

    for inst in instructions:
        if inst=='L':
            idx = (idx+3)%4
        elif inst=='R':
            idx = (idx+1)%4
        else:
            x += directions[idx][0]
            y += directions[idx][1]

    if (x==0 and y==0) or idx!=0:
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
        if sumL==sumR-nums[i]:
            return i
        sumL += nums[i]
        sumR -= nums[i]
    return -1


print('2. find_pivot_index: ', str((find_pivot_index([2,1,-1]))))


# 6. Given an array of integers, print the array in such a way that the first element is first maximum and second element is first minimum and so on.
# https://www.geeksforgeeks.org/alternative-sorting/

def print_alternate_sorting(nums):
    nums.sort()
    left, right =0, len(nums)-1
    while left<right:
        print(nums[right], end=' ')
        print(nums[left], end=' ')
        left+=1
        right-=1

print('3. print_alternate_sorting: ',print_alternate_sorting([3,1,5,3,7,9,2,5]))
print('ALternate Sorting', end='')
print_alternate_sorting([7, 1, 2, 3, 4, 5, 6])
print('\n ALternate sorting', end='')
print_alternate_sorting([1, 6, 9, 4, 3, 7, 8, 2])


# 3. Remove duplicates from linkedlist
class Node:
    def __init__(self,val):
        self.val = val
        self.next = None

class LinkedList:
    def __init__(self,head=None):
        self.head=head

    def append(self,val):
        if not self.head:
            self.head= Node(val)
            return
        cur = self.head
        while cur.next:
            cur=cur.next
        cur.next= Node(val)


    def remove_duplicates(self):
        prev=None
        cur=self.head
        duplicates=[]
        while cur:
            if cur.val in duplicates:
                prev.next =cur.next
            else:
                duplicates.append(cur.val)
                prev=cur
            cur=prev.next

    def print_list(self):
        cur=self.head
        while cur:
            print(cur.val,end ='->')
            cur=cur.next
        print()


ll = LinkedList()
ll.append(1)
ll.append(2)
ll.append(1)
ll.append(3)
ll.append(3)
ll.append(4)
ll.append(3)
print('\n4.  Remove duplicates from linked list\n')
ll.remove_duplicates()
ll.print_list()

# 5. https://leetcode.com/problems/minimum-size-subarray-sum/

def minSubArrayLen(nums,target):
    winStart, res, winSum = 0, -math.inf, 0
    for winEnd in range(len(nums)):
        winSum += nums[winEnd]
        while winSum>=target:
            res= min(res,winEnd-winStart+1)
            winSum-=nums[winStart]
            winStart+=1
        return 0 if res==-math.inf else res

print('5. minSubArrayLen', str(minSubArrayLen([2,3,1,2,4,3],7)))
print('5. minSubArrayLen', str(minSubArrayLen([1,4,4],4)))
print('5. minSubArrayLen', str(minSubArrayLen([1,1,1,1,1,1,1,1],11)))



# 8. https://leetcode.com/problems/string-compression/

def string_compression(chars):
    res=''
    prev=chars[0]
    count=0
    for i in range(len(chars)):
        if chars[i]==prev:
            count+=1
        else:
            if count==1:
                res += prev
            else:
                res += prev + str(count)
            count=1
            prev=chars[i]
    if count==1:
        res += prev
    else:
        res += prev + str(count)
    for i in range(len(res)):
        chars[i]=res[i]

    return len(res)

nums=["a","a","b","b","c","c","c"]
res=string_compression(nums)
print('\n6. string_compression, length- ', res, ' result- ',''.join(nums[:res]))

nums=["a","b","b","b","b","b","b","b","b","b","b","b","b"]
res=string_compression(nums)
print('\n6. string_compression, length- ', res, ' result- ',''.join(nums[:res]))


# Input:  [{"Bob","87"], {"Mike", "35"],{"Bob", "52"], {"Jason","35"], {"Mike", "55"], {"Jessica", "99"]]
# Output: 99
# Explanation: Since Jessica's average is greater than Bob's, Mike's and Jason's average.

def find_max_average(input):
    avgs ={}
    for name,score in input:
        if name in avgs:
            avgs[name][0]+= int(score)
            avgs[name][1]+=1
        else:
            avgs[name] = [int(score),1]
    maxScore =0
    for key,val in avgs.items():
        maxScore = max(maxScore, val[0]/val[1])
    return maxScore


avgs = [("Bob", "87"), ("Mike", "35"), ("Bob", "52"), ("Jason", "35"), ("Mike", "55"), ("Jessica", "99")]
print('7. find_max_average: ',find_max_average(avgs))

# leetcode -> l
def find_first_unique_char(string):
    chars ={}
    ordered=[]
    for ch in string:
        chars[ch]= chars.get(ch,0)+1
        if ch not in ordered:
            ordered.append(ch)
    for ch in ordered:
        if chars[ch]==1:
            return ch


string = 'leetcodelove'
print('8. find_first_unique_char',find_first_unique_char(string))


def find_maximum_occuring_char_instring(string):
    chars={}
    maxOccur, res= 0,''
    for ch in string:
        chars[ch]= chars.get(ch,0)+1
        if chars[ch]>maxOccur:
            maxOccur=chars[ch]
            res=ch
    return res


string = 'loeooeetcodelove'
print('9. find_maximum_occuring_char_instring',find_maximum_occuring_char_instring(string))

from heapq import *


def findMedianSortedArrays(nums1, nums2):
    totalLen = len(nums1) + len(nums2)
    needk2=False
    if totalLen%2==0:
        k1= (totalLen-1)//2
        needk2 =True
    else:
        k1 = totalLen//2

    i,j,k = 0,0,0
    m1,m2=0,0
    l1,l2 = len(nums1), len(nums2)
    while i<l1 and j<l2:
        if nums1[i]<nums2[j]:
            val=nums1[i]
            i+=1
        else:
            val= nums2[j]
            j+=1
        if k==k1 and not needk2:
            return val
        elif k==k1 and needk2:
            m1=val
        elif k>k1:
            return (m1+val)/2.0
        k+=1


print('10. findMedianSortedArrays', findMedianSortedArrays([1,2,4,4,5],[1,2,5,9,10]))


# #Problem #31 : Longest Uniform Substring
def longestUniformSubString(str):
    winStart, vals= 0, {}
    maxLen, res=0,[-1,-1]
    for winEnd in range(len(str)):
        right = str[winEnd]
        vals[right] = vals.get(right,0)+1
        while len(vals)>1:
            left = str[winStart]
            vals[left]-=1
            if vals[left]==0:
                del vals[left]
            winStart+=1
        if winEnd-winStart+1>maxLen:
            maxLen = winEnd-winStart+1
            res =[winStart,winEnd+1]
    return str[res[0]:res[1]]


print('11. longestUniformSubString: ', longestUniformSubString('abcccbbbbba'))


# https://www.geeksforgeeks.org/find-starting-indices-substrings-string-s-made-concatenating-words-listl/
def subStringIndexWithConcatationOfWordList(arr, words):
    res=[]
    word_dict ={}
    for word in words:
        word_dict[word] = word_dict.get(word,0)+1
    word_count= len(words)
    word_length = len(words[0])
    for i in range(len(arr)-word_count*word_length+1):
        word_seen={}
        for j in range(word_count):
            next_ind = i+ j*word_length
            next_word= arr[next_ind:next_ind+word_length]

            if next_word not in word_dict:
                break
            word_seen[next_word] = word_seen.get(next_word,0)+1
            if word_seen[next_word]> word_dict[next_word]:
                break
            if j+1==word_count:
                res.append(i)
    return res


print('12. subStringIndexWithConcatationOfWordList: ', subStringIndexWithConcatationOfWordList('catfoxcat', ['cat','fox']))


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
        temp_list =[0 for i in range(26)]
        for ch in word:
            temp_list[ord(ch)-ord('a')] +=1
        dic[tuple(temp_list)] = dic.get(tuple(temp_list),[]) + [word]
    return dic.values()

print('14. groupAnagrams: ', groupAnagrams(["eat","tea","tan","ate","nat","bat"]))


# https://www.geeksforgeeks.org/to-find-smallest-and-second-smallest-element-in-an-array/
def print2Smallest(arr):
    if len(arr)<=1:
        return arr
    first,sec = math.inf,math.inf
    for i in range(len(arr)):
        if arr[i]<first:
            second=first
            first=arr[i]
        elif arr[i]<second and arr[i]!=first:
            second=arr[i]
    return [first,second]

print('15. print2Smallest: ', print2Smallest([3,5,1,6,12,34,8,0,333]))


# 4. https://leetcode.com/problems/reaching-points/
def reachingPoints(sx,sy,tx,ty):
    while tx>=sx and ty>=sy:
        if tx==ty:
            return sx==tx and sy==ty
        elif tx>ty:
            if ty>sy:
                tx = tx%ty
            else:
                return (tx-sx)%sy == 0
        else:
            if tx>sx:
                ty= ty%tx
            else:
                return (ty-sy)%sx ==0
    return False


print('16.reaching points: ', str(reachingPoints(2,1,5,3)))
print('16.reaching points: ', str(reachingPoints(2,1,5,4)))
print('16.reaching points: ', str(reachingPoints(3,3,12,9)))


# 7. Graph - https://leetcode.com/problems/number-of-provinces/solution/
def findProvinceCount(isConnected):
    n = len(isConnected)
    visited =[False for j in range(n)]
    q = deque()
    count=0
    for i in range(n):
        if not visited[i]:
            visited[i]=True
            q.append(i)
            count+=1
            while q:
                node = q.popleft()
                for j in range(n):
                    if isConnected[node][j]==1 and not visited[j]:
                        visited[j]=True
                        q.append(j)
    return count

isConnected = [[1,1,0],[1,1,0],[0,0,1]]
print('17. Province count ', str(findProvinceCount(isConnected)))
isConnected = [[1,0,0],[0,1,0],[0,0,1]]
print('17. Province count ', str(findProvinceCount(isConnected)))


# https://www.geeksforgeeks.org/size-of-the-largest-trees-in-a-forest-formed-by-the-given-graph/
def addEdge(adj, u, v):
    adj[u].append(v)
    adj[v].append(u)


def largestTreeSize(nodes,edges):
    visited = [False for i in range(nodes)]
    maxLen =0

    for i in range(nodes):
        if not visited[i]:
           maxLen = max(maxLen,largest_treesize_helper(visited,edges,i))
    return maxLen


def largest_treesize_helper(visited,edges,i):
    visited[i] = True
    size=1
    for j in range(len(edges[i])):
        if not visited[edges[i][j]]:
            size += largest_treesize_helper(visited,edges,edges[i][j])
    return size

V = 6
edges = [[0, 1], [0, 2], [3, 4], [0,4],[3,5]]
edges = [[] for i in range(V)]

addEdge(edges, 0, 1)
addEdge(edges, 0, 2)
addEdge(edges, 3, 4)
addEdge(edges, 0, 4)
addEdge(edges, 3, 5)
print(edges)
print('18. largestTreeSize: ', str(largestTreeSize(V,edges)))


# https://leetcode.com/problems/high-five/
def highFive(items):
    vals={}
    result =[]
    for item in items:
        if item[0] not in vals:
            vals[item[0]]= [item[1]]
        else:
            vals[item[0]].append(item[1])
    for id,score in vals.items():
        score.sort()
        result.append([id, sum(score[-5:])//5])
    result.sort(key=lambda x:x[0])
    return result


scores = [[1,91],[1,92],[2,93],[2,97],[1,60],[2,77],[1,65],[1,87],[1,100],[2,100],[2,76]]
print('19. high Five: ', str(highFive(scores)))

#https://leetcode.com/problems/height-checker/submissions/
def heightChecker(heights):
    real = list(heights)
    real.sort()
    count=0
    for i in range(len(heights)):
        if heights[i]!=real[i]:
            count+=1
    return count

print('20. height wise stand', str(heightChecker([5,2,3,4,1])))

# https://leetcode.com/problems/coin-change/
def coinChange(coins,amount):
    if not coins or amount<=0:
        return 0
    dp = [math.inf for i in range(amount+1)]
    dp[0]=0
    for coin in coins:
        for j in range(coin, amount+1):
            dp[j]= min(dp[j],dp[j-coin]+1)
    return dp[amount] if dp[amount]<math.inf else -1


print('21. coin change: ', coinChange([1,3,5],11))


# https://leetcode.com/problems/trapping-rain-water/
def trapRainwater(height):
    if not height:
        return 0
    res=0
    left,right = 0, len(height)-1
    leftMax, rightMax = height[left],height[right]
    while left<right:
        if leftMax<rightMax:
            left+=1
            leftMax = max(leftMax,height[left])
            res += leftMax-height[left]
        else:
            right-=1
            rightMax = max(rightMax,height[right])
            res += rightMax-height[right]
    return res


print('22. trap Rainwater: ', str(trapRainwater([0,1,0,2,1,0,1,3,2,1,2,1])))


# https://www.geeksforgeeks.org/count-inversions-of-size-three-in-a-give-array/
def countInversions(arr):
    n = len(arr)
    count=0
    for i in range(1,n-1):
        small = 0
        for j in range(i+1,n):
            if arr[j]<arr[i]:
                small+=1
        large=0
        for j in range(i-1,-1,-1):
            if arr[j]>arr[i]:
                large +=1
        count += small*large
    return count


print('23. count inversions: ', countInversions([8,4,2,1]))

# https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/
def shortestSubArratWithSum_atleastK(nums,k):
    if not nums:
        return -1
    sums = [0 for i in range(len(nums)+1)]
    for i in range(1,len(sums)):
        sums[i]= sums[i-1] + nums[i-1]

    cur = len(nums)+1
    que=deque()
    for i in range(len(sums)):
        while que and sums[i]-sums[que[0]]>=k:
            cur= min(cur, i-que.popleft())
        while que and sums[i]<sums[que[-1]]:
            que.pop()
        que.append(i)
    return -1 if cur>len(nums) else cur

print('24. shortestSubArratWithSum_atleastK: ', shortestSubArratWithSum_atleastK([2,-1,2],3))


# https://leetcode.com/problems/longest-word-in-dictionary-through-deleting/
def findLongestWord(string, dictionary):
    ans=''
    for word in dictionary:
        if len(word)<len(ans) or (len(word)==len(ans) and word>ans):
            continue
        pos=-1
        for ch in word:
            if ch not in string[pos+1:]:
                pos=-1
                break
            pos= string.index(ch,pos+1)
        if pos!=-1:
            ans=word
    return ans

print('25. findLongestWord: ', findLongestWord('abpcplea',["ale","apple","monkey","plea"]))



# https://www.geeksforgeeks.org/lexicographical-maximum-substring-string/
def LexicographicalMaxString(str):
    maxStr =''
    for i in range(len(str)):
        maxStr = max(maxStr,str[i:])
    return maxStr

print('26. def LexicographicalMaxString: ', LexicographicalMaxString('acbacbc'))



# https://www.geeksforgeeks.org/josephus-problem-set-1-a-on-solution/
def josephusProblem(n,k):
    result = [0 for i in range(len(n))]
    for i in range(len(result)):
        result[i]=n[i]
    return helper_josephus(result,0,k)

def helper_josephus(result,start,k):
    if len(result)==1:
        return result[0]
    start = (start+k)%len(result)
    del result[start]
    return helper_josephus(result,start,k)


def josephus2(n,k):
    if n==1:
        return 1
    else:
        return (josephus2(n-1,k) +k-1)%n + 1

print('27. Josephus problem: ', josephusProblem([1,2,3,4,5,6,7,8,9,10,11,12,13,14],2))
print('27. Choosen place for Josephus to spare person: ', josephus2(14,2))


# https://www.geeksforgeeks.org/find-largest-word-dictionary-deleting-characters-given-string/
# Time Complexity: O(N*(K+n)) Here N is the length of dictionary and n is the length of given string ‘str’ and K – maximum length of words in the dictionary.
# Auxiliary Space: O(1)
def largestWordInDIctionaryBydeleteingCharsInString(dictionary,string):
    result=''
    length=0
    for word in dictionary:
        if len(word)>length and isSubSequence(word,string):
            result=word
            length=len(word)
    return result

def isSubSequence(word,string):
    l1 =len(word)
    l2 = len(string)
    i,j=0,0
    while i<l1 and j<l2:
        if word[i]==string[j]:
            i+=1
        j+=1
    if i==l1:
        return True


dict1 = ["ale", "apple", "monkey", "plea"];
str1 = "abpcplea" ;
print('28. Find longest string in dictionary: ', largestWordInDIctionaryBydeleteingCharsInString(dict1, str1));


def longestCommonSubstring(string1, string2):
    l1, l2 = len(string1), len(string2)
    dp= [[0 for j in range(l2+1)] for i in range(l1+1)]

    for i in range(l1+1):
        for j in range(l2+1):
            if i==0 or j==0:
                dp[i][j]=0
            elif string1[i-1]!=string2[j-1]:
                dp[i][j]= max(dp[i-1][j], dp[i][j-1])
            else:
                dp[i][j]=dp[i-1][j-1]+1
    return dp[l1][l2]


X = "AGGTAB"
Y = "GXTXAYB"
print("29. Length of LCS is ",longestCommonSubstring(X, Y))

# https://www.geeksforgeeks.org/print-all-prime-factors-of-a-given-number/
# Time - O(sqrt(n))
def primeFactorizers(n):
    result =[]
    while n%2==0:
        result.append(2)
        n=n/2

    for i in range(3, int(math.sqrt(n))+1,2):
        if n%i==0:
            result.append(i)
            n=n/i
    if n>2:
        result.append(n)
    return result

print('30. Prime Factorizers: ', str(primeFactorizers(315)))


def rob(nums):
    if not nums:
        return 0
    prev_prev, prev = 0, nums[0]
    cur=0
    for i in range(1,len(nums)):
        cur = max(prev, prev_prev+nums[i])
        prev_prev=prev
        prev=cur
    return prev

print('31. House Robber max Amount:', rob([1,2,3,1))
