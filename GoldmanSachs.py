# 1 https://leetcode.com/problems/robot-bounded-in-circle/
import math


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



# 5. https://leetcode.com/problems/minimum-size-subarray-sum/

def minSubArrayLen(nums,target):
    winStart,res,winSum =0,float('inf'),0
    for winEnd in range(len(nums)):
        winSum+=nums[winEnd]
        while winSum>=target:
            res = min(res,winEnd-winStart+1)
            winSum-=nums[winStart]
            winStart+=1
    return 0 if res == float('inf') else res


# 6. Given an array of integers, print the array in such a way that the first element is first maximum and second element is first minimum and so on.
# https://www.geeksforgeeks.org/alternative-sorting/

def print_alternate_sorting(nums):
    nums.sort()
    l=len(nums)
    i=0
    print('\n6. Alternate sorted list-')
    while i<(l+1)//2:
        print(nums[l-i-1],end='->')
        if l-i-1==i:
            break
        print(nums[i],end='->')
        i+=1



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





ll = LinkedList()
ll.append(1)
ll.append(2)
ll.append(1)
ll.append(3)
ll.append(3)
ll.append(4)
ll.append(3)

ll.remove_duplicates()
ll.print_list()

print('5. minSubArrayLen', str(minSubArrayLen([2,3,1,2,4,3],7)))
print('5. minSubArrayLen', str(minSubArrayLen([1,4,4],4)))
print('5. minSubArrayLen', str(minSubArrayLen([1,1,1,1,1,1,1,1],11)))

print_alternate_sorting([7, 1, 2, 3, 4, 5, 6])
print_alternate_sorting([1, 6, 9, 4, 3, 7, 8, 2])

nums=["a","a","b","b","c","c","c"]
res=string_compression(nums)
print('\n8. string_compression, length- ', res, ' result- ',''.join(nums[:res]))

nums=["a","b","b","b","b","b","b","b","b","b","b","b","b"]
res=string_compression(nums)
print('\n8. string_compression, length- ', res, ' result- ',''.join(nums[:res]))


# Input:  [{"Bob","87"}, {"Mike", "35"},{"Bob", "52"}, {"Jason","35"}, {"Mike", "55"}, {"Jessica", "99"}]
# Output: 99
# Explanation: Since Jessica's average is greater than Bob's, Mike's and Jason's average.

def find_max_average(input):
    avgs = {}
    for name, score in input:
        if name not in avgs:
            avgs[name] = [0, 0]
        avgs[name][0] += int(score)
        avgs[name][1] += 1
    maxAvg = 0
    for val in avgs.values():
        avg = val[0] / val[1]
        maxAvg = max(maxAvg, avg)
    return maxAvg


avgs = [("Bob", "87"), ("Mike", "35"), ("Bob", "52"), ("Jason", "35"), ("Mike", "55"), ("Jessica", "99")]
print(find_max_average(avgs))


def find_first_unique_char(string):
    chars = {}
    for i in range(len(string)):
        chars[string[i]] = chars.get(string[i], 0) + 1
    for i in range(len(string)):
        if chars[string[i]] == 1:
            return i
    return -1


string = 'leetcodelove'
print(find_first_unique_char(string))


def find_maximum_occuring_char_instring(string):
    chars = [0 for i in range(256)]
    for i in range(len(string)):
        chars[ord(string[i])] += 1
    maxVal = 0
    chr = -1
    for i in range(len(string)):
        if maxVal < chars[ord(string[i])]:
            maxVal = chars[ord(string[i])]
            chr = string[i]
    return chr


string = 'leetcodelove'
print(find_maximum_occuring_char_instring(string))

from heapq import *


def findMedianSortedArrays(self, nums1, nums2):
    nums1 = nums1 + nums2
    minHeap, maxHeap = [], []
    for num in nums1:
        if not maxHeap or -maxHeap[0] >= num:
            heappush(maxHeap, 0 if num == 0 else -num)
        else:
            heappush(minHeap, num)

        if len(maxHeap) > len(minHeap) + 1:
            heappush(minHeap, -heappop(maxHeap))
        elif len(minHeap) > len(maxHeap):
            heappush(maxHeap, 0 if minHeap[0] == 0 else -heappop(minHeap))
    if len(maxHeap) == len(minHeap):
        return -maxHeap[0] / 2.0 + minHeap[0] / 2.0
    return -maxHeap[0] / 1.0


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


print('longestUniformSubString: ', longestUniformSubString('abcccbbbbba'))


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


print('subStringIndexWithConcatationOfWordList: ', subStringIndexWithConcatationOfWordList('catfoxcat', ['cat','fox']))


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


print('stringToInt', myAtoi('12312312312323'))
print('stringToInt', myAtoi('-123123123'))
print('stringToInt', myAtoi('  -23123'))


# https://leetcode.com/problems/group-anagrams/
def groupAnagrams(strs):
    dic = dict([])
    for word in strs:
        temp_list =[0 for i in range(26)]
        for ch in word:
            temp_list[ord(ch)-ord('a')] +=1
        dic[tuple(temp_list)] = dic.get(tuple(temp_list),[]) + [word]
    result =[]
    for key, val in dic.items():
        result.append(val)
    return result

print('groupAnagrams: ', groupAnagrams(["eat","tea","tan","ate","nat","bat"]))


# https://www.geeksforgeeks.org/to-find-smallest-and-second-smallest-element-in-an-array/
def print2Smallest(arr):
    if len(arr)<=1:
        return arr
    first = second = math.inf
    for i in range(len(arr)):
        if arr[i]<first:
            second = first
            first=arr[i]
        elif arr[i]<second and arr[i]!=first:
            second=arr[i]
    return [first,second]



print('print2Smallest: ', print2Smallest([3,5,1,6,12,34,8,0,-23,333]))


# 4. https://leetcode.com/problems/reaching-points/
# 7. Graph - https://leetcode.com/problems/number-of-provinces/solution/
# https://www.geeksforgeeks.org/size-of-the-largest-trees-in-a-forest-formed-by-the-given-graph/
# https://leetcode.com/problems/high-five/
# https://leetcode.com/problems/trapping-rain-water/
# https://www.geeksforgeeks.org/josephus-problem-set-1-a-on-solution/
# https://www.geeksforgeeks.org/find-largest-word-dictionary-deleting-characters-given-string/
# https://www.geeksforgeeks.org/print-all-prime-factors-of-a-given-number/
# https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/
# https://leetcode.com/problems/longest-word-in-dictionary-through-deleting/
