# 1 https://leetcode.com/problems/robot-bounded-in-circle/

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


# 2. https://leetcode.com/problems/find-pivot-index/
def find_pivot_index(nums):
    sumL, sumR = 0, sum(nums)
    for i in range(len(nums)):
        print(sumL,sumR,i)
        sumR -= nums[i]
        if sumR==sumL:
            return i
        sumL+=nums[i]
    return -1


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


# 4. https://leetcode.com/problems/reaching-points/


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

# 7. Graph - https://leetcode.com/problems/number-of-provinces/solution/


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



print('1. isRobotBounded: ', str(isRobotBounded('GGLLGG')))
print('1. isRobotBounded: ', str(isRobotBounded('GGGLGLGLGG')))
print('1. isRobotBounded: ', str(isRobotBounded('GGGRGLGL')))
print('1. isRobotBounded: ', str(isRobotBounded('GG')))

print('2. find_pivot_index: ', str((find_pivot_index([2,1,-1]))))

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