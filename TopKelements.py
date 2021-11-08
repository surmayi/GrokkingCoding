from heapq import *
from collections import deque
# Time - O(NlogK) space - O(K)
def find_k_largest_numbers(nums, k):
    minHeap =[]
    for i in range(len(nums)):
        heappush(minHeap,nums[i])
        if len(minHeap)>k:
            heappop(minHeap)
    return list(minHeap)


print("1.Here are the top K numbers: " + str(find_k_largest_numbers([3, 1, 5, 12, 2, 11], 3)))
print("1.Here are the top K numbers: " + str(find_k_largest_numbers([5, 12, 11, -1, 12], 3)))

# Time - O(NlogK) , space - O(K)
def find_Kth_smallest_number(nums, k):
    maxHeap =[]
    for i in range(len(nums)):
        heappush(maxHeap,-nums[i])
        if len(maxHeap)>k:
            heappop(maxHeap)
    return -maxHeap[0]


print("2.Kth smallest number is: " + str(find_Kth_smallest_number([1, 5, 12, 2, 11, 5], 3)))
# since there are two 5s in the input array, our 3rd and 4th smallest numbers should be a '5'
print("2.Kth smallest number is: " + str(find_Kth_smallest_number([1, 5, 12, 2, 11, 5], 4)))
print("2.Kth smallest number is: " + str(find_Kth_smallest_number([5, 12, 11, -1, 12], 3)))


class Point:

  def __init__(self, x, y):
    self.x = x
    self.y = y

  def print_point(self):
    print("[" + str(self.x) + ", " + str(self.y) + "] ", end='')

  def getDistance(self):
      return self.x**2 + self.y**2

  def __lt__(self, other):
      return self.getDistance() > other.getDistance()


# O(NlogK) , O(K)
def find_closest_points(points, k):
    result = []
    maxHeap=[]
    for i in range(len(points)):
        heappush(maxHeap,(-points[i].getDistance(),points[i]))
        if len(maxHeap)>k:
            heappop(maxHeap)

    for i in range(k):
        result.append(heappop(maxHeap)[1])
    return result


result = find_closest_points([Point(1, 3), Point(3, 4), Point(2, -1)], 2)
print("Here are the k points closest the origin: ", end='')
for point in result:
    point.print_point()
result = find_closest_points([Point(1, 3), Point(1,2), Point(2, -1)], 1)
print("\nHere are the k points closest the origin: ", end='')
for point in result:
    point.print_point()


# O(KlogK) , O(K)
def find_closest_points2(points, k):
    maxHeap =[]
    for i in range(k):
        heappush(maxHeap,points[i])
    for i in range(k,len(points)):
        if points[i].getDistance() < maxHeap[0].getDistance():
            heappop(maxHeap)
            heappush(maxHeap,points[i])
    return list(maxHeap)


result = find_closest_points2([Point(1, 3), Point(1,2), Point(2, -1)], 1)
print("\nHere are the k points closest the origin: ", end='')
for point in result:
    point.print_point()
result = find_closest_points2([Point(1, 3), Point(3, 4), Point(2, -1)], 2)
print("\nHere are the k points closest the origin: ", end='')
for point in result:
    point.print_point()


# O(N logN), O(N)
def minimum_cost_to_connect_ropes(lengths):
    result=0
    minHeap=[]
    for l in lengths:
        heappush(minHeap,l)
    while len(minHeap)>1:
        s = heappop(minHeap) + heappop(minHeap)
        result+=s
        heappush(minHeap,s)
    return result


print("\nMinimum cost to connect ropes: " + str(minimum_cost_to_connect_ropes([1, 3, 11, 5])))
print("Minimum cost to connect ropes: " + str(minimum_cost_to_connect_ropes([3, 4, 5, 6])))
print("Minimum cost to connect ropes: " + str(minimum_cost_to_connect_ropes([1, 3, 11, 5, 2])))


# Time - O(N+NlogK)
# Space - O(N)
def find_k_frequent_numbers(nums,k):
    result=[]
    minHeap =[]
    vals={}
    for num in nums:
        vals[num]=vals.get(num,0)+1
    for key,val in vals.items():
        heappush(minHeap,(val,key))
        if len(minHeap)>k:
            heappop(minHeap)
    while minHeap:
        result.append(heappop(minHeap)[1])
    return result


print("Here are the K frequent numbers: " +  str(find_k_frequent_numbers([1, 3, 5, 12, 11, 12, 11], 2)))
print("Here are the K frequent numbers: " + str(find_k_frequent_numbers([5, 10, 11, 3, 9, 9, 11], 3)))


# Time - O(N logN)
# Space - O(N)
def sort_character_by_frequency(str):
    result=''
    maxHeap, vals=[],{}
    for ch in str:
        vals[ch]= vals.get(ch,0)+1
    for key,val in vals.items():
        heappush(maxHeap, (-val,key))
    while maxHeap:
        freq,val = heappop(maxHeap)
        while -freq>0:
            result+=val
            freq+=1
    return result


print("String after sorting characters by frequency: " +sort_character_by_frequency("Programming"))
print("String after sorting characters by frequency: " +sort_character_by_frequency("abcbab"))


class KthLargestNumberInStream:
    def __init__(self,nums,k):
        self.minHeap=[]
        self.k=k
        for num in nums:
            heappush(self.minHeap,num)
    # TIme - O(log K), spcae - O(K)
    def add(self,num):
        heappush(self.minHeap,num)
        while len(self.minHeap)>self.k:
            heappop(self.minHeap)
        return self.minHeap[0]


kthLargestNumber = KthLargestNumberInStream([3, 1, 5, 12, 2, 11], 4)
print("4th largest number is: " + str(kthLargestNumber.add(6)))
print("4th largest number is: " + str(kthLargestNumber.add(13)))
print("4th largest number is: " + str(kthLargestNumber.add(4)))


# O(K + logN), O(N)
def find_closest_elements(arr,K,X):
    ind = binary_search(arr,X)
    left,right = max(0, ind-1), min(len(arr)-1,ind+1)
    result =deque()
    result.append(arr[ind])
    while len(result)<K:
        if left>=0 and right<len(arr):
            if abs(arr[left]-X)<=abs(arr[right]-X):
                result.appendleft(arr[left])
                left-=1
            else:
                result.append(arr[right])
                right+=1
        elif left>=0:
            result.appendleft(arr[left])
            left-=1
        elif right<len(arr):
            result.append(arr[right])
            right+=1
        else:
            break
    return list(result)


def binary_search(arr,X):
    low,high = 0, len(arr)-1
    while low<=high:
        mid = low + (high-low)//2
        if arr[mid]==X:
            return mid
        elif arr[mid]<X:
            low=mid+1
        else:
            high=mid-1
    if low<=0:
        return 0
    return low-1


print("'K' closest numbers to 'X' are: " +str(find_closest_elements([5, 6, 7, 8, 9], 3, 7)))
print("'K' closest numbers to 'X' are: " +str(find_closest_elements([2, 4, 5, 6, 9], 3, 6)))
print("'K' closest numbers to 'X' are: " +str(find_closest_elements([2, 4, 5, 6, 9], 3, 10)))


# Time - O(NlogN + KlogN)
# Space - O(N)
def find_maximum_distinct_elements(arr,k):
    distinct =0
    if len(arr)<=k:
        return distinct
    vals, minHeap ={},[]
    for num in arr:
        vals[num]= vals.get(num,0)+1

    for key,val in vals.items():
        if val==1:
            distinct+=1
        else:
            heappush(minHeap,(val,key))
    while k>0 and minHeap:
        freq, num = heappop(minHeap)
        k-= freq-1
        if k>=0:
            distinct+=1
    if k>0:
        distinct-=k
    return distinct


print("Maximum distinct numbers after removing K numbers: " +str(find_maximum_distinct_elements([7, 3, 5, 8, 5, 3, 3], 2)))
print("Maximum distinct numbers after removing K numbers: " +str(find_maximum_distinct_elements([3, 5, 12, 11, 12], 3)))
print("Maximum distinct numbers after removing K numbers: " +str(find_maximum_distinct_elements([1, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5], 2)))


# Solution1 - TIme - O(N*logN), space - O(N)
def find_sum_of_elements(nums,k1,k2):
    minHeap=[]
    result=0
    for num in nums:
        heappush(minHeap,num)
    i=1
    while i<=k1:
        heappop(minHeap)
        i+=1
    while k2-k1-1>0:
        result+= heappop(minHeap)
        k2-=1
    return result


# Solution2 - Time O(NlogK2) , Space - O(K2)
def find_sum_of_elements2(nums,k1,k2):
    result=0
    maxHeap=[]
    for i in range(len(nums)):
        heappush(maxHeap, -nums[i])
        if len(maxHeap)==k2:
            heappop(maxHeap)
    for i in range(k2-k1-1):
        result+= -heappop(maxHeap)
    return result


print("Sum of all numbers between k1 and k2 smallest numbers: " +str(find_sum_of_elements([1, 3, 12, 5, 15, 11], 3, 6)))
print("Sum of all numbers between k1 and k2 smallest numbers: " +str(find_sum_of_elements([3, 5, 8, 7], 1, 4)))
print("Sum of all numbers between k1 and k2 smallest numbers: " +str(find_sum_of_elements2([1, 3, 12, 5, 15, 11], 3, 6)))
print("Sum of all numbers between k1 and k2 smallest numbers: " +str(find_sum_of_elements2([3, 5, 8, 7], 1, 4)))


# Time - O(N logN) space - O(N)
def rearrange_string(str):
    result=''
    maxHeap, vals = [],{}
    for ch in str:
        vals[ch] = vals.get(ch,0)+1
    for ch,freq in vals.items():
        heappush(maxHeap, (-freq,ch))
    prevCh, prevFreq =None,0
    while maxHeap:
        freq, ch = heappop(maxHeap)
        if prevCh and -prevFreq>0:
            heappush(maxHeap,(prevFreq,prevCh))
        result+=ch
        prevCh, prevFreq = ch, freq+1

    return result if len(result)==len(str) else ''


print("Rearranged string:  " + rearrange_string("aappp"))
print("Rearranged string:  " + rearrange_string("Programming"))
print("Rearranged string:  " + rearrange_string("aapa"))


# Time - O(NlogN) Space - O(N)
def reorganize_string(string,k):
    result=''
    maxHeap=[]
    vals={}
    for ch in string:
        vals[ch]= vals.get(ch,0)+1
    for ch,freq in vals.items():
        heappush(maxHeap,(-freq,ch))
    que = deque()
    while maxHeap:
        freq, ch = heappop(maxHeap)
        result += ch
        freq+=1
        que.append((freq,ch))
        if len(que)==k:
            freq,ch = que.popleft()
            if -freq>0:
                heappush(maxHeap, (freq,ch))
    return '' if len(result)!=len(string) else result


print("Reorganized string: " + reorganize_string("Programming", 5))
print("Reorganized string: " + reorganize_string("mmpp", 2))
print("Reorganized string: " + reorganize_string("aab", 2))
print("Reorganized string: " + reorganize_string("aapa", 3))


# TIme = O(NlogN), space - O(N)
def schedule_tasks(tasks, k):
    intervalCount = []
    count=0
    maxHeap = []
    vals = {}
    for t in tasks:
        vals[t] = vals.get(t, 0) + 1
    for t, freq in vals.items():
        heappush(maxHeap, (-freq, t))

    while maxHeap:
        waitQue= []
        n=k+1
        while n>0 and maxHeap:
            freq,ch = heappop(maxHeap)
            intervalCount.append(ch)
            count+=1
            if -freq>1:
                waitQue.append((freq+1,ch))
            n-=1

        for freq,ch in waitQue:
            heappush(maxHeap,(freq,ch))

        if maxHeap:
            while n>0:
                intervalCount.append('idle')
                count+=1
                n-=1
    return intervalCount


print("Minimum intervals needed to execute all tasks: " +str(schedule_tasks(['a', 'a','b', 'b','b','b','a','b', 'c', 'c'], 2)))
print("Minimum intervals needed to execute all tasks: " +str(schedule_tasks(['a', 'b', 'a'], 3)))


class Element:
    def __init__(self,number,freq,seq):
        self.number=number
        self.freq=freq
        self.seq=seq

    def __lt__(self, other):
        if self.freq!=other.freq:
            return self.freq>other.freq
        return self.seq>other.seq


class FrequencyStack:

    def __init__(self):
        self.maxHeap=[]
        self.freqMap={}
        self.seq=0

    # O(logN), O(N)
    def push(self,num):
        self.freqMap[num]= self.freqMap.get(num,0)+1
        heappush(self.maxHeap,Element(num,self.freqMap[num],self.seq))
        self.seq+=1

    # O(1) both
    def pop(self):
        num = heappop(self.maxHeap).number
        self.freqMap[num]-=1
        if self.freqMap[num]==0:
            del self.freqMap[num]
        return num


frequencyStack = FrequencyStack()
frequencyStack.push(1)
frequencyStack.push(2)
frequencyStack.push(3)
frequencyStack.push(2)
frequencyStack.push(1)
frequencyStack.push(2)
frequencyStack.push(5)
print(frequencyStack.pop())
print(frequencyStack.pop())
print(frequencyStack.pop())
