import math
from heapq import *

# K way merge
class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None

    def __lt__(self, other):
        return self.val < other.val


# Time - O(NlogK) N for traversal and logK for maintaining k elements in heap at a time, space- O(K)
def merge_lists(lists):
    resultHead = cur = None
    minHeap = []
    for list in lists:
        if list:
            heappush(minHeap, list)
    while minHeap:
        node = heappop(minHeap)
        if not resultHead:
            resultHead = cur = node
        else:
            cur.next = node
            cur = node
        if node.next:
            heappush(minHeap, node.next)

    return resultHead


l1 = ListNode(2)
l1.next = ListNode(6)
l1.next.next = ListNode(8)

l2 = ListNode(3)
l2.next = ListNode(6)
l2.next.next = ListNode(7)

l3 = ListNode(1)
l3.next = ListNode(3)
l3.next.next = ListNode(4)

result = merge_lists([l1, l2, l3])
print("Here are the elements form the merged list: ", end='')
while result is not None:
    print(str(result.val) + " ", end='')
    result = result.next


# Time - O(M + MlogK) - M is no. of lists, space - O(M)
def find_Kth_smallest(lists, k):
    minHeap = []
    for i in range(len(lists)):
        heappush(minHeap, (lists[i][0], i, 0))
    num, countK = 0, 0
    while minHeap:
        num, listInd, itemInd = heappop(minHeap)
        countK += 1
        if countK == k:
            return num
        wholeList = lists[listInd]
        if len(wholeList) > itemInd + 1:
            heappush(minHeap, (wholeList[itemInd + 1], listInd, itemInd + 1))
    return -1


print("\nKth smallest number is: " + str(find_Kth_smallest([[2, 6, 8], [3, 6, 7], [1, 3, 4]], 5)))
print("Kth smallest number is: " + str(find_Kth_smallest([[5, 8, 9], [1, 7]], 3)))


# time - O(MlogK) space - O(M)
def find_median(lists):
    n = 0
    minHeap = []
    for i in range(len(lists)):
        heappush(minHeap, (lists[i][0], i, 0))
        n += len(lists[i])
    if n % 2 == 0:
        k1 = n // 2
        needk2 = True
    else:
        k1 = (n + 1) // 2
        needk2 = False
    print(n, k1)
    median = []
    num, countK1 = 0, 0
    while minHeap:
        num, listInd, itemInd = heappop(minHeap)
        countK1 += 1
        print(countK1, num)
        if countK1 == k1:
            median.append(num)
            if not needk2:
                return median[0]
        if countK1 > k1:
            median.append(num)
            return sum(median) / 2
        wholeList = lists[listInd]
        if len(wholeList) > itemInd + 1:
            heappush(minHeap, (wholeList[itemInd + 1], listInd, itemInd + 1))


print("Median number is: " + str(find_median([[2, 6, 8], [3, 6, 7], [1, 3, 4]])))
print("Median number is: " + str(find_median([[5, 8, 9], [1, 7, 10]])))


# TIme - NlogM, space- O(M)
def merge_arrays(lists):
    result = []
    minHeap = []
    for i in range(len(lists)):
        heappush(minHeap, (lists[i][0], i, 0))
    while minHeap:
        num, listInd, itemInd = heappop(minHeap)
        result.append(num)
        wholeList = lists[listInd]
        if len(wholeList) > itemInd + 1:
            heappush(minHeap, (wholeList[itemInd + 1], listInd, itemInd + 1))
    return result


print("Merge K sorted arrays : " + str(merge_arrays([[2, 6, 8], [3, 6, 7], [1, 3, 4]])))
print("Merge K sorted arrays : " + str(merge_arrays([[5, 8, 9], [1, 7, 10]])))


# Time - NlogM, space - M
def find_smallest_range(lists):
    rangeStart,rangeEnd=0,math.inf
    curMax = -math.inf
    minHeap =[]
    for i in range(len(lists)):
        heappush(minHeap,(lists[i][0],i,0))
        curMax = max(curMax,lists[i][0])

    while len(minHeap)==len(lists):
        num, listInd, itemInd = heappop(minHeap)
        if rangeEnd-rangeStart> curMax-num:
            rangeEnd,rangeStart =curMax,num
        wholeList = lists[listInd]
        if len(wholeList)> itemInd+1:
            heappush(minHeap,(wholeList[itemInd+1],listInd,itemInd+1))
            curMax=max(curMax, wholeList[itemInd+1])
    return [rangeStart,rangeEnd]


print("Smallest range is: " + str(find_smallest_range([[1, 3, 8], [11, 12], [7, 8, 10]])))


# time - k*k*logK, space - O(K)
def find_k_largest_pairs(nums1, nums2, k):
    result=[]
    minHeap=[]
    for i in range(0,min(k,len(nums1))):
        for j in range(0,min(k,len(nums2))):
            if len(minHeap)<k:
                heappush(minHeap,(nums1[i]+nums2[j],i,j))
            else:
                if nums1[i]+nums2[j]<minHeap[0][0]:
                    break
                else:
                    heappop(minHeap)
                    heappush(minHeap,(nums1[i]+nums2[j],i,j))
    for s, i,j in minHeap:
        result.append([nums1[i],nums2[j]])

    return result


print("Pairs with largest sum are: " +str(find_k_largest_pairs([9, 8, 2], [6, 3, 1], 3)))
print("Pairs with largest sum are: " +str(find_k_largest_pairs([5, 2, 1], [2, -1], 3)))
