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
