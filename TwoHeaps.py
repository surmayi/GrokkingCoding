import heapq
from heapq import *

#1.Find median of Number stream

class Interval:
    def __init__(self, start, end):
        self.start = start
        self.end = end

class MedianOfAStream:
    def __init__(self):
        self.minHeap=[]
        self.maxHeap =[]

    def insert_num(self,num):  # O(logn) O(n)

        if not self.maxHeap or -self.maxHeap[0]>=num:
            heappush(self.maxHeap,-num)
        else:
            heappush(self.minHeap,num)

        if len(self.maxHeap)> len(self.minHeap)+1:
            heappush(self.minHeap,-heappop(self.maxHeap))
        elif len(self.minHeap)> len(self.maxHeap):
            heappush(self.maxHeap, -heappop(self.minHeap))


    def find_median(self): # O(1) , O(n)
        if len(self.maxHeap)==len(self.minHeap):
            return -self.maxHeap[0]/2.0 + self.minHeap[0]/2.0
        return -self.maxHeap[0]/1.0

# Sliding Window Median using heapq
class SlidingWindowMedian:
    def __init__(self):
        self.minHeap, self.maxHeap =[],[]

    def find_sliding_window_median(self,nums,k):  # Time- O(N*k) - k for heap inserts,  Space- O(k)- for heaps, O(n) for result array
        result = [0.0 for i in range(len(nums)-k+1)]
        for i in range(len(nums)):
            if not self.maxHeap or -self.maxHeap[0]>= nums[i]:
                heappush(self.maxHeap,-nums[i])
            else:
                heappush(self.minHeap,nums[i])
            self.adjustHeaps()

            if i>= k-1:
                if len(self.maxHeap)==len(self.minHeap):
                    result[i-k+1]= -self.maxHeap[0]/2.0 + self.minHeap[0]/2.0
                else:
                    result[i-k+1] = -self.maxHeap[0]/1.0
                if nums[i-k+1]<= -self.maxHeap[0]:
                    self.remove(self.maxHeap,-nums[i-k+1])
                else:
                    self.remove(self.minHeap,nums[i-k+1])
                self.adjustHeaps()
        return result


    def adjustHeaps(self):
        if len(self.maxHeap)> len(self.minHeap)+1:
            heappush(self.minHeap,-heappop(self.maxHeap))
        elif len(self.minHeap)>len(self.maxHeap):
            heappush(self.maxHeap,-heappop(self.minHeap))

    def remove(self,heap,element):
        ind = heap.index(element)
        heap[ind]= heap[-1]
        del heap[-1]
        if ind<len(heap):
            heapq._siftup(heap,ind)
            heapq._siftdown(heap,0,ind)

# O(NlogN+KlogN), where ‘N’ is the total number of projects and ‘K’ is the number of projects we are selecting.
# O(n)
def find_maximum_capital(capital, profits, projects, initialAmt):
    minHeapCap, maxHeapPro =[],[]

    for i in range(len(capital)):
        heappush(minHeapCap,(capital[i],i))
    available= initialAmt

    for _ in range(projects):
        while minHeapCap and minHeapCap[0][0]<= available:
            capital, i = heappop(minHeapCap)
            heappush(maxHeapPro,(-profits[i],i))
        if not maxHeapPro:
            break
        available += -heappop(maxHeapPro)[0]
    return available


def find_next_interval(intervals):
    result = [-1 for i in range(len(intervals))]
    maxEndHeap, maxStartHeap = [],[]

    for i in range(len(intervals)):
        heappush(maxEndHeap,(-intervals[i].end,i))
        heappush(maxStartHeap,(-intervals[i].start,i))

    for _ in range(len(intervals)):
        topEnd, endInd = heappop(maxEndHeap)
        if -topEnd<=-maxStartHeap[0][0]:
            topStart, startInd = heappop(maxStartHeap)
            while maxStartHeap and -maxStartHeap[0][0]>=-topEnd:
                topStart,startInd  = heappop(maxStartHeap)
            result[endInd]= startInd
            heappush(maxStartHeap,(topStart,startInd))
    return result


def main():
  medianOfAStream = MedianOfAStream()
  medianOfAStream.insert_num(3)
  medianOfAStream.insert_num(1)
  print("The median is: " + str(medianOfAStream.find_median()))
  medianOfAStream.insert_num(5)
  print("The median is: " + str(medianOfAStream.find_median()))
  medianOfAStream.insert_num(4)
  print("The median is: " + str(medianOfAStream.find_median()))

  slidingWindowMedian = SlidingWindowMedian()
  result = slidingWindowMedian.find_sliding_window_median([1, 2, -1, 3, 5], 2)
  print("Sliding window medians are: " + str(result))

  slidingWindowMedian = SlidingWindowMedian()
  result = slidingWindowMedian.find_sliding_window_median([1, 2, -1, 3, 5], 3)
  print("Sliding window medians are: " + str(result))

  print("Maximum capital: " +
        str(find_maximum_capital([0, 1, 2], [1, 2, 3], 2, 1)))
  print("Maximum capital: " +
        str(find_maximum_capital([0, 1, 2, 3], [1, 2, 3, 5], 3, 0)))



  result = find_next_interval(
    [Interval(2, 3), Interval(3, 4), Interval(5, 6)])
  print("Next interval indices are: " + str(result))

  result = find_next_interval(
    [Interval(3, 4), Interval(1, 5), Interval(4, 6)])
  print("Next interval indices are: " + str(result))


main()
