import heapq
from heapq import *

#1.Find median of Number stream
# Insert - logn
# find O(1)
class MedianOfAStream:
    def __init__(self):
        self.maxHeap, self.minHeap =[],[]

    def insert_num(self,num):
        if not self.maxHeap or -self.maxHeap[0]>=num:
            heappush(self.maxHeap, -num)
        else:
            heappush(self.minHeap,num)

        if len(self.maxHeap)> len(self.minHeap)+1:
            heappush(self.minHeap, -heappop(self.maxHeap))
        elif len(self.minHeap)> len(self.maxHeap):
            heappush(self.maxHeap,-heappop(self.minHeap))

    def find_median(self):
        if len(self.minHeap)==len(self.maxHeap):
            return -self.maxHeap[0]/2.0 + self.minHeap[0]/2.0
        return -self.maxHeap[0]/1.0

# Sliding Window Median using heapq
class SlidingWindowMedian:
    def __init__(self):
        self.minHeap, self.maxHeap =[],[]

    def find_sliding_window_median(self,nums,k):  # Time- O(N*k) - k for heap inserts,  Space- O(k)- for heaps, O(n) for result array
        result = [0.0 for i in range(len(nums)-k+1)]
        for i in range(len(nums)):
            if not self.maxHeap or nums[i]<= -self.maxHeap[0]:
                heappush(self.maxHeap,-nums[i])
            else:
                heappush(self.minHeap,nums[i])
            self.adjustHeaps()
            if i>=k-1:
                if len(self.maxHeap)==len(self.minHeap):
                    result[i-k+1] = -self.maxHeap[0]/2.0 + self.minHeap[0]/2.0
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
        heap[ind] =heap[-1]
        del heap[-1]
        heapify(heap)


# O(NlogN+KlogN), where ‘N’ is the total number of projects and ‘K’ is the number of projects we are selecting.
# O(n)
def find_maximum_capital(capital, profits, projects, initialAmt):
    minCapHeap, maxProfitHeap =[],[]
    for i in range(len(capital)):
        heappush(minCapHeap,(capital[i],i))
    available= initialAmt
    for _ in range(projects):
        while minCapHeap and minCapHeap[0][0]<=available:
            capital, i= heappop(minCapHeap)
            heappush(maxProfitHeap,(-profits[i],i))
        if not maxProfitHeap:
            break
        available += -heappop(maxProfitHeap)[0]
    return available


# Find the index of next interval in the list of intervals for each interval

class Interval:
    def __init__(self, start, end):
        self.start = start
        self.end = end

def find_next_interval(intervals):
    result =[-1 for i in range(len(intervals))]
    maxStartHeap, maxEndHeap=[],[]
    for i in range(len(intervals)):
        heappush(maxStartHeap,(-intervals[i].start,i))
        heappush(maxEndHeap, (-intervals[i].end, i))

    for _ in range(len(intervals)):
        topEnd, endInd = heappop(maxEndHeap)
        if -topEnd<= -maxStartHeap[0][0]:
            topStart,startInd = maxStartHeap[0]
            while maxStartHeap and -maxStartHeap[0][0]>=-topEnd:
                topStart,startInd= heappop(maxStartHeap)
            result[endInd]= startInd
            heappush(maxStartHeap, (-topStart,startInd))
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
  medianOfAStream.insert_num(6)
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
