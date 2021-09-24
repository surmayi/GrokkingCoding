class Intervals:
  def __init__(self, start, end):
    self.start = start
    self.end = end

  def print_interval(self):
    print("[" + str(self.start) + ", " + str(self.end) + "]", end='')



def merge(intervals):
    if len(intervals)<2:
        return intervals

    intervals.sort(key=lambda x:x.start)

    start = intervals[0].start
    end = intervals[0].end
    mergedIntervals=[]
    for i in range(1,len(intervals)):
        if intervals[i].start<=end:
            end = max(intervals[i].end,end)
        else:
            mergedIntervals.append(Interval(start,end))
            start = intervals[i].start
            end = intervals[i].end

    mergedIntervals.append(Interval(start,end))
    return mergedIntervals


def check_if_intervals_overlap(intervals):
    if len(intervals)<2:
        return False

    intervals.sort(key=lambda x:x.start)
    start = intervals[0].start
    end=intervals[0].end
    for i in range(1,len(intervals)):
        if intervals[i].start<=end:
            return True
        else:
            start = intervals[i].start
            end = intervals[i].end
    return False

def insert_new_interval_and_merge(intervals,new_interval):
    i,start,end=0,0,1
    merged=[]
    l=len(intervals)

    while i<l and new_interval[start]> intervals[i][end]: # until new-(1,4), int- (2,6)
        merged.append(intervals[i])
        i+=1

    while i<l and new_interval[end]>= intervals[i][start]:  # this means overlapping
        new_interval[start] = min(new_interval[start],intervals[i][start])
        new_interval[end] = max(new_interval[end],intervals[i][end])
        i+=1

    merged.append(new_interval)

    while i<l:
        merged.append(intervals[i])
        i+=1

    return merged


def intervals_intersection(intervals_a, intervals_b):
    i,j =0,0
    start,end=0,1
    l1,l2 = len(intervals_a), len(intervals_b)
    result=[]

    while i<l1 and j<l2:

        # a.start lies between start-end of b
        a_overlaps_b = intervals_b[j][start] <= intervals_a[i][start] <= intervals_b[j][end]

        b_overlaps_a = intervals_a[i][start] <= intervals_b[j][start] <= intervals_a[i][end]

        if a_overlaps_b or b_overlaps_a:
            result.append( [ max(intervals_a[i][start], intervals_b[j][start]) , min(intervals_a[i][end], intervals_b[j][end])] )

        if intervals_a[i][end]< intervals_b[j][end]:
            i+=1
        else:
            j+=1

    return result


def conflicting_appointments(intervals):
    intervals.sort(key=lambda x:x[0])
    start,end = 0,1
    for i in range(1, len(intervals)):
        if intervals[i-1][end]> intervals[i][start]:
            return False
    return True

def find_conflicting_appointments(intervals):
    intervals.sort(key=lambda x:x[0])
    result=[]
    start,end = 0,1
    checkInterval = intervals[0]
    for i in range(1, len(intervals)):
        if checkInterval[end]> intervals[i][start]:
            result.append([checkInterval, intervals[i]])
        else:
            checkInterval= intervals[i]
    return result


class Interval:
  def __init__(self, start, end):
    self.start = start
    self.end = end

  def __lt__(self, other):
      return self.end< other.end

from heapq import *
def find_room_count_for_meetings(meetings):
    meetings.sort(key=lambda x:x.start)
    rooms = 0
    minHeap =[]
    for meeting in meetings:
        while len(minHeap)>0 and meeting.start>=minHeap[0].end:
            heappop(minHeap)
        heappush(minHeap,meeting)
        rooms = max(rooms, len(minHeap))

    return rooms

def find_pointof_max_rooms_occupied(meetings):
    meetings.sort(key=lambda x:x.start)
    rooms = 0
    minHeap =[]
    pointMax=0
    for meeting in meetings:
        while len(minHeap)>0 and meeting.start>=minHeap[0].end:
            heappop(minHeap)
        heappush(minHeap,meeting)
        if rooms< len(minHeap):
            rooms = len(minHeap)
            pointMax = minHeap[0].start
    return pointMax


def find_minimum_platforms(trains_schedule):
    trains_schedule.sort(key= lambda x:x.start)
    platforms=0
    minHeap=[]
    for schedule in trains_schedule:
        while len(minHeap)>0 and schedule.start>= minHeap[0].end:
            heappop(minHeap)
        heappush(minHeap,schedule)
        platforms = max(platforms, len(minHeap))
    return platforms

class Jobs:
    def __init__(self,start,end,cpu):
        self.start = start
        self.end =end
        self.cpu = cpu

    def __lt__(self, other):
        return self.end< other.end

def find_max_cpu_load(jobs):
    jobs.sort(key=lambda x:x.start)
    curLoad, maxLoad = 0,0
    minHeap=[]
    for job in jobs:
        while len(minHeap)>0 and job.start>= minHeap[0].end:
            curLoad-=minHeap[0].cpu
            heappop(minHeap)
        heappush(minHeap,job)
        curLoad+=job.cpu
        maxLoad = max(maxLoad,curLoad)
    return maxLoad

def find_employee_free_time(schedules):
    allIntervals=[]
    for schedule in schedules:
        for int in schedule:
            allIntervals.append(int)

    schedules=[]
    allIntervals.sort(key=lambda x:x.start)
    start,end = allIntervals[0].start, allIntervals[0].end
    for i in range(1, len(allIntervals)):
        if allIntervals[i].start<end:
            end = max(allIntervals[i].end, end)
        else:
            schedules.append(Intervals(start,end))
            start,end = allIntervals[i].start, allIntervals[i].end
    allIntervals.append(Intervals(start,end))
    result =[]
    for i in range(1, len(allIntervals)):
        if allIntervals[i].start - allIntervals[i-1].end> 0:
            result.append(Intervals(allIntervals[i-1].end,allIntervals[i].start))
    return result


def main():
    print("Merged intervals: ", end='')

    for i in merge([Intervals(1, 4), Intervals(2, 5), Intervals(7, 9)]):
        i.print_interval()
    print()

    print("Merged intervals: ", end='')
    for i in merge([Intervals(6, 7), Intervals(2, 4), Intervals(5, 9)]):
        i.print_interval()
    print()

    print("Merged intervals: ", end='')
    for i in merge([Intervals(1, 4), Intervals(2, 6), Intervals(3, 5)]):
        i.print_interval()
    print()

    print("Overlap intervals? : ", end='')

    print(check_if_intervals_overlap([Intervals(1, 3), Intervals(4, 5), Intervals(7, 9)]))

    print("Overlap intervals? : ", end='')
    print(check_if_intervals_overlap([Intervals(6, 7), Intervals(2, 4), Intervals(5, 9)]))

    print("Overlap intervals? :", end='')
    print(check_if_intervals_overlap([Intervals(1, 4), Intervals(2, 6), Intervals(3, 5)]))

    print("Intervals after inserting the new interval: " + str(insert_new_interval_and_merge([[1, 3], [5, 7], [8, 12]], [4, 6])))
    print("Intervals after inserting the new interval: " + str(insert_new_interval_and_merge([[1, 3], [5, 7], [8, 12]], [4, 10])))
    print("Intervals after inserting the new interval: " + str(insert_new_interval_and_merge([[2, 3], [5, 7]], [1, 4])))

    print("Intervals Intersection: " + str(intervals_intersection([[1, 3], [5, 6], [7, 9]], [[2, 3], [5, 7]])))
    print("Intervals Intersection: " + str(intervals_intersection([[1, 3], [5, 7], [9, 12]], [[5, 10]])))

    print("Can attend all appointments: " + str(conflicting_appointments([[1, 4], [2, 5], [7, 9]])))
    print("Can attend all appointments: " + str(conflicting_appointments([[6, 7], [2, 4], [8, 12]])))
    print("Can attend all appointments: " + str(conflicting_appointments([[4, 5], [2, 3], [3, 6]])))

    print("Conflicting appointments: " + str(find_conflicting_appointments([[1, 4], [2, 5], [7, 9]])))
    print("Conflicting appointments: " + str(find_conflicting_appointments([[4,5], [2,3], [3,6], [5,7], [7,8]])))
    print("Conflicting appointments: " + str(find_conflicting_appointments([[4, 5], [2, 3], [3, 6]])))

    print("Minimum meeting rooms required: " + str(find_room_count_for_meetings([Interval(4, 5), Interval(2, 3), Interval(2, 4), Interval(3, 5)])))
    print("Minimum meeting rooms required: " +str(find_room_count_for_meetings([Interval(1, 4), Interval(2, 5), Interval(7, 9)])))
    print("Minimum meeting rooms required: " +str(find_room_count_for_meetings([Interval(6, 7), Interval(2, 4), Interval(8, 12)])))
    print("Minimum meeting rooms required: " +str(find_room_count_for_meetings([Interval(1, 4), Interval(2, 3), Interval(3, 6)])))
    print("Minimum meeting rooms required: " + str(find_room_count_for_meetings([Interval(4, 5), Interval(2, 3), Interval(2, 4), Interval(2, 5)])))

    print("Point of max rooms occupied: " + str(find_pointof_max_rooms_occupied([Interval(4, 5), Interval(2, 3), Interval(2, 4), Interval(3, 5)])))
    print("Point of max rooms occupied: " + str(find_pointof_max_rooms_occupied([Interval(4, 5), Interval(2, 3), Interval(2, 4), Interval(2, 5)])))

    print("Minimum platforms required: " + str(find_minimum_platforms([Interval(4, 5), Interval(2, 3), Interval(2, 4), Interval(3, 5)])))
    print("Minimum platforms required: " +str(find_minimum_platforms([Interval(1, 4), Interval(2, 5), Interval(7, 9)])))
    print("Minimum platforms required: " +str(find_minimum_platforms([Interval(6, 7), Interval(2, 4), Interval(8, 12)])))
    print("Minimum platforms required: " +str(find_minimum_platforms([Interval(1, 4), Interval(2, 3), Interval(3, 6)])))
    print("Minimum platforms required: " + str(find_minimum_platforms([Interval(4, 5), Interval(2, 3), Interval(2, 4), Interval(2, 5)])))

    print("Maximum CPU load at any time: " + str(find_max_cpu_load([Jobs(1, 4, 3), Jobs(2, 5, 4), Jobs(7, 9, 6)])))
    print("Maximum CPU load at any time: " + str(find_max_cpu_load([Jobs(6, 7, 10), Jobs(2, 4, 11), Jobs(8, 12, 15)])))
    print("Maximum CPU load at any time: " + str(find_max_cpu_load([Jobs(1, 4, 2), Jobs(2, 4, 1), Jobs(3, 6, 5)])))

    input = [[Intervals(1, 3), Intervals(5, 6)], [Intervals(2, 3), Intervals(6, 8)]]
    print("Free intervals: ", end='')
    for interval in find_employee_free_time(input):
        interval.print_interval()
    print()

    input = [[Intervals(1, 3), Intervals(9, 12)], [Intervals(2, 4)], [Intervals(6, 8)]]
    print("Free intervals: ", end='')
    for interval in find_employee_free_time(input):
        interval.print_interval()
    print()

    input = [[Intervals(1, 3)], [Intervals(2, 4)], [Intervals(3, 5), Intervals(7, 9)]]
    print("Free intervals: ", end='')
    for interval in find_employee_free_time(input):
        interval.print_interval()
    print()
main()