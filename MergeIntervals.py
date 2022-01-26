from __future__ import print_function


class Interval:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __lt__(self, other):
        return self.end < other.end

    def print_interval(self):
        print("[" + str(self.start) + ", " + str(self.end) + "]", end='')


# Time - O(NlogN) (for sorting), space- O(N)
def merged(intervals):
    if len(intervals) <= 1:
        return intervals
    merged = []
    intervals.sort(key=lambda x: x.start)
    start, end = intervals[0].start, intervals[0].end
    for i in range(1, len(intervals)):
        if intervals[i].start < end:
            end = max(end, intervals[i].end)
        else:
            merged.append(Interval(start, end))
            start, end = intervals[i].start, intervals[i].end
    merged.append(Interval(start, end))
    return merged


# TIme - O(NlogN) for sorting, space-O(N) for sorting
def check_if_intervals_overlap(intervals):
    intervals.sort(key=lambda x:x.start)
    for i in range(1,len(intervals)):
        if intervals[i-1].end>intervals[i].start:
            return True
    return False


# Time- O(N), space - O(N)
def insert_new_interval_and_merge(intervals, new_interval):
    merged = []
    i = 0
    while i < len(intervals) and intervals[i][1] < new_interval[0]:
        merged.append(intervals[i])
        i += 1

    while i < len(intervals) and new_interval[1] > intervals[i][0]:
        new_interval[0] = min(new_interval[0], intervals[i][0])
        new_interval[1] = max(new_interval[1], intervals[i][1])
        i += 1
    merged.append(new_interval)
    while i < len(intervals):
        merged.append(intervals[i])
        i += 1
    return merged


#   Time = O(N+M), space = O(1), apart from space required for result
def intervals_intersection(intervals_a, intervals_b):
    if not intervals_a or not intervals_b:
        return
    i, j = 0, 0
    l1, l2 = len(intervals_a), len(intervals_b)
    result = []
    while i < l1 and j < l2:

        a_overlaps_b = intervals_a[i][0] <= intervals_b[j][0] <= intervals_a[i][1]
        b_overlaps_a = intervals_b[j][0] <= intervals_a[i][0] <= intervals_b[j][1]

        if a_overlaps_b or b_overlaps_a:
            result.append([max(intervals_a[i][0], intervals_b[j][0]),
                           min(intervals_a[i][1], intervals_b[j][1])])

        if intervals_a[i][1] < intervals_b[j][1]:
            i += 1
        else:
            j += 1
    return result


#   Time - O(NlogN) for sorting, space -O(N) for sorting
def conflicting_appointments(intervals):
    intervals.sort(key=lambda x: x[0])
    for i in range(1, len(intervals)):
        if intervals[i - 1][1] > intervals[i][0]:
            return False
    return True


#  TIme - O(NlogN) Space- O(N)
def find_conflicting_appointments(intervals):
    result=[]
    intervals.sort(key=lambda x: x[0])
    start,end = intervals[0][0],intervals[0][1]
    for i in range(1,len(intervals)):
        if end>intervals[i][0]:
            result.append([[start,end],intervals[i]])
        else:
            start,end = intervals[i][0],intervals[i][1]
    return result


from heapq import *


# Time - O(NlogN)(due to sorting), space- O(N)
def find_room_count_for_meetings(meetings):
    meetings.sort(key=lambda x: x.start)
    minHeap = []
    rooms = 0
    for meeting in meetings:
        while minHeap and minHeap[0].end <= meeting.start:
            heappop(minHeap)
        heappush(minHeap, meeting)
        rooms = max(rooms, len(minHeap))
    return rooms


# Time - O(NlogN), space - O(N)
def find_pointof_max_rooms_occupied(meetings):
    meetings.sort(key=lambda x:x.start)
    rooms, point =0,None
    minHeap=[]
    for meeting in meetings:
        while minHeap and minHeap[0].end<=meeting.start:
            heappop(minHeap)
        heappush(minHeap,meeting)
        if len(minHeap)>rooms:
            rooms= len(minHeap)
            point= minHeap[0]
    return [point.start,point.end]


# time - O(NlogN), space O(N)
def find_minimum_platforms(trains_schedule):
    trains_schedule.sort(key=lambda x: x.start)
    minHeap = []
    platforms = 0
    for schedule in trains_schedule:
        while minHeap and minHeap[0].end <= schedule.start:
            heappop(minHeap)
        heappush(minHeap, schedule)
        platforms = max(platforms, len(minHeap))
    return platforms


class Jobs:
    def __init__(self, start, end, cpu_load):
        self.start = start
        self.end = end
        self.cpu_load = cpu_load

    # this function is used by heap to compare element during heapify/adjust/insert
    def __lt__(self, other):
        return self.end < other.end


# Time - O(NlogN), space- O(N)
def find_max_cpu_load(jobs):
    jobs.sort(key=lambda x: x.start)
    minHeap = []
    maxLoad, curLoad = 0, 0
    for job in jobs:
        while minHeap and minHeap[0].end <= job.start:
            curLoad -= minHeap[0].cpu_load
            heappop(minHeap)
        heappush(minHeap, job)
        curLoad += job.cpu_load
        maxLoad = max(maxLoad, curLoad)
    return maxLoad


# Time,space - O(N*M)
def find_employee_free_time(schedules):
    allIntervals = []
    for schedule in schedules:
        for sch in schedule:
            allIntervals.append(sch)
    allIntervals.sort(key=lambda x: x.start)

    start, end = allIntervals[0].start, allIntervals[0].end
    schedules = []
    for i in range(1, len(allIntervals)):
        if allIntervals[i].start < end:
            end = max(end, allIntervals[i].end)
        else:
            schedules.append(Interval(start, end))
            start, end = allIntervals[i].start, allIntervals[i].end
    schedules.append(Interval(start, end))
    result = []
    for i in range(1, len(schedules)):
        if schedules[i].start - schedules[i - 1].end > 0:
            result.append(Interval(schedules[i - 1].end, schedules[i].start))
    return result


class Employee:
    def __init__(self, interval, empInd, intInd):
        self.interval = interval
        self.empInd = empInd
        self.intInd = intInd

    def __lt__(self, other):
        return self.interval.start < other.interval.start


# Time - NlogK, space - O(K)
def find_employee_free_time2(schedule):
    result = []
    minHeap = []
    for i in range(len(schedule)):
        heappush(minHeap, Employee(schedule[i][0], i, 0))

    prevInterval = minHeap[0].interval
    while minHeap:
        topEmp = heappop(minHeap)
        if prevInterval.end < topEmp.interval.start:
            result.append(Interval(prevInterval.end, topEmp.interval.start))
            prevInterval = topEmp.interval
        else:
            if prevInterval.end < topEmp.interval.end:
                prevInterval = topEmp.interval
        empWholeSch = schedule[topEmp.empInd]
        if len(empWholeSch) > topEmp.intInd + 1:
            heappush(minHeap, Employee(empWholeSch[topEmp.intInd + 1], topEmp.empInd, topEmp.intInd + 1))
    return result


def main():
    print("1. Merged intervals: ", end='')

    for i in merged([Interval(1, 4), Interval(2, 5), Interval(7, 9)]):
        i.print_interval()
    print()

    print("1. Merged intervals: ", end='')
    for i in merged([Interval(6, 7), Interval(2, 4), Interval(5, 9)]):
        i.print_interval()
    print()

    print("1. Merged intervals: ", end='')
    for i in merged([Interval(1, 4), Interval(2, 6), Interval(3, 5)]):
        i.print_interval()
    print()

    print("2. Overlap intervals? : ", end='')

    print(check_if_intervals_overlap([Interval(1, 3), Interval(4, 5), Interval(7, 9)]))

    print("2. Overlap intervals? : ", end='')
    print(check_if_intervals_overlap([Interval(6, 7), Interval(2, 4), Interval(5, 9)]))

    print("2. Overlap intervals? :", end='')
    print(check_if_intervals_overlap([Interval(1, 4), Interval(2, 6), Interval(3, 5)]))

    print("3.Interval after inserting the new interval: " + str(
        insert_new_interval_and_merge([[1, 3], [5, 7], [8, 12]], [4, 6])))
    print("3.Interval after inserting the new interval: " + str(
        insert_new_interval_and_merge([[1, 3], [5, 7], [8, 12]], [4, 10])))
    print(
        "3.Interval after inserting the new interval: " + str(insert_new_interval_and_merge([[2, 3], [5, 7]], [1, 4])))

    print("4.Interval Intersection: " + str(intervals_intersection([[1, 3], [5, 6], [7, 9]], [[2, 3], [5, 7]])))
    print("4.Interval Intersection: " + str(intervals_intersection([[1, 3], [5, 7], [9, 12]], [[5, 10]])))

    print("5.Can attend all appointments: " + str(conflicting_appointments([[1, 4], [2, 5], [7, 9]])))
    print("5.Can attend all appointments: " + str(conflicting_appointments([[6, 7], [2, 4], [8, 12]])))
    print("5.Can attend all appointments: " + str(conflicting_appointments([[4, 5], [2, 3], [3, 6]])))

    print("6. Conflicting appointments: " + str(find_conflicting_appointments([[1, 4], [2, 5], [7, 9]])))
    print(
        "6. Conflicting appointments: " + str(find_conflicting_appointments([[4, 5], [2, 3], [3, 6], [5, 7], [7, 8]])))
    print("6. Conflicting appointments: " + str(find_conflicting_appointments([[4, 5], [2, 3], [3, 6]])))

    print("7.Minimum meeting rooms required: " + str(
        find_room_count_for_meetings([Interval(4, 5), Interval(2, 3), Interval(2, 4), Interval(3, 5)])))
    print("7.Minimum meeting rooms required: " + str(
        find_room_count_for_meetings([Interval(1, 4), Interval(2, 5), Interval(7, 9)])))
    print("7.Minimum meeting rooms required: " + str(
        find_room_count_for_meetings([Interval(6, 7), Interval(2, 4), Interval(8, 12)])))
    print("7.Minimum meeting rooms required: " + str(
        find_room_count_for_meetings([Interval(1, 4), Interval(2, 3), Interval(3, 6)])))
    print("7.Minimum meeting rooms required: " + str(
        find_room_count_for_meetings([Interval(4, 5), Interval(2, 3), Interval(2, 4), Interval(2, 5)])))

    print("8.Point of max rooms occupied: " + str(
        find_pointof_max_rooms_occupied([Interval(4, 5), Interval(2, 3), Interval(2, 4), Interval(3, 5)])))
    print("8.Point of max rooms occupied: " + str(
        find_pointof_max_rooms_occupied([Interval(4, 5), Interval(2, 3), Interval(2, 4), Interval(2, 5)])))

    print("9.Minimum platforms required: " + str(
        find_minimum_platforms([Interval(4, 5), Interval(2, 3), Interval(2, 4), Interval(3, 5)])))
    print("9.Minimum platforms required: " + str(
        find_minimum_platforms([Interval(1, 4), Interval(2, 5), Interval(7, 9)])))
    print("9.Minimum platforms required: " + str(
        find_minimum_platforms([Interval(6, 7), Interval(2, 4), Interval(8, 12)])))
    print("9.Minimum platforms required: " + str(
        find_minimum_platforms([Interval(1, 4), Interval(2, 3), Interval(3, 6)])))
    print("9.Minimum platforms required: " + str(
        find_minimum_platforms([Interval(4, 5), Interval(2, 3), Interval(2, 4), Interval(2, 5)])))

    print("10.Maximum CPU load at any time: " + str(find_max_cpu_load([Jobs(1, 4, 3), Jobs(2, 5, 4), Jobs(7, 9, 6)])))
    print(
        "10.Maximum CPU load at any time: " + str(find_max_cpu_load([Jobs(6, 7, 10), Jobs(2, 4, 11), Jobs(8, 12, 15)])))
    print("10.Maximum CPU load at any time: " + str(find_max_cpu_load([Jobs(1, 4, 2), Jobs(2, 4, 1), Jobs(3, 6, 5)])))

    input = [[Interval(1, 3), Interval(5, 6)], [Interval(2, 3), Interval(6, 8)]]
    print("11.Free intervals: ", end='')

    for interval in find_employee_free_time(input):
        interval.print_interval()
    print()
    input = [[Interval(1, 3), Interval(9, 12)], [Interval(2, 4)], [Interval(6, 8)]]
    print("11.Free intervals: ", end='')
    for interval in find_employee_free_time(input):
        interval.print_interval()
    print()

    input = [[Interval(1, 3)], [Interval(2, 4)], [Interval(3, 5), Interval(7, 9)]]
    print("11.Free intervals: ", end='')
    for interval in find_employee_free_time(input):
        interval.print_interval()
    print()

    input = [[Interval(1, 3), Interval(5, 6)], [
        Interval(2, 3), Interval(6, 8)]]
    print("Free intervals: ", end='')
    for interval in find_employee_free_time2(input):
        interval.print_interval()
    print()

    input = [[Interval(1, 3), Interval(9, 12)], [
        Interval(2, 4)], [Interval(6, 8)]]
    print("Free intervals: ", end='')
    for interval in find_employee_free_time2(input):
        interval.print_interval()
    print()

    input = [[Interval(1, 3)], [
        Interval(2, 4)], [Interval(3, 5), Interval(7, 9)]]
    print("Free intervals: ", end='')
    for interval in find_employee_free_time2(input):
        interval.print_interval()
    print()


main()
