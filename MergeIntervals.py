class Interval:
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




def main():
    print("Merged intervals: ", end='')

    for i in merge([Interval(1, 4), Interval(2, 5), Interval(7, 9)]):
        i.print_interval()
    print()

    print("Merged intervals: ", end='')
    for i in merge([Interval(6, 7), Interval(2, 4), Interval(5, 9)]):
        i.print_interval()
    print()

    print("Merged intervals: ", end='')
    for i in merge([Interval(1, 4), Interval(2, 6), Interval(3, 5)]):
        i.print_interval()
    print()

    print("Overlap intervals? : ", end='')

    print(check_if_intervals_overlap([Interval(1, 3), Interval(4, 5), Interval(7, 9)]))

    print("Overlap intervals? : ", end='')
    print(check_if_intervals_overlap([Interval(6, 7), Interval(2, 4), Interval(5, 9)]))

    print("Overlap intervals? :", end='')
    print(check_if_intervals_overlap([Interval(1, 4), Interval(2, 6), Interval(3, 5)]))

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


main()