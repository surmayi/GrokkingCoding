from heapq import *
def find_k_largest_numbers(nums, k):
    minHeap =[]
    for i in range(k):
        heappush(minHeap,nums[i])

    for i in range(k,len(nums)):
        if nums[i]>=minHeap[0]:
            heappop(minHeap)
            heappush(minHeap,nums[i])
    return list(minHeap)


print("Here are the top K numbers: " + str(find_k_largest_numbers([3, 1, 5, 12, 2, 11], 3)))

print("Here are the top K numbers: " + str(find_k_largest_numbers([5, 12, 11, -1, 12], 3)))


def find_Kth_smallest_number(nums, k):
    maxHeap =[]
    for i in range(k):
        heappush(maxHeap,-nums[i])

    for i in range(k,len(nums)):
        if nums[i]<= -maxHeap[0]:
            heappop(maxHeap)
            heappush(maxHeap,-nums[i])
    return -maxHeap[0]


print("Kth smallest number is: " + str(find_Kth_smallest_number([1, 5, 12, 2, 11, 5], 3)))
# since there are two 5s in the input array, our 3rd and 4th smallest numbers should be a '5'
print("Kth smallest number is: " + str(find_Kth_smallest_number([1, 5, 12, 2, 11, 5], 4)))
print("Kth smallest number is: " + str(find_Kth_smallest_number([5, 12, 11, -1, 12], 3)))


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



def find_closest_points(points, k):
    result = []
    maxHeap=[]
    for i in range(k):
        dist = points[i].x**2 + points[i].y**2
        heappush(maxHeap,(-dist,points[i]))
    for i in range(k,len(points)):
        dist = points[i].x**2 + points[i].y**2
        if dist<= -maxHeap[0][0]:
            heappop(maxHeap)
            heappush(maxHeap,(-dist,points[i]))

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