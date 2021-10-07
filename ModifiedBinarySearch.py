import math


def binary_search(arr, key):
    low,high = 0, len(arr)-1
    isAscending = arr[low]<arr[high]
    while low<=high:
        mid = low + (high-low)//2
        if key == arr[mid]:
            return mid
        if isAscending:
            if key<arr[mid]:
                high = mid-1
            else:
                low = mid+1
        else:
            if key<arr[mid]:
                low=mid+1
            else:
                high=mid-1
    return -1


def search_ceiling_of_a_number(arr, key):
    low,high = 0, len(arr)-1
    if key>arr[high]:
        return -1
    while low<=high:
        mid = low + (high-low)//2
        if key>=arr[mid]:
            low=mid+1
        else:
            high=mid-1
    return low


def search_floor_of_a_number(arr, key):
    low,high = 0, len(arr)-1
    if key<arr[low]:
        return -1
    while low<=high:
        mid = low + (high-low)//2
        if key>=arr[mid]:
            low=mid+1
        else:
            high=mid-1
    return high


def search_next_letter(arr, key):
    low,high= 0, len(arr)-1
    while low<=high:
        mid = low + (high-low)//2
        if arr[mid]<=key:
            low=mid+1
        else:
            high=mid-1
    return arr[low%len(arr)]


def search_previous_letter(arr, key):
    low,high = 0, len(arr)-1
    while low<=high:
        mid = low + (high-low)//2
        if arr[mid]<key:
            low=mid+1
        else:
            high=mid-1
    return arr[high%len(arr)]


def find_range(arr, key):
    result =[-1,-1]
    result[0] = find_range_helper(arr,key,True)
    if result[0]!=-1:
        result[1] = find_range_helper(arr,key,False)
    return result


def find_range_helper(arr,key,isMin):
    low,high = 0, len(arr)-1
    foundAt=-1
    while low<=high:
        mid = low + (high-low)//2
        if arr[mid]<key:
            low=mid+1
        elif arr[mid]>key:
            high=low-1
        else:
            foundAt=mid
            if isMin:
                high=mid-1
            else:
                low=mid+1
    return foundAt



class ArrayReader:
    def __init__(self,arr):
        self.arr=arr

    def get(self,index):
        if index>=len(self.arr):
            return math.inf
        return self.arr[index]


def search_in_infinite_array(reader, key):
    low,high = 0,1
    while key>reader.get(high):
        low,high = high+1, high+ (high-low+1)*2
    return search_in_infinite_array_helper(reader,key,low,high)


def search_in_infinite_array_helper(reader,key,low,high):
    while low<=high:
        mid = low + (high-low)//2
        if key==reader.get(mid):
            return mid
        if reader.get(mid)<key:
            low=mid+1
        else:
            high=mid-1
    return -1



def search_min_diff_element(arr, key):
    low,high = 0, len(arr)-1
    diff, res = math.inf,-1
    while low<=high:
        mid = low + (high-low)//2
        if abs(arr[mid]-key)<diff:
            diff = abs(arr[mid]-key)
            res=arr[mid]
        if arr[mid]<key:
            low=mid+1
        else:
            high=mid-1
    return res

#1,2,3,4
def find_max_in_bitonic_array(arr):
    low,high = 0, len(arr)-1
    while low<=high:
        mid = low + (high-low)//2
        left = arr[mid-1] if mid>0 else -math.inf
        right = arr[mid+1] if mid<len(arr)-1 else -math.inf
        if left<arr[mid]>right:
            return arr[mid]
        if arr[mid]>=arr[low]:
            low=mid+1
        else:
            high=mid-1
    return -1


# First, we find the peak emenent index then divide array in 2 parts from that index to first search in ascending order then descending order
def search_bitonic_array(arr, key):
    peakInd = find_peak_index(arr)
    res= search_bitonic_array_order(arr,key,0,peakInd)
    if res!=-1:
        return res
    return search_bitonic_array_order(arr,key,peakInd+1,len(arr)-1)


def find_peak_index(arr):
    low,high =0 ,len(arr)-1
    while low<=high:
        mid = low + (high-low)//2
        left = arr[mid-1] if mid>0 else float('-inf')
        right = arr[mid+1] if mid< len(arr)-1 else float('-inf')
        if left<arr[mid]>right:
            return mid
        if arr[mid]>arr[low]:   # For a rotated array, use arr[mid]>= arr[low], rest everything is same
            low=mid+1
        else:
            high=mid-1
    return -1


def search_bitonic_array_order(arr,key,low,high):
    order = arr[low]<arr[high]
    while low<=high:
        mid = low+ (high-low)//2
        if arr[mid]==key:
            return mid
        if order:
            if arr[mid]<key:
                low=mid+1
            else:
                high=mid-1
        else:
            if arr[mid]>key:
                low=mid+1
            else:
                high=mid-1
    return -1


def search_rotated_array(arr, key):
    low, high = 0, len(arr)-1
    while low<=high:
        mid = low + (high-low)//2
        if key==arr[mid]:
            return mid
        if arr[low]<arr[mid]:
            if key>=arr[low] and key< arr[mid]:
                high=mid-1
            else:
                low=mid+1
        else:
            if key>arr[mid] and key<=arr[high]:
                low=mid+1
            else:
                high=mid-1
    return -1


def search_rotated_with_duplicates(arr, key):
    low,high = 0, len(arr)-1
    while low<=high:
        mid = low + (high-low)//2
        if arr[mid]==key:
            return mid
        if arr[low]==arr[mid]==arr[high]:
            low+=1
            high-=1
        if arr[low]<arr[mid]:
            if key>=arr[low] and key<arr[mid]:
                high=mid-1
            else:
                low=mid+1
        else:
            if key>arr[mid] and key<=arr[high]:
                low=mid+1
            else:
                high=mid-1

    return -1


# [3,3,7,3]
#  0,1,2,3
def count_rotations(arr):
    low, high = 0, len(arr)-1
    while low<=high:
        mid = low + (high-low)//2
        left = arr[mid-1] if mid>0 else float('inf')
        right = arr[mid+1] if mid<len(arr)-1 else float('inf')
        if left>arr[mid]<right:
            return mid
        if arr[low]==arr[mid]==arr[high]:
            if low<high and arr[low]> arr[low+1]:
                return low+1
            low+=1
            if arr[high]<arr[high-1]:
                return high
            high-=1
        elif arr[mid]>=arr[low]:
            low=mid+1
        else:
            high=mid-1
    return 0




def main():
  print('1. binary_search any order: ',binary_search([4, 6, 10], 10))
  print('1. binary_search any order: ',binary_search([1, 2, 3, 4,4, 5, 6,6, 7], 5))
  print('1. binary_search any order: ',binary_search([10, 6, 4], 10))
  print('1. binary_search any order: ',binary_search([10, 6, 4], 4))
  
  print('2. search_ceiling_of_a_number: ',search_ceiling_of_a_number([4, 6, 10], 6))
  print('2. search_ceiling_of_a_number: ',search_ceiling_of_a_number([1, 3, 8, 10, 15], 12))
  print('2. search_ceiling_of_a_number: ',search_ceiling_of_a_number([4, 6, 10], 17))
  print('2. search_ceiling_of_a_number: ',search_ceiling_of_a_number([4, 6, 10], -1))

  print('3. search_floor_of_a_number: ',search_floor_of_a_number([4, 6, 10], 6))
  print('3. search_floor_of_a_number: ',search_floor_of_a_number([1, 3, 8, 10, 15], 12))
  print('3. search_floor_of_a_number: ',search_floor_of_a_number([4, 6, 10], 17))
  print('3. search_floor_of_a_number: ',search_floor_of_a_number([4, 6, 10], -1))

  print('4. search_next_letter: ',search_next_letter(['a', 'c', 'f', 'h'], 'f'))
  print('4. search_next_letter: ',search_next_letter(['a', 'c', 'f', 'h'], 'd'))
  print('4. search_next_letter: ', search_next_letter(['a', 'c', 'f', 'h'], 'b'))
  print('4. search_next_letter: ',search_next_letter(['a', 'c', 'f', 'h'], 'm'))

  print('5. search_previous_letter: ', search_previous_letter(['a', 'c', 'f', 'h'], 'f'))
  print('5. search_previous_letter: ', search_previous_letter(['a', 'c', 'f', 'h'], 'd'))
  print('5. search_previous_letter: ', search_previous_letter(['a', 'c', 'f', 'h'], 'b'))
  print('5. search_previous_letter: ', search_previous_letter(['a', 'c', 'f', 'h'], 'm'))

  print('6. find_range: ',find_range([4, 6, 6, 6, 9], 6))
  print('6. find_range: ',find_range([1, 3, 8, 10, 15], 10))
  print('6. find_range: ',find_range([1, 3, 8, 10, 15], 12))

  reader = ArrayReader([4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
  print("7. search_in_infinite_array: ",search_in_infinite_array(reader, 16))
  print("7. search_in_infinite_array: ",search_in_infinite_array(reader, 11))
  reader = ArrayReader([1, 3, 8, 10, 15])
  print("7. search_in_infinite_array: ",search_in_infinite_array(reader, 15))
  print("7. search_in_infinite_array: ",search_in_infinite_array(reader, 200))

  print('8. search_min_diff_element: ',search_min_diff_element([4, 6, 10], 7))
  print('8. search_min_diff_element: ',search_min_diff_element([4, 6, 10], 4))
  print('8. search_min_diff_element: ',search_min_diff_element([1, 3, 8, 10, 15], 12))
  print('8. search_min_diff_element: ',search_min_diff_element([4, 6, 10], 17))

  print('9. find_max_in_bitonic_array: ',find_max_in_bitonic_array([1, 3, 8, 12, 4, 2]))
  print('9. find_max_in_bitonic_array: ',find_max_in_bitonic_array([3, 8, 3, 1]))
  print('9. find_max_in_bitonic_array: ',find_max_in_bitonic_array([1, 3, 8, 12]))
  print('9. find_max_in_bitonic_array: ',find_max_in_bitonic_array([10, 9, 8]))

  print('10: search_bitonic_array: ',search_bitonic_array([1, 3, 8, 4, 3], 4))
  print('10: search_bitonic_array: ',search_bitonic_array([3, 8, 3, 1], 8))
  print('10: search_bitonic_array: ',search_bitonic_array([1, 3, 8, 12], 12))
  print('10: search_bitonic_array: ',search_bitonic_array([10, 9, 8], 10))
  print('10: search_bitonic_array: ',search_bitonic_array([1, 3, 8, 4, 3,2,1], 4))
  print('10: search_bitonic_array: ',search_bitonic_array([3, 4,5,6,7,8, 3, 1], 8))
  print('10: search_bitonic_array: ',search_bitonic_array([1, 3, 8, 11,12], 11))
  print('10: search_bitonic_array: ',search_bitonic_array([1,10, 9, 8,7], 1))

  print('11. search_rotated_array: ',search_rotated_array([10, 15, 1, 2, 3, 8], 2))
  print('11. search_rotated_array: ',search_rotated_array([4, 5, 7, 9, 10, -1, 2], 10))

  print('12. search_rotated_with_duplicates: ',search_rotated_with_duplicates([3, 7, 3, 3, 3], 7))
  print('12. search_rotated_with_duplicates: ',search_rotated_with_duplicates([4, 5, 7, 9, 9, 10, 10,-1, 2], 7))
  print('12. search_rotated_with_duplicates: ',search_rotated_with_duplicates([4, 4,5, 7, 9, 10, -1, 2,3,4], 7))

  print('13. count_rotations: ',count_rotations([10, 15, 1, 3, 8]))
  print('13. count_rotations: ',count_rotations([4, 5, 7, 9, 10, -1, 2]))
  print('13. count_rotations: ',count_rotations([1, 3, 8, 10]))
  print('13. count_rotations with duplicates: ', count_rotations([1, 3, 3, 8, 10]))
  print('13. count_rotations with duplicates: ', count_rotations([3,3,4,5,6,6,7,3]))

main()