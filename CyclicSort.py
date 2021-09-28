def cyclic_sort(nums):
    start,end = 1, len(nums)
    while start<end:
        if nums[start-1]!=start:
            ind = nums[start-1]-1
            nums[start-1], nums[ind]= nums[ind], nums[start-1]
        else:
            start+=1
    return nums


def find_missing_number(nums):
    start,end = 0, len(nums)
    while start<end:
        ind = nums[start]
        if ind<end and nums[start]!=start:
            nums[start], nums[ind]= nums[ind], nums[start]
        else:
            start+=1
    for i in range(end):
        if i != nums[i]:
            return i
    return end


def find_missing_numbers(nums):
    start,end = 0, len(nums)
    missingNumbers=[]
    while start<end:
        ind = nums[start]-1
        if nums[start]!=start and nums[start]!=nums[ind]:
            nums[start], nums[ind]= nums[ind], nums[start]
        else:
            start+=1
    for i in range(end):
        if i != nums[i]-1:
            missingNumbers.append(i+1)
    return missingNumbers


def find_duplicate(nums):
    start,end = 0, len(nums)
    while start<end:
        ind = nums[start]-1
        if nums[start]-1 !=start:
            if nums[start]==nums[ind]:
                return nums[start]
            nums[start],nums[ind]=nums[ind], nums[start]
        else:
            start+=1
    return -1


def find_duplicate_cyclic(nums):
    slow, fast = nums[0], nums[nums[0]]
    while slow!=fast:
        slow= nums[slow]
        fast = nums[nums[fast]]
    count=1
    cur = nums[slow]
    while cur!=slow:
        count+=1
        cur=nums[cur]
    return cur


def find_all_duplicates(nums):
    duplicates=[]
    start,end = 0, len(nums)
    while start<end:
        ind = nums[start]-1
        if ind!=start:
            if nums[ind]==nums[start]:
                duplicates.append(nums[start])
                start+=1
            else:
                nums[start], nums[ind]= nums[ind], nums[start]
        else:
            start+=1
    return duplicates


def find_corrupt_numbers(nums):
    start,end = 0, len(nums)
    while start<end:
        ind = nums[start]-1
        if ind!=start and nums[start]!=nums[ind]:
            nums[start], nums[ind]= nums[ind],nums[start]
        else:
            start+=1
    for i in range(end):
        if i != nums[i]-1:
            return [nums[i],i+1]
    return [-1,-1]


def find_first_smallest_missing_positive(nums):
    start,end = 0, len(nums)
    while start<end:
        ind = nums[start]-1
        if ind!=start and ind>=0 and ind<end:
            nums[start], nums[ind]= nums[ind], nums[start]
        else:
            start+=1
    for i in range(end):
        if i!=nums[i]-1:
            return i+1
    return -1


def find_first_k_missing_positive(nums, k):
    start,end = 0, len(nums)
    while start<end:
        ind = nums[start]-1
        if ind>=0 and ind<end and nums[start]!=nums[ind]:
            nums[ind], nums[start]= nums[start], nums[ind]
        else:
            start+=1
    remain= set()
    missing=[]
    for i in range(end):
        if len(missing)<k:
            if i+1!= nums[i]:
                missing.append(i+1)
                remain.add(nums[i])

    while len(missing)<k:
        if end+1 not in remain:
            missing.append(end+1)
        end+=1

    return missing

def main():
  print("1. cyclic_sort ",cyclic_sort([3, 1, 5, 4, 2]))
  print("1. cyclic_sort ",cyclic_sort([2, 6, 4, 3, 1, 5]))
  print("1. cyclic_sort ",cyclic_sort([1, 5, 6, 4, 3, 2]))

  print("2.find_missing_number ",find_missing_number([4, 0, 3, 1]))
  print("2.find_missing_number ",find_missing_number([8, 3, 5, 2, 4, 6, 0, 1]))

  print("3. find_missing_numbers ", find_missing_numbers([2, 3, 1, 8, 2, 3, 5, 1]))
  print("3. find_missing_numbers ",find_missing_numbers([2, 4, 1, 2]))
  print("3. find_missing_numbers ",find_missing_numbers([2, 3, 2, 1]))

  print("4. find_duplicate ",find_duplicate([1, 4, 4, 3, 2]))
  print("4. find_duplicate ",find_duplicate([2, 1, 3, 3, 5, 4]))
  print("4. find_duplicate ",find_duplicate([2, 4, 1, 4, 4]))

  print("5. find_duplicate without modifying ",find_duplicate([1, 1, 4, 3, 2]))
  print("5. find_duplicate without modifying ",find_duplicate([2, 1, 5, 3, 5, 4]))
  print("5. find_duplicate without modifying ",find_duplicate([2, 4, 1, 4, 4]))

  print("6: find_all_duplicates ",find_all_duplicates([3, 4, 4, 5, 5]))
  print("6: find_all_duplicates ",find_all_duplicates([5, 4, 7, 2, 3, 5, 3]))

  print("7. find_corrupt_numbers ", find_corrupt_numbers([3, 1, 2, 5, 2]))
  print("7. find_corrupt_numbers ",find_corrupt_numbers([3, 1, 2, 3, 6, 4]))

  print("8. find_first_smallest_missing_positive ", find_first_smallest_missing_positive([-3, 1, 5, 4, 2]))
  print("8. find_first_smallest_missing_positive ",find_first_smallest_missing_positive([3, -2, 0, 1, 2]))
  print("8. find_first_smallest_missing_positive ",find_first_smallest_missing_positive([3, 2, 5, 1]))

  print("9. find_first_k_missing_positive ",find_first_k_missing_positive([3, -1, 4, 5, 5], 3))
  print("9. find_first_k_missing_positive ",find_first_k_missing_positive([2, 3, 4], 3))
  print("9. find_first_k_missing_positive ",find_first_k_missing_positive([-2, -3, 4], 2))

main()
