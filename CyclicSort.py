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
    return count



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

main()