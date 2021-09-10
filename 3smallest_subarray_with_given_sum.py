def smallest_subarray_with_given_sum(s, arr):
  windowStart=0
  curSum=0
  smallestLen= float('inf')
  for windowEnd in range(len(arr)):
    curSum+=arr[windowEnd]
    while curSum>= s:
      smallestLen= min(smallestLen, windowEnd-windowStart+1)
      curSum-=arr[windowStart]
      windowStart+=1
  return smallestLen

def main():
  print("Smallest subarray length: " + str(smallest_subarray_with_given_sum(7, [2, 1, 5, 2, 3, 2])))
  print("Smallest subarray length: " + str(smallest_subarray_with_given_sum(7, [2, 1, 5, 2, 8])))
  print("Smallest subarray length: " + str(smallest_subarray_with_given_sum(8, [3, 4, 1, 1, 6])))


main()
