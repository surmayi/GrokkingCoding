def max_sub_array_of_size_k(k, arr):
    windowStart,windowSum,maxSum = 0,0, float('-inf')
    for windowEnd in range(len(arr)):
        if windowEnd>=k:
            maxSum = max(maxSum,windowSum)
            windowSum-=arr[windowStart]
            windowStart+=1
        windowSum+=arr[windowEnd]
    return maxSum


def main():
  print("Maximum sum of a subarray of size K: " + str(max_sub_array_of_size_k(3, [2, 1, 5, 1, 3, 2])))
  print("Maximum sum of a subarray of size K: " + str(max_sub_array_of_size_k(2, [2, 3, 4, 1, 5])))


main()