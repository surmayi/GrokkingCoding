def find_averages_of_subarrays(K,arr):
    windowStart=0
    result=[]
    windowSum=0
    for windowEnd in range(len(arr)):
        if windowEnd>=K:
            result.append(windowSum/K)
            windowSum-=arr[windowStart]
            windowStart+=1
        windowSum+=arr[windowEnd]
    return result


arr = [1, 3, 2, 6, -1, 4, 1, 8, 2]
print(find_averages_of_subarrays(5,arr))