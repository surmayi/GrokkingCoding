
# Both time and space - O(N*C)
def solve_knapsack(profits, weights, capacity):
    n = len(weights)
    if n!=len(profits) or n<=0 or capacity<=0:
        return 0
    dp = [[0 for j in range(capacity+1)] for i in range(n)]

    for j in range(1,capacity+1):
        if weights[0]<=capacity:
            dp[0][j]=profits[0]

    for i in range(1,n):
        for j in range(1,capacity+1):
            if weights[i]<=j:
                profit1= dp[i-1][j]
                profit2 = profits[i] + dp[i-1][j-weights[i]]
                dp[i][j]= max(profit1,profit2)
            else:
                dp[i][j]= dp[i-1][j]
    print('get_selected_weights', str(get_selected_weights(dp,profits,weights,capacity)))
    return dp[n-1][capacity]

# time- O(N)
def get_selected_weights(dp,profits,weights,capacity):
    result =[]
    n= len(profits)
    totalProfit = dp[n-1][capacity]
    for i in range(n-1,0,-1):
        if totalProfit!=dp[i-1][capacity]:
            result.append(weights[i])
            capacity-=weights[i]
            totalProfit-=profits[i]
    if totalProfit>0:
        result.append(weights[0])
    return result


print(solve_knapsack([1, 6, 10, 16], [1, 2, 3, 5], 5))
print(solve_knapsack([1, 6, 10, 16], [1, 2, 3, 5], 6))
print(solve_knapsack([1, 6, 10, 16], [1, 2, 3, 5], 7))


def solve_knapsack_space_optimized(profits, weights, capacity):
    n = len(profits)
    dp = [0 for j in range(capacity+1)]
    for j in range(1,capacity+1):
        if weights[0]<=j:
            dp[j]= profits[0]
    for i in range(1,n):
        for j in range(capacity,-1,-1):
            if weights[i]<=j:
                dp[j] = max(dp[j], profits[i]+ dp[j-weights[i]])
    return dp[capacity]



print(solve_knapsack_space_optimized([1, 6, 10, 16], [1, 2, 3, 5], 5))
print(solve_knapsack_space_optimized([1, 6, 10, 16], [1, 2, 3, 5], 6))
print(solve_knapsack_space_optimized([1, 6, 10, 16], [1, 2, 3, 5], 7))

# Both time and space - O(N*S)
def can_partition(nums):
    s = sum(nums)
    if s%2!=0:
        return False
    s=s//2
    n = len(nums)
    dp = [[False for j in range(s+1)] for i in range(n)]

    for i in range(n):
        dp[i][0]=True

    for j in range(1,s+1):
        dp[0][j]= nums[0]==j

    for i in range(1,n):
        for j in range(1,s+1):
            if dp[i-1][j]:
                dp[i][j]=True
            else:
                dp[i][j]= dp[i-1][j-nums[i]]
    return dp[n-1][s]




print("Can partition: " + str(can_partition([1, 2, 3, 4])))
print("Can partition: " + str(can_partition([1, 1, 3, 4, 7])))
print("Can partition: " + str(can_partition([2, 3, 4, 6])))

# time - O(N*S), space- O(S)
def can_partition_space_optimized(nums):
    s= sum(nums)
    if s%2!=0:
        return False
    n = len(nums)
    s=s//2
    dp = [False for j in range(s+1)]
    dp[0]=True

    for i in range(1,n):
        for j in range(s,-1,-1):
            if not dp[j] and nums[i]<=j:
                dp[j] = dp[j-nums[i]]
    return dp[s]


print("Can partition space optimized : " + str(can_partition_space_optimized([1, 2, 3, 4])))
print("Can partition space optimized : " + str(can_partition_space_optimized([1, 1, 3, 4, 7])))
print("Can partition space optimized : " + str(can_partition_space_optimized([2, 3, 4, 6])))


# Time - O(N*S), S is sum and n is len of nums. space - O(S)
# this is similar to above except instead of calculating s, we are already provided it.
def can_partition_for_given_sum(nums, sum):
    dp = [False for i in range(sum+1)]

    dp[0] =True

    for j in range(1,sum+1):
        dp[j]= nums[0]==j
    n= len(nums)
    for i in range(1,n):
        for j in range(sum,-1,-1):
            if not dp[j] and nums[i]<=j:
                dp[j]= dp[j-nums[i]]
    return dp[sum]


print("Can partition for given sum : " + str(can_partition_for_given_sum([1, 2, 3, 7], 6)))
print("Can partition for given sum : " + str(can_partition_for_given_sum([1, 2, 7, 1, 5], 10)))
print("Can partition for given sum : " + str(can_partition_for_given_sum([1, 3, 4, 8], 6)))


# Both space and time - O(N*S)
def min_diff_partition_set(nums):
    s = sum(nums)
    n= len(nums)
    halfS = int(s/2)

    dp = [[False for j in range(halfS+1)] for i in range(n)]

    for i in range(n):
        dp[i][0]= True

    for j in range(1,halfS+1):
        dp[0][j] = nums[0]==j

    for i in range(1,n):
        for j in range(1,halfS+1):
            if dp[i-1][j]:
                dp[i][j]=dp[i-1][j]
            else:
                if nums[i]<=j:
                    dp[i][j]= dp[i-1][j-nums[i]]
    sum1=0
    for j in range(halfS,-1,-1):
        if dp[n-1][j]:
            sum1=j
            break
    sum2 = s-sum1

    return abs(sum1-sum2)


print("Partition Set Sum Min Difference : " + str(min_diff_partition_set([1, 2, 3, 9])))
print("Partition Set Sum Min Difference : " + str(min_diff_partition_set([1, 2, 7, 1, 5])))
print("Partition Set Sum Min Difference : " + str(min_diff_partition_set([1, 3, 100, 4])))

# Both space and time - O(N*S)
def count_subsets(nums, sum):
    n = len(nums)
    dp = [[0 for j in range(sum+1)] for i in range(n)]

    for i in range(n):
        dp[i][0]=1
    for j in range(1,sum+1):
        dp[0][j]= 1 if nums[0]==j else 0

    for i in range(1,n):
        for j in range(1,sum+1):
            dp[i][j]= dp[i-1][j]
            if nums[i]<=j:
                dp[i][j] += dp[i-1][j-nums[i]]
    return dp[n-1][sum]


print("Total number of subsets " + str(count_subsets([1, 1, 2, 3], 4)))
print("Total number of subsets: " + str(count_subsets([1, 2, 7, 1, 5], 9)))


# time - O(N*S), space- O(S)
def count_subsets_space_optimised(nums, sum):
    n = len(nums)
    dp = [0 for j in range(sum+1)]
    dp[0]=1
    for j in range(1,sum+1):
        dp[j]= 1 if nums[0]==j else 0

    for i in range(1,n):
        for j in range(sum,-1,-1):
            if nums[i]<=j:
                dp[j]+=dp[j-nums[i]]
    return dp[sum]


print("Total number of subsets with O(S) space: " + str(count_subsets_space_optimised([1, 1, 2, 3], 4)))
print("Total number of subsets with O(S) space: " + str(count_subsets_space_optimised([1, 2, 7, 1, 5], 9)))

# time - O(N*S), space - O(S)
def find_target_subsets(nums, target):
    sumN = sum(nums)
    n=len(nums)
    s = sumN+target
    if s%2!=0:
        return 0
    s=s//2
    dp = [0 for j in range(s+1)]
    dp[0]=1
    for j in range(1,s+1):
        dp[j]= nums[0]==j
    for i in range(1,n):
        for j in range(s,-1,-1):
            if nums[i]<=j:
                dp[j] += dp[j-nums[i]]
    return dp[s]

print("Total ways to place +/- to reach target sum: " + str(find_target_subsets([1, 1, 1, 3], 2)))
print("Total ways to place +/- to reach target sum: " + str(find_target_subsets([1, 2, 7, 1], 9)))
print("Total ways to place +/- to reach target sum: " + str(find_target_subsets([1, 2, 7, 1], 8)))















