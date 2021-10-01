def find_subsets(nums):
    subsets=[[]]
    for num in nums:
        for set1 in subsets:
            subsets = subsets+[set1+[num]]
    return subsets


def find_subsets_with_duplicates(nums):
    subsets=[[]]
    start, end = 0,0
    for i in range(len(nums)):
        start=0
        if i>0 and nums[i]==nums[i-1]:
            start = end+1
        end = len(subsets)-1
        for j in range(start,end+1):
            subsets = subsets + [subsets[j] + [nums[i]]]
    return subsets


def find_permutations(nums):
    result=[]
    find_permutations_helper(nums,0,[],result)
    return result


def find_permutations_helper(nums,index,curSet,result):
    if len(nums)==index:
        result.append(curSet)
        return
    for i in range(len(curSet)+1):
        newSet = list(curSet)
        newSet.insert(i,nums[index])
        find_permutations_helper(nums,index+1,newSet,result)


# Both Space and Time = n*2^n , where n is no. of alphas for temp list and 2n for permutations getting doubled for each iteration.
def find_letter_case_string_permutations(st):
    permutations=[st]
    for i in range(len(st)):
        if st[i].isalpha():
            for j in range(len(permutations)):
                temp = list(permutations[j])
                temp[i]= temp[i].swapcase()
                permutations.append(''.join(temp))
    return permutations


def main():

  print("Here is the list of subsets: " + str(find_subsets([1, 3])))
  print("Here is the list of subsets: " + str(find_subsets([1, 5, 3])))

  print("Here is the list of subsets: " + str(find_subsets_with_duplicates([1, 3, 3])))
  print("Here is the list of subsets: " + str(find_subsets_with_duplicates([1, 5, 3, 3])))

  print("Here are all the permutations: " + str(find_permutations([1, 3, 5])))

  print("String permutations are: " +str(find_letter_case_string_permutations("ad52")))
  print("String permutations are: " +str(find_letter_case_string_permutations("ab7c")))


main()