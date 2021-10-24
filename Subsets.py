
# O(N*2^N)
def find_subsets(nums):
    subsets = [[]]
    for num in nums:
        for set1 in subsets:
            subsets = subsets + [set1 + [num]]
    return subsets


# O(N*N^2)
def find_subsets_with_duplicates(nums):
    subsets = [[]]
    start, end = 0, 0
    for i in range(len(nums)):
        start = 0
        if i > 0 and nums[i] == nums[i - 1]:
            start = end + 1
        end = len(subsets) - 1
        for j in range(start, end + 1):
            subsets = subsets + [subsets[j] + [nums[i]]]
    return subsets



# O(N*N!)
def find_permutations(nums):
    result = []
    find_permutations_helper(nums, 0, [], result)
    return result


def find_permutations_helper(nums, index, curSet, result):
    if len(nums) == index:
        result.append(curSet)
        return
    for i in range(len(curSet) + 1):
        newSet = list(curSet)
        newSet.insert(i, nums[index])
        find_permutations_helper(nums, index + 1, newSet, result)


# Both Space and Time = n*2^n , where n is no. of alphas for temp list and 2n for permutations getting doubled for each iteration.
def find_letter_case_string_permutations(st):
    permutations = [st]
    for i in range(len(st)):
        if st[i].isalpha():
            for j in range(len(permutations)):
                temp = list(permutations[j])
                temp[i] = temp[i].swapcase()
                permutations.append(''.join(temp))
    return permutations


# Both Space and time = O(N * 2^N)
def generate_valid_parentheses(num):
    result = []
    paran = [0 for i in range(num * 2)]
    generate_valid_parentheses_helper(num, 0, 0, paran, 0, result)
    return result


def generate_valid_parentheses_helper(num, openC, closeC, paran, ind, result):
    if num == openC and num == closeC:
        result.append(''.join(paran))
    else:
        if openC < num:
            paran[ind] = '('
            generate_valid_parentheses_helper(num, openC + 1, closeC, paran, ind + 1, result)
        if openC > closeC:
            paran[ind] = ')'
            generate_valid_parentheses_helper(num, openC, closeC + 1, paran, ind + 1, result)


# Both space and time = O(N * 2^N)
def generate_generalized_abbreviation(word):
    result = []
    generate_generalized_abbreviation_helper(word, 0, 0, [], result)
    return result


def generate_generalized_abbreviation_helper(word, ind, count, curList, result):
    if len(word) == ind:
        if count != 0:
            curList.append(str(count))
        result.append(''.join(curList))
        return
    generate_generalized_abbreviation_helper(word, ind + 1, count + 1, list(curList), result)

    if count != 0:
        curList.append(str(count))
    newList = list(curList)
    newList.append(word[ind])
    generate_generalized_abbreviation_helper(word, ind + 1, 0, newList, result)


# Time - O(N*2^N) space = O(2^N)
def diff_ways_to_evaluate_expression(exp):
    return diff_ways_to_evaluate_expression_helper(exp, {})


def diff_ways_to_evaluate_expression_helper(exp, map):
    if exp in map:
        return map[exp]
    result = []
    if '+' not in exp and '-' not in exp and '*' not in exp:
        result.append(int(exp))
    else:
        for i in range(len(exp)):
            ch = exp[i]
            if not ch.isdigit():
                leftpart = diff_ways_to_evaluate_expression_helper(exp[:i], map)
                rightpart = diff_ways_to_evaluate_expression_helper(exp[i + 1:], map)
                for chl in leftpart:
                    for chr in rightpart:
                        if ch == '+':
                            result.append(chl + chr)
                        elif ch == '-':
                            result.append(chl - chr)
                        elif ch == '*':
                            result.append(chl * chr)
    map[exp] = result
    return result

from collections import deque
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

    def printTree(self):
        root = self
        que = deque()
        que.append(root)
        while que:
            levelsize = len(que)
            while levelsize>0:
                node = que.popleft()
                print(node.val, end='->')
                levelsize-=1
                if node.left:
                    que.append(node.left)
                if node.right:
                    que.append(node.right)
            print(' ')

# Time - O(N*2^N) space = O(2^N)
def find_unique_trees(n):
    if n <= 0:
        return []
    return find_unique_trees_helper(1, n)


def find_unique_trees_helper(start, end):
    result = []

    if start > end:
        result.append(None)
        return result

    for i in range(start, end + 1):
        leftTree = find_unique_trees_helper(start, i - 1)
        rightTree = find_unique_trees_helper(i + 1, end)
        for left in leftTree:
            for right in rightTree:
                root = TreeNode(i)
                root.left = left
                root.right = right
                result.append(root)
    return result


def count_trees(n):
    return count_trees_helper(n,{})


def count_trees_helper(n,map):
    if n in map:
        return map[n]
    if n<=1:
        return 1
    count =0
    for i in range(1,n+1):
        countLeftTrees = count_trees_helper(i-1,map)
        countRightTrees = count_trees_helper(n-i,map)
        count += (countLeftTrees*countRightTrees)
    map[n]=count
    return count


def main():
    print("1.Here is the list of subsets: " + str(find_subsets([1, 3])))
    print("1.Here is the list of subsets: " + str(find_subsets([1, 5, 3])))

    print("2.Here is the list of subsets: " + str(find_subsets_with_duplicates([1, 3, 3])))
    print("2.Here is the list of subsets: " + str(find_subsets_with_duplicates([1, 5, 3, 3])))

    print("3.Here are all the permutations: " + str(find_permutations([1, 3, 5])))

    print("4.String permutations are: " + str(find_letter_case_string_permutations("ad52")))
    print("4.String permutations are: " + str(find_letter_case_string_permutations("ab7c")))

    print("5.All combinations of balanced parentheses are: " + str(generate_valid_parentheses(2)))
    print("5.All combinations of balanced parentheses are: " + str(generate_valid_parentheses(3)))

    print("6.Generalized abbreviation are: " + str(generate_generalized_abbreviation("BAT")))
    print("6.Generalized abbreviation are: " + str(generate_generalized_abbreviation("code")))

    print("7.Expression evaluations: " + str(diff_ways_to_evaluate_expression("1+2*3")))
    print("7.Expression evaluations: " + str(diff_ways_to_evaluate_expression("2*3-4-5")))

    trees = find_unique_trees(2)
    print('8. All unique tree structures')
    for tree in trees:
        tree.printTree()

    trees = find_unique_trees(3)
    for tree in trees:
        tree.printTree()

    print("\n9.Total trees: " + str(count_trees(2)))
    print("9.Total trees: " + str(count_trees(3)))

main()
