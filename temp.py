class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def max_sum_path(root):
    maxSumPath = {}
    max_sum_path_helper(root, 0, [], 0, maxSumPath)
    return maxSumPath[max(maxSumPath,key=maxSumPath.get)]


def max_sum_path_helper(node, curSum, curPath, maxSum, maxSumPath):
    if not node:
        return
    curSum += node.val
    curPath.append(node.val)
    if not node.left and not node.right:
        if curSum not in maxSumPath:
            maxSumPath[curSum]=list(curPath)
    else:
        max_sum_path_helper(node.left, curSum, curPath, maxSum, maxSumPath)
        max_sum_path_helper(node.right, curSum, curPath, maxSum, maxSumPath)

    del curPath[-1]
    #curSum -= node.val


def max_sum(root):
    return max_sum_helper(root, 0)


def max_sum_helper(node, curSum):
    if not node:
        return 0
    curSum += node.val
    if not node.left and not node.right:
        return curSum
    return max(max_sum_helper(node.left, curSum), max_sum_helper(node.right, curSum))


def main():
    root = TreeNode(12)
    root.left = TreeNode(7)
    root.right = TreeNode(1)
    root.left.left = TreeNode(4)
    root.right.left = TreeNode(10)
    root.right.right = TreeNode(5)
    #print("Tree max sum " + str(max_sum(root)))
    print("Tree paths with max sum " + str(max_sum_path(root)))


main()
