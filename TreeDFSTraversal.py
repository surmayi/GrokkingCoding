class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None


# O(N)
def has_path(root, sumRequired):
    if not root:
        return False
    if root.val == sumRequired and not root.left and not root.right:
        return True
    return has_path(root.left, sumRequired - root.val) or has_path(root.right, sumRequired - root.val)


def find_paths(root, required_sum):
    allPaths = []
    find_paths_helper(root, required_sum, [], allPaths)
    return allPaths

# O(NlogN)
def find_paths_helper(node, sumRequired, curPath, allPaths):
    if not node:
        return

    curPath.append(node.val)
    if node.val == sumRequired and not node.left and not node.right:
        allPaths.append(list(curPath))
    else:
        find_paths_helper(node.left, sumRequired - node.val, curPath, allPaths)
        find_paths_helper(node.right, sumRequired - node.val, curPath, allPaths)
    del curPath[-1]


def return_all_paths(root):
    allPaths = []
    all_paths_helper(root, [], allPaths)
    return allPaths


def all_paths_helper(node, curPath, allPaths):
    if not node:
        return

    curPath.append(node.val)

    if not node.left and not node.right:
        allPaths.append(list(curPath))
    else:
        all_paths_helper(node.left, curPath, allPaths)
        all_paths_helper(node.right, curPath, allPaths)
    del curPath[-1]


# O(N)
def max_sum_path(root):
    return max_sum_path_helper(root,0)


def max_sum_path_helper(node,curSum):
    if not node:
        return 0
    curSum+=node.val
    if not node.left and not node.right:
        return curSum
    return max(max_sum_path_helper(node.left,curSum), max_sum_path_helper(node.right,curSum))


# O(N)
def find_paths2(root):
    maxPath=[]
    find(root,0,[],0,maxPath)
    return maxPath[-1]


def find(node,curSum,curPath,maxSum,maxPath):
    if not node:
        return
    curSum+=node.val
    curPath.append(node.val)
    if not node.left and not node.right and curSum>maxSum:
        maxSum = curSum
        maxPath.append(list(curPath))
    find(node.left,curSum,curPath,maxSum,maxPath)
    find(node.right, curSum, curPath, maxSum, maxPath)
    del curPath[-1]


 # O(N)
def sum_of_paths_asIntegers(root):
    return sumofpathhelper(root,0)

def sumofpathhelper(node,curSum):
    if not node:
        return 0
    curSum = curSum*10 + node.val
    if not node.left and not node.right:
        return curSum
    return sumofpathhelper(node.left,curSum)+ sumofpathhelper(node.right,curSum)


#O(N)
def find_path_with_sequence(root, seq):
    target = 0
    for i in range(len(seq)):
        target = target * 10 + seq[i]
    return find_path_with_sequence_helper(root, target, 0)


def find_path_with_sequence_helper(node, target, curVal):
    if not node:
        return False
    curVal = curVal * 10 + node.val
    if not node.right and not node.left:
        return curVal == target
    return find_path_with_sequence_helper(node.left, target, curVal) or find_path_with_sequence_helper(node.right,
                                                                                                       target, curVal)


def find_path2_with_sequence(root, seq):
    return find_path2_with_sequence_helper(root, seq, 0)


def find_path2_with_sequence_helper(node, seq, curInd):
    if not node:
        return False
    if curInd >= len(seq) or seq[curInd] != node.val:
        return False
    if not node.left and not node.right and curInd == len(seq) - 1:
        return True
    return find_path2_with_sequence_helper(node.left, seq, curInd + 1) or find_path2_with_sequence_helper(node.right,
                                                                                                          seq,
                                                                                                          curInd + 1)

# O(N)
def count_paths(node, S):
    if not node:
        return 0
    if not node.left and not node.right and node.val == S:
        return 1
    return count_paths(node.left, S-node.val) + count_paths(node.right, S-node.val)


def count_paths_from_anynode(root, S):
    return count_paths_from_anynode_helper(root, S, [])


def count_paths_from_anynode_helper(node, S, curPath):
    if not node:
        return 0
    curPath.append(node.val)
    pathCount,pathSum =0,0
    for i in range(len(curPath)-1,-1,-1):
        pathSum+=curPath[i]
        if pathSum==S:
            pathCount+=1

    pathCount += count_paths_from_anynode_helper(node.left,S,curPath)
    pathCount += count_paths_from_anynode_helper(node.right, S, curPath)

    del curPath[-1]
    return pathCount


# O(N)
class TreeDiameter:
    def __init__(self):
        self.diameter = 0
    def getTreeDiameter(self,root):
        self.calculateHeight(root)
        return self.diameter

    def calculateHeight(self,node):
        if not node:
            return 0
        leftTreeHeight = self.calculateHeight(node.left)
        rightTreeHeight = self.calculateHeight(node.right)

        if leftTreeHeight and rightTreeHeight:
            diameter = leftTreeHeight+rightTreeHeight+1
            self.diameter =max(self.diameter,diameter)
        return max(leftTreeHeight,rightTreeHeight)+1


# O(N)
class TreePath:
    def __init__(self):
        self.maxPath = float('-inf')

    def getTreeMaxPath(self,root):
        self.calculateMaxPath(root)
        return self.maxPath

    def calculateMaxPath(self,node):
        if not node:
            return 0
        leftTreemaxPath = self.calculateMaxPath(node.left)
        rightTreemaxPath = self.calculateMaxPath(node.right)

        leftTreemaxPath = max(leftTreemaxPath,0)
        rightTreemaxPath = max(rightTreemaxPath,0)

        maxPath = leftTreemaxPath+rightTreemaxPath+node.val
        self.maxPath =max(self.maxPath,maxPath)
        return max(leftTreemaxPath,rightTreemaxPath) + node.val


def find_maximum_path_sum(root):
    treePath = TreePath()
    return treePath.getTreeMaxPath(root)


def main():
    root = TreeNode(12)
    root.left = TreeNode(7)
    root.right = TreeNode(1)
    root.left.left = TreeNode(9)
    root.right.left = TreeNode(10)
    root.right.right = TreeNode(5)
    print("1. Tree has path: " + str(has_path(root, 23)))
    print("1. Tree has path: " + str(has_path(root, 16)))

    root = TreeNode(12)
    root.left = TreeNode(7)
    root.right = TreeNode(1)
    root.left.left = TreeNode(4)
    root.right.left = TreeNode(10)
    root.right.right = TreeNode(15)
    required_sum = 23
    print("2. Tree paths with required_sum " + str(required_sum) +
          ": " + str(find_paths(root, required_sum)))

    print("3. All paths " + ": " + str(return_all_paths(root)))

    print("4a. Max Sum of all paths: " + str(max_sum_path(root)))

    print("4b. Path with max sum: " + str(find_paths2(root)))

    root = TreeNode(1)
    root.left = TreeNode(0)
    root.right = TreeNode(1)
    root.left.left = TreeNode(1)
    root.right.left = TreeNode(6)
    root.right.right = TreeNode(5)
    print("5. Total Sum of Path Numbers: " + str(sum_of_paths_asIntegers(root)))

    root = TreeNode(12)
    root.left = TreeNode(7)
    root.right = TreeNode(1)
    root.left.left = TreeNode(4)
    root.right.left = TreeNode(10)
    root.right.right = TreeNode(15)
    print("5. Total Sum of Path Numbers: " + str(sum_of_paths_asIntegers(root)))

    root = TreeNode(1)
    root.left = TreeNode(0)
    root.right = TreeNode(1)
    root.left.left = TreeNode(1)
    root.right.left = TreeNode(6)
    root.right.right = TreeNode(5)

    print("6. Tree has path sequence: " + str(find_path_with_sequence(root, [1, 0, 7])))
    print("6. Tree has path sequence: " + str(find_path_with_sequence(root, [1, 1, 6])))
    print("6. Tree has path sequence: " + str(find_path2_with_sequence(root, [1, 0, 7])))
    print("6. Tree has path sequence: " + str(find_path2_with_sequence(root, [1, 1, 6])))

    root = TreeNode(12)
    root.left = TreeNode(7)
    root.right = TreeNode(1)
    root.left.left = TreeNode(4)
    root.right.left = TreeNode(10)
    root.right.right = TreeNode(5)
    print("7. Tree has paths from root: " + str(count_paths(root, 23)))

    root = TreeNode(1)
    root.left = TreeNode(7)
    root.right = TreeNode(9)
    root.left.left = TreeNode(6)
    root.left.right = TreeNode(5)
    root.right.left = TreeNode(2)
    root.right.right = TreeNode(3)
    print("7. Tree has paths from root: " + str(count_paths(root, 12)))

    root = TreeNode(12)
    root.left = TreeNode(7)
    root.right = TreeNode(1)
    root.left.left = TreeNode(4)
    root.right.left = TreeNode(10)
    root.right.right = TreeNode(5)
    print("8. Tree has paths from any node: " + str(count_paths_from_anynode(root, 11)))

    root = TreeNode(1)
    root.left = TreeNode(7)
    root.right = TreeNode(9)
    root.left.left = TreeNode(6)
    root.left.right = TreeNode(5)
    root.right.left = TreeNode(2)
    root.right.right = TreeNode(3)
    print("8. Tree has paths from any node: " + str(count_paths_from_anynode(root, 12)))

    treeDiameter = TreeDiameter()
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.right.left = TreeNode(5)
    root.right.right = TreeNode(6)
    print("9. Tree Diameter: " + str(treeDiameter.getTreeDiameter(root)))
    root.left.left = None
    root.right.left.left = TreeNode(7)
    root.right.left.right = TreeNode(8)
    root.right.right.left = TreeNode(9)
    root.right.left.right.left = TreeNode(10)
    root.right.right.left.left = TreeNode(11)
    print("9. Tree Diameter: " + str(treeDiameter.getTreeDiameter(root)))

    treePath = TreePath()
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.right.left = TreeNode(5)
    root.right.right = TreeNode(6)
    print("10. Tree Max Path: " + str(treePath.getTreeMaxPath(root)))
    root.left.left = None
    root.right.left.left = TreeNode(7)
    root.right.left.right = TreeNode(8)
    root.right.right.left = TreeNode(9)
    root.right.left.right.left = TreeNode(10)
    root.right.right.left.left = TreeNode(11)
    print("10. Tree Max Path: " + str(treePath.getTreeMaxPath(root)))


    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)

    print("11. Maximum Path Sum: " + str(find_maximum_path_sum(root)))
    root.left.left = TreeNode(1)
    root.left.right = TreeNode(3)
    root.right.left = TreeNode(5)
    root.right.right = TreeNode(6)
    root.right.left.left = TreeNode(7)
    root.right.left.right = TreeNode(8)
    root.right.right.left = TreeNode(9)
    print("11. Maximum Path Sum: " + str(find_maximum_path_sum(root)))

    root = TreeNode(-1)
    root.left = TreeNode(-3)
    print("11. Maximum Path Sum: " + str(find_maximum_path_sum(root)))

main()
