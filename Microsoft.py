# https://leetcode.com/problems/count-good-nodes-in-binary-tree/
import math
class TreeNode:
    def __init__(self,val):
        self.val=val
        self.left=None
        self.right=None


def goodNodes(root):
    stack = [(root,-math.inf)]
    res=0
    while stack:
        node,max_so_far = stack.pop()
        if max_so_far<=node.val:
            res+=1
        if node.left:
            stack.append((node.left,max(max_so_far,node.val)))
        if node.right:
            stack.append((node.right,max(max_so_far,node.val)))
    return res


root = TreeNode(12)
root.left = TreeNode(7)
root.right = TreeNode(1)
root.left.left = TreeNode(9)
root.right.left = TreeNode(10)
root.right.right = TreeNode(5)
root.right.left.left = TreeNode(20)
root.right.left.right = TreeNode(17)
print('1. Count Good Nodes: ', str(goodNodes(root)))


# https://leetcode.com/problems/sign-of-the-product-of-an-array/
def arraySign(nums):
    if not nums:
        return 0
    count=0
    for num in nums:
        if num==0:
            return 0
        elif num<0:
            count+=1
    if count%2:
        return -1
    return 1


print('2. Return Array Sign after product: ', str(arraySign([-1,-2,-3,-4,3,2,1])))