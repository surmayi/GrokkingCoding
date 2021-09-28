from collections import deque
class TreeNode:
    def __init__(self,val):
        self.val = val
        self.left,self.right = None, None
        self.next =None

    def print_level_order(self):
        nextLevelRoot = self
        while nextLevelRoot:
            current = nextLevelRoot
            nextLevelRoot = None
            while current:
                print(str(current.val) + " ", end='')
                if not nextLevelRoot:
                    if current.left:
                        nextLevelRoot = current.left
                    elif current.right:
                        nextLevelRoot = current.right
                current = current.next
            print()

    def print_tree(self):
        current = self
        while current:
            print(str(current.val) + " ", end='')
            current = current.next


def levelOrderTraversal(root):
    if not root:
        return
    result=[]
    que=deque()
    que.append(root)

    while que:
        levelSize= len(que)
        temp=[]
        while levelSize>0:
            node = que.popleft()
            temp.append(node.val)
            levelSize-=1
            if node.left:
                que.append(node.left)
            if node.right:
                que.append(node.right)
        result.append(temp)
    return result


def reverseLevelOrderTraversal(root):
    if not root:
        return
    result =deque()
    que =deque()
    que.append(root)
    while que:
        levelSize= len(que)
        temp=[]
        while levelSize>0:
            node = que.popleft()
            temp.append(node.val)
            levelSize-=1
            if node.left:
                que.append(node.left)
            if node.right:
                que.append(node.right)
        result.appendleft(temp)
    return result


# Use temp as a deque and append from left/right based on whether it is left to right or right to left
def zigzagTraversal(root):
    if not root:
        return
    result=[]
    que=deque()
    que.append(root)
    leftToRight=True
    while que:
        levelSize = len(que)
        temp = deque()
        while levelSize>0:
            node = que.popleft()
            if leftToRight:
                temp.append(node.val)
            else:
                temp.appendleft(node.val)
            levelSize-=1
            if node.left:
                que.append(node.left)
            if node.right:
                que.append(node.right)
        result.append(list(temp))
        leftToRight = not leftToRight
    return result

def findLevelAvg(root):
    if not root:
        return
    result=[]
    que= deque()
    que.append(root)
    while que:
        levelSize = len(que)
        i=0
        levelSum = 0
        while i<levelSize:
            node = que.popleft()
            levelSum+= node.val
            i+=1
            if node.left:
                que.append(node.left)
            if node.right:
                que.append(node.right)
        result.append(levelSum/levelSize)
    return result


def findLargestNodeForEachLevel(root):
    if not root:
        return -1
    result=[]
    que= deque()
    que.append(root)
    while que:
        levelSize = len(que)
        i=0
        levelMax = 0
        while i<levelSize:
            node = que.popleft()
            levelMax=max(levelMax,node.val)
            i+=1
            if node.left:
                que.append(node.left)
            if node.right:
                que.append(node.right)
        result.append(levelMax)
    return result

def findMinDepth(root):
    if not root:
        return 0
    que=deque()
    que.append(root)
    minDepth =0
    while que:
        levelSize = len(que)
        minDepth+=1
        while levelSize>0:
            node = que.popleft()
            levelSize-=1
            if not node.left and not node.right:
                return minDepth
            if node.left:
                que.append(node.left)
            if node.right:
                que.append(node.right)
    return -1


def findMaxDepth(root):
    if not root:
        return 0
    que=deque()
    que.append(root)
    maxDepth = 0
    while que:
        levelSize=len(que)
        maxDepth+=1
        while levelSize>0:
            node= que.popleft()
            levelSize-=1
            if node.left:
                que.append(node.left)
            if node.right:
                que.append(node.right)
    return maxDepth


def findLevelOrderSuccessorOfKey(root,key):
    if not root:
        return
    que=deque()
    que.append(root)
    found=False
    while que:
        levelSize = len(que)
        while levelSize>0:
            node = que.popleft()
            if found:
                return node
            if node.val==key:
                found=True
            levelSize-=1
            if node.left:
                que.append(node.left)
            if node.right:
                que.append(node.right)
    return

def connectLevelOrderNodes(root):
    if not root:
        return
    que=deque()
    que.append(root)
    while que:
        levelSize=len(que)
        prev=None
        while levelSize>0:
            node = que.popleft()
            if prev:
                prev.next=node
            prev=node
            levelSize-=1
            if node.left:
                que.append(node.left)
            if node.right:
                que.append(node.right)
        if prev:
            prev.next=None
    return

def connectAllNodesLevelOrderTraversal(root):
    if not root:
        return
    que=deque()
    que.append(root)
    prev=root
    while que:
        levelSize = len(que)
        while levelSize>0:
            node = que.popleft()
            prev.next=node
            prev=node
            levelSize-=1
            if node.left:
                que.append(node.left)
            if node.right:
                que.append(node.right)
    prev.next=None
    return

def lastNodeOfEachLevel(root):
    if not root:
        return
    result=[]
    que = deque()
    que.append(root)
    while que:
        levelSize=len(que)
        last=None
        while levelSize>0:
            node= que.popleft()
            last=node
            levelSize-=1
            if node.left:
                que.append(node.left)
            if node.right:
                que.append(node.right)
        result.append(last)
    return result


def firstNodeOfEachLevel(root):
    if not root:
        return
    result=[]
    que=deque()
    que.append(root)
    while que:
        first=None
        levelSize = len(que)
        while levelSize>0:
            node=que.popleft()
            levelSize-=1
            if not first:
                first=node
            if node.left:
                que.append(node.left)
            if node.right:
                que.append(node.right)
        result.append(first)
    return result


def main():
  root = TreeNode(12)
  root.left = TreeNode(7)
  root.right = TreeNode(1)
  root.left.left = TreeNode(9)
  root.right.left = TreeNode(10)
  root.right.right = TreeNode(5)
  print("1. Level order traversal: " + str(levelOrderTraversal(root)))

  root = TreeNode(1)
  root.left = TreeNode(2)
  root.right = TreeNode(3)
  root.left.left = TreeNode(4)
  root.left.right = TreeNode(5)
  root.right.left = TreeNode(6)
  root.right.right = TreeNode(7)
  print("1. Level order traversal: " + str(levelOrderTraversal(root)))

  root = TreeNode(12)
  root.left = TreeNode(7)
  root.right = TreeNode(1)
  root.left.left = TreeNode(9)
  root.right.left = TreeNode(10)
  root.right.right = TreeNode(5)
  print("2. Reverse Level order traversal: " + str(reverseLevelOrderTraversal(root)))

  root = TreeNode(1)
  root.left = TreeNode(2)
  root.right = TreeNode(3)
  root.left.left = TreeNode(4)
  root.left.right = TreeNode(5)
  root.right.left = TreeNode(6)
  root.right.right = TreeNode(7)
  print("2. Reverse Level order traversal: " + str(reverseLevelOrderTraversal(root)))

  root = TreeNode(12)
  root.left = TreeNode(7)
  root.right = TreeNode(1)
  root.left.left = TreeNode(9)
  root.right.left = TreeNode(10)
  root.right.right = TreeNode(5)
  root.right.left.left = TreeNode(20)
  root.right.left.right = TreeNode(17)
  print("3. Zigzag traversal: " + str(zigzagTraversal(root)))

  root = TreeNode(1)
  root.left = TreeNode(2)
  root.right = TreeNode(3)
  root.left.left = TreeNode(4)
  root.left.right = TreeNode(5)
  root.right.left = TreeNode(6)
  root.right.right = TreeNode(7)
  print("3. Zigzag traversal: " + str(zigzagTraversal(root)))

  root = TreeNode(12)
  root.left = TreeNode(7)
  root.right = TreeNode(1)
  root.left.left = TreeNode(9)
  root.right.left = TreeNode(10)
  root.right.right = TreeNode(5)
  root.right.left.left = TreeNode(20)
  root.right.left.right = TreeNode(17)
  print("4. Find level Avg : " + str(findLevelAvg(root)))

  root = TreeNode(1)
  root.left = TreeNode(2)
  root.right = TreeNode(3)
  root.left.left = TreeNode(4)
  root.left.right = TreeNode(5)
  root.right.left = TreeNode(6)
  root.right.right = TreeNode(7)
  print("4. Find level Avg : " + str(findLevelAvg(root)))

  root = TreeNode(12)
  root.left = TreeNode(7)
  root.right = TreeNode(1)
  root.left.left = TreeNode(9)
  root.left.right = TreeNode(2)
  root.right.left = TreeNode(10)
  root.right.right = TreeNode(5)
  print("4. Find level Avg : " + str(findLevelAvg(root)))

  root = TreeNode(12)
  root.left = TreeNode(7)
  root.right = TreeNode(1)
  root.left.left = TreeNode(9)
  root.right.left = TreeNode(10)
  root.right.right = TreeNode(5)
  root.right.left.left = TreeNode(20)
  root.right.left.right = TreeNode(17)
  print("5. Find level max node : " + str(findLargestNodeForEachLevel(root)))

  root = TreeNode(1)
  root.left = TreeNode(2)
  root.right = TreeNode(3)
  root.left.left = TreeNode(4)
  root.left.right = TreeNode(5)
  root.right.left = TreeNode(6)
  root.right.right = TreeNode(7)
  print("5. Find level max node : " + str(findLargestNodeForEachLevel(root)))

  root = TreeNode(12)
  root.left = TreeNode(7)
  root.right = TreeNode(1)
  root.left.left = TreeNode(9)
  root.left.right = TreeNode(2)
  root.right.left = TreeNode(10)
  root.right.right = TreeNode(5)
  print("5. Find level max node : " + str(findLargestNodeForEachLevel(root)))

  root = TreeNode(12)
  root.left = TreeNode(7)
  root.right = TreeNode(1)
  root.right.left = TreeNode(10)
  root.right.right = TreeNode(5)
  print("6. Tree Min Depth: " + str(findMinDepth(root)))
  root.left.left = TreeNode(9)
  root.right.left.left = TreeNode(11)
  print("6. Tree Min Depth: " + str(findMinDepth(root)))
  root.left.left.left = TreeNode(19)
  print("6. Tree Min Depth: " + str(findMinDepth(root)))

  root = TreeNode(12)
  root.left = TreeNode(7)
  root.right = TreeNode(1)
  root.right.left = TreeNode(10)
  root.right.right = TreeNode(5)
  print("7. Tree Max Depth: " + str(findMaxDepth(root)))
  root.left.left = TreeNode(9)
  root.right.left.left = TreeNode(11)
  print("7. Tree Max Depth: " + str(findMaxDepth(root)))
  root.right.left.left.right = TreeNode(19)
  print("7. Tree Max Depth: " + str(findMaxDepth(root)))

  root = TreeNode(12)
  root.left = TreeNode(7)
  root.right = TreeNode(1)
  root.left.left = TreeNode(9)
  root.right.left = TreeNode(10)
  root.right.right = TreeNode(5)
  root.right.left.right = TreeNode(15)
  result = findLevelOrderSuccessorOfKey(root, 12)
  if result:
    print("8. findLevelOrderSuccessorOfKey: ",12, result.val)
  result = findLevelOrderSuccessorOfKey(root, 9)
  if result:
    print("8. findLevelOrderSuccessorOfKey: ",9, result.val)
  result = findLevelOrderSuccessorOfKey(root, 5)
  if result:
    print("8. findLevelOrderSuccessorOfKey: ",5, result.val)
  result = findLevelOrderSuccessorOfKey(root, 15)
  if not result:
    print("8. findLevelOrderSuccessorOfKey: ",15, result)

  root = TreeNode(12)
  root.left = TreeNode(7)
  root.right = TreeNode(1)
  root.left.left = TreeNode(9)
  root.right.left = TreeNode(10)
  root.right.right = TreeNode(5)
  connectLevelOrderNodes(root)

  print("9. connectLevelOrderNodes using 'next' pointer: ")
  root.print_level_order()

  root = TreeNode(12)
  root.left = TreeNode(7)
  root.right = TreeNode(1)
  root.left.left = TreeNode(9)
  root.right.left = TreeNode(10)
  root.right.right = TreeNode(5)
  connectAllNodesLevelOrderTraversal(root)
  print("10. connectAllNodesLevelOrderTraversal using 'next' pointer: ")
  root.print_tree()

  root = TreeNode(1)
  root.left = TreeNode(2)
  root.right = TreeNode(3)
  root.left.left = TreeNode(4)
  root.left.right = TreeNode(5)
  root.right.left = TreeNode(6)
  root.right.right = TreeNode(7)
  connectAllNodesLevelOrderTraversal(root)
  print("\n10. connectAllNodesLevelOrderTraversal using 'next' pointer: ")
  root.print_tree()


  root = TreeNode(12)
  root.left = TreeNode(7)
  root.right = TreeNode(1)
  root.left.left = TreeNode(9)
  root.right.left = TreeNode(10)
  root.right.right = TreeNode(5)
  root.left.left.left = TreeNode(3)
  result = lastNodeOfEachLevel(root)
  print("\n11.Tree right view: ")
  for node in result:
    print(str(node.val) + " ", end='')

  root = TreeNode(12)
  root.left = TreeNode(7)
  root.right = TreeNode(1)
  root.left.left = TreeNode(9)
  root.right.left = TreeNode(10)
  root.right.right = TreeNode(5)
  root.left.left.left = TreeNode(3)
  result = firstNodeOfEachLevel(root)
  print("\n12.Tree Left view: ")
  for node in result:
    print(str(node.val) + " ", end='')

main()