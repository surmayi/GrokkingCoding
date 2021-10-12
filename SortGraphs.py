from collections import deque

# Both space and time - O(V+E)
def topological_sort(vertices, edges):
    if vertices<=0:
        return []
    sortedList=[]
    inDegree= {i:0 for i in range(vertices)}
    graph = {i:[] for i in range(vertices)}

    for parent, child in edges:
        graph[parent].append(child)
        inDegree[child]+=1

    source = deque()
    for key, val in inDegree.items():
        if val==0:
            source.append(key)

    while source:
        vertex = source.popleft()
        sortedList.append(vertex)
        for child in graph[vertex]:
            inDegree[child]-=1
            if inDegree[child]==0:
                source.append(child)

    if len(sortedList) !=vertices:
        return []

    return sortedList


print("Topological sort: " +
      str(topological_sort(4, [[3, 2], [3, 0], [2, 0], [2, 1]])))
print("Topological sort: " +
      str(topological_sort(5, [[4, 2], [4, 3], [2, 0], [2, 1], [3, 1]])))
print("Topological sort: " +
      str(topological_sort(7, [[6, 4], [6, 2], [5, 3], [5, 4], [3, 0], [3, 1], [3, 2], [4, 1]])))


# Find if a given Directed Graph has a cycle in it or not.

def check_for_cycle(vertices,edges):
    if vertices<=0:
        return []
    sortedList=[]
    inDegree= {i:0 for i in range(vertices)}
    graph = {i:[] for i in range(vertices)}

    for parent, child in edges:
        graph[parent].append(child)
        inDegree[child]+=1

    source = deque()
    for key, val in inDegree.items():
        if val==0:
            source.append(key)

    while source:
        vertex = source.popleft()
        sortedList.append(vertex)
        for child in graph[vertex]:
            inDegree[child]-=1
            if inDegree[child]==0:
                source.append(child)

    if len(sortedList) !=vertices:
        return True

    return False

print("check_for_cycle: " +
      str(check_for_cycle(4, [[3, 2], [3, 0], [2, 0], [2, 1]])))
print("check_for_cycle: " +
      str(check_for_cycle(5, [[4, 2], [4, 3], [2, 0], [2, 1], [3, 4]])))
print("check_for_cycle: " +
      str(check_for_cycle(7, [[6, 4], [6, 2], [5, 3], [5, 4], [3, 0], [3, 1], [3, 5], [4, 1]])))


# Both time and space - O(V+E)
# Similar problem - can all courses be taken by a student
def is_scheduling_possible(tasks, preReqs):
    if tasks<=1:
        return True
    inDegree = {i:0 for i in range(tasks)}
    graph = {i:[] for i in range(tasks)}

    for child,parent in preReqs:
        graph[parent].append(child)
        inDegree[child]+=1

    que = deque()
    for key,val in inDegree.items():
        if val==0:
            que.append(key)
    completed=0
    while que:
        task = que.popleft()
        completed+=1
        for child in graph[task]:
            inDegree[child]-=1
            if inDegree[child]==0:
                que.append(child)
    if completed==tasks:
        return True
    return False


print("Is scheduling possible: " +
      str(is_scheduling_possible(3, [[0, 1], [1, 2]])))
print("Is scheduling possible: " +
      str(is_scheduling_possible(3, [[0, 1], [1, 2], [2, 0]])))
print("Is scheduling possible: " +
      str(is_scheduling_possible(6, [[0, 4], [1, 4], [3, 2], [1, 3]])))


# Both time and space - O(V+E)
# Similar problem is finding the order in which all courses can be taken
def find_scheduling_order(tasks, prerequisites):
  sortedOrder = []
  if tasks<=1:
    return []
  inDegree ={i:0 for i in range(tasks)}
  graph = {i:[] for i in range(tasks)}

  for parent, child in prerequisites:
    graph[parent].append(child)
    inDegree[child]+=1

  que= deque()
  for key, val in inDegree.items():
    if val ==0:
      que.append(key)

  while que:
    task = que.popleft()
    sortedOrder.append(task)
    for child in graph[task]:
      inDegree[child]-=1
      if inDegree[child]==0:
        que.append(child)
  if len(sortedOrder)!=tasks:
    return []
  return sortedOrder


print("Is scheduling possible, if Yes, get Order : " + str(find_scheduling_order(3, [[0, 1], [1, 2]])))
print("Is scheduling possible, if Yes, get Order : " +
      str(find_scheduling_order(3, [[0, 1], [1, 2], [2, 0]])))
print("Is scheduling possible, if Yes, get Order : " +
      str(find_scheduling_order(6, [[2, 5], [0, 5], [0, 4], [1, 4], [3, 2], [1, 3]])))



# TIme - O(V!*E), space - O(V!*E) - worst case is when each task has no prereq
def all_possible_orders(tasks, preReqs):
    finalResult=[]
    sortedOrder=[]
    if tasks<=0:
        return []
    inDegree ={i:0 for i in range(tasks)}
    graph = {i:[] for i in range(tasks)}

    for parent, child in preReqs:
        graph[parent].append(child)
        inDegree[child]+=1

    que = deque()
    for key,val in inDegree.items():
        if val==0:
            que.append(key)

    helper(inDegree,graph,que,sortedOrder,finalResult)
    return finalResult


def helper(inDegree,graph,que,sortedOrder,finalResult):
    if que:
        for task in que:
            sortedOrder.append(task)
            unsortedTasks =deque(que)
            unsortedTasks.remove(task)
            for child in graph[task]:
                inDegree[child]-=1
                if inDegree[child]==0:
                    unsortedTasks.append(child)
            helper(inDegree,graph,unsortedTasks,sortedOrder,finalResult)

            sortedOrder.remove(task)
            for child in graph[task]:
                inDegree[child]+=1
    if len(sortedOrder)==len(inDegree):
        finalResult.append(list(sortedOrder))


print("Task Orders: ", all_possible_orders(3, [[0, 1], [1, 2]]))
print("Task Orders: ", all_possible_orders(4, [[3, 2], [3, 0], [2, 0], [2, 1]]))
print("Task Orders: ",all_possible_orders(6, [[2, 5], [0, 5], [0, 4], [1, 4], [3, 2], [1, 3]]))


# Both time and space - O(V+N) , N is the no. of words and V is no. of distinct chars in all words
def find_alien_dictionary_order(words):
    sortedOrder=[]
    if len(words)<=1:
        return ''
    inDegree={}
    graph ={}

    for word in words:
        for ch in word:
            inDegree[ch]=0
            graph[ch]=[]

    for i in range(len(words)-1):
        w1, w2 = words[i],words[i+1]
        l = min(len(w1),len(w2))
        for j in range(l):
            parent, child = w1[j],w2[j]
            if parent!=child:
                inDegree[child]+=1
                graph[parent].append(child)
                break
    que=deque()
    for key, val in inDegree.items():
        if val==0:
            que.append(key)
    while que:
        ch = que.popleft()
        sortedOrder.append(ch)
        for child in graph[ch]:
            inDegree[child]-=1
            if inDegree[child]==0:
                que.append(child)

    return ''.join(sortedOrder) if len(sortedOrder)==len(inDegree) else ''


print("Character order: " + find_alien_dictionary_order(["ba", "bc", "ac", "cab"]))
print("Character order: " + find_alien_dictionary_order(["cab", "aaa", "aab"]))
print("Character order: " + find_alien_dictionary_order(["ywx", "wz", "xww", "xz", "zyy", "zwz"]))


# both time and space - O(V+N) - V is count of distinct numbers and N is total rules.
def can_construct_unique_subsequence(originalSeq, sequences):
    if not originalSeq or not sequences:
        return False
    inDegree={}
    graph={}
    for seq in sequences:
        for num in seq:
            inDegree[num]=0
            graph[num]=[]

    for seq in sequences:
        for i in range(1,len(seq)):
            parent, child = seq[i-1],seq[i]
            graph[parent].append(child)
            inDegree[child]+=1

    if len(inDegree)!=len(originalSeq):
        return False

    seq=deque()
    for key,val in inDegree.items():
        if val==0:
            seq.append(key)


    sortedOrder=[]
    while seq:
        if len(seq)>1:
            return False
        if originalSeq[len(sortedOrder)] != seq[0]:
            return False
        node = seq.popleft()
        sortedOrder.append(node)
        for child in graph[node]:
            inDegree[child]-=1
            if inDegree[child]==0:
                seq.append(child)
    if len(sortedOrder) == len(originalSeq):
        return True
    return False


print("Can construct: " +str(can_construct_unique_subsequence([1, 2, 3, 4], [[1, 2], [2, 3], [3, 4]])))
print("Can construct: " +str(can_construct_unique_subsequence([1, 2, 3, 4], [[1, 2], [2, 3], [2, 4]])))
print("Can construct: " +str(can_construct_unique_subsequence([3, 1, 4, 2, 5], [[3, 1, 5], [1, 4, 2, 5]])))


# both time and space - O(V+E) - nodes, edges(rules)
def find_minimum_height_tree_roots(nodes,edges):
    if nodes<=1:
        return []
    inDegree = {i:0 for i in range(nodes)}
    graph = {i:[] for i in range(nodes)}

    for edge in edges:
        n1, n2 = edge[0],edge[1]
        graph[n1].append(n2)
        graph[n2].append(n1)
        inDegree[n1]+=1
        inDegree[n2]+=1

    que=deque()
    for key,val in inDegree.items():
        if val==1:
            que.append(key)

    nodeCount = nodes
    while nodeCount>2:
        levelSize = len(que)
        nodeCount -=levelSize
        while levelSize>0:
            node = que.popleft()
            levelSize -=1
            for child in graph[node]:
                inDegree[child] -=1
                if inDegree[child]==1:
                    que.append(child)
    return list(que)


print("Roots of MHTs: " +str(find_minimum_height_tree_roots(5, [[0, 1], [1, 2], [1, 3], [2, 4]])))
print("Roots of MHTs: " +str(find_minimum_height_tree_roots(4, [[0, 1], [0, 2], [2, 3]])))
print("Roots of MHTs: " +str(find_minimum_height_tree_roots(4, [[0, 1], [1, 2], [1, 3]])))