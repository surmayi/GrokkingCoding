def TwoSumOfIndexes(arr,target):
    left=0
    right=len(arr)-1
    for ind in range(len(arr)):
        twoSum = arr[left]+ arr[right]
        if twoSum==target:
            return [left,right]
        elif twoSum < target:
            left+=1
        else:
            right-=1
    return [-1,-1]


print('1. TwoSumOfIndexes: ', TwoSumOfIndexes([1,2,3,4,5,6],6))

def removeDuplicatesFromSortedList_NoExtraSpace(arr):
    cur, nxt =1,1
    while nxt<len(arr):
        if arr[cur-1]!=arr[nxt]:
            arr[cur] = arr[nxt]
            cur+=1
            nxt+=1
        else:
            nxt+=1
    return arr[:cur]

print('2. removeDuplicatesFromSortedList_NoExtraSpace: ', removeDuplicatesFromSortedList_NoExtraSpace([2,3,3,3,6,6,9]))

def removeAllOccurancesOfKeyInUnsortedList(arr,key):
    cur, nxt = 0,0
    while nxt<len(arr):
        if arr[nxt]!=key:
            arr[cur]=arr[nxt]
            cur+=1
        nxt+=1
    return arr[:cur]

print('3. removeAllOccurancesOfKeyInUnsortedList: ', removeAllOccurancesOfKeyInUnsortedList([3, 2, 3, 6, 3, 10, 9, 3],3))


def sortListAfterSquaringContainingNegatives(arr):
    sqlist = [0 for i in range(len(arr))]
    left=0
    right = ind= len(arr)-1
    while left<right:
        leftsq =arr[left]**2
        rightsq = arr[right]**2
        if leftsq>rightsq:
            sqlist[ind]= leftsq
            left+=1
        else:
            sqlist[ind]=rightsq
            right-=1
        ind-=1
    return sqlist

print('4. sortListAfterSquaringContainingNegatives: ',sortListAfterSquaringContainingNegatives([-3, -1, 0, 1, 2]))


def findZeroSumTriplets(arr):
    res=[]
    arr.sort()
    for ind,val in enumerate(arr):
        if ind>0 and arr[ind]==arr[ind-1]:
            continue
        left,right = ind+1, len(arr)-1
        while left<right:
            s= val+arr[left]+arr[right]
            if s==0:
                res.append([val,arr[left],arr[right]])
                left+=1
                right-=1
                while left<right and arr[left]==arr[left-1]:
                    left+=1
                while right>0 and arr[right]==arr[right+1]:
                    right-=1
            elif s<0:
                left+=1
            else:
                right-=1
    return res

print('5. findZeroSumTriplets: ', findZeroSumTriplets([-5, 2, -1, -2, 3]))
print('5. findZeroSumTriplets: ', findZeroSumTriplets([-3, 0, 1, 2, -1, 1, -2]))

def findSumOfTripletThatisClosestToTarget(arr,target):
    arr.sort()
    res,diff=0,float('inf')
    for ind, val in enumerate(arr):
        if ind>0 and arr[ind]==arr[ind-1]:
            continue
        left, right = ind+1, len(arr)-1
        while left<right:
            s= val+arr[left]+arr[right]
            if abs(s-target)<diff:
                diff=abs(s-target)
                res=s
            if s<target:
                left+=1
            elif s>target:
                right-=1
            else:
                return s
    return res

print('6. findSumOfTripletThatisClosestToTarget: ',findSumOfTripletThatisClosestToTarget([-2, 0, 1, 2],2))
print('6. findSumOfTripletThatisClosestToTarget: ',findSumOfTripletThatisClosestToTarget([-3, -1, 1, 2],1))


def NumberOfTripletsWithSmallerSumThanTarget(arr,target):
    arr.sort()
    count=0
    for ind,val in enumerate(arr):
        if ind>0 and arr[ind]==arr[ind-1]:
            continue
        left, right = ind+1, len(arr)-1
        while left<right:
            s= val+arr[left]+arr[right]
            if s<target:
                count+= right-left
                left+=1
            else:
                right-=1
    return count

print('7. NumberOfTripletsWithSmallerSumThanTarget: ', NumberOfTripletsWithSmallerSumThanTarget([-1, 4, 2, 1, 3],5))


def ListOfTripletsWithSmallerSumThanTarget(arr,target):
    arr.sort()
    res=[]
    for ind,val in enumerate(arr):
        if ind>0 and arr[ind]==arr[ind-1]:
            continue
        left, right = ind+1, len(arr)-1
        while left<right:
            s = val + arr[left]+ arr[right]
            if s<target:
                count=right-left-1
                while count>=0:
                    res.append([val,arr[left],arr[right-count]])
                    count-=1
                left+=1
            else:
                right-=1
    return res

print('8. ListOfTripletsWithSmallerSumThanTarget: ',ListOfTripletsWithSmallerSumThanTarget([-1, 4, 2, 1, 3], 5))

from collections import deque

def productOfContiguousSubArrayLessThanTarget(arr,target):
    res= []
    prod, winStart=1,0
    for winEnd in range(len(arr)):
        prod*=arr[winEnd]
        while prod>=target and winStart<len(arr):
            left=arr[winStart]
            prod //= left
            winStart+=1
        que= deque()
        for i in range(winEnd, winStart-1,-1):
            que.appendleft(arr[i])
            res.append(list(que))
    return res

print('9. productOfContiguousSubArrayLessThanTarget: ', productOfContiguousSubArrayLessThanTarget([8, 2, 6, 5],50))

def dutchNationalFlagProblem(arr): #Sort all 0,1,2s
    zeros, ones, twos= 0,0, len(arr)-1
    while ones<=twos:
        if arr[ones]==0:
            arr[zeros],arr[ones]= arr[ones],arr[zeros]
            zeros+=1
            ones+=1
        elif arr[ones]==1:
            ones+=1
        else:
            arr[ones], arr[twos]= arr[twos], arr[ones]
            twos-=1
    return arr

print('10. dutchNationalFlagProblem: ', dutchNationalFlagProblem([2, 2, 0, 1, 2, 0]))


# Quadruple Sum to Target (medium)

def search_helper(arr, target, one, two, quads):
    three, four = two+1, len(arr)-1
    while three<four:
        s= arr[one]+arr[two]+arr[three]+arr[four]
        if s==target:
            quads.append([arr[one],arr[two],arr[three],arr[four]])
            three+=1
            four-=1
            while three<four and arr[three]==arr[three-1]:
                three+=1
            while four>three and arr[four]==arr[four+1]:
                four-=1
        elif s<target:
            three+=1
        else:
            four-=1

def search_qudruplets(arr,target):
    quads=[]
    arr.sort()
    for i in range(len(arr)-3):
        for j in range(i+1, len(arr)-2):
            search_helper(arr,target,i,j,quads)
    return quads

print('11. search_qudruplets: ', search_qudruplets([4, 1, 2, -1, 1, -3],1))
print('11. search_qudruplets: ', search_qudruplets([2, 0, -1, 1, -2, 2],2))


def compareStringContainingBackspaces(arr1,arr2):
    q1= q2 = deque()
    for i in range(len(arr1)):
        if arr1[i]=='#':
            q1.pop()
        else:
            q1.append(arr1[i])
    for i in range(len(arr2)):
        if arr2[i]=='#':
            q2.pop()
        else:
            q2.append(arr2[i])

    while q1 and q2:
        if q1.pop()!=q2.pop():
            return False
    if q1 or q2:
        return False
    return True

print('12. compareStringContainingBackspaces: ', compareStringContainingBackspaces("xp#", "xyz##"))
print('12. compareStringContainingBackspaces: ', compareStringContainingBackspaces("xy#z", "xyz#"))


def minSubArrayWindowToMaketheWholeArraySorted(arr):
    winStart, winEnd = 0, len(arr)-1
    while winStart<winEnd and arr[winStart]<arr[winStart+1]:
        winStart+=1
    if winStart==winEnd:
        return 0
    while winEnd>0 and arr[winEnd]>arr[winEnd-1]:
        winEnd-=1
    minEle, maxEle = float('inf'), float('-inf')
    for i in range(winStart,winEnd+1):
        minEle = min(minEle,arr[i])
        maxEle = max(maxEle,arr[i])

    while winStart>=0 and arr[winStart]>minEle:
        winStart-=1
    while winEnd<len(arr) and arr[winEnd]<maxEle:
        winEnd+=1

    return winEnd-winStart-1

print('13. minSubArrayWindowToMaketheWholeArraySorted: ', minSubArrayWindowToMaketheWholeArraySorted([1, 2, 5, 3, 7, 10, 9, 12]))
print('13. minSubArrayWindowToMaketheWholeArraySorted: ', minSubArrayWindowToMaketheWholeArraySorted([-1,3, 2, 1, 4,5,6]))

