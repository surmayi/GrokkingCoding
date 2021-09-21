# All Sliding Window Problems

def findAvg(arr,k):
    winStart =0
    res=[]
    winSum=0
    for winEnd in range(len(arr)):
        winSum+= arr[winEnd]
        if winEnd>=k-1:
            res.append(winSum/k)
            winSum-=arr[winStart]
            winStart+=1
    return res

print('1. findAvg ',findAvg([1, 3, 2, 6, -1, 4, 1, 8, 2], 5))


def max_sum_window(arr,k):
    winStart =0
    winSum =0
    winMax= float('-inf')
    for winEnd in range(len(arr)):
        winSum+=arr[winEnd]
        if winEnd>=k-1:
            winMax= max(winMax,winSum)
            winSum-=arr[winStart]
            winStart+=1
    return winMax

print('2. max_sum_window: ',max_sum_window([2, 1, 5, 1, 3, 2],3))


def smallestSubArrayGreaterThanEqualTarget(arr,target):
    winStart, winSum, winSize = 0,0,0
    for winEnd in range(len(arr)):
        while winSum>=target:
            winSize = min(winSize,winEnd-winStart)
            winSum-=arr[winStart]
            winStart+=1
        winSize = winEnd-winStart
        winSum+=arr[winEnd]
    return winSize

print('3. smallestSubArrayGreaterThanEqualTarget: ',smallestSubArrayGreaterThanEqualTarget( [2, 1, 5, 2, 3, 2],7))


def longestSubstringWithUtmostKdistinctChars(arr,k):
    winStart, winSize, vals = 0,0,{}
    for winEnd in range(len(arr)):
        right = arr[winEnd]
        vals[right]= vals.get(right,0)+1
        while len(vals)>k:
            left = arr[winStart]
            vals[left]-=1
            if vals[left]==0:
                del vals[left]
            winStart+=1
        winSize = max(winSize, winEnd-winStart+1)
    return winSize

print('4. longestSubstringWithUtmostKdistinctChars: ',longestSubstringWithUtmostKdistinctChars("araaci",2))

def maxFruitsIn2Baskets(arr):
    winStart, maxLen, vals =0,0, {}
    for winEnd in range(len(arr)):
        right = arr[winEnd]
        vals[right]= vals.get(right,0)+1
        while len(vals)>2:
            left = arr[winStart]
            vals[left]-=1
            if vals[left]==0:
                del vals[left]
            winStart+=1
        maxLen= max(maxLen, winEnd-winStart+1)
    return maxLen

print('5. maxFruitsIn2Baskets: ',maxFruitsIn2Baskets(['A', 'B', 'C', 'B', 'B', 'C']))


def longestSubstringWithNoRepeatingChars(arr):
    winStart, winLen, vals =0,0,{}
    for winEnd in range(len(arr)):
        right = arr[winEnd]
        if right in vals:
            winStart =max(winStart,vals[right]+1)
        vals[right]=winEnd
        winLen = max(winLen, winEnd-winStart+1)
    return winLen

print('6. longestSubstringWithNoRepeatingChars: ', longestSubstringWithNoRepeatingChars('aabccacbdb'))


def longestSubStringWithSameCharAfterReplacement(arr,k):
    winStart, maxLen, vals = 0,0, {}
    for winEnd in range(len(arr)):
        right = arr[winEnd]
        vals[right]= vals.get(right,0)+1
        while len(vals)>k+1:
            left = arr[winStart]
            vals[left]-=1
            if vals[left]==0:
                del vals[left]
            winStart+=1
        maxLen = max(maxLen, winEnd-winStart+1)
    return maxLen

print('7. longestSubStringWithSameCharAfterReplacement: ', longestSubStringWithSameCharAfterReplacement('abccde',2))


def longestContiguousArrayWithAll1s(arr,k):
    winStart, maxLen, vals=0,0,{}
    for winEnd in range(len(arr)):
        right = arr[winEnd]
        vals[right]= vals.get(right,0)+1
        while vals[0]>k:
            left = arr[winStart]
            vals[left]-=1
            winStart+=1
        maxLen= max(maxLen,winEnd-winStart+1)
    return maxLen

print('8. longestContiguousArrayWithAll1s: ', longestContiguousArrayWithAll1s([0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1], 3))

def findAnagramIndexes(arr,pattern):
    winStart,matched, result, valsP = 0,0, [],{}
    for ch in pattern:
        valsP[ch]= valsP.get(ch,0)+1
    for winEnd in range(len(arr)):
        right = arr[winEnd]
        if right in valsP:
            valsP[right]-=1
            if valsP[right]==0:
                matched+=1
        if len(valsP)==matched:
            result.append(winStart)
        if winEnd>= len(pattern)-1:
            left = arr[winStart]
            if left in valsP:
                if valsP[left]==0:
                    matched-=1
                valsP[left]+=1
            winStart+=1
    return result


print('9. findAnagramIndexes: ', findAnagramIndexes('abbcabc','abc'))


def CheckStringContainsPermutationOfPattern(arr,pattern):
    winStart, matched, valsP = 0,0,{}
    for ch in pattern:
        valsP[ch]= valsP.get(ch,0)+1
    for winEnd in range(len(arr)):
        right = arr[winEnd]
        if right in valsP:
            valsP[right]-=1
            if valsP[right]==0:
                matched+=1
        if len(valsP)==matched:
            return True
        if winEnd>= len(pattern)-1:
            left = arr[winStart]
            if left in valsP:
                if valsP[left]==0:
                    matched-=1
                valsP[left]+=1
            winStart+=1
    return False

print('10. CheckStringContainsPermutationOfPattern: ', CheckStringContainsPermutationOfPattern('oidbcaf','abco'))
print('10. CheckStringContainsPermutationOfPattern: ', CheckStringContainsPermutationOfPattern('oidbcaf','abc'))


def smallestWindowContainingPattern(arr,pattern):
    winStart, matched, winSize, res, valsP = 0,0,float('inf'), [-1,-1],{}
    for ch in pattern:
        valsP[ch]= valsP.get(ch,0)+1

    for winEnd in range(len(arr)):
        right = arr[winEnd]
        if right in valsP:
            valsP[right]-=1
            if valsP[right]==0:
                matched+=1
        if len(valsP)== matched:
            if winSize> winEnd-winStart+1:
                winSize= winEnd-winStart+1
                res= [winStart,winEnd+1]
        if winEnd>= len(pattern)-1:
            left =arr[winStart]
            if left in valsP:
                if valsP[left]==0:
                    matched-=1
                valsP[left]+=1
            winStart+=1
    return arr[res[0]:res[1]]


print('11. smallestWindowContainingPattern: ', smallestWindowContainingPattern('abdbca','abc'))

def subStringIndexWithConcatationOfWordList(arr, words):
    res=[]
    dict_words={}
    for word in words:
        dict_words[word] = dict_words.get(word,0)+1
    word_count=len(dict_words)
    word_length= len(words[0])
    for i in range(len(arr)- word_count*word_length+1):
        word_seen ={}
        for j in range(word_count):
            next_word_ind = j*word_length+i
            next_word = arr[next_word_ind:next_word_ind+word_length]

            if next_word not in dict_words:
                break

            word_seen[next_word]= word_seen.get(next_word,0)+1
            if word_seen[next_word]>1:
                break

            if j+1== word_count:
                res.append(i)
    return res


print('12. subStringIndexWithConcatationOfWordList: ', subStringIndexWithConcatationOfWordList('catfoxcat', ['cat','fox']))
