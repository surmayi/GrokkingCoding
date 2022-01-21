#https://leetcode.com/problems/number-of-good-pairs/
def numIdenticalPairs(nums):
    count = 0
    for i in range(len(nums) - 1):
        for j in range(i + 1, len(nums)):
            if nums[i] == nums[j]:
                count += 1
    return count


print('1. No. of Good pairs: ', numIdenticalPairs([1,2,3,1,1,3]))


# https://leetcode.com/problems/decompress-run-length-encoded-list/
def decompressRLElist(nums):
    result = []
    for i in range(0, len(nums), 2):
        freq, val = nums[i], nums[i + 1]
        result = result + [val] * freq
    return result


print('2. Decompress Run-Length Encoded List: ', decompressRLElist([1,2,3,1,1,3]))


# https://leetcode.com/problems/reverse-words-in-a-string-iii/
def reverseWords(s) -> str:
    s = s.strip()
    if not s:
        return ''
    s = s.split()
    result = []
    for word in s:
        temp = []
        for ch in word:
            temp.append(ch)
        while temp:
            result.append(temp.pop())
        result.append(' ')
    result = ''.join(result[:-1])
    return result


print('3. Reverse words in String: ', reverseWords("Let's take LeetCode contest"))


# https://leetcode.com/problems/find-the-highest-altitude/
def largestAltitude(gain):
    gain = [0] + gain
    for i in range(1, len(gain)):
        gain[i] += gain[i - 1]
    return max(gain)


print('4. Largest Altitude Cyclist: ', largestAltitude([-5,1,5,0,-7]))
print('4. Largest Altitude Cyclist: ', largestAltitude([-4,-3,-2,-1,4,3,2]))
