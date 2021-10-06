def find_missing_number(nums):
    x1=1
    n = len(nums)+1
    for i in range(2,n+1):
        x1^=i
    x2=nums[0]
    for i in range(1,n-1):
        x2^=nums[i]

    return x1^x2


def find_single_number(arr):
    x1=0
    for n in arr:
        x1^=n
    return x1


# 1. Find the xor of all numbers which is actually the xor of n1 and n2
# 2. Then find the bit that is different in this XOR, getting differ_bit using -> (x & (1<<i))!=0 (we search for i)
# 3. Then we check each number with this differ_bit, if the & of that number gives 0, XOR it with n1 otherwise with n2,
def find_single_numbers(nums):
    n1xn2=0
    for num in nums:
        n1xn2^=num
    differ_bit=1
    while (differ_bit & n1xn2) ==0:
        differ_bit<<=1
    n1,n2=0,0
    for num in nums:
        if (differ_bit & num) ==0:
            n1^=num
        else:
            n2^=num
    return [n1,n2]


# 1. complement of 1010 is 0101. 1010 ^ 1111 gives 0101.
# first we count number of bits in 1010, which is 4
# pow(2,bit_count) = pow(2,4) gives 10000. 10000-1 gives 1111
# finally 1010 ^ 1111 will give 0101
# Time complexity id O(b) where b is number of bits in the number
def calculate_bitwise_complement(num):
    bit_count, n = 0, num
    while n>0:
        bit_count+=1
        n>>=1

    all_bits_one = pow(2,bit_count)-1
    return all_bits_one^num


def count_number_of_set_bits(num):
    count,n=0,num
    while n>0:
        count+= n&1
        n>>=1
    return count


def check_if_number_is_odd(num):
    return bool(num&1)


def flip_and_invert_image(matrix):
    l = len(matrix)
    for i in range(l):
        for j in range((l+1)//2):
            matrix[i][j], matrix[i][l-j-1] = matrix[i][l-j-1]^1 ,matrix[i][j]^1
    return matrix


def main():
  arr = [1, 5, 2, 6, 4]
  print('Missing number is:' + str(find_missing_number(arr)))

  arr = [1, 4, 2, 1, 3, 2, 3]
  print('Single number missing is:' + str(find_single_number(arr)))

  print('2 missing numbers are:' + str(find_single_numbers([1, 4, 2, 1, 3, 5, 6, 2, 3, 5])))
  print('2 missing  numbers are:' + str(find_single_numbers([2, 1, 3, 2])))

  print('Bitwise complement is: ' + str(calculate_bitwise_complement(8)))
  print('Bitwise complement is: ' + str(calculate_bitwise_complement(10)))

  print('count_number_of_set_bits: '+ str(count_number_of_set_bits(10)))
  print('count_number_of_set_bits: ' + str(count_number_of_set_bits(8)))

  print('check_if_number_is_odd: '+ str(check_if_number_is_odd(10)))
  print('check_if_number_is_odd: ' + str(check_if_number_is_odd(99)))

  print('flip_and_invert_image',str(flip_and_invert_image([[1,0,1], [1,1,1], [0,1,1]])))
  print('flip_and_invert_image',str(flip_and_invert_image([[1,1,0,0],[1,0,0,1],[0,1,1,1],[1,0,1,0]])))

main()