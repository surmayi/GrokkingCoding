class Node:
    def __init__(self, value, next=None):
        self.value = value
        self.next = next


    def print_list(self):
        temp = self
        while temp is not None:
          print(str(temp.value) + " ", end='')
          temp = temp.next
        print()


def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False


def find_cycle_length(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    count = 1
    cur = slow.next
    while cur != slow:
        cur = cur.next
        count += 1
    return count


def find_cycle_start(head):
    l = find_cycle_length(head)
    slow = fast = head
    while l > 0:
        fast = fast.next
        l -= 1
    while slow != fast:
        slow = slow.next
        fast = fast.next
    return slow


def findDigitsSq(num):
    s = 0
    while num > 0:
        s += (num % 10) ** 2
        num //= 10
    return s


def find_happy_number(num):
    slow = fast = num
    while True:
        slow = findDigitsSq(slow)
        fast = findDigitsSq(findDigitsSq(fast))
        if slow == fast:
            break
    if slow == 1:
        return True
    return False


def find_middle_of_linked_list(head):
    slow=fast = head
    while fast and fast.next:
        slow = slow.next
        fast=fast.next.next
    return slow


def reverseList(head):
    prev=None
    while head:
        nxt = head.next
        head.next = prev
        prev=head
        head=nxt
    return prev

# In constant space O(1)
def is_palindromic_linked_list(head):
    slow=fast=head
    while fast and fast.next:
        slow=slow.next
        fast = fast.next.next

    head2 = reverseList(slow)
    head2copy = head2
    while head and head2:
        if head.value!=head2.value:
            break
        head = head.next
        head2= head2.next
    reverseList(head2copy)
    if not head2 and not head:
        return True
    return False


# In constant space O(1)
def reorder(head):
    slow = fast = head
    while fast and fast.next:
        slow= slow.next
        fast= fast.next.next
    head2 = reverseList(slow)
    while head and head2:
        tmp = head.next
        head.next = head2
        head=tmp
        tmp = head2.next
        head2.next = head
        head2=tmp
    if head:
        head.next=None


def getNextInd(cur, nums, is_forward):
    direction = nums[cur]>=0
    if direction!=is_forward:
        return -1
    nextInd = (cur+ nums[cur])%len(nums)
    if nextInd==cur:
        return -1
    return nextInd


def circular_array_loop_exists(nums):
    for i in range(len(nums)):
        slow=fast=i
        is_forward = nums[i]>=0
        while True:
            slow = getNextInd(slow,nums,is_forward)
            fast = getNextInd(fast,nums,is_forward)
            if fast!=-1:
                fast = getNextInd(fast,nums,is_forward)
            if slow==-1 or fast==-1 or slow==fast:
                break
        if slow!=-1 and slow==fast:
            return True
    return False



def main():
    head = Node(1)
    head.next = Node(2)
    head.next.next = Node(3)
    head.next.next.next = Node(4)
    head.next.next.next.next = Node(5)
    head.next.next.next.next.next = Node(6)
    print("LinkedList has cycle: " + str(has_cycle(head)))

    head.next.next.next.next.next.next = head.next.next
    print("LinkedList has cycle: " + str(has_cycle(head)))

    head.next.next.next.next.next.next = head.next.next.next
    print("LinkedList has cycle: " + str(has_cycle(head)))

    head = Node(1)
    head.next = Node(2)
    head.next.next = Node(3)
    head.next.next.next = Node(4)
    head.next.next.next.next = Node(5)
    head.next.next.next.next.next = Node(6)
    head.next.next.next.next.next.next = head.next.next
    print("LinkedList cycle length: " + str(find_cycle_length(head)))

    head.next.next.next.next.next.next = head.next.next.next
    print("LinkedList cycle length: " + str(find_cycle_length(head)))

    head = Node(1)
    head.next = Node(2)
    head.next.next = Node(3)
    head.next.next.next = Node(4)
    head.next.next.next.next = Node(5)
    head.next.next.next.next.next = Node(6)

    head.next.next.next.next.next.next = head.next.next
    print("LinkedList cycle start: " + str(find_cycle_start(head).value))

    head.next.next.next.next.next.next = head.next.next.next
    print("LinkedList cycle start: " + str(find_cycle_start(head).value))

    head.next.next.next.next.next.next = head
    print("LinkedList cycle start: " + str(find_cycle_start(head).value))

    print(find_happy_number(23))
    print(find_happy_number(12))

    head = Node(1)
    head.next = Node(2)
    head.next.next = Node(3)
    head.next.next.next = Node(4)
    head.next.next.next.next = Node(5)

    print("Middle Node: " + str(find_middle_of_linked_list(head).value))

    head.next.next.next.next.next = Node(6)
    print("Middle Node: " + str(find_middle_of_linked_list(head).value))

    head.next.next.next.next.next.next = Node(7)
    print("Middle Node: " + str(find_middle_of_linked_list(head).value))

    head = Node(2)
    head.next = Node(4)
    head.next.next = Node(6)
    head.next.next.next = Node(4)
    head.next.next.next.next = Node(2)

    print("Is palindrome: " + str(is_palindromic_linked_list(head)))

    head.next.next.next.next.next = Node(2)
    print("Is palindrome: " + str(is_palindromic_linked_list(head)))

    head = Node(2)
    head.next = Node(4)
    head.next.next = Node(6)
    head.next.next.next = Node(8)
    head.next.next.next.next = Node(10)
    head.next.next.next.next.next = Node(12)
    reorder(head)
    head.print_list()

    print(circular_array_loop_exists([1, 2, -1, 2, 2]))
    print(circular_array_loop_exists([2, 2, -1, 2]))
    print(circular_array_loop_exists([2, 1, -1, -2]))

main()
