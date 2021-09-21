from __future__ import print_function


class Node:
  def __init__(self, value, next=None):
    self.value = value
    self.next = next

  def print_list(self):
    temp = self
    while temp is not None:
      print(temp.value, end=" ")
      temp = temp.next
    print()


def reverse_sub_list(head, p,q):
  i=1
  cur=head
  prev=None
  while cur and i<p:
    prev=cur
    cur=cur.next
    i+=1
  first = prev
  last = cur
  i,prev=0,None
  while cur and i<q-p+1:
    nxt=cur.next
    cur.next = prev
    prev=cur
    cur=nxt
    i+=1

  if first:
    first.next=prev
  else:
    head=prev
  last.next=cur
  return head


def rever_first_k_elements(head,k):
  last = cur= head
  prev=None
  i=0
  while cur and i<k:
    nxt=cur.next
    cur.next=prev
    prev=cur
    cur=nxt
    i+=1

  last.next=cur

  return prev

def lengthOfLinkedList(head):
  if not head:
    return 0
  return 1+ lengthOfLinkedList(head.next)

def reverse_in_two_parts(head):
  n= lengthOfLinkedList(head)
  head1 = reverse_sub_list(head, 1, n // 2)
  if n%2==0:
    reverse_sub_list(head1,n//2+1,n)
  else:
    reverse_sub_list(head1,n//2+2,n)
  return head1


def reverse_every_k_elements(head,k):
  cur,prev= head, None
  while True:
    prevListLast, curListLast= prev,cur

    i=0
    while cur and i<k:
      nxt = cur.next
      cur.next= prev
      prev=cur
      cur=nxt
      i+=1

    if prevListLast:
      prevListLast.next = prev
    else:
      head=prev

    curListLast.next = cur
    if not cur:
      break

    prev=curListLast

  return head


def reverse_alternate_k_elements(head, k):
  cur, prev = head, None
  reversal =True
  while True:
    prevListLast , curListLast = prev,cur
    i=0
    if reversal:
      while cur and i<k:
        nxt = cur.next
        cur.next = prev
        prev=cur
        cur=nxt
        i+=1
      if prevListLast:
        prevListLast.next = prev
      else:
        head =prev
      reversal = False
    else:
      while cur and i <k:
        curListLast = cur
        cur=cur.next
        i+=1
      reversal=True
    curListLast.next=cur
    if not cur:
      break
    prev= curListLast

  return head


def rotateKTimesFromLast(head,k):
  n = lengthOfLinkedList(head)
  p= k%n
  if p ==0:
    return head
  l= n-p-1
  cur =head
  while l>0:
    cur=cur.next
    l-=1
  first= cur.next
  last = cur

  while cur.next:
    cur=cur.next
  cur.next= head
  head= first
  last.next=None

  return head




def main():
  head = Node(1)
  head.next = Node(2)
  head.next.next = Node(3)
  head.next.next.next = Node(4)
  head.next.next.next.next = Node(5)

  print("Nodes of original LinkedList are: ", end='')
  head.print_list()
  result = reverse_sub_list(head, 2, 4)
  print("Nodes of reversed LinkedList are: ", end='')
  result.print_list()

  head = Node(1)
  head.next = Node(2)
  head.next.next = Node(3)
  head.next.next.next = Node(4)
  head.next.next.next.next = Node(5)

  result = rever_first_k_elements(head,3)
  print("Nodes of reversed LinkedList are: ", end='')
  result.print_list()

  head = Node(1)
  head.next = Node(2)
  head.next.next = Node(3)
  head.next.next.next = Node(4)
  head.next.next.next.next = Node(5)
  head.next.next.next.next.next = Node(6)

  result = reverse_in_two_parts(head)
  print("Nodes of reversed LinkedList with even count are: ", end='')
  result.print_list()

  head = Node(1)
  head.next = Node(2)
  head.next.next = Node(3)
  head.next.next.next = Node(4)
  head.next.next.next.next = Node(5)
  head.next.next.next.next.next = Node(6)
  head.next.next.next.next.next.next = Node(7)

  result = reverse_in_two_parts(head)
  print("Nodes of reversed LinkedList with odd count are: ", end='')
  result.print_list()

  head = Node(1)
  head.next = Node(2)
  head.next.next = Node(3)
  head.next.next.next = Node(4)
  head.next.next.next.next = Node(5)
  head.next.next.next.next.next = Node(6)
  head.next.next.next.next.next.next = Node(7)
  head.next.next.next.next.next.next.next = Node(8)
  head.next.next.next.next.next.next.next.next = Node(9)

  result = reverse_every_k_elements(head,3)
  print("Nodes of LinkedList with every k elements reversed are : ", end='')
  result.print_list()

  head = Node(1)
  head.next = Node(2)
  head.next.next = Node(3)
  head.next.next.next = Node(4)
  head.next.next.next.next = Node(5)
  head.next.next.next.next.next = Node(6)
  head.next.next.next.next.next.next = Node(7)
  head.next.next.next.next.next.next.next = Node(8)

  result = reverse_every_k_elements(head,3)
  print("Nodes of LinkedList with every k elements reversed are : ", end='')
  result.print_list()

  head = Node(1)
  head.next = Node(2)
  head.next.next = Node(3)
  head.next.next.next = Node(4)
  head.next.next.next.next = Node(5)
  head.next.next.next.next.next = Node(6)
  head.next.next.next.next.next.next = Node(7)
  head.next.next.next.next.next.next.next = Node(8)

  print("Nodes of original LinkedList are: ", end='')
  head.print_list()
  result = reverse_alternate_k_elements(head, 2)
  print("Nodes of alternate k nodes reversed LinkedList are: ", end='')
  result.print_list()

  head = Node(1)
  head.next = Node(2)
  head.next.next = Node(3)
  head.next.next.next = Node(4)
  head.next.next.next.next = Node(5)
  head.next.next.next.next.next = Node(6)
  head.next.next.next.next.next.next = Node(7)
  head.next.next.next.next.next.next.next = Node(8)

  print("Nodes of original LinkedList are: ", end='')
  head.print_list()
  result = reverse_alternate_k_elements(head, 3)
  print("Nodes of alternate k nodes reversed LinkedList are: ", end='')
  result.print_list()

  head = Node(1)
  head.next = Node(2)
  head.next.next = Node(3)
  head.next.next.next = Node(4)
  head.next.next.next.next = Node(5)
  head.next.next.next.next.next = Node(6)

  print("Nodes of original LinkedList are: ", end='')
  head.print_list()
  result = rotateKTimesFromLast(head, 3)
  print("Nodes of rotated LinkedList are: ", end='')
  result.print_list()

  head = Node(1)
  head.next = Node(2)
  head.next.next = Node(3)
  head.next.next.next = Node(4)
  head.next.next.next.next = Node(5)

  print("Nodes of original LinkedList are: ", end='')
  head.print_list()
  result = rotateKTimesFromLast(head, 8)
  print("Nodes of rotated LinkedList are: ", end='')
  result.print_list()

main()