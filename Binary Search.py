#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#binary search
list1 = [5, 2, 13, 1, 20, 0]  
list1.sort()
value = 5
low = 0
high = len(list1) - 1  
for i in range(len(list1)):
    if low > high:
        break

    mid = (low + high) // 2
    
    if list1[mid] == value:
        print("Value found at index", mid)
        break
    elif list1[mid] < value:
        low = mid + 1
    else:
        high = mid - 1
else:
    print("Value not found")

