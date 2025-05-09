#!/usr/bin/env python
# coding: utf-8

# In[2]:


import random
import matplotlib.pyplot as plt

# Ask for number of flips
num_flips = int(input("How many times do you want to flip the coin? "))

# Initialize counts and cumulative lists
heads = tails = 0
#We use these lists to store the cumulative number of heads and tails after each flip, so we can visualize them on a line chart.
heads_count = []
tails_count = []

# Flip and track results
for i in range(num_flips):
    if random.choice(['H', 'T']) == 'H':
        heads += 1
    else:
        tails += 1
    heads_count.append(heads)
    tails_count.append(tails)

# Print results
print(f"\nAfter {num_flips} flips:")
print(f"Heads: {heads}")
print(f"Tails: {tails}")

# Plot results
plt.plot(range(1, num_flips + 1), heads_count, label="Heads", color='skyblue', marker='o')
plt.plot(range(1, num_flips + 1), tails_count, label="Tails", color='salmon', marker='x')
plt.title(f"Coin Flip Results ({num_flips} Flips)")
plt.xlabel("Flip Number")
plt.ylabel("Cumulative Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

