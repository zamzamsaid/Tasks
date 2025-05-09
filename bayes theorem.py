#!/usr/bin/env python
# coding: utf-8

# In[3]:


P_D = 0.01
P_T_given_D = 0.9
P_T_given_not_D = 0.05
P_not_D = 1 - P_D

# Total probability of testing positive
P_T = P_T_given_D * P_D + P_T_given_not_D * P_not_D

# Bayes' Theorem: P(Disease | Positive Test)
P_D_given_T = (P_T_given_D * P_D) / P_T

print(f"Disease | Positive Test = {P_D_given_T:.3f}")


# ## A Markov Chain is a type of stochastic process that describes a system which:
# 
# Has discrete states (like weather: sunny, cloudy, rainy).
# 
# Moves between those states probabilistically.
# 
# The probability of moving to a new state depends only on the current state, not the full history (this is the Markov property).

# In[10]:


import numpy as np
import matplotlib.pyplot as plt

# State names
states = ['Sunny', 'Cloudy', 'Rainy']
state_dict = {0: 'Sunny', 1: 'Cloudy', 2: 'Rainy'}

# Transition matrix
#Each row represents the current state, and each column gives the probability of moving to the next state.From Sunny:

#80% chance it stays Sunny


#15% chance it becomes Cloudy

#5% chance it becomes Rainy

P = np.array([
    [0.8, 0.15, 0.05],  # Sunny -> ...
    [0.2, 0.6, 0.2],    # Cloudy -> ...
    [0.1, 0.3, 0.6]     # Rainy -> ...
])

# Simulation parameters
n_steps = 50
current_state = 0  # Start from Sunny
state_sequence = [current_state]

# Simulate the Markov chain
for _ in range(n_steps - 1):
    current_state = np.random.choice([0, 1, 2], p=P[current_state])
    state_sequence.append(current_state)

# Convert to state names
state_names = [state_dict[s] for s in state_sequence]

# Plot the states over time
plt.figure(figsize=(12, 3))
plt.plot(state_sequence, drawstyle='steps-post', marker='o', color='teal')
plt.yticks([0, 1, 2], states)
plt.title("Markov Chain: Weather State Over Time")
plt.xlabel("Time Step")
plt.ylabel("Weather State")
plt.grid(True)
plt.tight_layout()
plt.show()


# ## Problem Summary:
# Arrival rate (λ) = 8 calls/hour → mean time between arrivals = 1/8 hour
# 
# Service rate (μ) = 10 calls/hour → mean service time = 1/10 hour
# 
# Simulate for 100 customers
# 
# Use Exponential distribution to model arrival and service times
# 
# Find:
# 
# Time each customer spends in the system (waiting + service)
# 
# When congestion is highest
# 
# Average time in system
# 
# Visualize with boxplot
# 
# We will simulate 100 customers.
# 
# On average, 8 people call per hour (arrive).
# 
# One agent serves them at a rate of 10 per hour

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Parameters
num_customers = 100
arrival_rate = 8    # lambda
service_rate = 10   # mu

# Simulate inter-arrival times and service times
inter_arrival_times = np.random.exponential(scale=1/arrival_rate, size=num_customers)
arrival_times = np.cumsum(inter_arrival_times)

service_times = np.random.exponential(scale=1/service_rate, size=num_customers)

# Initialize tracking variables
start_times = np.zeros(num_customers)
end_times = np.zeros(num_customers)
time_in_system = np.zeros(num_customers)

# Process first customer
start_times[0] = arrival_times[0]
end_times[0] = start_times[0] + service_times[0]
time_in_system[0] = end_times[0] - arrival_times[0]

# Process remaining customers
for i in range(1, num_customers):
    start_times[i] = max(arrival_times[i], end_times[i - 1])  # Wait if previous customer still being served
    end_times[i] = start_times[i] + service_times[i]
    time_in_system[i] = end_times[i] - arrival_times[i]

# Boxplot of time in system
plt.figure(figsize=(8, 4))
sns.boxplot(y=time_in_system, color='skyblue')
plt.title("Distribution of Time Spent in the System (100 Customers)")
plt.ylabel("Time in System (Hours)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Average time in system
avg_time = np.mean(time_in_system)
print(f"Average time spent in the system: {avg_time:.2f} hours")

# Congestion analysis (number of customers in system over time)
timeline = np.linspace(0, end_times[-1], 500)
customers_in_system = [np.sum((arrival_times <= t) & (end_times > t)) for t in timeline]

plt.figure(figsize=(10, 4))
plt.plot(timeline, customers_in_system, color='coral')
plt.title("Number of Customers in the System Over Time")
plt.xlabel("Time (Hours)")
plt.ylabel("Customers in System")
plt.grid(True)
plt.tight_layout()
plt.show()

# Peak congestion time
max_customers = np.max(customers_in_system)
peak_time = timeline[np.argmax(customers_in_system)]
print(f"Peak congestion: {max_customers} customers at time {peak_time:.2f} hours")

