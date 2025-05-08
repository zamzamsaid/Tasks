#!/usr/bin/env python
# coding: utf-8

# ## inferential statistics

# In[21]:



from scipy import stats
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

# Your code
zvalue = stats.norm.ppf(0.975)
print("Z value: ", zvalue)

sd = 1
sample_mean = 3.2  # Sample mean of ice creams per week
n = 30
ME = zvalue * (sd/sqrt(30))
# Calculate the confidence interval
lower = sample_mean - ME
print("Lower Bound= ", lower)
upper = sample_mean + ME
print("Upper Bound= ", upper)

# Plotting the normal distribution
# x will represent "ice cream per week" values (from 2.00 to 4.00)
x = np.linspace(2.0, 4.0, 1000)  # Ice cream per week range from 2.00 to 4.00
y = stats.norm.pdf(x, sample_mean, sd / sqrt(n))  # Probability density

plt.figure(figsize=(10, 5))

# Plot the normal distribution
plt.plot(x, y, label='Normal Distribution', color='blue')

# Shading the confidence interval area
plt.fill_between(x, y, where=(x >= lower) & (x <= upper), color='skyblue', alpha=0.5, label='95% Confidence Interval')

# Adding vertical lines for the mean and bounds
plt.axvline(sample_mean, color='black', linestyle='--', label='Sample Mean (Ice Cream per Week)')
plt.axvline(lower, color='red', linestyle='--', label='Lower Bound')
plt.axvline(upper, color='green', linestyle='--', label='Upper Bound')

# Adjusting y-axis from 0 to 2 (as per your requirement)
plt.ylim(0, 2.5)

# Labels and title
plt.title('95% Confidence Interval for Ice Cream per Week')
plt.xlabel('Ice Cream per Week')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()


# In[23]:


SM = 78
SP = 80
SD = 2.5
n = 40
Z = (SM-SP)/(SD/sqrt(n))
Z
alpha = 0.05

# Step 3: Determine the critical z-value for two-tailed test at 95% confidence
z_critical = stats.norm.ppf(1 - alpha/2)
print(f"Critical z-value: ±{z_critical:.2f}")

# Step 4: Decision
if abs(Z) > z_critical:
    print("Reject the null hypothesis: The machine is not working properly.")
else:
    print("Fail to reject the null hypothesis: No strong evidence the machine is malfunctioning.")


# In[25]:


mp = 2           # Population mean
ms = 1.8      # Sample mean
sd = 0.15         # Sample standard deviation
n = 10           # Sample size
df = n - 1       # Degrees of freedom
alpha = 0.01     # Significance level for 99% confidence

# Step 1: Hypotheses
# H0: mu = 2
# H1: mu < 2 (left-tailed test)

t_score = (ms - mp) / (sd /sqrt(n))
print(f"T-score: {t_score:.2f}")

# Step 3: Get critical t-value (left-tailed)
t_critical = stats.t.ppf(alpha, df)
print(f"Critical t-value: {t_critical:.2f}")

# Step 4: Decision
if t_score < t_critical:
    print("Reject the null hypothesis: The average lifespan is significantly less than 2 years.")
else:
    print("Fail to reject the null hypothesis: No strong evidence that lifespan is less than 2 years.")

