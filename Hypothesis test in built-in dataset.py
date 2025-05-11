#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import pandas as pd

# Load dataset
tips = sns.load_dataset("tips")
tips


# In[3]:


from scipy.stats import ttest_ind

# Split the tip amounts by gender
male_tips = tips[tips['sex'] == 'Male']['tip']
female_tips = tips[tips['sex'] == 'Female']['tip']

# Perform independent t-test
#Calculates the mean and standard deviation of both groups.

#Computes the t-statistic using the formula for independent samples.

#Determines the p-value associated with that t-statistic.

t_stat, p_value = ttest_ind(male_tips, female_tips)

print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.3f}")

alpha = 0.05

if p_value <= alpha:
    print("Reject the null hypothesis: There is a significant difference in tip amounts between genders.")
else:
    print("Fail to reject the null hypothesis: No significant difference in tip amounts between genders.")


# ## Step 2: Formulate Hypotheses
# We will test:
# 
# Null Hypothesis (H₀): There is no difference in average tips between male and female customers.
# 
# Alternative Hypothesis (H₁): There is a difference in average tips between male and female customers.
# 
# This is a two-tailed independent t-test.

# ## Step 4: Interpret the Result
# If p_value <= 0.05, we reject the null hypothesis and conclude that there is a statistically significant difference.
# 
# If p_value > 0.05, we fail to reject the null, meaning no significant difference is observed.

# # Table of differance between Independent t-test  &  Paired t-test
# 
# | Test Type          | Function            | When to Use                                | Example                               |
# | ------------------ | ------------------- | ------------------------------------------ | ------------------------------------- |
# | Independent t-test | `stats.ttest_ind()` | Comparing **different groups**             | Male vs Female tips                   |
# | Paired t-test      | `stats.ttest_rel()` | Comparing **same subjects before & after** | Before vs After treatment in patients |
# 

# # T-statistic vs P-value 
# 
# 
# | Term        | Meaning                                                           |
# | ----------- | ----------------------------------------------------------------- |
# | T-statistic | How far apart the sample means are (in standard error units)      |
# | P-value     | The probability the observed difference happened by random chance |
# 
# example:
# 
# t-statistic = 2.5 → The group means are 2.5 standard errors apart.
# 
# p-value = 0.014 → There’s a 1.4% chance of observing this difference if there's no real difference.
# 
# Since 0.014 < 0.05, we’d reject the null and say the difference is statistically significant.
