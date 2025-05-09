#!/usr/bin/env python
# coding: utf-8

# ## Normal Distribution

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import kurtosis, skew

# Set style
sns.set(style="whitegrid")
np.random.seed(0)

# Generate data
right_skewed = np.random.exponential(scale=2, size=1000)
left_skewed = -np.random.exponential(scale=2, size=1000)

# Function to compute stats
def get_stats(data):
    return {
        "mean": np.mean(data),
        "median": np.median(data),
        "skewness": skew(data),
        "kurtosis": kurtosis(data)  # excess kurtosis
    }

# Get statistics
stats_right = get_stats(right_skewed)
stats_left = get_stats(left_skewed)

# Plot setup
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Right-skewed
sns.histplot(right_skewed, kde=True, bins=30, ax=axes[0], color='skyblue')
axes[0].axvline(stats_right["mean"], color='red', linestyle='--', label=f'Mean = {stats_right["mean"]:.2f}')
axes[0].axvline(stats_right["median"], color='green', linestyle='-', label=f'Median = {stats_right["median"]:.2f}')
axes[0].text(4, 0.35,
             f'Skewness = {stats_right["skewness"]:.2f}\nKurtosis = {stats_right["kurtosis"]:.2f}',
             bbox=dict(facecolor='white', alpha=0.7))
axes[0].set_title('Right-Skewed Distribution')
axes[0].legend()

# Left-skewed
sns.histplot(left_skewed, kde=True, bins=30, ax=axes[1], color='lightcoral')
axes[1].axvline(stats_left["mean"], color='red', linestyle='--', label=f'Mean = {stats_left["mean"]:.2f}')
axes[1].axvline(stats_left["median"], color='green', linestyle='-', label=f'Median = {stats_left["median"]:.2f}')
axes[1].text(-8, 0.35,
             f'Skewness = {stats_left["skewness"]:.2f}\nKurtosis = {stats_left["kurtosis"]:.2f}',
             bbox=dict(facecolor='white', alpha=0.7))
axes[1].set_title('Left-Skewed Distribution')
axes[1].legend()

plt.tight_layout()
plt.show()

