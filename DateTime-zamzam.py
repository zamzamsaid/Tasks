#!/usr/bin/env python
# coding: utf-8

# In[46]:


#import pandas
import pandas as pd
from datetime import date, time, timedelta


# In[47]:


#Load the dataset 
df = pd.read_csv('daily-minimum-temperatures-in-me.csv')
df


# In[48]:


df.head()


# In[5]:


df.describe()


# In[6]:


df.info()


# In[49]:


df['Date'] = pd.to_datetime(df['Date'])
df


# In[50]:


print(df.dtypes)


# In[52]:


df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day


# In[54]:


df['day_of_week'] = df['Date'].dt.day_name()
df


# In[55]:


df['Daily minimum temperatures'] = pd.to_numeric(df['Daily minimum temperatures'], errors='coerce')


# In[56]:


df_1985 = df[df['year'] == 1985]
below_10_1985 = df_1985[df_1985['Daily minimum temperatures'] < 10]
print(f"Days in 1985 with temperature below 10°C: {len(below_10_1985)}")


# In[57]:


yearly_avg_temp = df.groupby('year')['Daily minimum temperatures'].mean()


# In[58]:


max_avg_year = yearly_avg_temp.idxmax()
min_avg_year = yearly_avg_temp.idxmin()

print(f"Year with highest avg min temperature: {max_avg_year} ({yearly_avg_temp[max_avg_year]:.2f}°C)")
print(f"Year with lowest avg min temperature: {min_avg_year} ({yearly_avg_temp[min_avg_year]:.2f}°C)")


# ## Summary
# 
# After successfully loading and processing the Daily Minimum Temperatures dataset, I converted the date column to datetime format and extracted useful time-based features such as year and month. To understand temperature patterns over time, I calculated the average minimum temperature for each year. According to the analysis:
# 
# The highest average minimum temperature was recorded in 1988, with an average of 11.97°C.
# 
# The lowest average minimum temperature occurred in 1984, averaging 10.62°C.
# 
# Additionally, I examined temperature patterns in 1985 and found that there were several days with temperatures falling below 10°C, which is consistent with typical winter conditions.
# 
