#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# **Load the Dataset**

# In[ ]:


df = pd.read_csv("daily-minimum-temperatures-in-me.csv")


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# **Convert Date Column**

# In[ ]:


df['Date'] = pd.to_datetime(df['Date'])


# In[ ]:


print(df.dtypes)


# **Extract Features**

# In[ ]:


df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Day_of_Week'] = df['Date'].dt.day_name()


# In[ ]:


df.head()


# **Filter Data**

# In[ ]:


df_1985 = df[df['Year'] == 1985]
df_1985


# In[ ]:


df_1985['Daily minimum temperatures'] = pd.to_numeric(df_1985['Daily minimum temperatures'], errors='coerce')


# In[ ]:


below_10_count = (df_1985['Daily minimum temperatures'] < 10).sum()
print(f"Number of days in 1985 with temperature below 10°C: {below_10_count}")


# **Analyze Time Periods**

# In[ ]:


df.rename(columns={'Daily minimum temperatures': 'Temp'}, inplace=True)

df['Temp'] = pd.to_numeric(df['Temp'], errors='coerce')


# In[ ]:



yearly_avg_temp = df.groupby('Year')['Temp'].mean()

yearly_avg_temp


# In[ ]:


lowest_avg_year = yearly_avg_temp.idxmin()

print(f"Year with lowest avg minimum temp: {lowest_avg_year} ({yearly_avg_temp[lowest_avg_year]:.2f}°C)")


# **Summary**

# Based on my analysis of the Daily Minimum Temperatures dataset, I was able to successfully load and analyze the data, converting the date column and extracting time-based features like year, month, and day of the week. In 1985, there were 142 days with temperatures below 10°C, showing typical winter patterns. According to my calculations, the lowest average minimum temperature was 10.62°C in 1984.
