#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd


# In[4]:


df = pd.read_csv('sample_sales_data.csv')
df


# In[5]:


#Find all orders where the TotalAmount is greater than 500
df[df['TotalAmount'] > 500]


# In[6]:


#List all orders in the "Electronics" category
df[df['Category']=='Electronics']


# In[7]:


#Find all orders where the Category is "Clothing" and the Status is
#"Completed."
df[(df['Category'] == 'Clothing') & (df['Status'] == 'Completed')]


# In[8]:


#Find orders where the Quantity is greater than 5 or the Status is "Pending."
df[(df['Quantity'] > 5) | (df['Status'] == 'Pending')]


# In[10]:


#Find orders where the TotalAmount is between 200 and 1000.
df[df['TotalAmount'].between(200,1000)]


# In[12]:


#List all orders where the CustomerName starts with "A."
df[df['CustomerName'].str.startswith('A', na=False)]


# In[13]:


#Find all orders where the Category contains "Home."
df[df['Category'].str.contains('Home')]


# In[14]:


#Find orders where the CustomerName is missing (NaN)
df[df['CustomerName'].isnull()]


# In[15]:


#Find orders where the OrderDate is not missing.
df[df['OrderDate'].notnull()]


# In[16]:


#Filter the dataset to include only rows where the Status is "Completed."
df[df['Status'] == 'Completed']


# In[18]:


df1 = df[["OrderID", "CustomerName", "TotalAmount"]]
df1


# In[20]:


mask = df["Price"] > 100
df2 = df[mask]
df2


# In[24]:


#Filter rows where the TotalAmount exceeds 500 and select only the OrderID
#and Status columns.
filtered_results= df[df["TotalAmount"] > 500][["OrderID", "Status"]]
filtered_results


# In[22]:


#Select the first 10 rows and display their OrderID and TotalAmount .
df[["OrderID", "TotalAmount"]].iloc[:10]


# In[23]:


df.drop("Status", axis=1)


# In[25]:


filtered_results.to_csv("high_value_orders.csv")

