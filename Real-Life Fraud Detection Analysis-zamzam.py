#!/usr/bin/env python
# coding: utf-8

# In[4]:


#import pandas
import pandas as pd


# ### Phase 1: Understanding the Dataset

# In[5]:


#Load the dataset into a pandas DataFrame
df = pd.read_csv('Synthetic_Financial_datasets_log.csv')
df


# In[5]:


#Display the first few rows to familiarize yourself with the structure and contents
df.head()
#by using head() it show us the first 5 rows


# In[6]:


#List the columns and explain the importance of each in the context of fraud detection.
df.columns


# ### according to our dataset each column what mean:
# step: mean time where 1 mean one hour
# type mean transaction types include CASH-IN, CASH-OUT, DEBIT, PAYMENT, and TRANSFER.
# 
# amount: mean transaction amount in the local currency.
# 
# nameOrig: mean customer initiating the transaction it like the ID .
# 
# oldbalanceOrg: mean initial balance before the transaction.
# 
# newbalanceOrig: mean new balance after the transaction.
# 
# nameDest: mean transaction's recipient customer.
# 
# oldbalanceDest: mean initial recipient's balance before the transaction.if Not applicable for customers identified by 'M'.
# 
# newbalanceDest: mean new recipient's balance after the transaction. Not applicable for 'M' (Merchants).
# 
# isFraud :Identifies transactions conducted by fraudulent agents aiming to deplete customer accounts through transfers and cash-outs.
# 
# isFlaggedFraud: indicates whether a transaction was flagged by a fraud detection system as potentially fraudulent, regardless of whether it actually was fraud.

# ### Phase 2: Data Cleaning and Exploration

# In[7]:


#Check for missing or inconsistent values.
df.isnull()


# In[9]:


#here i copy the data so when I drop any data and then i need it will easy to recover it
#always we should have another copy to be in safe side.
df1 =df.copy()
df1


# In[10]:


df1 = df1.drop(['nameOrig', 'nameDest'], axis=1)
#i drop these to because they are unique values the ID
#will not tell me if it's a fraudulent user.


# In[14]:


#Calculate the total number of transactions and categorize them by type.
totalTransactions = len(df)
transactionsType = df['type'].value_counts() #value_counts() it will gives us the count of each transaction type
print("Transactions : type")
print(transactionsType)


# In[20]:


#Identify the percentage of fraudulent transactions and compare them across transaction types.
#first find the fraudulent transactions
# Total number of fraudulent transactions
fraudulentTransactions = df[df['isFraud'] == 1].shape[0]
print(f"Fraudulent transactions: {fraudulentTransactions}")
print("\n")
#Identify the percentage of fraudulent transactions and compare 
#them across transaction types.

# we use Group by for transaction type and calculate the count of fraudulent transactions
fraudByType = df[df['isFraud'] == 1].groupby('type').size()

# find total number of transactions by type
transactionsByType = df['type'].value_counts()

# find the percentage of fraudulent transactions by type
fraudPercentageWithType = (fraudByType / transactionsByType) * 100

# print result
print("Fraudulent transactions percentage by type:")
print(fraudPercentageWithType)


# ### Conlusion of Fraudulent transactions and their type
# CASH_IN, DEBIT, PAYMENT have NaN values for fraudulent transactions, which indicates that no fraudulent transactions occurred in those categories.
# 
# TRANSFER: 0.77% is the highest fraud rates 
# 
# CASH_OUT: 0.18%  is low percentage
# 
# from this we see that highst fraudulent transactions use is the TRANSFER

# In[27]:


# Examine the distribution of transaction amounts (mean, median, 
#standard deviation) for both fraudulent and non-fraudulent 
#transactions.

#fraudulent transactions:
fraudulentTransactions = df[df['isFraud'] == 1]['amount'].describe()
print("Fraudulent transaction amount :\n",fraudulentTransactions[['mean', '50%', 'std']])

print("\n")
#non-fraudulent transactions:
nonFraudulentTransactions =  df[df['isFraud'] == 0]['amount'].describe()
print("Non-Fraudulent transaction amount :\n",nonFraudulentTransactions[['mean', '50%', 'std']])


# ### Phase 3: Real-Life Fraud Detection Analysis

# In[28]:


#Identify and flag transactions exceeding the legal limit (amount > 
#200,000) as potentially fraudulent (isFlaggedFraud).
#isFlaggedFraud should be 1 as in table either 0 or 1 1 it mean yes isFlaggedFraud
df[(df['isFlaggedFraud']==1) & (df['amount']>200000)]


# In[29]:


#Find patterns in fraud-related transactions, such as the time step, 
#type, or transaction amount
# Group fraudulent transactions by 'step' and 'type' and calculate the mean transaction amount
fraudPattern = df.groupby(['step', 'type'])['amount'].mean()

# Display results
print("Fraudulent transaction patterns (step, type, mean amount):")
print(fraudPattern)


# ### Concluion of fraud pattern
# TRANSFER has the highest average transaction amounts especially in later steps (step 741, step 742, step 743) the mean amounts reach millions.
# 
# CASH_IN, CASH_OUT, and DEBIT tend to have lower average amounts compared to TRANSFER in most steps
# 
# PAYMENT and DEBIT are lower across all steps

# In[37]:


#Group transactions by type and identify which types have the highest volume and value.
#Group by type and counts the number of transactions for each type.
transction = df.groupby('type').size()

# find the type with the highest volume
highestVolumeType = transction.idxmax()
highestVolumeValue = grouped_by_type.max()

# Print the result
print(f"Transaction type with the highest volume: {highestVolumeType} ({highestVolumeValue} transactions)")


# In[41]:


# Filter fraudulent transactions
fraudulent_df = df[df['isFraud'] == 1]

# Count repeated involved customers in fraudulent transactions both origin and destination
origFraudCount = fraudulent_df['nameOrig'].value_counts()
destFraudCount = fraudulent_df['nameDest'].value_counts()

# Combine the results to get all customers involved in fraud
fraudCustomerCount = origFraudCount.append(destFraudCount)
print(fraudCustomerCount)


# ## Critical Thinking Task
# Write a short note to the company describing one scenario where 
# a legitimate transaction might appear fraudulent. Suggest ways to 
# improve fraud detection without flagging such cases incorrectly.
# 
# Scenario:
# 
# A legitimate transaction might appear fraudulent when a customer suddenly makes a large international transfer after a long history of small, local transactions. For example, a customer who regularly sends small amounts (e.g., $100) to local recipients may one day make a significant transfer of $5,000 to an international account. Due to the sudden deviation in transaction size and destination, the fraud detection system may flag this as fraudulent, even though the transaction is legitimate.
# 
# ### Ways to improve fraud detection without flagging such cases incorrectly:
# 
# Fraud Detection Machine Learning Models:
# 
# Build or implement machine learning models that use not just transaction history, but also a wider array of contextual factors like :device fingerprinting, geolocation, etc. to identify fraud more accurately, reducing false positives.
# 
# Time-Based Behavioral Features: 
# 
# Construct features that capture the time dynamics of transactions (e.g., time since last transaction, average transaction size over time). This will help the model understand whether the transaction is unusual in the context of the customer's transaction timeline.

# In[1]:


import matplotlib.pyplot as plt


# In[6]:


#Line Plot 
plt.plot(df.index[:1000], df['amount'][:1000])  # Only first 1000 rows for clarity
plt.title("Transaction Amount Over Time")
plt.xlabel("Transaction Index")
plt.ylabel("Amount")
plt.show()


# In[7]:


#Bar Chart
#chart show us the data grouped by transaction type and calculate the total amount for each type
type_group = df.groupby('type')['amount'].sum()
plt.bar(type_group.index, type_group.values)
plt.title("Total Transaction Amount by Type")
plt.xlabel("Transaction Type")
plt.ylabel("Total Amount")
plt.show()


# In[10]:


#Pie Chart
#this will show us the number of transactions for each transaction type as percentage 
type_counts = df['type'].value_counts()
plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%')
plt.title("Transaction Type Distribution")
plt.show()


# In[16]:


#Scatter Plot 
# this scatter plot will show us the relationship between old balance and transaction amount
plt.scatter(df['oldbalanceOrg'], df['amount'],color='navy')
plt.title("Old Balance vs Transaction Amount")
plt.xlabel("Old Balance Origin")
plt.ylabel("Transaction Amount")
plt.show()


# ## Correlation Analysis
# Types of Correlation:
# 1. Positive Correlation: When one variable increases, the other also increases.
# 
# 2. Negative Correlation: When one variable increases, the other decreases.
# 
# 3. No Correlation: No relationship between the variables.
# 
# ------------------------
# Between 0 and 1: Positive correlation.
# 
# Between -1 and 0: Negative correlation.

# In[23]:


# 1:Select only numeric columns
dfNumeric = df.select_dtypes(include=["number"])
dfNumeric


# In[24]:


#2: Compute correlatio matrix

correlationmatrix = dfNumeric.corr()
print(correlationmatrix)


# In[27]:


#3:Visualize correlation matrix Using heatmap
import seaborn as sns
# Set figure size
plt.figure(figsize=(10, 8))

# Create the heatmap
sns.heatmap(correlationmatrix, annot=True, cmap="coolwarm", fmt=".2f")

# Add a title
plt.title("Correlation Matrix Heatmap - Transactions")
plt.show()


# ## Conclusion of Correlation:
# oldbalanceDest & newbalanceDest are (0.98) Very strong positive correlation when one increases, the other does too.
# 
# amount & newbalanceDest (0.46) Moderate positive correlation — higher transaction amounts often lead to higher balance at destination.
# 
# isFraud & other variables (in most near 0) Very little linear correlation 
