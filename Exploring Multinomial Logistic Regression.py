#!/usr/bin/env python
# coding: utf-8

# In[19]:


# Required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[18]:


# Step 1: Load the dataset with the correct separator
df = pd.read_csv('student-por.csv', sep=';')
df


# In[7]:


# Optional: Strip whitespace from column names
df.columns = df.columns.str.strip()


# In[9]:


# Step 2: Recode the target variable G3 into Low, Medium, High
def recode_grade(g3):
    if g3 <= 9:
        return 'Low'
    elif g3 <= 14:
        return 'Medium'
    else:
        return 'High'

df['G3_cat'] = df['G3'].apply(recode_grade)

# Show the number of observations in each category
print("Class distribution:\n", df['G3_cat'].value_counts())


# In[10]:


# Step 3: Plot a numeric predictor vs G3 category
sns.boxplot(x='G3_cat', y='absences', data=df)
plt.title('Absences by Performance Category')
plt.xlabel('Performance Category')
plt.ylabel('Absences')
plt.show()


# In[11]:


# Step 4: Select predictors and prepare features/target
features = ['studytime', 'absences', 'failures', 'goout']
X = df[features]
y = df['G3_cat']


# In[12]:


# Encode the target variable (Low, Medium, High)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Class labels:", list(le.classes_))  # ['High', 'Low', 'Medium']


# In[13]:


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


# In[14]:


# Step 5: Fit the multinomial logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)


# In[15]:


# Step 6: Interpret one coefficient
coef_df = pd.DataFrame(model.coef_, columns=features)
coef_df['Class'] = [le.classes_[i] for i in range(len(coef_df))]
print("\nModel Coefficients:")
print(coef_df.set_index('Class'))


# In[16]:


# Step 7: Evaluate model with confusion matrix and accuracy
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print("\nConfusion Matrix:\n", conf_matrix)
print("Accuracy: {:.2f}%".format(acc * 100))


# In[17]:


# Step 8: Short summary
summary = f"""
Summary:
- Most common performance category: {df['G3_cat'].value_counts().idxmax()}
- Absences vary significantly across performance levels (as seen in the plot).
- Model accuracy: {acc:.2f}
- For example, a higher 'studytime' tends to decrease the odds of being in the 'Low' category.
"""
print(summary)

