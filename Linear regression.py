#!/usr/bin/env python
# coding: utf-8

# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load example dataset
df = sns.load_dataset('tips')

# Select only numeric columns manually
numeric_df = df.select_dtypes(include='number')

# Compute correlation matrix
corr_matrix = numeric_df.corr()

# Plot correlation heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()


# In[ ]:





# In[ ]:





# In[20]:


df.info()


# In[22]:


df


# In[14]:


from seaborn import PairGrid
import numpy as np

# Function to show correlation in the top triangle
def corrfunc(x, y, **kws):
    r = np.corrcoef(x, y)[0, 1] #it will returns a value between -1 and 1
    ax = plt.gca() # function is used to get the current axes in the current figure.
    ax.annotate(f"{r:.2f}", xy=(.5, .5), xycoords=ax.transAxes,
                ha='center', va='center', fontsize=12)

# Create the grid
g = PairGrid(df)
g.map_upper(corrfunc)                  # Correlation values
g.map_lower(sns.scatterplot)          # Scatter plots
g.map_diag(sns.histplot, kde=True)    # Histograms with KDE

plt.suptitle("Pair Plot with Correlation & Distributions", y=1.02)
plt.show()


# In[ ]:


#This line adds the correlation coefficient value r (formatted to two decimal places) to the center of the plot, aligned both horizontally and vertically, with a font size of 12. It is placed using axes-relative coordinates, so the annotation appears at the middle of the axes regardless of the data range.


# In[ ]:


#PairGrid(df): This creates a PairGrid object that takes a DataFrame df. A PairGrid is a grid of subplots where each subplot shows the relationship between two variables in df. 


# Upper Triangle: Displays correlation coefficients (how strongly variables are related).
# 
# Lower Triangle: Displays scatter plots (visual relationships between pairs of variables).
# 
# Diagonal: Displays histograms with KDE (the distribution of each individual variable).

# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from seaborn import PairGrid

# Load dataset
df = sns.load_dataset('tips')

# Select numeric columns and make a copy to avoid warnings
numeric_df = df.select_dtypes(include='number').copy()

# Convert 'size' to a continuous-like variable by adding small noise
np.random.seed(42)
numeric_df['size_continuous'] = numeric_df['size'] + np.random.uniform(-0.3, 0.3, size=len(numeric_df))

# Columns to analyze
columns_to_plot = ['total_bill', 'tip', 'size_continuous']

# -----------------------------
# Correlation Heatmap
# -----------------------------
# Compute correlation matrix
corr_matrix = numeric_df[columns_to_plot].corr()

# Plot heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Correlation Matrix (with Continuous-like Size)')
plt.tight_layout()
plt.show()

# Pair Plot with Correlation Annotations

# Function to compute and annotate Pearson correlation
def corrfunc(x, y, **kws):
    r = np.corrcoef(x, y)[0, 1]
    ax = plt.gca()
    ax.annotate(f"{r:.2f}", xy=(.5, .5), xycoords=ax.transAxes,
                ha='center', va='center', fontsize=12)

# Create PairGrid
g = PairGrid(numeric_df[columns_to_plot])
g.map_upper(corrfunc)
g.map_lower(sns.scatterplot)
g.map_diag(sns.histplot, kde=True)

plt.suptitle("Pair Plot with Continuous-like Size", y=1.02)
plt.tight_layout()
plt.show()


# In[ ]:


import pandas as pd


# In[25]:


#Load the dataset into a pandas DataFrame
data = pd.read_csv('Real estate.csv')
data


# ##  Step 1: Load and Prepare the Data

# In[26]:


import pandas as pd

# Load the dataset
data = pd.read_csv('Real estate.csv')

# Rename columns to make them easier to use
data.columns = [
    'No', 'transaction_date', 'house_age', 'distance_to_MRT',
    'num_convenience_stores', 'latitude', 'longitude', 'price_per_unit'
]

# Drop 'No' column it's just an index
data = data.drop(columns=['No'])


# ## Step 2: Define Features (X) and Target (y)

# In[27]:


# Features and target variable
X = data.drop(columns=['price_per_unit'])  # all columns except the target
y = data['price_per_unit']


# ## Step 3: Split Data & Train Model

# In[28]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Train-test split
#80% for training (X_train, y_train)
#20% for testing (X_test, y_test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)


# ## Step 4: Evaluate the Model

# In[33]:


# Print model performance
print("R² Score:", r2_score(y_test, y_pred)) # R² Score: Tells how much of the variance in price is explained by the model (closer to 1 is better).
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))#  RMSE: The average prediction error in the same units as the price (e.g., $ per square meter).
# Show coefficients
coeff_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print(coeff_df)


# ## 
# | Feature                  | Coefficient | Meaning                                               |
# | ------------------------ | ----------- | ----------------------------------------------------- |
# | `transaction_date`       | +5.44       | Prices **increased** over time (positive trend).      |
# | `house_age`              | -0.27       | Older houses slightly **reduce** price.               |
# | `distance_to_MRT`        | -0.0047     | Greater distance from MRT = **lower** price.          |
# | `num_convenience_stores` | +1.09       | More stores nearby = **higher** price.                |
# | `latitude`               | +229        | Being **further north** strongly **increases** price. |
# | `longitude`              | -29.5       | Being **further east** slightly **decreases** price.  |
# 

# A positive coefficient = as the feature increases, price increases.
# 
# A negative one = feature increase leads to lower price.
# 
# Your linear regression model explains most of the variation in house prices.
# 
# Location, transaction date, and amenities are important predictors.
# 
# There's room to improve:
# 
# Try polynomial regression, feature scaling, or adding interaction terms.
# 
# You can also explore residual plots or regularization (Ridge/Lasso) to refine it.

# # Visualization by Plot Predictions

# In[31]:


import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs. Predicted House Price")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.tight_layout()
plt.show()


# R² Score ≈ 0.68
# This means the model explains about 68% of the variance in house prices.
# 
# It’s a moderately strong model — not perfect, but it's capturing the main trends.
# 
#  What is RMSE?
# RMSE stands for Root Mean Squared Error. It's a measure of how far off your predictions are from the actual values.
# 
# In simple terms:
# 
# It tells you, on average, how much your model’s predictions deviate from the true values.
# 
# RMSE ≈ 7.31
# On average, the model’s predictions are off by about 7.3 price units.
# Since house prices in your dataset range from about 10 to 120, an average error of 7.31 is moderate. Not perfect, but not terrible either.
# 
# 
# Whether that’s good depends on the range of prices (from your scatter plot, prices range from ~10 to over 100), so some errors might be significant.
# 
