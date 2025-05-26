#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install mysql-connector-python


# ## dealing with null value can be by imputer (mean,median,mode),fill,predictive model

# ## dealing with null value by imputer

# In[2]:


import mysql.connector
import pandas as pd
from sklearn.impute import SimpleImputer

# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Zamzam@2001",
    database="CodeAcademyDB"
)

# Step 1: Read data into a DataFrame
query = "SELECT * FROM products_new"
df = pd.read_sql(query, conn)

# Step 2: Check for missing values
print(df.isnull().sum())

# Step 3: Impute missing values in specific columns
imputer = SimpleImputer(strategy='mean')  # You can also use 'median', 'most_frequent', etc.
df[['standard_cost', 'list_price']] = imputer.fit_transform(df[['standard_cost', 'list_price']])

# Step 4: (Optional) Update the values back to MySQL
cursor = conn.cursor()

for index, row in df.iterrows():
    update_query = """
        UPDATE products_new
        SET standard_cost = %s, list_price = %s
        WHERE product_id = %s
    """
    cursor.execute(update_query, (row['standard_cost'], row['list_price'], row['product_id']))

conn.commit()
cursor.close()


print("Missing values imputed and updated successfully.")


# In[ ]:





# # types of update in mysql
# 
# | Method                               | Use Case                               | Notes                            |
# | ------------------------------------ | -------------------------------------- | -------------------------------- |
# | `UPDATE`                             | Simple row update                      | Manual and specific              |
# | `REPLACE INTO`                       | Replace entire row                     | Deletes and re-inserts           |
# | `INSERT ... ON DUPLICATE KEY UPDATE` | Insert if not exists, update if exists | Best for upsert                  |
# | `UPDATE ... JOIN`                    | Bulk update using another table        | Efficient for large datasets     |
# | `to_sql` with SQLAlchemy             | Full DataFrame load                    | Replaces or appends entire table |
# | Stored Procedure                     | Encapsulated server-side logic         | Good for reusable updates        |
# 

# ##  dealing with null values by prdictive model

# In[8]:


import mysql.connector
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Zamzam@2001",
    database="CodeAcademyDB"
)

# Step 1: Load data
df = pd.read_sql("SELECT * FROM products_new", conn)

# Step 2: Print null counts
print("Before prediction:\n", df.isnull().sum())

# Step 3: Predict missing `description` only if there are missing values
if df['description'].isnull().sum() > 0:
    # Encode text in 'product_name' (required for ML model)
    encoder = LabelEncoder()
    df['product_name'] = encoder.fit_transform(df['product_name'].astype(str))

    # Split data into rows with and without description
    df_known = df[df['description'].notnull()]
    df_unknown = df[df['description'].isnull()]

    # Features to use for prediction
    features = ['product_name', 'standard_cost', 'list_price', 'category_id']

    X_train = df_known[features]
    y_train = df_known['description']

    X_pred = df_unknown[features]

    # Handle any missing values in features (e.g., list_price, standard_cost)
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_pred = imputer.transform(X_pred)

    # Train a classification model
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # Predict missing descriptions
    predicted_descriptions = clf.predict(X_pred)

    # Fill predictions into original DataFrame
    df.loc[df['description'].isnull(), 'description'] = predicted_descriptions

    # Step 4: Update MySQL with the predicted values
    cursor = conn.cursor()
    for index, row in df[df['description'].notnull()].iterrows():
        cursor.execute(
            "UPDATE products_new SET description = %s WHERE product_id = %s",
            (row['description'], row['product_id'])
        )
    conn.commit()
    cursor.close()

    print("\n✅ Missing 'description' values predicted and updated to MySQL.")

else:
    print("✅ No missing 'description' values to predict.")
print("After prediction:\n", df.isnull().sum())
# Close MySQL connection


# ## hiw can identify which method detecting outlier
# Plot a histogram and Q-Q plot for your numeric columns (like standard_cost and list_price).
# 
# This helps you see if the data looks normal (bell-shaped) or skewed (lopsided).
# 
# If the data looks roughly bell-shaped and Q-Q plot points align closely to the line → Z-score is good.
# If data looks skewed, or Q-Q plot points deviate a lot → IQR is better.

# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

# Assume df is already loaded from MySQL as before

# 1. Visualize outliers with boxplots
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.boxplot(x=df['standard_cost'])
plt.title('Boxplot of standard_cost')

plt.subplot(1,2,2)
sns.boxplot(x=df['list_price'])
plt.title('Boxplot of list_price')

plt.show()

# 2. Distribution plots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df['standard_cost'], kde=True)
plt.title('Distribution of standard_cost')

plt.subplot(1, 2, 2)
sns.histplot(df['list_price'], kde=True)
plt.title('Distribution of list_price')

plt.tight_layout()
plt.show()

# 2. Detect outliers using Z-score
z_scores = np.abs(stats.zscore(df[['standard_cost', 'list_price']].dropna()))
print("\nZ-scores of first 5 rows:\n", z_scores[:5])

# Threshold for Z-score (common choice is 3)
z_threshold = 3
outliers_z = (z_scores > z_threshold).any(axis=1)
print(f"\nNumber of outliers detected by Z-score: {sum(outliers_z)}")

# 3. Detect outliers using IQR method for each column
def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data < lower_bound) | (data > upper_bound)

outliers_iqr_standard_cost = detect_outliers_iqr(df['standard_cost'])
outliers_iqr_list_price = detect_outliers_iqr(df['list_price'])
outliers_iqr = outliers_iqr_standard_cost | outliers_iqr_list_price
print(f"Number of outliers detected by IQR method: {outliers_iqr.sum()}")

# 4. Remove outliers from DataFrame (choose method you prefer)
# Here we remove rows where either column is an outlier by IQR
df_clean = df.loc[~outliers_iqr].copy()

print(f"Original data size: {df.shape}")
print(f"Data size after removing outliers: {df_clean.shape}")

# Now you can continue with imputation or prediction on df_clean instead of df
# After your prediction and filling missing 'description' in df_clean
cursor = conn.cursor()

for index, row in df_clean[df_clean['description'].notnull()].iterrows():
    cursor.execute(
        "UPDATE products_new SET description = %s WHERE product_id = %s",
        (row['description'], row['product_id'])
    )

conn.commit()
cursor.close()


# # Removing duplicates: Eliminating duplicate records to ensure each entry is unique and relevant.
# # Removing duplicate rows
# # data = data.drop_duplicates()
# 

# # encoding types -> One Hot Encoding,Label Encoding,Ordinal Encoding
# 

# In[6]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# --------------------------
from sklearn.preprocessing import OneHotEncoder

print("1. One-Hot Encoding for 'color'")

# Initialize OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse=False, dtype=int, handle_unknown='ignore')

# Reshape and fit-transform
color_encoded = one_hot_encoder.fit_transform(df[['color']])

# Get column names from encoder
color_feature_names = one_hot_encoder.get_feature_names_out(['color'])

# Create DataFrame with encoded features
df_one_hot = pd.DataFrame(color_encoded, columns=color_feature_names, index=df.index)

# Concatenate with original DataFrame
df = pd.concat([df, df_one_hot], axis=1)

print("Last 20 rows of One-Hot Encoded 'color':")
print(df_one_hot.tail(20))

# --------------------------
print("\n2. Label Encoding for 'brand'")

# Label Encoding for 'brand'
df['brand'] = df['brand'].fillna('Unknown')  # handle None values
label_encoder = LabelEncoder()
df['brand_encoded'] = label_encoder.fit_transform(df['brand'])

# Mapping for inspection
brand_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Encoding - Brand Mapping:", brand_mapping)

# Show the last 20 rows
print(df[['brand', 'brand_encoded']].tail(20))


# --------------------------
print("\n3. Ordinal Encoding for 'size_category'")
# Ordinal Encoding for 'size_category'
# Clean size_category strings
df['size_category'] = df['size_category'].astype(str).str.strip().str.title()

# Replace any unknown or unexpected values with 'Unknown'
df.loc[~df['size_category'].isin(['Small', 'Medium', 'Large', 'XLarge']), 'size_category'] = 'Unknown'

# Ordinal encode with fixed categories
size_order = ['Unknown', 'Small', 'Medium', 'Large', 'XLarge']
ordinal_encoder = OrdinalEncoder(categories=[size_order])
df['size_encoded'] = ordinal_encoder.fit_transform(df[['size_category']])

print(df[['size_category', 'size_encoded']])

# --------------------------
print("\n4. Final DataFrame preview with encodings")
print(df[['color', 'brand', 'size_category', 'brand_encoded', 'size_encoded'] + list(df_one_hot.columns)].head())


# In[7]:


print(df.columns)


# In[9]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import mysql.connector

# 1. Connect to MySQL and load your data (example)
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Zamzam@2001",
    database="CodeAcademyDB"
)
query = "SELECT * FROM products_new"
df = pd.read_sql(query, conn)

# --- Outlier detection and removal ---

# Detect outliers via IQR method on 'standard_cost' and 'list_price'
def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data < lower_bound) | (data > upper_bound)

outliers_std_cost = detect_outliers_iqr(df['standard_cost'])
outliers_list_price = detect_outliers_iqr(df['list_price'])
outliers = outliers_std_cost | outliers_list_price

print(f"Original data size: {df.shape}")
df_clean = df.loc[~outliers].copy()
print(f"Data size after removing outliers: {df_clean.shape}")

# --- Encoding ---

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# --------------------------
from sklearn.preprocessing import OneHotEncoder

print("1. One-Hot Encoding for 'color'")

# Initialize OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse=False, dtype=int, handle_unknown='ignore')

# Reshape and fit-transform
color_encoded = one_hot_encoder.fit_transform(df[['color']])

# Get column names from encoder
color_feature_names = one_hot_encoder.get_feature_names_out(['color'])

# Create DataFrame with encoded features
df_one_hot = pd.DataFrame(color_encoded, columns=color_feature_names, index=df.index)

# Concatenate with original DataFrame
df = df.drop(columns=[col for col in df.columns if col.startswith('color_')], errors='ignore')
df = pd.concat([df, df_one_hot], axis=1)


print("Last 20 rows of One-Hot Encoded 'color':")
print(df_one_hot.tail(20))

# --------------------------
print("\n2. Label Encoding for 'brand'")

# Label Encoding for 'brand'
df['brand'] = df['brand'].fillna('Unknown')  # handle None values
label_encoder = LabelEncoder()
df['brand_encoded'] = label_encoder.fit_transform(df['brand'])

# Mapping for inspection
brand_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Encoding - Brand Mapping:", brand_mapping)

# Show the last 20 rows
print(df[['brand', 'brand_encoded']].tail(20))


# --------------------------
print("\n3. Ordinal Encoding for 'size_category'")
# Ordinal Encoding for 'size_category'
# Clean size_category strings
df['size_category'] = df['size_category'].astype(str).str.strip().str.title()

# Replace any unknown or unexpected values with 'Unknown'
df.loc[~df['size_category'].isin(['Small', 'Medium', 'Large', 'XLarge']), 'size_category'] = 'Unknown'

# Ordinal encode with fixed categories
size_order = ['Unknown', 'Small', 'Medium', 'Large', 'XLarge']
ordinal_encoder = OrdinalEncoder(categories=[size_order])
df['size_encoded'] = ordinal_encoder.fit_transform(df[['size_category']])

print(df[['size_category', 'size_encoded']])

# --------------------------
print("\n4. Final DataFrame preview with encodings")
print(df[['color', 'brand', 'size_category', 'brand_encoded', 'size_encoded'] + list(df_one_hot.columns)].head())

# --- Automatically add missing columns to MySQL before update ---

cursor = conn.cursor()

def add_column_if_not_exists(cursor, table_name, column_name, data_type):
    cursor.execute(f"""
        SELECT COUNT(*) 
        FROM information_schema.COLUMNS 
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s AND COLUMN_NAME = %s
    """, (table_name, column_name))
    if cursor.fetchone()[0] == 0:
        alter_sql = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {data_type}"
        print(f"Adding column {column_name} to {table_name} with type {data_type}")
        cursor.execute(alter_sql)
    else:
        print(f"Column {column_name} already exists in {table_name}")

table_name = 'products_new'

# Prepare columns and types (encoded + one-hot)
cols_and_types = {
    'brand_encoded': 'INT',
    'size_encoded': 'FLOAT',
}

for col in df_one_hot.columns:
    cols_and_types[col] = 'TINYINT(1)'

# Add missing columns if needed
for col, dtype in cols_and_types.items():
    add_column_if_not_exists(cursor, table_name, col, dtype)

conn.commit()

# --- Update database with encoded columns and one-hot columns ---

cols_to_update = list(cols_and_types.keys())
key_col = 'product_id'

for idx, row in df_clean.iterrows():
    set_clause = ", ".join([f"{col} = %s" for col in cols_to_update])
    sql = f"UPDATE {table_name} SET {set_clause} WHERE {key_col} = %s"
    values = [row[col] for col in cols_to_update] + [row[key_col]]
    cursor.execute(sql, values)

conn.commit()
cursor.close()
conn.close()

print("Database update completed.")


# ## perform Data scaling and normalization
# ### Min-max scaling and Standardization (Z-Score normalization)

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import mysql.connector

# --- 1. Connect to MySQL and load data ---
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Zamzam@2001",
    database="CodeAcademyDB"
)
query = "SELECT * FROM products_new"
df = pd.read_sql(query, conn)

# --- 2. Outlier Detection and Removal ---
def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data < lower_bound) | (data > upper_bound)

outliers_std_cost = detect_outliers_iqr(df['standard_cost'])
outliers_list_price = detect_outliers_iqr(df['list_price'])
outliers = outliers_std_cost | outliers_list_price

print(f"Original data size: {df.shape}")
df_clean = df.loc[~outliers].copy()
print(f"Data size after removing outliers: {df_clean.shape}")

# --- 3. One-Hot Encoding for 'color' ---
print("1. One-Hot Encoding for 'color'")
one_hot_encoder = OneHotEncoder(sparse=False, dtype=int, handle_unknown='ignore')
color_encoded = one_hot_encoder.fit_transform(df[['color']])
color_feature_names = one_hot_encoder.get_feature_names_out(['color'])
df_one_hot = pd.DataFrame(color_encoded, columns=color_feature_names, index=df.index)
df = df.drop(columns=[col for col in df.columns if col.startswith('color_')], errors='ignore')
df = pd.concat([df, df_one_hot], axis=1)
print(df_one_hot.tail(20))

# --- 4. Label Encoding for 'brand' ---
print("\n2. Label Encoding for 'brand'")
df['brand'] = df['brand'].fillna('Unknown')
label_encoder = LabelEncoder()
df['brand_encoded'] = label_encoder.fit_transform(df['brand'])
brand_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Encoding - Brand Mapping:", brand_mapping)
print(df[['brand', 'brand_encoded']].tail(20))

# --- 5. Ordinal Encoding for 'size_category' ---
print("\n3. Ordinal Encoding for 'size_category'")
df['size_category'] = df['size_category'].astype(str).str.strip().str.title()
df.loc[~df['size_category'].isin(['Small', 'Medium', 'Large', 'Xlarge']), 'size_category'] = 'Unknown'
size_order = ['Unknown', 'Small', 'Medium', 'Large', 'Xlarge']
ordinal_encoder = OrdinalEncoder(categories=[size_order])
df['size_encoded'] = ordinal_encoder.fit_transform(df[['size_category']])
print(df[['size_category', 'size_encoded']])

# --- 6. Scaling and Normalization ---
print("\n4. Scaling and Normalization")
numeric_cols = ['standard_cost', 'list_price']
minmax_scaled_cols = [f"{col}_minmax" for col in numeric_cols]
standard_scaled_cols = [f"{col}_zscore" for col in numeric_cols]

# MinMax Scaling
minmax_scaler = MinMaxScaler()
df[minmax_scaled_cols] = minmax_scaler.fit_transform(df[numeric_cols])

# Standard Scaling
standard_scaler = StandardScaler()
df[standard_scaled_cols] = standard_scaler.fit_transform(df[numeric_cols])

print(df[minmax_scaled_cols + standard_scaled_cols].head())

# Add scaled columns to df_clean
df_clean[minmax_scaled_cols + standard_scaled_cols] = df[minmax_scaled_cols + standard_scaled_cols]

# --- 7. Final Preview ---
print("\n5. Final DataFrame Preview")
print(df[['color', 'brand', 'size_category', 'brand_encoded', 'size_encoded'] + list(df_one_hot.columns)].head())

# --- 8. Automatically Add Columns to MySQL if Missing ---
cursor = conn.cursor()

def add_column_if_not_exists(cursor, table_name, column_name, data_type):
    cursor.execute("""
        SELECT COUNT(*) 
        FROM information_schema.COLUMNS 
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s AND COLUMN_NAME = %s
    """, (table_name, column_name))
    if cursor.fetchone()[0] == 0:
        alter_sql = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {data_type}"
        print(f"Adding column {column_name} to {table_name} with type {data_type}")
        cursor.execute(alter_sql)
    else:
        print(f"Column {column_name} already exists in {table_name}")

table_name = 'products_new'

# Columns to add: encodings + one-hot + scaled
cols_and_types = {
    'brand_encoded': 'INT',
    'size_encoded': 'FLOAT',
}

for col in df_one_hot.columns:
    cols_and_types[col] = 'TINYINT(1)'

for col in minmax_scaled_cols + standard_scaled_cols:
    cols_and_types[col] = 'FLOAT'

# Create columns if missing
for col, dtype in cols_and_types.items():
    add_column_if_not_exists(cursor, table_name, col, dtype)

conn.commit()

# --- 9. Update MySQL with New Values ---
print("\n6. Updating database...")
cols_to_update = list(cols_and_types.keys())
key_col = 'product_id'

for idx, row in df_clean.iterrows():
    set_clause = ", ".join([f"{col} = %s" for col in cols_to_update])
    sql = f"UPDATE {table_name} SET {set_clause} WHERE {key_col} = %s"
    values = [row[col] for col in cols_to_update] + [row[key_col]]
    cursor.execute(sql, values)

conn.commit()
cursor.close()
conn.close()

print("✅ Database update completed.")


# ## sklearn.pipeline

# In[2]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Specify column names
numeric_features = ['standard_cost', 'list_price']
categorical_features = ['color', 'brand', 'size_category']

# Numeric pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine both
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])


# In[3]:


# Make sure all columns exist and are the right type
df[categorical_features] = df[categorical_features].astype(str)

# Fit and transform the full data
preprocessed_data = preprocessor.fit_transform(df)

# Convert the transformed array back to DataFrame
# Get column names from transformers
num_cols = numeric_features
cat_cols = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features)

# Combine all column names
all_cols = list(num_cols) + list(cat_cols)

# Final preprocessed DataFrame
df_preprocessed = pd.DataFrame(preprocessed_data.toarray() if hasattr(preprocessed_data, 'toarray') else preprocessed_data,
                               columns=all_cols,
                               index=df.index)

print(df_preprocessed.head())


# In[7]:


import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Zamzam@2001",
    database="CodeAcademyDB"
)
cursor = conn.cursor()

# Add columns to MySQL if needed
for col in df_preprocessed.columns:
    add_column_if_not_exists(cursor, table_name, col, 'FLOAT')
for idx, row in df_preprocessed.iterrows():
    set_clause = ', '.join([f"{col} = %s" for col in df_preprocessed.columns])
    
for idx, row in df_preprocessed.iterrows():
    set_clause = ', '.join([f"{col} = %s" for col in df_preprocessed.columns])
    
    # Convert NumPy data types to native Python types
    values = [v.item() if hasattr(v, 'item') else v for v in row.values]
    
    product_id = df.loc[idx, 'product_id']
    if hasattr(product_id, 'item'):
        product_id = product_id.item()
    values.append(product_id)

    sql = f"UPDATE {table_name} SET {set_clause} WHERE product_id = %s"
    cursor.execute(sql, values)
conn.commit()


# Summery of pipeline:
# 
# 🔹 Pipeline: Lets us build a sequence of steps (like clean, scale, encode).
# 
# 🔹 StandardScaler: Standardizes numerical data (mean = 0, std = 1).
# 
# 🔹 OneHotEncoder: Converts categories into binary columns.
# 
# 🔹 SimpleImputer: Fills missing values.
# 
# 🔹 ColumnTransformer: Applies different transformations to different columns.
# 
# | Step | Action                                  | Purpose                                  |
# | ---- | --------------------------------------- | ---------------------------------------- |
# | 1    | Define numeric & categorical columns    | So we know what to scale vs encode       |
# | 2    | Create pipeline for numeric             | Fill missing + scale                     |
# | 3    | Create pipeline for categorical         | Fill missing + OneHotEncode              |
# | 4    | Combine with `ColumnTransformer`        | Apply the right transform to each column |
# | 5    | Transform the DataFrame                 | All in one line                          |
# | 6    | Convert to DataFrame + Add column names | For MySQL insertion                      |
# | 7    | Add columns and update MySQL            | Save it permanently                      |
# 
# ⚙️ What is StandardScaler()?
# StandardScaler is a preprocessing tool from Scikit-learn that standardizes (normalizes) numerical features.
# 
# 🔍 What it does:
# It transforms your numeric columns so they have:
# 
# Mean = 0
# 
# Standard deviation = 1
# 
# Why use StandardScaler?
# Because many machine learning models (especially linear models, SVMs, neural networks) work better when features are on the same scale. Otherwise, features with large values dominate the model.
# 

# In[ ]:




