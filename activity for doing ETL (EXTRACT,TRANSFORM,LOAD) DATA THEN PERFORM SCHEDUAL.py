#!/usr/bin/env python
# coding: utf-8

# In[28]:


pip install schedule


# In[ ]:


import pandas as pd
import pymysql
import schedule
import time

def run_etl():
    # Step 1: Extract
    file_paths = [
        "store_sales_1.csv",
        "store_sales_2.csv",
        "store_sales_3.csv"
    ]
    dataframes = [pd.read_csv(path) for path in file_paths]

    # Step 2: Transform
    def transform_data(df, store_id):
        df = df.dropna(subset=['Qty', 'Unit_Price', 'CustomerID']).copy()
        df['Qty'] = df['Qty'].astype(int)
        df['Unit_Price'] = df['Unit_Price'].astype(float)
        df['SaleDate'] = pd.to_datetime(df['SaleDate'])
        df['Total_Price'] = df['Qty'] * df['Unit_Price']
        df['Total_Price_OMR'] = df['Total_Price'] * 0.385

        if 'ProductName' in df.columns:
            df['ProductName'] = df['ProductName'].astype(str).str.strip().str.title()
        if 'CustomerName' in df.columns:
            df['CustomerName'] = df['CustomerName'].astype(str).str.strip().str.title()

        df['StoreID'] = store_id
        return df

    transformed_dfs = [transform_data(df, i) for i, df in enumerate(dataframes, start=1)]
    all_data = pd.concat(transformed_dfs, ignore_index=True)
    print(all_data.head())

    # Step 3: Load to MySQL
    try:
        connection = pymysql.connect(
            host='localhost',
            user='root',
            password='Zamzam@2001',
            database='store',
            cursorclass=pymysql.cursors.DictCursor
        )
        cursor = connection.cursor()

        # Insert distinct Products
        products = all_data[['ProductName']].drop_duplicates()
        for product in products['ProductName']:
            try:
                cursor.execute("INSERT INTO Product (ProductName) VALUES (%s)", (product,))
            except pymysql.err.IntegrityError:
                pass  # ignore duplicates

        # Insert distinct Customers
        customer_cols = ['CustomerID']
        if 'CustomerName' in all_data.columns:
            customer_cols.append('CustomerName')
        if 'ContactInfo' in all_data.columns:
            customer_cols.append('ContactInfo')

        customers = all_data[customer_cols].drop_duplicates()
        for _, row in customers.iterrows():
            try:
                cursor.execute(
                    "INSERT INTO Customer (CustomerID, CustomerName, ContactInfo) VALUES (%s, %s, %s)",
                    (
                        row['CustomerID'],
                        row.get('CustomerName', None),
                        row.get('ContactInfo', None)
                    )
                )
            except pymysql.err.IntegrityError:
                pass

        # Insert distinct Stores
        store_cols = ['StoreID']
        if 'StoreName' in all_data.columns:
            store_cols.append('StoreName')
        if 'Location' in all_data.columns:
            store_cols.append('Location')

        stores = all_data[store_cols].drop_duplicates()
        for _, row in stores.iterrows():
            try:
                cursor.execute(
                    "INSERT INTO Store (StoreID, StoreName, Location) VALUES (%s, %s, %s)",
                    (
                        row['StoreID'],
                        row.get('StoreName', None),
                        row.get('Location', None)
                    )
                )
            except pymysql.err.IntegrityError:
                pass

        # Map ProductName to ProductID for Sale insertion
        cursor.execute("SELECT ProductID, ProductName FROM Product")
        product_map = {row['ProductName']: row['ProductID'] for row in cursor.fetchall()}

        # Insert Sales
        insert_sales_query = """
            INSERT INTO Sale 
            (ProductID, CustomerID, StoreID, Qty, Unit_Price, SaleDate, CurrencyType, Total_Price, Total_Price_OMR)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        for _, row in all_data.iterrows():
            product_id = product_map.get(row['ProductName'])
            if product_id is None:
                continue  # skip if product missing in DB

            try:
                cursor.execute(insert_sales_query, (
                    product_id,
                    row['CustomerID'],
                    row['StoreID'],
                    int(row['Qty']),
                    float(row['Unit_Price']),
                    row['SaleDate'].strftime('%Y-%m-%d %H:%M:%S'),
                    row.get('CurrencyType', None),
                    float(row.get('Total_Price', 0)),
                    float(row.get('Total_Price_OMR', 0))
                ))
            except Exception as e:
                print(f"Sale insert error for row {row}: {e}")

        connection.commit()
        print("✅ Data loading successful!")

    except pymysql.MySQLError as e:
        print(f"❌ MySQL error: {e}")

    finally:
        if connection:
            cursor.close()
            connection.close()
            print("MySQL connection closed.")

    print("✅ ETL job completed.")


# Schedule the ETL to run every Monday at 02:00 AM
schedule.every().monday.at("02:00").do(run_etl)

print("Scheduler started. Waiting for scheduled time...")

while True:
    schedule.run_pending()
    time.sleep(60)  # wait 60 seconds between checks


# ## 🔁 Summary:
# 
# Extracts, cleans, and transforms sales data from CSVs.
# 
# Loads cleaned data into a structured MySQL database.
# 
# Schedules the job to repeat automatically every week at 2:00 AM Monday.
