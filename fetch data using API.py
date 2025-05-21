#!/usr/bin/env python
# coding: utf-8

# ## Understand the API
# Before writing any code, you need to know:
# 
# Base URL (e.g., https://fakestoreapi.com)
# 
# Endpoints (e.g., /products, /products/1)
# 
# HTTP Method (usually GET for fetching data)
# 
# Authentication required? (e.g., API key, token, etc.)
# 
# Format of response (usually JSON)

# In[1]:


pip install requests


# In[2]:


import requests

# URL of the Fake Store API
url = 'https://fakestoreapi.com/products'

# Send GET request to the API
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()  # Convert JSON to Python dict/list
    # Print out the data
    for product in data:
        print(f"Title: {product['title']}")
        print(f"Price: ${product['price']}")
        print(f"Category: {product['category']}")
        print("-" * 40)
else:
    print("Failed to fetch data:", response.status_code)


# # Fetch Data

# In[6]:


import requests

url = "https://my-json-server.typicode.com/horizon-code-academy/fake-movies-api/movies"

def get_movies():
    try:
        response = requests.get(url)
        response.raise_for_status()
        movies = response.json()
        display_movies(movies)
    except requests.exceptions.RequestException as e:
        print("Error fetching movies:", e)

def display_movies(movies):
    print("🎬 Movie List\n" + "-" * 40)
    for movie in movies:
        print(f"Title   : {movie.get('Title', 'N/A')}")
        print(f"Year    : {movie.get('Year', 'N/A')}")
        print(f"Runtime : {movie.get('Runtime', 'N/A')}")
        print(f"Poster  : {movie.get('Poster', 'N/A')}")
        print("-" * 40)

get_movies()


# # Convert to JSON

# In[8]:


import requests
import json

url = "https://my-json-server.typicode.com/horizon-code-academy/fake-movies-api/movies"

def get_movies():
    try:
        response = requests.get(url)
        response.raise_for_status()
        movies = response.json()
        
        # 🎯 Print as JSON
        print("🎬 Movie List \n" + "-" * 40)
        print(json.dumps(movies, indent=4))  # Pretty print

    except requests.exceptions.RequestException as e:
        print("Error fetching movies:", e)

get_movies()


# # Convert to CSV 

# In[11]:


with open("movies.txt", "w", encoding='utf-8') as f:
    for movie in movies:
        f.write(f"Title: {movie.get('Title', 'N/A')}\n")
        f.write(f"Year: {movie.get('Year', 'N/A')}\n")
        f.write(f"Runtime: {movie.get('Runtime', 'N/A')}\n")
        f.write(f"Poster: {movie.get('Poster', 'N/A')}\n")
        f.write("-" * 40 + "\n")


# # Convert to TXT (Readable Text File)

# In[10]:


import csv
import requests

url = "https://my-json-server.typicode.com/horizon-code-academy/fake-movies-api/movies"
response = requests.get(url)
movies = response.json()

# Define the CSV file
with open("movies.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["Title", "Year", "Runtime", "Poster"])
    writer.writeheader()
    writer.writerows(movies)


# In[15]:


import requests
import json

THAWANI_API_KEY = 'rRQ26GcsZzoEhbrP2HZvLYDbn9C9et'  # Your API key

def create_checkout_session(course_name, price_omr, registration_id, customer_name):
    url = "https://uatcheckout.thawani.om/api/v1/checkout/session"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "thawani-api-key": THAWANI_API_KEY
    }
    
    payload = {
        "client_reference_id": str(registration_id),
        "mode": "payment",
        "products": [
            {
                "name": course_name,
                "quantity": 1,
                "unit_amount": int(price_omr * 1000)  # convert OMR to Baisa
            }
        ],
        "success_url": "http://127.0.0.1:7000/succeed",
        "cancel_url": "http://127.0.0.1:7000/cancel",
        "metadata": {
            "Customer name": customer_name,
            "order id": registration_id
        }
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print("Response from Thawani:", json.dumps(data, indent=4))
        session_id = data.get("data", {}).get("session_id")
        return session_id
    else:
        print("Failed to create session:", response.status_code, response.text)
        return None

def get_checkout_session(session_id):
    url = f"https://uatcheckout.thawani.om/api/v1/checkout/session/{session_id}"
    headers = {
        "Accept": "application/json",
        "thawani-api-key": THAWANI_API_KEY
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to get session:", response.status_code, response.text)
        return None

# Example usage:
if __name__ == "__main__":
    # Simulate registration data
    course_name = "Python Course"
    price_omr = 20.0
    registration_id = 12345
    customer_name = "John Doe"
    
    # Step 1: Create checkout session
    session_id = create_checkout_session(course_name, price_omr, registration_id, customer_name)
    
    if session_id:
        print("Checkout session created:", session_id)
        
        # Step 2: Get checkout session info (optional)
        session_info = get_checkout_session(session_id)
        
        if session_info and session_info.get("success"):
            # Build payment URL with your public key (replace with your real key)
            public_key = "HGvTMLDssJghr9tlN9gr4DVYt0qyBy"
            payment_url = f"https://uatcheckout.thawani.om/pay/{session_id}?key={public_key}"
            print("Redirect user to payment URL:", payment_url)
        else:
            print("Error retrieving session info")
    else:
        print("Failed to create checkout session")

