#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install httpx beautifulsoup4')
import bs4  # beautifulsoup
import httpx
# scrape page HTML
url = "https://web-scraping.dev/product/1"
response = httpx.get(url)
assert response.status_code == 200, f"Failed to fetch {url}, got {response.status_code}"

# parse HTML
soup = bs4.BeautifulSoup(response.text, "html.parser")
product = {}
product['name'] = soup.select_one("h3.product-title").text
product['price'] = soup.select_one("span.product-price").text
product['description'] = soup.select_one("p.product-description").text
product['features'] = {}
feature_tables = soup.select(".product-features table")
for row in feature_tables[0].select("tbody tr"):
    key, value = row.select("td")
    product['features'][key.text] = value.text

# show results
from pprint import pprint
print("scraped product:")
pprint(product)
{'description': 'Indulge your sweet tooth with our Box of Chocolate Candy. '
                'Each box contains an assortment of rich, flavorful chocolates '
                'with a smooth, creamy filling. Choose from a variety of '
                'flavors including zesty orange and sweet cherry. Whether '
                "you're looking for the perfect gift or just want to treat "
                'yourself, our Box of Chocolate Candy is sure to satisfy.',
 'features': {'brand': 'ChocoDelight',
              'care instructions': 'Store in a cool, dry place',
              'flavors': 'Available in Orange and Cherry flavors',
              'material': 'Premium quality chocolate',
              'purpose': 'Ideal for gifting or self-indulgence',
              'sizes': 'Available in small, medium, and large boxes'},
 'name': 'Box of Chocolate Candy',
 'price': '$9.99 '}


# In[2]:


import pandas as pd


# In[3]:


colors = 'https://en.wikipedia.org/wiki/List_of_colors:_A%E2%80%93F'
cols_def = pd.read_html(colors)


# In[4]:


#how many table
len(cols_def)


# In[5]:


cols_def[0]


# In[ ]:


# step 1: requests → To fetch the page.

#beautifulsoup4 → To extract data from the HTML.


# In[19]:


#step 2: Install them using pip
get_ipython().system('pip install requests beautifulsoup4')


# In[17]:


#Step 3: Basic Scraping – Get Text from a Webpage
import requests
from bs4 import BeautifulSoup

# Step 1: Fetch the page
url = "https://web-scraping.dev/product/1"
response = requests.get(url)
html = response.text

# Step 2: Parse the HTML
soup = BeautifulSoup(html, "html.parser")

# Step 3: Extract data
title = soup.select_one("h3.product-title").text
price = soup.select_one("span.product-price").text
description = soup.select_one("p.product-description").text

print("Title:", title)
print("Price:", price)
print("Description:", description)


# In[22]:


#Step 4: Get an Image URL and Download It
# Extract image tag using correct class name
img_tag = soup.select_one("img.product-img")

if img_tag is not None:
    img_url = img_tag['src']
    img_data = requests.get(img_url).content
    with open("product.jpg", "wb") as f:
        f.write(img_data)
    print("Image downloaded successfully!")
else:
    print("Image not found.")


# In[23]:


from IPython.display import Image
Image("product.jpg")


# In[24]:


import requests
from bs4 import BeautifulSoup

# Send a GET request to the URL
url = 'https://iproyal.com/pricing/'
response = requests.get(url)

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Find all the cards containing the pricing information
cards = soup.find_all('a', class_='card')

# Iterate over each card and extract the necessary information
for card in cards:
    # Extract the proxy type (e.g., "Residential", "ISP", "Datacenter")
    proxy_type = card.find('h3', class_='tp-headline-s sm:tp-headline-m')
    if proxy_type:
        proxy_type = proxy_type.text.strip()
    
    # Extract the price information (e.g., "from $1.75/GB")
    price = card.find('div', class_='tp-body-xs md:v3-tp-body-s')
    if price:
        price = price.text.strip()

    # Extract the description (e.g., "Get accurate data and stay anonymous from anywhere...")
    description = card.find('p')
    if description:
        description = description.text.strip()

    # Print the extracted data
    print(f"Proxy Type: {proxy_type}")
    print(f"Price: {price}")
    print(f"Description: {description}")
    print("-" * 50)


# In[ ]:


#requests: This library is used to send HTTP requests and get the response. It's used here to request the webpage content.

#BeautifulSoup: A library for parsing HTML and XML documents. It's used here to parse the content of the webpage and extract the relevant data.
#response.text: This retrieves the HTML content of the page returned in the response.

