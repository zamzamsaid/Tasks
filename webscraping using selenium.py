#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install selenium


# In[2]:


import selenium


# In[5]:


pip install webdriver-manager


# In[3]:


pip install --upgrade selenium webdriver-manager


# In[3]:


from selenium import webdriver


# In[1]:


import time 


# In[10]:


from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# Set up and launch Chrome
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

# Open Google
driver.get("https://www.google.com")
print(driver.title)

# Find the search box and search for "GeeksforGeeks"
search_box = driver.find_element(By.NAME, "q")
search_box.send_keys("GeeksforGeeks", Keys.RETURN)

# Wait 20 seconds so you can see the result
time.sleep(20)

# Close the browser
driver.quit()


# In[ ]:





# Steps:
# Target Website: Choose an e-commerce website such as Books to Scrape, which is designed for practice. https://books.toscrape.com/
# Data to Extract:
# Book title
# Price
# Availability (In stock/Out of stock)
# Star rating (if available)
# Tasks:
# Write a script to extract the data for all books on the first page.
# Save the extracted data in a CSV file with columns: Title, Price, Availability, Star Rating.
# Advanced Challenge:
# Calculate the total number of books and the average price of all books.
# 
# 

# In[13]:


from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import json

# Setup Chrome driver (headless)
options = webdriver.ChromeOptions()
options.add_argument('--headless')
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Target URL
url = "https://books.toscrape.com/"
driver.get(url)

books = driver.find_elements(By.CSS_SELECTOR, "article.product_pod")

book_data = []

# //*[@id="default"]/div/div/div/div/section/div[2]/ol/li[1]/article/h3/a the orignal path for title but reduce to the .//h3/a

for book in books:
    # Title: use XPath (relative) Starting from the current book element, find an <a> tag inside an <h3> tag, at any depth within the book element.

    title = book.find_element(By.XPATH, ".//h3/a").get_attribute("title").strip()
    
    # Price: use CSS selector
    price_text = book.find_element(By.CSS_SELECTOR, "p.price_color").text.strip()
    price = float(price_text.replace('£', ''))
    
    # Availability: use Class Name
    availability = book.find_element(By.CLASS_NAME, "availability").text.strip()
    
    # Star Rating: use XPath (relative)
    star_element = book.find_element(By.TAG_NAME, "p")
    
    classes = star_element.get_attribute("class").split()
    star_rating = next((cls for cls in classes if cls != "star-rating"), "None")

    book_data.append({
        "Title": title,
        "Price": price,
        "Availability": availability,
        "Star Rating": star_rating
    })


# Save to JSON file
with open("books_data.json", "w", encoding="utf-8") as f:
    json.dump(book_data, f, indent=4, ensure_ascii=False)

# Summary statistics
total_books = len(book_data)
avg_price = sum(book["Price"] for book in book_data) / total_books if total_books else 0

print(f"Total books scraped: {total_books}")
print(f"Average book price: £{avg_price:.2f}")
print("Data saved to 'books_data.json'")

driver.quit()


# In[ ]:





# ACTIVITY:
# 
# Objective: Extract headlines and links to news articles from a news website.
# Steps:
# Target Website: Use a simple and public news website like BBC Technology News or similar.
# 
# 
# Data to Extract:
# Headline of the article
# URL of the article
# Brief summary (if available)
# 
# 
# Tasks:
# Write a script to extract the latest headlines and links from the main page of the chosen news section.
# Save the extracted data in JSON format with keys: Headline, URL, and Summary.
# 

# In[5]:


from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import json

# Set up headless Chrome browser
options = webdriver.ChromeOptions()
options.add_argument("--headless")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# URL of the specific article
url = "https://www.bbc.com/future/article/20250523-the-surprising-health-benefits-of-taking-creatine-powder"
driver.get(url)

# Extract the headline
headline = driver.find_element(By.TAG_NAME, "h1").text.strip()

# Extract the summary
try:
    summary = driver.find_element(By.XPATH, "//meta[@name='description']").get_attribute("content").strip()
except:
    try:
        summary = driver.find_element(By.CSS_SELECTOR, "article p").text.strip()
    except:
        summary = "Summary not available"

# Extract the image URL using XPath
try:
    image_element = driver.find_element(By.XPATH, '//*[@id="main-content"]/article/figure[1]/div/div/img')
    image_url = image_element.get_attribute("src")
except:
    image_url = "Image not available"

# Prepare data in JSON format
article_data = {
    "Headline": headline,
    "URL": url,
    "Summary": summary,
    "Image URL": image_url
}

# Save to JSON file
with open("specific_bbc_article.json", "w", encoding="utf-8") as f:
    json.dump(article_data, f, indent=4, ensure_ascii=False)

# Print output
print(json.dumps(article_data, indent=4, ensure_ascii=False))

# Close the browser
driver.quit()


# In[9]:


from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import time
import json

# Set up headless Chrome browser
options = webdriver.ChromeOptions()
options.add_argument("--headless")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Start from the BBC Future main page
start_url = "https://www.bbc.com/future"
driver.get(start_url)

# Keep scrolling until we get at least 15 unique article links
SCROLL_PAUSE_TIME = 2
MAX_SCROLLS = 10
MIN_ARTICLES = 15

article_links = set()

for _ in range(MAX_SCROLLS):
    # Scroll down
    driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
    time.sleep(SCROLL_PAUSE_TIME)
    
    # Find article links
    elements = driver.find_elements(By.CSS_SELECTOR, 'a[href*="/future/article/"]')
    for elem in elements:
        href = elem.get_attribute('href')
        if href:
            article_links.add(href)
    
    # Stop if we have enough
    if len(article_links) >= MIN_ARTICLES:
        break

# Convert set to list
article_links = list(article_links)[:MIN_ARTICLES]  # Get only first 15

# Visit each article and extract data
articles_data = []

for url in article_links:
    driver.get(url)
    time.sleep(1)  # Allow the page to load

    try:
        headline = driver.find_element(By.TAG_NAME, "h1").text.strip()
    except:
        headline = "Headline not available"

    try:
        summary = driver.find_element(By.XPATH, "//meta[@name='description']").get_attribute("content").strip()
    except:
        try:
            summary = driver.find_element(By.CSS_SELECTOR, "article p").text.strip()
        except:
            summary = "Summary not available"

    try:
        image_element = driver.find_element(By.CSS_SELECTOR, "#main-content article figure img")
        image_url = image_element.get_attribute("src")
    except:
        image_url = "Image not available"

    articles_data.append({
        "Headline": headline,
        "URL": url,
        "Summary": summary,
        "Image URL": image_url
    })

# Save to JSON
with open("bbc_future_articles.json", "w", encoding="utf-8") as f:
    json.dump(articles_data, f, indent=4, ensure_ascii=False)

# Output results
print(f"Extracted {len(articles_data)} articles.")
print(json.dumps(articles_data[:3], indent=4, ensure_ascii=False))

# Close browser
driver.quit()


# In[5]:


from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
import time
import json

# Setup driver
driver = webdriver.Chrome()

# Open BBC Future page
base_url = "https://www.bbc.com/future"
driver.get(base_url)
time.sleep(3)

# Scroll to load more articles
for _ in range(5):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

# Collect article links
article_links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/future/article/']")
links = []
for link in article_links:
    href = link.get_attribute("href")
    if href and href not in links:
        links.append(href)

print(f"Found {len(links)} articles.")

article_data = []

# Loop through links
for url in links:
    try:
        driver.get(url)
        time.sleep(2)

        headline = driver.find_element(By.TAG_NAME, "h1").text.strip()
        paragraphs = driver.find_elements(By.CSS_SELECTOR, "main p")
        full_text = " ".join([p.text for p in paragraphs if p.text.strip()])

        try:
            image_element = driver.find_element(By.CSS_SELECTOR, "main img")
            image_url = image_element.get_attribute("src")
        except:
            image_url = "No image found"

        article_data.append({
            "Headline": headline,
            "URL": url,
            "Summary": full_text,
            "Image": image_url
        })

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        continue

driver.quit()

# Save as CSV and JSON
df = pd.DataFrame(article_data)
df.to_csv("bbc_future_articles.csv", index=False)
with open("bbc_future_articles.json", "w", encoding="utf-8") as f:
    json.dump(article_data, f, ensure_ascii=False, indent=2)

print(f"Scraped {len(article_data)} articles successfully.")


# In[ ]:




