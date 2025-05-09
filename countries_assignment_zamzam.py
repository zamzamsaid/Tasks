#!/usr/bin/env python
# coding: utf-8

# In[33]:


#a. Read the File:
file = open('countries_population.txt', 'r')
lines = file.readlines()
file.close()

#b. Process the Data:
countries_population = []
for line in lines:
    line = line.strip() 

    if line != '' and ',' in line: 
        country, population = line.split(',')
        country = country.strip()
        population = int(population.strip())
        countries_population.append((country, population))

for country, population in countries_population:
    print(f"{country}: {population}")
    
print("\n")

#c. Display Information:
#1 Total number of countries
print("Total number of counties: ",len(countries_population))
print("\n")
#2 The country with the highest population
#(I use lambda because when i do normal max not work ,I search and i see we can replace with lambda )
#lambda allow compare based in seconed value which is index is 1
highest = max(countries_population, key=lambda p: p[1])
print("Country with the highest population:", highest)
print("\n")
#3 The country with the lowest population
lowest = min(countries_population, key=lambda p: p[1])
print("Country with the lowest population:", lowest)
print("\n")
#4 The average population
total_population = 0
for country, population in countries_population:
    total_population += population
average_population = total_population / len(countries_population)
print("The average population:", int(average_population))
print("\n")
#5 Sort all countries by their population in descending order and display
#the top 5 countries.
sort_population = sorted(countries_population,key=lambda p:p[1], reverse=True)
print("The top 5 sortd population: ",sort_population[:5])

