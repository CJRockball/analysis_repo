#%%
import requests
import pandas as pd
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup
import re
from bs4.element import Comment

import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
#import os
#os.environ['MOZ_HEADLESS'] = '1'

#%% Beautiful soup

url2 = 'https://www.propertyforsale.com.sg/private-residential-rental-transactions?district=13&project=18+WOODSVILLE&submit='

driver = webdriver.Firefox()

driver.get(url2)
p_source = driver.page_source
#print(p_source)
#print(driver.find_element(By.TAG_NAME, 'h2').text)

#%%


def get_data(list_dict):
    
    table_stuff = driver.find_element(By.ID, 'records_list')
    data_list = table_stuff.text.split('\n')

    for item in data_list:
        item_dict = {}
        item_item = item.split(' ')
        sqft_item = item_item[10].split('-')
        item_dict['year'] = int(item_item[1])
        item_dict['month'] = item_item[0]
        item_dict['bedrooms'] = int(item_item[9])
        item_dict['sqft'] = int(sqft_item[0]) + 50
        item_dict['price'] = int(item_item[11].replace(",", ""))
        
        list_dict.append(item_dict)
    
    return list_dict
    
list_dict = []
for i in range(1,5):
    try:
        driver.find_element(By.XPATH, f"//a[@data-dt-idx={str(i)}]").click()
        list_dict = get_data(list_dict)
    except:
        print('No more pages')
        
#%%
driver.quit()
# %%

df = pd.DataFrame(data=list_dict)
df['price_sqft'] = round(df['price'] / df['sqft'],2)

# Make date column
df['date'] = pd.to_datetime(df.year.astype(str) + df.month.astype(str), format='%Y%B')

display(df.head())
print(df.info())

#%%

df.to_parquet('pp_data/18woodsville.parquet')

# %%

display(df.head(10))




# %%

df_mean = df.groupby(['date', 'bedrooms'])['price_sqft'].mean().to_frame().reset_index()

# %%

display(df_mean.head())
# %%

one_room = df.loc[df['bedrooms'] == 1, ['date', 'price_sqft']]
two_room = df.loc[df['bedrooms'] == 2, ['date', 'price_sqft']]
three_room = df.loc[df['bedrooms'] == 3, ['date', 'price_sqft']]

plt.figure()
plt.plot(one_room.date, one_room.price_sqft, color='blue', label='one')
plt.plot(two_room.date, two_room.price_sqft, color='orange', label='two')
plt.plot(three_room.date, three_room.price_sqft, color='green', label='three')
plt.grid()
plt.legend()
plt.ylabel('price per sq foot')
plt.xlabel('date')
plt.show()

# %%

one_mean = df_mean.loc[df_mean['bedrooms'] == 1, ['date','price_sqft']]
two_mean = df_mean.loc[df_mean['bedrooms'] == 2, ['date','price_sqft']]
three_mean = df_mean.loc[df_mean['bedrooms'] == 3, ['date','price_sqft']]

plt.figure()
plt.plot(one_mean.date, one_mean.price_sqft, color='blue', label='one')
plt.plot(two_mean.date, two_mean.price_sqft, color='orange', label='two')
plt.plot(three_mean.date, three_mean.price_sqft, color='green', label='three')
plt.grid()
plt.legend()
plt.ylabel('price per sq foot')
plt.xlabel('data')
plt.show()

# %%
