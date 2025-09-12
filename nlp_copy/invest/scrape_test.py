#%%
import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
from bs4.element import Comment

import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
os.environ['MOZ_HEADLESS'] = '1'

# %%

url = 'https://www.investing.com/equities/volvo-b-earnings'

r = requests.get(url, headers={'User-Agent':'Mozilla/5.0'})
soup = BeautifulSoup(r.text, 'html.parser')


#%%

print(r.text)

# %% try requests-html
from requests_html import HTMLSession

session = HTMLSession()

url = 'https://www.investing.com/equities/volvo-b-earnings'
response = session.get(url)

r = response.html.html
print(r)



# %%

url = 'https://www.investing.com/equities/volvo-b-earnings'
driver = webdriver.Firefox()

driver.get(url)
p_source = driver.page_source

abstract_html = driver.find_element(By.ID, 'earningsHistory497') #class='abstract-response-placeholder js-abstract-response-placeholder']")
print(abstract_html.get_attribute('outerHTML'))
                



# %%

str_table = abstract_html.text
str_list = str_table.split('\n')
str_list2 = [item.split(' ') for item in str_list]

for item in str_list:
    print(item)

# print(abstract_html.text)

# %% Get table save to df

df = pd.DataFrame(data=str_list2)

display(df.head())

df['Release Date'] = df[0].astype(str) + df[1].astype(str) + df[2].astype(str)
df['Period End'] = df[3]
df['EPS'] = df[4].astype(str)
df['EPS Forecast'] = df[7].astype(str)
df['Revenue'] = df[8].astype(str)
df['Revenue Forecast'] = df[11].astype(str)
df.drop(columns=[0,1,2,3,4,5,6,7,8,9,10,11], inplace=True)
display(df)

#%%

url = 'https://www.investing.com/equities/'

driver = webdriver.Firefox()
driver.get(url)
p_source = driver.page_source

select = Select(driver.find_element(By.ID, 'stocksFilter'))
time.sleep(2)
select.select_by_visible_text('OMX Stockholm 30')
time.sleep(2)

stock_table = driver.find_element(By.ID, 'cross_rate_markets_stocks_1')
omx_source = driver.page_source

driver.quit()

# %% Use beautiful soup to get stock links
 
soup = BeautifulSoup(omx_source, 'html.parser')

table_test = soup.find_all('tr', {'id':'pair_958894'})

tr_elements = soup.find_all('tr', attrs={'id':True})

data_list = []
for item in tr_elements:
    data_list.append((item['id'], item.findChild("a")['title'], item.findChild("a")['href']))
    
df_data = pd.DataFrame(data=data_list, columns=['id', 'name', 'link'])    
df_data = df_data.iloc[:30,:]



# %% Use link to open page, get table
# Open page with selenium, get swedish stocks page, save html str
data_range = df_data.shape[0]

for i in range(data_range):
    test_string = df_data.loc[i,'link']
    if '?' in test_string:
        a = test_string.split('?')
        b = '?' + a[1]
    else:
        a = test_string, ''
        b = ''


    url = 'https://www.investing.com' + a[0] + '-earnings' + b
    print(url)
    
    start = time.time()
    driver = webdriver.Firefox()

    driver.get(url)
    p_source = driver.page_source

    for _ in range(4):
        try:
            driver.find_element(By.LINK_TEXT, "Show more").click()
        except:
            print("An exception occurred")
        
        
    abstract_html = driver.find_element(By.XPATH, f"//table[@class='genTbl openTbl ecoCalTbl earnings earningsPageTbl']") 

    str_table = abstract_html.text
    str_list = str_table.split('\n')
    str_list2 = [item.split(' ') for item in str_list]

    # print(abstract_html.text)
    df = pd.DataFrame(data=str_list2)

    #display(df.head())
    df['Release Date'] = df[0].astype(str) + df[1].astype(str) + df[2].astype(str)
    df['Period End'] = df[3]
    df['EPS'] = df[4].astype(str)
    df['EPS Forecast'] = df[7].astype(str)
    df['Revenue'] = df[8].astype(str)
    df['Revenue Forecast'] = df[11].astype(str)
    df.drop(columns=[0,1,2,3,4,5,6,7,8,9,10,11], inplace=True)
    df = df.iloc[1:,:]
    #display(df)

    fname = a[0].split('/')[2]
    df.to_csv(f'{fname}.csv', index=False)
    #time.sleep(10)
    driver.quit()
    print(time.time() - start)

#%% Sort data in table, and clean
import numpy as np


def read_in():
    files = os.listdir('stock_data')
    #print(files)
    df = pd.DataFrame(columns=['EPS', 'EPS Forecast', 'Revenue', 'Revenue Forecast'])

    for a_file in files:
        # Open file
        df_temp = pd.read_csv(f'stock_data/{a_file}')
        # Set up types
        df_temp['Release Date'] = pd.to_datetime(df_temp['Release Date'], format='%b%d,%Y')
        df_temp['Period End'] = pd.to_datetime(df_temp['Period End'], format='%m/%Y')
        df_temp['EPS'] = df_temp['EPS'].replace({"--": np.nan}, regex=True).astype(float)
        df_temp['EPS Forecast'] = df_temp['EPS Forecast'].replace({"--": np.nan}, regex=True).astype(float)
        df_temp['Revenue'] = df_temp['Revenue'].replace({r'[A-z]+':"", "--": np.nan}, regex=True).astype(float)
        df_temp['Revenue Forecast'] = df_temp['Revenue Forecast'].replace({",":".", r'[A-z]+':"", "--": np.nan}, regex=True).astype(float)
        
        df_temp.to_parquet(f"stock_data/{a_file.split('.')[0]}.parquet")
        
    #display(df_temp.info())


read_in()

# %% 

def read_in():
    files = os.listdir('stock_data/parquet')
    #print(files)
    df = pd.DataFrame(columns=['EPS', 'EPS Forecast', 'Revenue', 'Revenue Forecast'])
    for a_file in files:
        no_list = ['atlas-copco-a.parquet', 'investor.parquet', 'kinnevik-investment-b.parquet', 'nibe-industrier-b.parquet']

        if a_file not in no_list:
            # Open file
            df_temp = pd.read_parquet(f'stock_data/parquet/{a_file}')
            # Get newest report
            a = df_temp.loc[1,['EPS', 'EPS Forecast','Revenue', 'Revenue Forecast']].to_frame().T
            a['name'] = a_file
            # Get previous quarter EPS
            a['previous EPS'] = df_temp.loc[2,['EPS']].to_numpy()
            df = pd.concat([df,a])
            df['EPS'] = df.EPS.astype(float)
            df['EPS Forecast'] = df['EPS Forecast'].astype(float)
    #display(df)

    def beat_fcn(row):
        if row.EPS > row['EPS Forecast']:
            return 1
        else: return -1

    def perc_fcn(row):
        if row.EPS < 0 and row['EPS Forecast'] > 0:
            return round(-(row['EPS Forecast'] - row.EPS)/row['EPS Forecast'] * 100,1)
        elif row.EPS < 0 and row['EPS Forecast'] < 0:
             return round((row['EPS Forecast'] - row.EPS)/row['EPS Forecast'] * 100,1)
        else:
            return round((row.EPS - row['EPS Forecast']) / row['EPS Forecast'] * 100,1)
        

    df['EPS beat'] = df.apply(beat_fcn, axis=1)
    df['EPS_perc_forecast'] = df.apply(perc_fcn, axis=1)
    display(df.sort_values(by='Revenue'))
    
read_in()


# %%

display(df)
# %%
