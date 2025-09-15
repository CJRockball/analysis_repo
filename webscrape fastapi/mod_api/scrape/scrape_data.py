#%%
%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
import re
import time
import matplotlib.pyplot as plt

import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options

import os

cwd = os. getcwd()


#%%
url = 'https://www.99.co/singapore/condos-apartments/18-woodsville#'
url2 = 'https://www.99.co/singapore/condos-apartments/the-poiz-residences#'
url3 = 'https://www.99.co/singapore/condos-apartments/the-venue-residences#'
url4 = 'https://www.99.co/singapore/condos-apartments/sant-ritz'

#driver = webdriver.Firefox()
#driver = webdriver.Chrome()
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=options)

driver.get(url2)


#driver.find_element(By.XPATH,f"//a[@aria-label='Close']").click() 
#WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, f"//*[@aria-label='Close']"))).click()


#p_source = driver.page_source
#print(p_source)

# #%%
# url1 = 'https://www.99.co/singapore/condos-apartments/18-woodsville#'
# url2 = 'https://www.99.co/singapore/condos-apartments/the-poiz-residences#'
# url3 = 'https://www.99.co/singapore/condos-apartments/the-venue-residences#'
# url4 = 'https://www.99.co/singapore/condos-apartments/sant-ritz'

## Firefox
# #options = Options()
# #options.add_argument("--headless")
# #driver = webdriver.Firefox(options=options)
## Chrome
# options = webdriver.ChromeOptions()
# options.add_argument("--headless=new")
#options.add_argument("--start-maximized")
# driver = webdriver.Chrome(options=options)
# try:
#     driver.get(url3)
# except:
#     print("Didn't load")


#%%

dev_name = 'intero' #'one-leicester' #'parc-aston' #'woodsville-28' #'nin-residence' #'sennett-residences' #'the-venue-residences' # 'the-poiz-residences' #  'sant-ritz' #'18-woodsville'
no_head = False
fname = f'data/{dev_name}'
path = os.path.join(cwd, fname)
print(path)
 
if not os.path.exists(path):
  os.mkdir(path)
  print("Folder %s created!" % path)
else:
  print("Folder %s already exists" % path)


def load_page(dev_name, no_head=True):
    url = f'https://www.99.co/singapore/condos-apartments/{dev_name}'
    
    if no_head:
        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--start-maximized")
        driver = webdriver.Chrome(options=options)        
        try:
            driver.get(url)
        except:
            print("Didn't load")
    else:
        options = webdriver.ChromeOptions()
        options.add_argument("--start-maximized")
        driver = webdriver.Chrome(options=options)

        driver.get(url)

    return driver

driver = load_page(dev_name, no_head)

#%%
# Switch to banner
#WebDriverWait(driver, 20).until(EC.frame_to_be_available_and_switch_to_it((By.XPATH,"//iframe[@name='intercom-banner-frame']")))
iframe = driver.find_element(By.XPATH, "//iframe[@name='intercom-banner-frame']")
driver.switch_to.frame(iframe)
# Close banner
driver.find_element(By.XPATH,f"//div[@aria-label='Close']").click() 
# Switch back
driver.switch_to.parent_frame() #.default_content()

#%%

print(driver.current_url)


#%% Create dict of housing specs

data_dict_keys = ['address', 'district', 'nhood', 'proj_size', \
    'built', 'tenure','units', 'blocks', 'floors', 'bedrooms', 'developer']
web_data_keys = ['Address', 'District', 'Neighbourhood', 'Project Size', \
    'Built year', 'Tenure', 'Number of Units', 'Blocks', 'Floors', 'Bedrooms', 'Developer']
new_web_data_keys = ['Address', 'District', 'Neighbourhood', 'Property type', 'Project Size', \
    'TOP date', 'Tenure', 'Number of Units', 'Blocks', 'Floors', 'Bedrooms', 'Developer']


def list_function(header, specs):
    specs_list = []
    item0 = None
    for item in specs:
        if item.text != item0:
            #print(item.text)
            specs_list.append(item.text)
            item0 = item.text
                
    header_list = []
    for item in header:
        #print(item.text)
        header_list.append(item.text)
        
    web_data = {}
    for key,val in zip(header_list,specs_list):
        web_data[key] = val
    
    return web_data


def new_list_function(specs):
    
    for item in specs:
        specs_tmp = item.text.split('\n')
    #print(specs_tmp) 
    # print(specs_tmp.index("Address"))
    
    specs_list = []
    for wd_key in new_web_data_keys:
        data_pos = specs_tmp.index(wd_key) + 1
        #print(specs_tmp[data_pos - 1], specs_tmp[data_pos])
        specs_list.append(specs_tmp[data_pos])
        
    web_data = {}
    for key,val in zip(web_data_keys,specs_list):
        web_data[key] = val
    
    return web_data


specs = driver.find_elements(By.CLASS_NAME, "_377zm")

if len(specs) < 1:
    specs = driver.find_elements(By.CLASS_NAME, "DetailsTable_tableLayout__HFC8y")
    web_data = new_list_function(specs)
else:
    header = driver.find_elements(By.CLASS_NAME, "_3Yq9T")
    web_data = list_function(header, specs)

data_dict = {}
for dd_keys, wd_keys in zip(data_dict_keys, web_data_keys):
    try:
        value = web_data[wd_keys]
        #print("Key exists in the dictionary.")
        data_dict[dd_keys] = value
    except KeyError:
        #print("Key does not exist in the dictionary.")
        data_dict[dd_keys] = None

df_data = pd.DataFrame([data_dict])
display(df_data.head())

#%%

path1 = os.path.join(path,'proj_data.csv')
df_data.to_csv(path1, index=False)

#%% Download sales data

sales_data_list = []
banana = True
i = 2
while banana:
    
    try:
        sales_data = driver.find_elements(By.CLASS_NAME, '_1TB50') #'TransactionsListWithProfit_tableRow__kUycO') #

        for item in sales_data:
            sales_data_list.append(item.text.split('\n'))
            
        driver.find_element(By.XPATH,f"//a[@aria-label='Page {i}']").click() 
        print('worked', i, len(sales_data_list))
        i += 1
        time.sleep(2)
    except:
        print('No more data')
        print(len(sales_data_list))
        banana = False
        driver.find_element(By.XPATH,f"//a[@aria-label='Page 1']").click() 

#%%

sales_dict = {'date':[], 'block':[], 'beds':[], 'psf':[], 'area':[], 'price':[]}

for item in sales_data_list:
    try:
        sales_dict['date'].append(item[0])
        sales_dict['block'].append(item[1])
        if item[3][0].isdigit():
            sales_dict['beds'].append(item[3][0])
        else:
            sales_dict['beds'].append('0')
        
        sales_dict['psf'].append(item[4][1:].replace(',',''))
        sales_dict['area'].append(item[5].replace(',',''))
        
        temp_price = item[6][1:].replace(',', '')
        if re.match('[0-9]{1}\.[0-9]{3}M', temp_price):
            temp_price2 = temp_price[:-1].replace('.','')
            temp_price2 = temp_price2 + '000'
            sales_dict['price'].append(temp_price2)
        else:
            sales_dict['price'].append(temp_price)
    except:
        print('error')
        
        
#%%
df_sales = pd.DataFrame(sales_dict)
df_sales['date'] = pd.to_datetime(df_sales['date'], format='%m/%Y')
df_sales['block'] = df_sales['block'].astype(int)
df_sales['beds'] = df_sales['beds'].astype(int)
df_sales['psf'] = df_sales['psf'].astype(int)
df_sales['area'] = df_sales['area'].astype(int)
df_sales['price'] = df_sales['price'].astype(int)

display(df_sales.head())
print(df_sales.info())

#%%

path2 = os.path.join(path,'sales_data.csv')
df_sales.to_csv(path2, index=False)

# %% Get rent data

# Click rent button
# Works (actual name is different from eplorer)
#i = driver.find_element(By.CLASS_NAME,'dniCg._1AoIA.KOuH9._2rhE-') #.text() #.click()

#driver.find_element(By.XPATH,"//p[@class='dniCg _1AoIA KOuH9 _2rhE-'][contains(., 'Rent')]").click() 
# driver.find_element(By.XPATH, "//p[text()='Rent']").click()
# time.sleep(2)
#driver.find_element(By.CLASS_NAME, '_2j1I_').click()  #***************************************
# Need to click twice to get to work
rent_btn = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, "//p[text()='Rent']")))
#rent_btn = WebDriverWait(driver,20).until(EC.element_to_be_clickable((By.XPATH,"//p[@class='dniCg _1AoIA KOuH9 _2rhE-']")))
rent_btn.click()
time.sleep(2)
#driver.find_element(By.XPATH,"//p[@class='dniCg _1AoIA KOuH9 _2rhE-'][contains(., 'Rent')]").click()

#%% Download rent data

rent_data_list = []
banana = True
i = 2
while banana:
    
    try:
        rent_data = driver.find_elements(By.CLASS_NAME, '_1TB50') #'TransactionsListWithProfit_tableRow__kUycO') #'_1TB50')
        
        for item in rent_data:
            rent_data_list.append(item.text.split('\n'))
            
        driver.find_element(By.XPATH,f"//a[@aria-label='Page {i}']").click() 
        print('worked', i, len(rent_data_list))
        i += 1
        time.sleep(2)
    except:
        print('No more data')
        print(len(rent_data_list))   
        banana = False
        #driver.find_element(By.XPATH,f"//a[@aria-label='Page 1']").click() 



#%%
rent_data_dict = {'date':[], 'beds':[], 'psf':[], 'area':[], 'price':[]}

for item in rent_data_list:
    try:
        if re.match('[0-1][0-9]\/20[0-9]{2}', item[0]):
            rent_data_dict['date'].append(item[0])
        else:
            rent_data_dict['date'].append('01/1970')
            print('0',item[0])
        if item[1][0].isdigit():
            rent_data_dict['beds'].append(item[1][0])
        else:
            rent_data_dict['beds'].append('0')
            #print('1', item[1][0], item)
        if re.match('[0-9]{1,2}\.[0-9]{1,2}',item[2][1:]):
            rent_data_dict['psf'].append(item[2][1:])
        else:
            rent_data_dict['psf'].append('0.0')
            print('2', item[2][1:])
        if item[3].split('-')[0].isdigit() or re.match('[0-9]{1},[0-9]{3}',item[3].split('-')[0]):
            rent_data_dict['area'].append(item[3].split('-')[0])
        else:
            rent_data_dict['area'].append('0')
            print('3', item[3].split('-')[0])
        if re.match('[0-9]{1},[0-9]{3}', item[4][1:]):
            rent_data_dict['price'].append(item[4][1:])
        else:
            rent_data_dict['price'].append('0')
            print('4', item[4])
    except: 
        print('Data Issues', item)    

df = pd.DataFrame(rent_data_dict)
df['date'] = pd.to_datetime(df['date'], format='%m/%Y')
df['beds'] = df['beds'].astype(int)
df['psf'] = df['psf'].astype(float)
df['area'] = df['area'].str.replace(',','').astype(int) + 50
df['price'] = df['price'].str.replace(',','').astype(int)

#%%
display(df.head())
print(df.info())

print(rent_data_list[0])
print(len(rent_data_list))
print(df.isnull().sum())

#%%

path3 = os.path.join(path,'rent_data.csv')
df.to_csv(path3, index=False)

# %%

driver.quit()




# %%
