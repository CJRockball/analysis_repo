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
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
#import os
#os.environ['MOZ_HEADLESS'] = '1'

#%% Scrape PRL abstracts

for i in range(122,124):
    # DL from this year vol 122, 123 and issues
    for j in range(1,27):
        if i == 123 & j > 6:
            break
        else:
            df_prl = pd.DataFrame(columns=['title', 'authors','citation'])
            print(f'Volume: {i}, issue: {j}')
            
            url = f'https://pubs.aip.org/aip/apl/issue/{i}/{j}'

            r = requests.get(url, headers = {"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(r.text, 'html.parser')

            title_html = soup.find_all(class_="customLink item-title")
            author_html = soup.find_all(class_="al-authors-list")
            articleid_html = soup.find_all('a', attrs={'data-articleid':True})
            citation_html = soup.find_all('div', class_='ww-citation-primary')
    
            print('html', len(title_html), len(author_html), len(articleid_html), len(citation_html))
    
            art_list = [item.text.strip('\n') for item in title_html]
            auth_list = [item.text.strip('\n') for item in author_html]
            citation = [item.text.strip() for item in citation_html]

            articles_on_page_list = [item['data-articleid'] for item in articleid_html]

            print('lists', len(art_list), len(auth_list), len(citation), len(articles_on_page_list))
    
            
            a = list(zip(art_list, auth_list, citation))
            df_temp = pd.DataFrame(data=a, columns=['title', 'authors', 'citation'])

            df_prl = pd.concat([df_prl, df_temp], axis=0, ignore_index=True)


            driver = webdriver.Firefox()

            driver.get(url)
            p_source = driver.page_source

            abstract_list = []
            for abs_num in articles_on_page_list:

                article_link = driver.find_element(By.XPATH,f"//a[@data-articleid={str(abs_num)}]") 
                #print(article_link.get_attribute("outerHTML"))

                driver.execute_script("arguments[0].click();", article_link)
                time.sleep(2)
                abstract_html = driver.find_element(By.XPATH, f"//div[@id='abstract-{str(abs_num)}']") #class='abstract-response-placeholder js-abstract-response-placeholder']")
                #print(abstract_html.get_attribute('outerHTML'))
                
                abstract_list.append(abstract_html.text)
                time.sleep(10)

            driver.quit()
            
            df_prl['abstract'] = abstract_list
            
            df_prl.to_csv(f'prl_data/vol{i}_issue{j}', index=False)
    




# %%

display(df_prl.head())
print(df_prl.shape)


# %% try requests-html
from requests_html import HTMLSession

session = HTMLSession()

url = 'https://pubs.aip.org/aip/apl/issue/123/5'
response = session.get(url)

r = response.html.html
print(r)
#%%