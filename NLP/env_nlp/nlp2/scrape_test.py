#%%
import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
from bs4.element import Comment

#url = 'https://di.se/live/'
url = 'https://www.cia.gov/the-world-factbook/countries/singapore/'

r = requests.get(url)

print(type(r))

# %%

print(r.text)

# %%


soup = BeautifulSoup(r.text, 'html.parser')
headings = soup.find_all(re.compile("^h[1-6]$"))

#soup.select("h3 mt30")
#soup.select('href="/the-world-factbook/field/coastline"')
#name_list = soup.find_all(class_='mt30')


# %%
print(type(headings))
header_list = []
for i in range(len(headings)):
    header_list.append(headings[i].text)

#print(soup.label.text)
#print(soup.get_text())
# print(name_list[7])
# print(len(name_list))


# %%



def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)

#html = urllib.request.urlopen('http://www.nytimes.com/2009/12/21/us/21storm.html').read()
test_text = text_from_html(r.text)

# %%

soup = BeautifulSoup(r.text)

# kill all script and style elements
for script in soup(["script", "style"]):
    script.extract()    # rip it out

# get text
text2 = soup.get_text()

#%%


split_dict = {}
for i in range(len(header_list)-1):
    split_dict[header_list[i]] = text2.partition(header_list[i])[2].partition(header_list[i+1])[0]

df_singapore = pd.DataFrame.from_dict(split_dict, orient='index')

# %%

display(df_singapore.index)

# %%

print(header_list)

# %% ============== APL ======================

url = 'https://pubs.aip.org/aip/apl/issue/123/5'

r = requests.get(url, headers = {"User-Agent": "Mozilla/5.0"})

#%%

soup = BeautifulSoup(r.text, 'html.parser')
print(r.text)

# %%

name_list = soup.find_all(class_="fb-item-title")

print(type(name_list))
print(len(name_list))
print(name_list)

# %%

#
art_set = soup.find_all(class_="customLink item-title")

print(type(art_set))
art_list = []
for i in range(len(art_set)):
    art_list.append(art_set[i].text.strip('\n'))

#print(art_list)

# %%

auth_set = soup.find_all(class_="al-authors-list")

auth_list = []
for i in range(len(auth_set)):
    auth_list.append(auth_set[i].text.strip('\n'))

# print(auth_list)

# %%

print(art_list)
print( auth_list)

# %%

a = list(zip(art_list, auth_list))

df_prl = pd.DataFrame(data=a, columns=['title','authors'])

# %%

display(df_prl.head())

# %%

#abs_set = soup.find_all(class_="abstract-2904860")

b = soup.find(id="abstract-2905553")

print(b)







# %%
