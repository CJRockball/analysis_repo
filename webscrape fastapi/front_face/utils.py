import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import folium

import os
import time
from selenium import webdriver

from front_face.map_util import map_data


def time_plt(df, bedrooms):
    df_2bed = df.loc[df.beds == bedrooms, :]
    df_month = df_2bed.groupby('date')['psf'].mean().reset_index()
    df_month['date'] = df_month['date'].astype('datetime64[ns]')
    
    df_month['psf_ma'] = df_month['psf'].rolling(3).mean()
    
    sns.set()
    plt.figure()
    plt.plot(df_month.date, df_month.psf, lw=2, label='PSF')
    plt.plot(df_month.date, df_month.psf_ma, color='orange', lw=2, label='PSF 3MA')
    plt.title("PSF vs Time")
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Time')
    plt.ylabel('PSF [SGD]')
    plt.savefig('front_face/static/plot1.png')

    return


def data_fcn(data, bedrooms):
    # Split up data on 3 df
    info_df = data.json()['proj_info']
    df_info = pd.DataFrame(info_df)
    #print(df_info.shape)

    rent_df = data.json()['rent_data']
    df_rent = pd.DataFrame(rent_df)
    #print(df_rent.shape)
    
    sales_df = data.json()['sales_data']
    df_sales = pd.DataFrame(sales_df)
    #print(df_sales.shape)
    
    # Plot rental psf
    time_plt(df_rent, bedrooms)
    
    return    


def map_fcn(district=12):
    district_data = map_data()['features'][district]
    geo_dict = {'type':"FeatureCollection",'features':[district_data]}

    m = folium.Map(width=650,height=400,location=[1.3539024,103.8202006], zoom_start=11)
# [1.329590,103.868190]

    folium.GeoJson(geo_dict,
               style_function=lambda x: {'fillColor':x['properties']['fill_color'], 
                                         'fillOpacity':0.7,
                                         "color": "black",'weight':2}
               ).add_to(m)
    
    m.save('front_face/templates/map.html')

    fn='map.html'
    tmpurl=f'file://{os.getcwd()}/front_face/templates/{fn}'
    
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--start-maximized")
    browser = webdriver.Chrome(options=options)  
    browser.get(tmpurl)

    time.sleep(5)  #Give the map tiles some time to load
    browser.save_screenshot('front_face/static/map.png')
    browser.quit()

    return

