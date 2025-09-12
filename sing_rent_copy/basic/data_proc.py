import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import requests
import json

disp = FastAPI()

disp.mount("/static", StaticFiles(directory="basic/static"), name="static")
templates = Jinja2Templates(directory='basic/templates')


@disp.get('/')
def index():
    return {'display':'world'}


def plot_graf(df):
    #NEed to get the data in the right format for seaborn.
    #Needs to be transformed back for axis labels
    df['date'] = pd.to_datetime(df['date']).apply(lambda date: date.toordinal())
    print(df.info())
    sns.set()
    
    plt.figure()
    #sns.lmplot(x='date', y='price/sqft', data=df)
    sns.regplot(x='date', y='price/sqft', data=df)
    # plt.plot(df.date, df['price/sqft'])
    # plt.xticks(rotation=-45, ha='left')
    # plt.grid()
    plt.savefig('basic/static/plot1.png')
    return 


@disp.get('/get_data')
def get_data(request: Request):
    test_url = 'http://localhost:8000/all_data'
    test_response = requests.get(test_url)
    response_dict = json.loads(test_response.text)
    df = pd.DataFrame(data=response_dict, columns=['index', 'year', 'month', 'project',
                                                   'road', 'districs','bedrooms','sqft',
                                                   'price','price/sqft','date'])
    
    df_3bd = df.groupby(['bedrooms','date'])['price/sqft'].mean().to_frame().reset_index()
    
    df_3bd1 = df_3bd.loc[df_3bd.bedrooms == 3, :]
    plot_graf(df_3bd1)
    
    
    
    return templates.TemplateResponse('index.html', {'request':request})


@disp.get('/data_search')
def data_search_get(request:Request):    
    return templates.TemplateResponse('index2.html', {'request':request})


@disp.post('/data_search')
def data_search_post(request:Request, project:str=Form(...)):
    test_url = 'http://localhost:8000/data?project=' + str(project)

    test_response = requests.get(test_url)
    response_dict = json.loads(test_response.text)
    return templates.TemplateResponse('index2.html', {'request':request, 'db_rentals':response_dict})
