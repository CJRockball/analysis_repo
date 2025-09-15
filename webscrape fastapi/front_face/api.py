import requests
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
from front_face.utils import data_fcn, map_fcn

app = FastAPI()
app.mount('/static', StaticFiles(directory='front_face/static'), name='static')
templates = Jinja2Templates(directory='front_face/templates')

@app.get('/')
def root():
    return {'hello':'world'}


@app.get('/get/{name}')
def get_project(name:str):
    db_data = requests.get(f'http://localhost:8000/get_project/{name}')
    data_fcn(db_data)
     
    return db_data.json()


@app.get('/project_data')
def data_search_get(request:Request):    
    return templates.TemplateResponse('index.html', {'request':request})


@app.post('/project_data')
def display_data(request:Request, project:str=Form(...), bedrooms:int=Form(...)):
    data_url = f'http://localhost:8000/get_project/{project}'
    db_data = requests.get(data_url)
    
    data_fcn(db_data, bedrooms)
    map_fcn()

    return templates.TemplateResponse('index2.html', {'request':request, 'project':project, 'bedrooms': bedrooms})
    