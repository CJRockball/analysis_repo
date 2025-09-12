from fastapi import FastAPI
import pandas as pd
from basic.db_help import get_data, get_x_data, project_search

app = FastAPI()

@app.get('/')
def index():
    return {'hello':'world'}


@app.get('/all_data')
def data():
    rows = get_data()
    return list(rows)


@app.get('/data/{lines}')
def x_data(lines):
    rows = get_x_data(lines)
    return list(rows)

@app.get('/data/')
def project_data(project:str):
    rows = project_search(project)
    return list(rows)