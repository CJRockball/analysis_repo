from fastapi import FastAPI, Depends

import sqlalchemy as sa
from sqlalchemy import create_engine, insert
from sqlalchemy.orm import sessionmaker, Session

from models import DevName, ProjectData, SalesData, RentData
from db_scrape import engine, get_db_session
from schemas import DevProjectName, ProjectName, ProjectNames


app = FastAPI()


@app.get('/')
def root():
    return {'hello':'world'}


@app.get("/get", response_model=ProjectNames)
def get_proj(db=Depends(get_db_session)):
    data = db.query(DevName).all()
    data_list = []
    for item in data:
        data_list.append(item.dict_out())
    return {'projects': [item.dict_out() for item in data]} #data_list

@app.get('/get/{proj_name}', response_model=ProjectName)
def project(proj_name:str, db: Session=Depends(get_db_session)):
    return db.query(DevName).filter(DevName.dev_name == proj_name).first().dict_out()


@app.get('/rent_data/{proj_name}')
def rent_data(proj_name:str, db: Session=Depends(get_db_session)):
    proj_data = db.query(DevName).filter(DevName.dev_name == proj_name).first().dict_out()
    proj_data_id = proj_data['dev_id']
    
    data =  db.query(RentData).filter(RentData.proj_id  == proj_data_id).all()
    data_list = []
    for item in data:
        data_list.append(item.dict_out())
    return  data_list
    

@app.post('/project') #, response_model=ProjectName)
def create_project(payload: DevProjectName, db: Session=Depends(get_db_session)):
    #print(payload)
    c1 = DevName(dev_name=payload.model_dump()['dev_name'])
    db.add(c1)
    db.commit()
  
    return db.query(DevName).filter(DevName.dev_name == payload.model_dump()['dev_name']).first().dict_out()

