#%%
import sqlalchemy as sa
from sqlalchemy import create_engine, insert
from sqlalchemy.orm import sessionmaker

from models import DevName, ProjectData, SalesData, Base

import pandas as pd
import time
import os


#%%
# Connect to db
engine = create_engine('sqlite:///scrape.db')
Session = sessionmaker(bind = engine)
session = Session()
Base.metadata.create_all(engine)


#%%
## ----------------------  ## Check data folder add new data
"""
1. Check all file names
2. set up load
    1. Check if project is in db
    2. Read project or skip
"""

def get_proj():
    data = session.query(DevName).all()

    for name in data:
        print(name.dict_out())

    return

def get_proj_names():
    fname = 'mod_api/scrape/data'
    cwd = os.getcwd()
    path = os.path.join(cwd,fname)

    files = os.listdir(path)
    
    return files


def check_projects(project_names):
    
    # Get all projects in db from DenNames
    data = session.query(DevName).all()
    
    current_projects = []
    for item in data:
        current_projects.append(item.dict_out()['dev_name'])
        
    for name in project_names:
        print(name)
        if name not in current_projects:
            load_project(name)
            
    return 


def load_project(file_name):
    fname = f'mod_api/scrape/data/{file_name}'
    cwd = os.getcwd()
    path = os.path.join(cwd,fname)
    files = os.listdir(path)
    
    insert_dev_name(file_name)
    insert_dev_data(file_name)
    insert_sales(file_name)
    insert_rent(file_name)
    
    print(f'=== {file_name} DONE===')
    
    return 

def insert_dev_name(file_name):
    session.add(DevName(dev_name = file_name))
    session.commit()
    return

   
def insert_dev_data(file_name):
    fname = f'mod_api/scrape/data/{file_name}/proj_data.csv'
    cwd = os.getcwd()
    path = os.path.join(cwd,fname)
    
    df = pd.read_csv(path)
    
    data = session.query(DevName).filter(DevName.dev_name == file_name).first().dict_out()
    insert_id = data['dev_id']

    df['proj_id'] = insert_id
    insert_dict = df.to_dict(orient='records')
    
    session.execute(insert(ProjectData),list(insert_dict))
    session.commit()
    
        

def insert_sales(file_name):
    fname = f'mod_api/scrape/data/{file_name}/sales_data.csv'
    cwd = os.getcwd()
    path = os.path.join(cwd,fname)
    df = pd.read_csv(path)
    
    data = session.query(DevName).filter(DevName.dev_name == file_name).first().dict_out()
    insert_id = data['dev_id']

    df['proj_id'] = insert_id
    insert_dict_list = df.to_dict(orient='records')
        
    connection = engine.connect()
    metadata_obj = sa.MetaData()
    tbl = sa.Table("sales_datas", metadata_obj, autoload_with=engine)

    connection.execute(tbl.insert(), insert_dict_list)
    connection.commit()
    
    return


def insert_rent(file_name):
    fname = f'mod_api/scrape/data/{file_name}/rent_data.csv'
    cwd = os.getcwd()
    path = os.path.join(cwd,fname)
    df = pd.read_csv(path)
    
    data = session.query(DevName).filter(DevName.dev_name == file_name).first().dict_out()
    insert_id = data['dev_id']

    df['proj_id'] = insert_id
    insert_dict_list = df.to_dict(orient='records')
        
    connection = engine.connect()
    metadata_obj = sa.MetaData()
    tbl = sa.Table("rent_datas", metadata_obj, autoload_with=engine)

    connection.execute(tbl.insert(), insert_dict_list)
    connection.commit()
    
    return



#%%


project_names = get_proj_names()

check_projects(project_names)


# %%

get_proj()


# %%
