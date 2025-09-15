import sqlalchemy as sa
from sqlalchemy import create_engine, insert
from sqlalchemy.orm import sessionmaker

from models import DevName, ProjectData, SalesData
from db_scrape import Base

import pandas as pd
import time

# Connect to db
engine = create_engine('sqlite:///scrape.db')
Session = sessionmaker(bind = engine)
session = Session()
Base.metadata.create_all(engine)
# Load DevName
def insert_dev_name(name='18woodsville'):
    c1 = DevName(dev_name=name)
    session.add(c1)
    session.commit()
    
def insert_dev_data(name):
    df = pd.read_csv('proj_data.csv')
    
    data = session.query(DevName).filter(DevName.dev_name == name).first().dict_out()
    insert_id = data['dev_id']

    df['proj_id'] = insert_id
    insert_dict = df.to_dict(orient='records')
    
    session.execute(insert(ProjectData),list(insert_dict))
    session.commit()
    
        

def insert_sales(name):
    df = pd.read_csv('sales_data.csv')
    
    data = session.query(DevName).filter(DevName.dev_name == name).first().dict_out()
    insert_id = data['dev_id']

    df['proj_id'] = insert_id
    insert_dict_list = df.to_dict(orient='records')
        
    connection = engine.connect()
    metadata_obj = sa.MetaData()
    tbl = sa.Table("sales_datas", metadata_obj, autoload_with=engine)

    connection.execute(tbl.insert(), insert_dict_list)
    connection.commit()
    
    return


def insert_rent(name):
    df = pd.read_csv('rent_data.csv')
    
    data = session.query(DevName).filter(DevName.dev_name == name).first().dict_out()
    insert_id = data['dev_id']

    df['proj_id'] = insert_id
    insert_dict_list = df.to_dict(orient='records')
        
    connection = engine.connect()
    metadata_obj = sa.MetaData()
    tbl = sa.Table("rent_datas", metadata_obj, autoload_with=engine)

    connection.execute(tbl.insert(), insert_dict_list)
    connection.commit()
    
    return


insert_dev_name('Poiz') #'18woodsville')
# insert_dev_data('18woodsville')
# insert_sales('18woodsville')
# insert_rent('18woodsville')