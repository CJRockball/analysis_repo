import sqlalchemy as sa
from sqlalchemy import create_engine, insert
from sqlalchemy.orm import sessionmaker

from models import DevName, ProjectData, SalesData
from db_scrape import Base

import pandas as pd
import time
import os

# Connect to db
engine = create_engine('sqlite:///scrape.db')
Session = sessionmaker(bind = engine)
session = Session()
Base.metadata.create_all(engine)

## ----------------------  INSERT SINGLE DATA AND READ OUT WITH ORM
# Load DevName
def insert_dev_name(name='18woodsville'):
    c1 = DevName(dev_name=name)
    session.add(c1)
    session.commit()

#insert_dev_name(name='18woodsville')
#insert_dev_name(name='Poiz')

# # Prints out one post as a dict using method
data = session.query(DevName).all()
for item in data:
    print(item.dict_out())


## Check data folder add new data






