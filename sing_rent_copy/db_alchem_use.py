import sqlalchemy as sa
from sqlalchemy import create_engine, insert
from sqlalchemy.orm import sessionmaker

from db_alchem import DevName, ProjectData, SalesData

import pandas as pd
import time

# Connect to db
engine = create_engine('sqlite:///scrape.db')
Session = sessionmaker(bind = engine)
session = Session()

## ----------------------  INSERT SINGLE DATA AND READ OUT WITH ORM
##Make data and insert in table dev_names
c1 = DevName(dev_name='House1')
session.add(c1)
session.commit()

# #Create data and insert into table project_datas
# data = ProjectData(
#     proj_id     = 1,
#     adress      = 'Address 4 Meyappa Chettiar Road · 358453',
#     district    = 'D13',
#     nhood       = 'Toa Payoh',
#     proj_size   = 'Medium',
#     built       = '2019',
#     tenure      = '99 years',
#     units       = 731,
#     blocks      = 6,
#     floors      = 18,
#     bedrooms    = '1,2,3,4,5',
#     developer   = 'MCC Land',
# )
# session.add(data)
# session.commit()

# # Prints out one post as a dict using method
# data = session.query(DevName).first().dict_out()
# print(data)

# # Print project data
# data = session.query(ProjectData).first().dict_out()
# print(data)

#-------------------------------------------------------------------------
# # Add more data table dev_names
# c1 = DevName(dev_name='House2')
# session.add(c1)
# session.commit()

# # Prints out one post as a dict using method
# data = session.query(DevName).all()
# for item in data:
#     print(item.dict_out())

# Filter and return one post
# data = session.query(DevName).filter(DevName.dev_name == 'House1').first().dict_out()
# print(data)
# print(data['dev_id'])


# # Get project id to find project data
# def get_data(proj_name):
#     data = session.query(DevName).filter(DevName.dev_name == proj_name).first().dict_out()
#     proj_id = data['dev_id']   

#     search_data = session.query(ProjectData).filter(ProjectData.proj_id == proj_id).first().dict_out()
#     print(search_data)
#     return

# get_data('House1')

# Multiple insert ORM
#session.execute(insert(DevName),[{'dev_name':'House3'},{'dev_name':'House4'}])
#session.commit()



# -------------------------------
# Explore inspection object
# connection = engine.connect()
# inspection = sa.inspect(engine)
# a = inspection.get_table_names()
# print(a)


# #---------------------- INSERT DATA
# # Make one insert with Core
# metadata_obj = sa.MetaData()
# ins_table = sa.Table("dev_names", metadata_obj, autoload_with=engine)
# connection = engine.connect()

# stmt = ins_table.insert().values(dev_name='House3')

# connection.execute(stmt)
# connection.commit()

#Table.query.limit(10).offset(10).all()

#-----------------------------------
# # Make multiple executions
# with engine.connect() as connection:
#     metadata_obj = sa.MetaData()
#     ins_table = sa.Table("dev_names", metadata_obj, autoload_with=engine)

#     result = connection.execute(ins_table.select())
#     for row in result:
#         print(row)

#     stmt = ins_table.insert().values(dev_name='House4')

#     connection.execute(stmt)
#     connection.commit()
    
#     result = connection.execute(ins_table.select())
#     for row in result:
#         print(row)


#-------------------------------------------------
# ## Insert multiple data
# #connection = engine.connect()
# insert_list = ['House5', 'House6', 'House7', 'House8', 'House9']
# insert_dict_list = [{'dev_name': i} for i in insert_list]

# tbl = sa.Table('dev_names', sa.MetaData(), autoload_with=engine)
# ins = tbl.insert()

# with engine.begin() as conn:
#     # This will automatically commit
#     conn.execute(ins, insert_dict_list)


# ----------------------- INSERT CSV file

# def insert_csv(name="House1"):
#     df = pd.read_csv('sales_data.csv')
#     df = df.iloc[:3,:]
    
#     data = session.query(DevName).filter(DevName.dev_name == name).first().dict_out()
#     insert_id = data['dev_id']

#     df['proj_id'] = insert_id
#     insert_dict_list = df.to_dict(orient='records')
        
#     connection = engine.connect()
#     metadata_obj = sa.MetaData()
#     tbl = sa.Table("sales_datas", metadata_obj, autoload_with=engine)

#     connection.execute(tbl.insert(), insert_dict_list)
#     connection.commit()
    
#     return


# insert_csv()


##----------------------------
# data = session.query(SalesData).all()
# for item in data:
#     print(item.dict_out())

# data = session.query(SalesData).first().dict_out()
# print(data)

# data = session.query(SalesData).first().dict_out()
# print(data)

