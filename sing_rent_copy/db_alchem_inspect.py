import sqlalchemy as sa
from sqlalchemy import create_engine, insert
from sqlalchemy.orm import sessionmaker

from db_alchem import DevName, ProjectData, SalesData, RentData

import pandas as pd
import time
#from types import SQLTable

# Connect to db
engine = create_engine('sqlite:///scrape.db')
Session = sessionmaker(bind = engine)
session = Session()

# # Create an inspector
# inspector = sa.inspect(engine)
# tables_dbo = inspector.get_table_names()


def get_table_size(session: sessionmaker, table_class) -> int:
    """
    Get the number of rows a table has
    
    :param session: An SQLAlchemy session
    :param table_class: A class that inherits from `sqlalchemy.Base` and represents a table 
    :return: integer
    """
    with session() as session:
        return session.scalar(sa.select(sa.func.count()).select_from(table_class))


nr_rows = get_table_size(Session, RentData)
print(nr_rows)


##----------------------------
# data = session.query(SalesData).all()
# for item in data:
#     print(item.dict_out())

# data = session.query(RentData).all()
# for item in data:
#     print(item.dict_out())

# data = session.query(ProjectData).all()
# for item in data:
#     print(item.dict_out())



# data = session.query(SalesData).first().dict_out()
# print(data)

# data = session.query(SalesData).first().dict_out()
# print(data)




