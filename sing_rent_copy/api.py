from fastapi import FastAPI, Depends

import sqlalchemy as sa
from sqlalchemy import create_engine, insert
from sqlalchemy.orm import sessionmaker, Session

from models import DevName, ProjectData, SalesData, RentData #, get_db_session, db_init
from db_scrape import engine
import schemas

app = FastAPI()

session1 = sessionmaker(bind=engine, expire_on_commit=False)

def get_table_size(session: sessionmaker, table_class) -> int:
    """
    Get the number of rows a table has
    
    :param session: An SQLAlchemy session
    :param table_class: A class that inherits from `sqlalchemy.Base` and represents a table 
    :return: integer
    """
    with session() as session:
        return session.scalar(sa.select(sa.func.count()).select_from(table_class))


nr_rows = get_table_size(session1, RentData)
print(nr_rows)



# @app.on_event("startup")
# def on_startup():
#     db_init()


@app.get('/')
def root():
    return {'hello':'world'}


# @app.get("/get")
# def get_proj(db=session): #Depends(get_db_session)):
#     return db.query(DevName).first().dict_out()



# def project(db: Session, proj_name:str):
#     return db.query(models.DevName).filter(models.DevName.dev_name == proj_name).first()





