from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQL_URL = 'sqlite:///c:\\Users\\PatCa\\Documents\\PythonScripts\\fast_api_setup\\api_test\\db_setup\\book_db.sqlite'

engine = create_engine(SQL_URL, connect_args={'check_same_thread':False})

ses = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()