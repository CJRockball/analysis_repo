from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

DATABASE_URL = 'sqlite:///scrape.db'

engine = create_engine(DATABASE_URL, connect_args={'check_same_thread': False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db_session():
    with SessionLocal() as session:
        yield session


class Base(DeclarativeBase):
    pass


def db_init():
    Base.metadata.create_all(SessionLocal)
    return