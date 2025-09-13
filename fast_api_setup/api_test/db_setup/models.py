from sqlalchemy import Column, Integer, String
from api_test.db_setup.db import Base

class Books(Base):
    __tablename__ = 'book'

    id = Column(Integer, primary_key=True)
    title = Column(String)
    author = Column(String)
    genre = Column(String)
    height = Column(Integer)
    publisher = Column(String) 
    
    def dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'author': self.author,
            'genre': self.genre,
            'height': self.height,
            'publisher': self.publisher
        }


