import uuid
from datetime import datetime

from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class ProductModel(Base):
    __tablename__ = 'product'
    
    product_id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    store = Column(Integer, nullable=False)
    
    def dict(self):
        return {
            'product_id': self.product_id,
            'name': self.name,
            'store': self.store
        }



