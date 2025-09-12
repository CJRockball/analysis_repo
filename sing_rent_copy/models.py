from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, Numeric, String, ForeignKey, DateTime
#from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import sessionmaker, relationship

from db_scrape import Base



class DevName(Base):
    __tablename__ = 'dev_names'
    
    dev_id = Column(Integer(), autoincrement=True, primary_key=True)
    dev_name = Column(String(), nullable=False, unique=True)
    
    def dict_out(self):
        return {'dev_id': self.dev_id, 'dev_name': self.dev_name}
    

class ProjectData(Base):
    __tablename__ = 'project_datas'
    
    proj_id     = Column(Integer(), ForeignKey('dev_names.dev_id'), primary_key=True)
    address      = Column(String(200))
    district    = Column(String(3))
    nhood       = Column(String(200))
    proj_size   = Column(String(50))
    built       = Column(String(50))
    tenure      = Column(String(50))
    units       = Column(Integer())
    blocks      = Column(Integer())
    floors      = Column(Integer())
    bedrooms    = Column(String(50))
    developer   = Column(String(200))

    # Establish one to one relationship
    dev_name = relationship('DevName', uselist=False)

    def dict_out(self):
        return {
        'proj_id': self.proj_id,
        'address': self.address,
        'district': self.district,
        'nhood': self.nhood,
        'proj_size': self.proj_size,
        'built': self.built,
        'tenure': self.tenure,
        'units': self.units,
        'blocks': self.blocks,
        'floors': self.floors,
        'bedrooms': self.bedrooms,
        'developer': self.developer,
        }


class SalesData(Base):
    __tablename__ = 'sales_datas'

    sales_id    = Column(Integer(), autoincrement=True, primary_key=True)
    proj_id     = Column(Integer(), ForeignKey('dev_names.dev_id')) 
    date        = Column(String(50))
    block       = Column(Integer())
    beds        = Column(Integer())
    psf         = Column(Integer())
    area        = Column(Integer())
    price       = Column(Integer())

    def dict_out(self):
        return {
            'sales_id': self.sales_id,
            'proj_id': self.proj_id,
            'date' : self.date,
            'block' : self.block,
            'beds' : self.beds,
            'psf' : self.psf,
            'area' : self.area,
            'price' : self.price
        }

class RentData(Base):
    __tablename__ = 'rent_datas'

    rent_id     = Column(Integer(), autoincrement=True, primary_key=True)
    proj_id     = Column(Integer(), ForeignKey('dev_names.dev_id')) 
    date        = Column(String(50))
    beds        = Column(Integer())
    psf         = Column(Integer())
    area        = Column(Integer())
    price       = Column(Integer())

    def dict_out(self):
        return {
            'rent_id': self.rent_id,
            'proj_id': self.proj_id,
            'date' : self.date,
            'beds' : self.beds,
            'psf' : self.psf,
            'area' : self.area,
            'price' : self.price
        }


 
    
