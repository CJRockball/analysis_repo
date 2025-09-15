from pydantic import BaseModel
from typing import List

class DevProjectName(BaseModel):
    dev_name: str

class ProjectName(DevProjectName):
    dev_id: int

class ProjectNames(BaseModel):
    projects: List[ProjectName]

class ProjectData(ProjectName):
    address: str
    district: str
    nhood: str
    proj_size: str
    built: str
    tenure: str
    units: int
    blocks: int
    floors: int
    bedrooms: str
    developer: str
    
class ProjectList(ProjectData):
    proj_info: List[ProjectData]
        
class SalesData(BaseModel):
    sales_id: int
    proj_id: int
    date: str
    block: int
    beds: int
    psf: float
    area: int
    price: int
        
class SalesDatas(SalesData):
    sales_data: List[SalesData]
        

class RentData(BaseModel):
    rent_id: int
    proj_id: int
    date: str
    beds: int
    psf: float
    area: int
    price: int
    
class RentDatas(RentData):
    rent_data: List[RentData]


class DataOut(ProjectList): #ProjectData):
    rent_data: List[RentData]
    sales_data: List[SalesData]
      
        
        