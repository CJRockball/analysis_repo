from pydantic import BaseModel
from typing import List

class DevProjectName(BaseModel):
    dev_name: str

class ProjectName(DevProjectName):
    dev_id: int

class ProjectNames(BaseModel):
    projects: List[ProjectName]

