from fastapi import FastAPI
from mod_api.repo.unit_of_work import UnitOfWork
from mod_api.repo.project_repository import ProjectRepository
from mod_api.service.project_service import ProjectService
from mod_api.web.api.schema import ProjectName, ProjectData, DataOut, ProjectList

app = FastAPI()

@app.get('/')
def root():
    return {'hello':'world'}

@app.get('/get_project/{name}') #, response_model=DataOut)
def get_projects(name:str):
    """Returns all project data"""
    # Inject database
    with UnitOfWork() as unit_of_work:
        # Inst repository
        repo = ProjectRepository(unit_of_work.session)
        # Inst service
        project_service = ProjectService(repo)
        # Use service obj get_data, Call repository get data, 
        # get data from db, inst data holding obj, return data holding obj
        data = project_service.get_project(name)
        # Use data obj dict method to return to api
    return data.dict()

@app.get('/get_project_info/{name}') #, response_model=ProjectList)
def get_projects(name:str):
    """ Returns just the project info"""
    # Inject database
    with UnitOfWork() as unit_of_work:
        # Inst repository
        repo = ProjectRepository(unit_of_work.session)
        # Inst service
        project_service = ProjectService(repo)
        # Use service obj get_data, Call repository get data, 
        # get data from db, inst data holding obj, return data holding obj
        data = project_service.get_project_info(name)
        # Use data obj dict method to return to api
    return data.dict_s()


