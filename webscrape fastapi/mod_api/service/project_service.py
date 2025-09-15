from mod_api.repo.project_repository import ProjectRepository
from mod_api.service.exceptions import ProjectNotFoundError

class ProjectService:
    def __init__(self, project_repository: ProjectRepository):
        self.project_repository = project_repository
    
    def get_project(self, project_name):
        project = self.project_repository.get(project_name)
        if project is not None:
            return project
        raise ProjectNotFoundError(f'Order with id {project_name} not found')
    
    def get_project_info(self, project_name):
        project = self.project_repository.get_project_info(project_name)
        if project is not None:
            return project
        raise ProjectNotFoundError(f'Order with id {project_name} not found')
