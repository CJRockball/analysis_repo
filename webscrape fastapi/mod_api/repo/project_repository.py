from models import DevName, ProjectData, SalesData, RentData
from mod_api.service.projects import Project


class ProjectRepository:
    def __init__(self,session):
        self.session = session

    def _get_proj_nr(self, proj_name):
        developer_id = self.session.query(DevName).filter(DevName.dev_name == proj_name).first().dict_out()
        return developer_id
    
    def get(self, proj_name):
        developer_id = self._get_proj_nr(proj_name)
        dev_nr = developer_id['dev_id']
        if developer_id is not None:
            project_data = self.session.query(ProjectData).filter(ProjectData.proj_id == dev_nr).first().dict_out()
            
            del project_data['proj_id']
            data_dict = {**developer_id, **project_data}
            data_list = [data_dict]
            print(data_list)
            project_object = Project(data_list) #**data_dict)
            
            rent_data =  self.session.query(RentData).filter(RentData.proj_id  == dev_nr).all()
            rent_data_list = []
            for item in rent_data:
                rent_data_list.append(item.dict_out())
            project_object.add_rent_data(rent_data_list)
            
            sales_data = self.session.query(SalesData).filter(SalesData.proj_id  == dev_nr).all()
            sales_data_list = []
            for item in sales_data:
                sales_data_list.append(item.dict_out())
            project_object.add_sales_data(sales_data_list)
            
            return project_object
        
        
    def get_project_info(self, proj_name):
        developer_id = self._get_proj_nr(proj_name)
        dev_nr = developer_id['dev_id']
        if developer_id is not None:
            project_data = self.session.query(ProjectData).filter(ProjectData.proj_id == dev_nr).first().dict_out()
            
            del project_data['proj_id']
            data_dict = {**developer_id, **project_data}
            data_list = [data_dict]
            
            project_object = Project(data_list)
            print(project_object.dict_s())
            
            return project_object       