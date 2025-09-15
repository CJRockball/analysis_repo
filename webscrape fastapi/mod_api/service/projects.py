

class Project:
    def __init__(self, proj_list): #dev_id, dev_name, address, district, nhood, proj_size, built, tenure, units, blocks, floors, bedrooms, developer):
        # self.dev_name = dev_name
        # self.dev_id = dev_id
        # self.address = address
        # self.district = district
        # self.nhood = nhood
        # self.proj_size = proj_size
        # self.built = built
        # self.tenure = tenure 
        # self.units = units
        # self.blocks = blocks
        # self.floors = floors
        # self.bedrooms = bedrooms
        # self.developer = developer
        self.proj_info = proj_list
        
    def add_rent_data(self, rent_list):
        self.rent_data = rent_list
        return
    
    def add_sales_data(self, sales_list):
        self.sales_data = sales_list
        
        
        
    def dict(self):
        return {
            # 'dev_name': self.dev_name,
            # 'dev_id':self.dev_id,
            # 'address':self.address,
            # 'district':self.district,
            # 'nhood':self.nhood,
            # 'proj_size':self.proj_size,
            # 'built':self.built,
            # 'tenure':self.tenure,
            # 'units':self.units,
            # 'blocks':self.blocks,
            # 'floors':self.floors,
            # 'bedrooms':self.bedrooms,
            # 'developer':self.developer,
            'proj_info':self.proj_info,
            'rent_data':self.rent_data,
            'sales_data':self.sales_data
                }      
        
    def dict_s(self):
        return {
            'proj_info':self.proj_info
                }      
        