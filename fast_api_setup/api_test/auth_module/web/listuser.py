

class ListUser:
    def __init__(self, id, email, password, hashed_password): #, user_=None):
        #self._book = user_
        self.id = id
        self.email = email
        self.password = password
        self.hashed_password = hashed_password
        
        
    def dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'password': self.password,
            'hashed_password':self.hashed_password
        }