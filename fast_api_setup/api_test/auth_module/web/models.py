
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from api_test.auth_module.web.database import Base

#Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String, unique=True, index=True)
    hashed_password = Column(String, unique=True)

    def dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'password': self.password,
            'hashed_password': self.hashed_password
        }



