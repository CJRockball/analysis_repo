from pydantic import BaseModel, EmailStr
from typing import List


class BaseUser(BaseModel):
    email: EmailStr
    
    class Config:
        orm_mode = True
        
class CreateUser(BaseUser):
    password: str
    
class User(CreateUser):
    id: int
    
class UserDB(BaseUser):
    hashed_password: str
    
    
class GetUserSchema(BaseModel):
    users: List[UserDB]