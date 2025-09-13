from fastapi import FastAPI, Depends, status
from sqlalchemy.orm import Session
from passlib.context import CryptContext

import api_test.auth_module.web.models as models
from api_test.auth_module.web.schemas import CreateUser, User, GetUserSchema, UserDB
from api_test.auth_module.web.database import SessionLocal, engine
from api_test.auth_module.web.listuser import ListUser

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')

def get_password_hash(password:str) -> str:
    return pwd_context.hash(password)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        

@app.get('/')
def home():
    return {'helloo':'world'}
    
@app.post('/add', status_code=status.HTTP_201_CREATED, response_model=UserDB)
def add(data: CreateUser, db:Session=Depends(get_db)):    
    new_user = data.dict()
    hashed_password = get_password_hash(new_user['password'])
    new_user_add = models.User(email=new_user['email'], password=new_user['password'], hashed_password=hashed_password)
    db.add(new_user_add)
    db.commit()
    
    return_payload = new_user_add.dict()
    return return_payload

@app.get('/get_users', response_model=GetUserSchema)
def get_users(db:Session=Depends(get_db)):
    records = db.query(models.User).all()
    results = [ListUser(**record.dict()) for record in records]
    
    return {'users':[result.dict() for result in results]} 




