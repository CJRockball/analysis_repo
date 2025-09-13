from fastapi import FastAPI, Depends, status
from starlette.responses import Response

import api_test.db_setup.models as models
from api_test.db_setup.db import ses, engine
from api_test.db_setup.schemas import CreateBookSchema, BookSchema, GetBookSchema
from api_test.db_setup.books import Book
import pathlib

PROJECT_PATH = pathlib.Path(__file__).resolve().parent
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

def get_db():
    db = ses()
    try:
        yield db
    finally:
        db.close()
    
@app.get('/')
def home():
    return {'hello':'world'}


@app.post('/add', status_code=status.HTTP_201_CREATED, response_model=BookSchema)
def add_book(payload: CreateBookSchema, db:ses=Depends(get_db)):
    record = payload.dict()['product']
    #new_book = models.Books(id=1, title='New Book', author='Steve Banana', genre='Horror', height=233, publisher='Waco Book')
    new_book = models.Books(**record[0])
    db.add(new_book)
    db.commit()
    return_payload = new_book.dict()
    return return_payload


@app.get('/all', response_model=GetBookSchema)
def get_all(db: ses = Depends(get_db)):
    records = db.query(models.Books).all()
    results =  [Book(**record.dict()) for record in records]
    return {'books': [result.dict() for result in results]}


@app.put('/update/{book_id}', response_model=BookSchema)
def update(book_id:int, book_details:CreateBookSchema, db:ses=Depends(get_db)):
    new_book = book_details.dict()['product']
    record = db.query(models.Books).filter(models.Books.id == book_id).first()   

    for key,value in new_book[0].items():
        print(key, value)
        setattr(record, key, value)
    db.commit()
    result = Book(**record.dict())     

    return result.dict()

    
@app.delete('/delete/{book_id}', status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete_book(book_id:int, db:ses=Depends(get_db)):
    del_book = db.query(models.Books).filter(models.Books.id == book_id).first()
    db.delete(del_book)
    db.commit()
    return
    





