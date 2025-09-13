from typing import Optional

from fastapi import FastAPI, Depends, Request, Form, status, HTTPException
from starlette.responses import Response
from sqlalchemy.orm import Session

from api_test.bookstore.web.app import app
from api_test.bookstore.book_service.exceptions import BookNotFoundError
from api_test.bookstore.repo.unit_of_work import UnitOfWork
from api_test.bookstore.repo.book_repository import BookRepository
from api_test.bookstore.book_service.book_service import BookService
from api_test.bookstore.web.api.schemas import CreateBookSchema, BookSchema, DisplayBookSchema, GetBookSchema
from api_test.bookstore.book_service.exceptions import BookNotFoundError



@app.get('/')
def home():
    return {'hello':'world'}


@app.post('/add', status_code=status.HTTP_201_CREATED, response_model=BookSchema)
def add_book(payload: CreateBookSchema):
    with UnitOfWork() as unit_of_work:
        repo = BookRepository(unit_of_work.session)
        book_service = BookService(repo)
        new_book = payload.dict()['product']
        new_book = book_service.add_book_item(new_book)
        unit_of_work.commit()
        
        return_payload = new_book.dict()
        return return_payload


@app.get('/all', response_model=GetBookSchema)
def get_all(limit:Optional[int]=None):
    with UnitOfWork() as unit_of_work:
        repo = BookRepository(unit_of_work.session)
        book_service = BookService(repo)
        results = book_service.list_books(limit=limit)
    return {'books':[result.dict() for result in results]}


@app.put('/update/{book_id}', response_model=BookSchema)
def update(book_id:int, book_details:CreateBookSchema):
    try:
        with UnitOfWork() as unit_of_work:
            repo = BookRepository(unit_of_work.session)
            book_service = BookService(repo)
            book = book_details.dict()['product']
            book = book_service.update_book(book_id=book_id, items=book)
            unit_of_work.commit()
        return book.dict()
    except BookNotFoundError:
        raise HTTPException(status_code=404, detail=f'Book with ID {book_id} not found')
        
        

    
@app.delete('/delete/{book_id}', status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete_book(book_id:int):
    try:
        with UnitOfWork() as unit_of_work:
            repo = BookRepository(unit_of_work.session)
            book_service = BookService(repo)
            book_service.delete_book(book_id=book_id)
            unit_of_work.commit()
        return
    except BookNotFoundError:
        raise HTTPException(status_code=404, detail=f'Book with ID {book_id} not found')

