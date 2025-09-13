from pydantic import BaseModel
from typing import List

class BookSchema(BaseModel):
    title: str
    author: str
    genre: str
    height: int
    publisher: str 

class DisplayBookSchema(BookSchema):
    id: int
 
class CreateBookSchema(BaseModel):
    product: List[BookSchema]

class GetBookSchema(BaseModel):
    books: List[DisplayBookSchema]
 
    