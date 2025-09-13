from api_test.bookstore.repo.models import Books
from api_test.bookstore.book_service.books import Book

class BookRepository:
    def __init__(self, session):
        self.session = session
    
    def get(self, id_):
        book = self.session.query(Books).filter(Books.id == int(id_)).first()
        if book is not None:
            return Book(**book.dict())
    
        
    def add(self, item):
        record = Books(**item[0])
        self.session.add(record)
        return Book(**record.dict(), book_=record)
    
    def list(self, limit=None, **filters):
        query = self.session.query(Books)
        records = query.filter_by(**filters).limit(limit).all()
        return [Book(**record.dict()) for record in records]
    
    def update(self, id_, **payload):
        record = self.session.query(Books).filter(Books.id == int(id_)).first()

        for key, value in payload['items'][0].items():
            setattr(record, key, value)
        return Book(**record.dict())
    
    def delete(self, id_):
        self.session.delete(self.session.query(Books).filter(Books.id == int(id_)).first())

