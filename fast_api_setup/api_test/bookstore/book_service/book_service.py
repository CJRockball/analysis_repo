from api_test.bookstore.repo.book_repository import BookRepository
from api_test.bookstore.book_service.exceptions import BookNotFoundError

class BookService:
    def __init__(self, book_repository:BookRepository):
        self.book_repository = book_repository

    def list_books(self, **filters):
        limit = filters.pop('limit', None)
        return self.book_repository.list(limit, **filters)

    def add_book_item(self, item):
        return self.book_repository.add(item)
    
    def update_book(self, book_id, **payload):
        book = self.book_repository.get(book_id)
        if book is None:
            raise BookNotFoundError(f'Book with ID {book_id} not found')
        return self.book_repository.update(book_id, **payload)    
    
    def delete_book(self, book_id):
        book = self.book_repository.get(book_id)
        if book is None:
            raise BookNotFoundError(f'Book with ID {book_id} not found')
        return self.book_repository.delete(book_id)
    



