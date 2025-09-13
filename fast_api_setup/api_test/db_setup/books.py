

class Book:
    def __init__(self, id, title, author, genre, height, publisher):
        self.id = id
        self.title = title
        self.author = author
        self.genre = genre
        self.height = height
        self.publisher = publisher
        
        
    def dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'author': self.author,
            'genre': self.genre,
            'height': self.height,
            'publisher': self.publisher
        }
