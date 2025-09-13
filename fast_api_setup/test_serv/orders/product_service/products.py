import requests
from test_serv.orders.product_service.exceptions import ProductNotFoundError, APIIntegrationError, InvalidActionError


class Product:
    def __init__(self, product_id, name, store, product_=None):
        self._product = product_
        self.product_id = product_id
        self.name = name
        self.store = store
        
        
    def dict(self):
        return {
            'product_id':self.product_id,
            'name': self.name,
            'store':self.store
        }