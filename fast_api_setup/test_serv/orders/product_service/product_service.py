from test_serv.orders.product_service.exceptions import ProductNotFoundError
from test_serv.orders.repository.product_repository import ProductRepository


class ProductService:
    def __init__(self, product_repository: ProductRepository):
        self.product_repository = product_repository

    def get_product(self, product_id):
        product = self.product_repository.get(product_id)
        if product is not None:
            return product
        raise ProductNotFoundError(f'Order with id {product_id} not found')

    def list_products(self, **filters):
        limit = filters.pop('limit', None)
        return self.product_repository.list(limit, **filters)
    
    def add_product_item(self,item):
        return self.product_repository.add(item)


    def update_product(self, product_id, **payload):
        product = self.product_repository.get(product_id)
        if product is None:
            raise ProductNotFoundError(f'Product with ID {product_id} not found')
        return self.product_repository.update(product_id, **payload)
    
    def delete_product(self, product_id):
        product = self.product_repository.get(product_id)
        if product is None:
            raise ProductNotFoundError(f'Product with ID {product_id} not found')
        return self.product_repository.delete(product_id)