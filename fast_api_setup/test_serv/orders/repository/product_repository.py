from test_serv.orders.repository.models import ProductModel
from test_serv.orders.product_service.products import Product

class ProductRepository:
    def __init__(self, session):
        self.session = session
        
    def add(self, items):
        record = ProductModel(**items[0])
        self.session.add(record)
        return Product(**record.dict(), product_=record)

    def _get(self, id_):
        return self.session.query(ProductModel).filter(ProductModel.product_id == int(id_)).first()

    def get(self, id_):
        product = self._get(id_)
        if product is not None:
            return Product(**product.dict())

    def list(self, limit=None, **filters):
        query = self.session.query(ProductModel)
        records = query.filter_by(**filters).limit(limit).all()
        return [Product(**record.dict()) for record in records]
        
    def update(self, id_, **payload):
        record = self._get(id_) #get db data
        
        # if 'items' in payload:
        #     self.session.delete(**record.dict())
        #     record = payload['items']

        #print(record.dict())
        #print(payload['items'])
        
        for key,value in payload['items'][0].items():
            #print(key, value)
            setattr(record, key, value)
        return Product(**record.dict())  
        
    def delete(self, id_):
        self.session.delete(self._get(id_))
        
        