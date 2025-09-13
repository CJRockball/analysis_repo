from typing import Optional

from fastapi import HTTPException
from starlette import status
from starlette.responses import Response

from test_serv.orders.product_service.exceptions import ProductNotFoundError
from test_serv.orders.repository.product_repository import ProductRepository
from test_serv.orders.product_service.product_service import ProductService
from test_serv.orders.web.app import app
from test_serv.orders.web.api.schemas import GetProductsSchema, CreateProductSchema, ProductItemSchema
from test_serv.orders.repository.unit_of_work import UnitOfWork

@app.get('/products', response_model=GetProductsSchema)
def get_products(limit:Optional[int]=None):
    with UnitOfWork() as unit_of_work:
        repo = ProductRepository(unit_of_work.session)
        product_service = ProductService(repo)
        results = product_service.list_products(limit=limit)
        return {'products': [result.dict() for result in results]}
        

@app.post('/products', status_code=status.HTTP_201_CREATED, response_model=ProductItemSchema) #GetProductsSchema)
def add_product(payload: CreateProductSchema):
    with UnitOfWork() as unit_of_work:
        repo = ProductRepository(unit_of_work.session)
        product_service = ProductService(repo)
        product = payload.dict()['product']
        product = product_service.add_product_item(product)
        unit_of_work.commit()
        return_payload = product.dict()
        return return_payload

@app.get('/products/{product_id}', response_model=ProductItemSchema)
def get_product(product_id: int):
    try:
        with UnitOfWork() as unit_of_work:
            repo = ProductRepository(unit_of_work.session)
            product_service = ProductService(repo)
            product = product_service.get_product(product_id=product_id)
        return product.dict()
    except ProductNotFoundError:
        raise HTTPException(status_code=404, detail=f'Product with ID {product_id} not found')

@app.put('/products/{product_id}', response_model=ProductItemSchema)
def update_product(product_id: int, product_details:CreateProductSchema):
    try:
        with UnitOfWork() as unit_of_work:
            repo = ProductRepository(unit_of_work.session)
            product_service = ProductService(repo)
            product = product_details.dict()['product']
            product = product_service.update_product(product_id=product_id, items=product)
            unit_of_work.commit()
        return product.dict()
    except ProductNotFoundError:
        raise HTTPException(status_code=404, detail=f'Product with ID {product_id} not found')

@app.delete('/product/{prouct_id}', status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete_product(product_id:int):
    try:
        with UnitOfWork() as unit_of_work:
            repo = ProductRepository(unit_of_work.session)
            product_service = ProductService(repo)
            product_service.delete_product(product_id=product_id)
            unit_of_work.commit()
        return
    except ProductNotFoundError:
        raise HTTPException(status_code=404, detail=f'Product with ID {product_id} not found')




