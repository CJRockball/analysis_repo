from pydantic import BaseModel, conlist
from typing import List


class ProductItemSchema(BaseModel):
    product_id: int
    name: str
    store: int
    
class CreateProductSchema(BaseModel):
    product: conlist(ProductItemSchema)

class GetProductSchema(ProductItemSchema):
    id: int

class GetProductsSchema(BaseModel):
    products: List[ProductItemSchema]    

