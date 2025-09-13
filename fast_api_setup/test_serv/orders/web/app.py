from fastapi import FastAPI

app = FastAPI(debug=True)

from test_serv.orders.web.api import api