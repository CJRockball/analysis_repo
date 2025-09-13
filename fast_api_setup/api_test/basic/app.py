from fastapi import FastAPI, Response, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.templating import Jinja2Templates
import starlette.status as status
from pydantic import BaseModel
import pathlib

PROJECT_PATH = pathlib.Path(__file__).resolve().parent
TEMP_PATH = PROJECT_PATH / "templates"
#STAT_PATH = PROJECT_PATH / "static"
templates = Jinja2Templates(directory=TEMP_PATH)

app = FastAPI()

class User(BaseModel):
    name:str
    pw:str

@app.get('/', )
def index():
    return {"Hello":"World"}


@app.get('/form', response_class=HTMLResponse)
def input_page(request: Request):
    return templates.TemplateResponse('index.html', {'request':request})

@app.post('/return_name')
def return_name(name: str=Form(...)):
    return {'Name':name}

@app.post('/return_data', response_model=User)
def return_data(name:str=Form(...), pw:str=Form(...)):
    return User(name=name, pw=pw)

@app.post('/form1', response_model=User, response_class=HTMLResponse)
def output_page(request:Request, name:str=Form(...), pw:str=Form(...)):
    return templates.TemplateResponse('index.html', {'request':request, 'name':name, 'pw':pw})
    
    
  
    
    
    