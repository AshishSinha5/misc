import uvicorn as uvicorn
import fastapi
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()


class Address(BaseModel):
    street: str
    country: str = "India"
    zipcode: int

class Person(BaseModel):
    first_name: str
    last_name: Optional[str] 
    address: Optional[Address]

@app.get("/")
def index():
    response = fastapi.responses.HTMLResponse(content="<h1>Hello World! <a href='/api'>/api</a></h1>")
    return response

@app.post("/api")
def api(person : Person):
    print(f"Here - {person}")
    return person.address

if __name__ == '__main__':
    uvicorn.run(app)
    