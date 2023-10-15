import flask
import json
from typing import Optional
from pydantic import BaseModel

app = flask.Flask(__name__)

class Address(BaseModel):
    street: str
    country: str = "India"
    zipcode: int

class Person(BaseModel):
    first_name: str
    last_name: Optional[str] 
    address: Optional[Address]

@app.route('/')
def index():
    return 'Hello World! <a href="/api">/api</a>'


@app.route('/api', methods=['POST'])
def api():
    data = flask.request.json or {"message": "Hello World!"}
    person = Person(**data)
    print(f"Here - {person}") 
    return person.model_dump_json()


if __name__ == '__main__':
    app.run(debug=True)
