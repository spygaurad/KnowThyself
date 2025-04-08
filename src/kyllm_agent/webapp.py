from fastapi import FastAPI

app = FastAPI()

@app.get("/get_model")
def read_root():
    return {"Hello": "World"}