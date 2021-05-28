from fastapi import FastAPI
from mangum import Mangum

from api.v1.api import router as api_router

app = FastAPI(title="Serverless Lambda REST FastAPI")
app.include_router(api_router, prefix="/api/v1")

@app.get("/", tags=["Endpoint Test"])
def main_endpoint_test():
    return {"message": "Welcome to the root endpoint"}


handler = Mangum(app=app)