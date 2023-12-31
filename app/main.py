from fastapi import FastAPI
# from app.api.v1.routes import router as api_v1_router
from app.api.v1.router import router as api_v1_router

app = FastAPI()

app.include_router(api_v1_router, prefix="/api/v1")
