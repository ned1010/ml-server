from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

#import routers 
from .routers import ingest
from .routers import trychat
from .routers import streamchat



app = FastAPI()

#middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}
    

@app.get("/api/ml/health")
async def health():
    return {"message": "ML server is running OK"}


#Include routers 
app.include_router(ingest.router, prefix='/api/ml', tags=['/ingest'])
app.include_router(trychat.router, prefix='/api/ml', tags=['/chat'])
app.include_router(streamchat.router, prefix='/api/ml', tags=['/streamchat'])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
