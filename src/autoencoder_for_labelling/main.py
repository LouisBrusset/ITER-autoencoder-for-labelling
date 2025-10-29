import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import data_router, checking_router, training_router, visualization_router, labels_router


# === GLOBAL SETUP ===

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("data/uploaded", exist_ok=True)
os.makedirs("data/synthetic", exist_ok=True)
os.makedirs("models/saved_model", exist_ok=True)
os.makedirs("models/saved_architechture", exist_ok=True)
os.makedirs("results/labels", exist_ok=True)
os.makedirs("results/latents", exist_ok=True)
os.makedirs("results/projections2d", exist_ok=True)
os.makedirs("results/reconstructions", exist_ok=True)

app.include_router(data_router.router)
app.include_router(checking_router.router)
app.include_router(training_router.router)
app.include_router(visualization_router.router)
app.include_router(labels_router.router)


@app.get("/")
async def root():
    return {"message": "Autoencoder Training API is running!"}


