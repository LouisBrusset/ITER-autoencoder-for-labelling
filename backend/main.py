import os
# import warnings
# import pkg_resources

# Conf os + warnings
# os.environ['PYTORCH_NVFUSER_DISABLE'] = '1'
# os.environ['PYTORCH_NVFUSER_DISABLE_FALLBACK'] = '1'
# warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
# warnings.filterwarnings("ignore", category=FutureWarning, module="pkg_resources")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import data_router, checking_router, training_router, visualization_router


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
os.makedirs("models/saved", exist_ok=True)

app.include_router(data_router.router)
app.include_router(checking_router.router)
app.include_router(training_router.router)
app.include_router(visualization_router.router)


@app.get("/")
async def root():
    return {"message": "Autoencoder Training API is running!"}


