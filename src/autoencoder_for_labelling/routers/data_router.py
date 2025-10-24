import os

from fastapi import APIRouter, HTTPException, UploadFile, File

import numpy as np
import pandas as pd

from autoencoder_for_labelling.data.dataset import generate_synthetic_data

router = APIRouter()


@router.get("/data-options")
async def get_data_options():
    """Give available data options"""
    uploaded_files = os.listdir("data/uploaded")
    synthetic_files = os.listdir("data/synthetic")
    return {
        "uploaded_files": uploaded_files, 
        "synthetic_files": synthetic_files
    }


@router.get("/current-dataset")
async def get_current_dataset():
    """Yield the currently loaded dataset"""
    try:
        if os.path.exists("data/current_dataset.npz"):
            data = np.load("data/current_dataset.npz")
            response_data = {
                "loaded": True, 
                "filename": str(data.get('filename', 'current_dataset.npz'))
            }
            dataset = data['data']
            response_data["shape"] = list(dataset.shape) if hasattr(dataset, 'shape') else "Unknown"
            return response_data
        return {"loaded": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load-dataset")
async def load_dataset(filename: str, data_type: str = "synthetic"):
    """Load a specific dataset file as the current dataset"""
    try:
        if data_type == "synthetic":
            filepath = f"data/synthetic/{filename}"
        else:
            filepath = f"data/uploaded/{filename}"
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="File not found in directory")

        # Load the data
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
            data = df.values
            labels = np.zeros(len(data))
        else:
            loaded = np.load(filepath)
            if 'data' in loaded:
                data = loaded['data']
            elif 'arr_0' in loaded:
                data = loaded['arr_0']
            else:
                data = loaded[loaded.files[0]]
            labels = loaded['labels'] if 'labels' in loaded else np.zeros(len(data))

        # Type conversion safe for serialization
        data_for_save = data.copy() if hasattr(data, 'copy') else data
        labels_for_save = labels.copy() if hasattr(labels, 'copy') else labels
        
        np.savez(
            "data/current_dataset.npz", 
            data=data_for_save, 
            labels=labels_for_save, 
            filename=filename
        )

        response_data = {
            "status": "success", 
            "filename": filename
        }
        if hasattr(data, 'shape'):
            response_data["data_shape"] = list(data.shape)
        if hasattr(labels, 'shape'):
            response_data["labels_shape"] = list(labels.shape)

        return response_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete-dataset")
async def delete_dataset(filename: str, data_type: str = "synthetic"):
    """Delete a specific dataset file"""
    try:
        if data_type == "synthetic":
            filepath = f"data/synthetic/{filename}"
        else:
            filepath = f"data/uploaded/{filename}"
        
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="No file found in directory")
        
        os.remove(filepath)

        # If it was the current dataset, remove that as well
        if os.path.exists("data/current_dataset.npz"):
            current_data = np.load("data/current_dataset.npz")
            if current_data.get('filename', '') == filename:
                os.remove("data/current_dataset.npz")
        
        return {
            "status": "success", 
            "filename": filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create-synthetic-data")
async def create_synthetic_data(n_samples: int = 1000, input_dim: int = 20, n_anomalies: int = 50):
    """Create synthetic data and save to file"""
    try:
        train_tensor, _ = generate_synthetic_data(n_samples=n_samples, input_dim=input_dim, n_anomalies=n_anomalies)
        all_data = train_tensor.numpy()
        labels = np.zeros(n_samples)
        labels[-n_anomalies:] = 1
        
        filename = f"synthetic_data_{n_samples}_{input_dim}.npz"
        filepath = f"data/synthetic/{filename}"
        np.savez(filepath, data=all_data, labels=labels)
        
        return {
            "status": "success", 
            "filename": filename, 
            "data_shape": all_data.shape, 
            "anomalies_count": n_anomalies
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    """Upload a file and save to uploaded data directory"""
    try:
        if not file.filename.endswith(('.csv', '.npz')):
            raise HTTPException(status_code=400, detail="Only CSV and NPZ files are supported")
        filepath = f"data/uploaded/{file.filename}"
        
        content = await file.read()
        with open(filepath, "wb") as f:
            f.write(content)
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(filepath)
            data_shape = df.shape
        else:
            data = np.load(filepath)
            data_shape = data['data'].shape if 'data' in data else data.shape
        
        return {
            "status": "success", 
            "filename": file.filename, 
            "data_shape": data_shape
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
