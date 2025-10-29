import os

from fastapi import APIRouter, HTTPException

import numpy as np
import torch


router = APIRouter()


@router.get("/latent-space")
async def get_latent_space(subset: str | None = 'validation', deterministic: bool = True):
    """
    Compute and return 2D UMAP projection of the latent space for the current dataset and model.

    Query parameter `subset` controls which samples are used for projection:
        - 'train' or 'training' : use the training subset
        - 'validation' or 'val' : use the validation subset (default)
        - 'all' : use the entire dataset
    """
    try:
        if not os.path.exists("models/current_model.pth") or not os.path.exists("data/current_dataset.npz"):
            raise HTTPException(status_code=400, detail="First load a dataset and a model")
        
        data = np.load("data/current_dataset.npz", allow_pickle=True)
        all_data = data['data']

        # Determine train/validation indices; legacy support: if no is_train assume all training
        if 'is_train' in data:
            is_train = np.array(data['is_train'], dtype=bool)
        else:
            is_train = np.ones(len(all_data), dtype=bool)
        train_indices = np.where(is_train)[0]
        val_indices = np.where(~is_train)[0]

        # choose subset
        s = (subset or 'validation').lower()
        if s in ('validation', 'val'):
            if len(val_indices) == 0:
                raise HTTPException(status_code=400, detail='No validation samples available — create dataset with a validation split')
            dataset = torch.FloatTensor(all_data[val_indices])
            working_indices = val_indices
        elif s in ('train', 'training'):
            if len(train_indices) == 0:
                raise HTTPException(status_code=400, detail='No training samples available')
            dataset = torch.FloatTensor(all_data[train_indices])
            working_indices = train_indices
        elif s in ('all', 'global', 'dataset'):
            dataset = torch.FloatTensor(all_data)
            working_indices = np.arange(len(all_data))
        else:
            raise HTTPException(status_code=400, detail=f"Unknown subset '{subset}'. Use 'train', 'validation' or 'all'.")

        # Determine labels and latent/projection data strictly from the "current_*" files.
        # If any current file is missing or incompatible with the requested subset, return an error.
        labels = None
        latent_representations = None
        latent_2d = None

        # Helper to map saved arrays by provided indices into the requested working_indices order
        def map_by_indices(meta, key='data'):
            if key not in meta:
                return None
            arr = np.array(meta[key])
            if 'indices' in meta:
                src_idx = np.array(meta['indices'], dtype=int)
                pos = {int(v): i for i, v in enumerate(src_idx)}
                positions = []
                for gi in working_indices:
                    if int(gi) not in pos:
                        return None
                    positions.append(pos[int(gi)])
                return arr[positions]
            else:
                # accept only if shapes match full dataset or the working subset
                if arr.shape[0] == len(all_data):
                    return arr[working_indices]
                if arr.shape[0] == len(working_indices):
                    return arr
                return None

        # Require current projection and latents to be present and mappable
        cur_proj = os.path.join('results', 'current_projection2d.npz')
        cur_lat = os.path.join('results', 'current_latents.npz')
        if not os.path.exists(cur_proj) or not os.path.exists(cur_lat):
            raise HTTPException(status_code=400, detail='current_projection2d.npz and current_latents.npz must be present in results and compatible with the requested subset')

        # load and map projection
        meta_proj = np.load(cur_proj, allow_pickle=True)
        mapped_proj = map_by_indices(meta_proj, 'data')
        if mapped_proj is None:
            raise HTTPException(status_code=400, detail='current_projection2d.npz is not compatible with the requested subset or indices are missing')
        latent_2d = np.array(mapped_proj)

        # extract labels from projection or latents (require at least one)
        proj_labels_mapped = None
        if 'labels' in meta_proj:
            proj_labels_mapped = map_by_indices(meta_proj, 'labels')

        meta_lat = np.load(cur_lat, allow_pickle=True)
        mapped_lat = map_by_indices(meta_lat, 'data')
        if mapped_lat is None:
            raise HTTPException(status_code=400, detail='current_latents.npz is not compatible with the requested subset or indices are missing')
        latent_representations = np.array(mapped_lat)

        lat_labels_mapped = None
        if 'labels' in meta_lat:
            lat_labels_mapped = map_by_indices(meta_lat, 'labels')

        # labels must come from currents only
        if proj_labels_mapped is not None:
            labels = np.array(proj_labels_mapped)
        elif lat_labels_mapped is not None:
            labels = np.array(lat_labels_mapped)
        else:
            raise HTTPException(status_code=400, detail='No labels found in current_projection2d.npz or current_latents.npz')

        # Prepare data for Chart.js, include latent vector if available
        points = []
        for i_local, (x, y) in enumerate(latent_2d):
            global_i = int(working_indices[i_local])
            pt = {
                "x": float(x),
                "y": float(y),
                "label": int(labels[i_local]) if i_local < len(labels) else 0,
                "original_index": global_i
            }
            if latent_representations is not None and i_local < latent_representations.shape[0]:
                try:
                    pt['latent'] = latent_representations[i_local].tolist()
                except Exception:
                    pt['latent'] = None
            points.append(pt)

        return {
            "points": points,
            "latent_dim": int(latent_representations.shape[1]) if latent_representations is not None else None,
            "original_dim": int(dataset.shape[1])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reconstruct")
async def reconstruct_sample(index: int = 0, subset: str | None = None):
    """Return original and reconstructed data for a given sample index.

    Parameters:
    - index: integer. Interpretation depends on `subset`:
        * subset='val' -> index is taken relative to the validation subset (0..n_val-1)
        * subset='train' -> index is taken relative to the training subset (0..n_train-1)
        * subset omitted -> index is treated as a global dataset index (0..n-1)
    """
    try:
        # Check files
        if not os.path.exists("data/current_dataset.npz"):
            raise HTTPException(status_code=400, detail="No dataset loaded")
        if not os.path.exists("models/current_model.pth"):
            raise HTTPException(status_code=400, detail="No model loaded")
        
        # Load data and determine indices
        data = np.load("data/current_dataset.npz", allow_pickle=True)
        all_data = data['data']
        if 'is_train' in data:
            is_train = np.array(data['is_train'], dtype=bool)
        else:
            is_train = np.ones(len(all_data), dtype=bool)

        train_indices = np.where(is_train)[0]
        val_indices = np.where(~is_train)[0]

        # Interpret index depending on requested subset
        if subset == 'val':
            if len(val_indices) == 0:
                raise HTTPException(status_code=400, detail='No validation samples available — create dataset with a validation split')
            if index < 0 or index >= len(val_indices):
                raise HTTPException(status_code=400, detail=f"Index out of range (0..{len(val_indices)-1}) for validation set")
            global_idx = int(val_indices[index])
        elif subset == 'train':
            if len(train_indices) == 0:
                raise HTTPException(status_code=400, detail='No training samples available')
            if index < 0 or index >= len(train_indices):
                raise HTTPException(status_code=400, detail=f"Index out of range (0..{len(train_indices)-1}) for training set")
            global_idx = int(train_indices[index])
        else:
            # treat index as a global index
            n_samples = all_data.shape[0]
            if index < 0 or index >= n_samples:
                raise HTTPException(status_code=400, detail=f"Index out of range (0..{n_samples-1})")
            global_idx = int(index)

        # Only use current_reconstructions.npz; fail if missing or incompatible
        cur_recon = os.path.join('results', 'current_reconstructions.npz')
        if not os.path.exists(cur_recon):
            raise HTTPException(status_code=400, detail='current_reconstructions.npz must be present in results and compatible with the requested subset')
        try:
            rm = np.load(cur_recon, allow_pickle=True)
            if 'indices' in rm:
                idxs = np.array(rm['indices'], dtype=int)
                matches = np.where(idxs == int(global_idx))[0]
                if matches.size > 0:
                    reconstruction = rm['data'][int(matches[0])].tolist()
                else:
                    raise HTTPException(status_code=400, detail='Requested sample not found in current_reconstructions.npz')
            else:
                # require the file to cover the full dataset or the working subset length
                if 'data' in rm and int(global_idx) < rm['data'].shape[0]:
                    reconstruction = rm['data'][int(global_idx)].tolist()
                else:
                    raise HTTPException(status_code=400, detail='current_reconstructions.npz is not compatible with the requested subset or indices are missing')
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=500, detail='Failed to read current_reconstructions.npz')

        # original sample from the global dataset
        original = all_data[global_idx].tolist()

        return {
            "index": int(global_idx),
            "original": original,
            "reconstruction": reconstruction,
            "message": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
