import os

from fastapi import APIRouter, HTTPException

import numpy as np
import torch
import umap

from autoencoder_for_labelling.models.autoencoder import SimpleAutoencoder

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

        # Determine labels: prefer any labels embedded in saved inference artifacts (projection, latents, reconstructions),
        # otherwise fall back to dataset labels or zeros.
        labels = None
        proj_labels = None
        latents_labels = None
        recon_labels = None

        # Try to find latest projection matching dataset and extract labels
        try:
            proj_dir = os.path.join('results', 'projections2d')
            if os.path.exists(proj_dir):
                pfiles = [os.path.join(proj_dir, f) for f in os.listdir(proj_dir) if f.endswith('.npz')]
                pmatches = []
                for pf in pfiles:
                    try:
                        meta = np.load(pf, allow_pickle=True)
                        dsname = str(meta.get('filename',''))
                        if dsname == str(data.get('filename','current_dataset')):
                            ts = int(meta.get('timestamp', os.path.getmtime(pf))) if 'timestamp' in meta else int(os.path.getmtime(pf))
                            pmatches.append((ts, pf))
                    except Exception:
                        continue
                if pmatches:
                    pmatches.sort(key=lambda x: x[0], reverse=True)
                    chosen_proj = pmatches[0][1]
                    try:
                        meta = np.load(chosen_proj, allow_pickle=True)
                        if 'labels' in meta:
                            proj_labels = np.array(meta['labels'])
                    except Exception:
                        proj_labels = None
        except Exception:
            proj_labels = None

        # Try latents
        try:
            lat_dir = os.path.join('results', 'latents')
            if os.path.exists(lat_dir):
                lfiles = [os.path.join(lat_dir, f) for f in os.listdir(lat_dir) if f.endswith('.npz')]
                lmatches = []
                for lf in lfiles:
                    try:
                        lm = np.load(lf, allow_pickle=True)
                        dsname = str(lm.get('filename',''))
                        if dsname == str(data.get('filename','current_dataset')):
                            lts = int(lm.get('timestamp', os.path.getmtime(lf))) if 'timestamp' in lm else int(os.path.getmtime(lf))
                            lmatches.append((lts, lf))
                    except Exception:
                        continue
                if lmatches:
                    lmatches.sort(key=lambda x: x[0], reverse=True)
                    chosen_lat = lmatches[0][1]
                    try:
                        lm = np.load(chosen_lat, allow_pickle=True)
                        if 'labels' in lm:
                            latents_labels = np.array(lm['labels'])
                    except Exception:
                        latents_labels = None
        except Exception:
            latents_labels = None

        # Try reconstructions
        try:
            recon_dir = os.path.join('results', 'reconstructions')
            if os.path.exists(recon_dir):
                rfiles = [os.path.join(recon_dir, f) for f in os.listdir(recon_dir) if f.endswith('.npz')]
                rmatches = []
                for rf in rfiles:
                    try:
                        rm = np.load(rf, allow_pickle=True)
                        dsname = str(rm.get('filename',''))
                        if dsname == str(data.get('filename','current_dataset')):
                            rts = int(rm.get('timestamp', os.path.getmtime(rf))) if 'timestamp' in rm else int(os.path.getmtime(rf))
                            rmatches.append((rts, rf))
                    except Exception:
                        continue
                if rmatches:
                    rmatches.sort(key=lambda x: x[0], reverse=True)
                    chosen_recon = rmatches[0][1]
                    try:
                        rm = np.load(chosen_recon, allow_pickle=True)
                        if 'labels' in rm:
                            recon_labels = np.array(rm['labels'])
                    except Exception:
                        recon_labels = None
        except Exception:
            recon_labels = None

        # select labels in order: projection > latents > recon > dataset
        if proj_labels is not None:
            labels = proj_labels[working_indices] if len(proj_labels) == len(all_data) else None
        if labels is None and latents_labels is not None:
            labels = latents_labels[working_indices] if len(latents_labels) == len(all_data) else None
        if labels is None and recon_labels is not None:
            labels = recon_labels[working_indices] if len(recon_labels) == len(all_data) else None
        if labels is None:
            if 'labels' in data:
                full_labels = data['labels']
                labels = full_labels[working_indices]
            else:
                labels = np.zeros(len(working_indices), dtype=int)
        
        input_dim = dataset.shape[1]
        state = torch.load('models/current_model.pth')
        encoding_dim = None
        if isinstance(state, dict):
            if 'encoder.2.weight' in state:
                encoding_dim = state['encoder.2.weight'].shape[0]
        if encoding_dim is None:
            encoding_dim = 8
        model = SimpleAutoencoder(input_dim=input_dim, encoding_dim=encoding_dim)
        model.load_state_dict(state)
        model.eval()
        
        # Calculate latent space & UMAP projection to 2D (reuse saved projection/latents if present)
        latent_representations = None
        latent_2d = None

        # attempt to load latest saved projection and latents for this dataset
        try:
            proj_dir = os.path.join('results', 'projections2d')
            chosen_proj = None
            if os.path.exists(proj_dir):
                pfiles = [os.path.join(proj_dir, f) for f in os.listdir(proj_dir) if f.endswith('.npz')]
                pmatches = []
                for pf in pfiles:
                    try:
                        meta = np.load(pf, allow_pickle=True)
                        dsname = str(meta.get('filename',''))
                        if dsname == str(data.get('filename','current_dataset')):
                            ts = int(meta.get('timestamp', os.path.getmtime(pf))) if 'timestamp' in meta else int(os.path.getmtime(pf))
                            pmatches.append((ts, pf))
                    except Exception:
                        continue
                if pmatches:
                    pmatches.sort(key=lambda x: x[0], reverse=True)
                    chosen_proj = pmatches[0][1]

            if chosen_proj is not None:
                meta = np.load(chosen_proj, allow_pickle=True)
                if 'data' in meta:
                    latent_2d = np.array(meta['data'])
                # try to find matching latents file
                lat_dir = os.path.join('results', 'latents')
                if os.path.exists(lat_dir):
                    lfiles = [os.path.join(lat_dir, f) for f in os.listdir(lat_dir) if f.endswith('.npz')]
                    lmatches = []
                    for lf in lfiles:
                        try:
                            lm = np.load(lf, allow_pickle=True)
                            dsname = str(lm.get('filename',''))
                            if dsname == str(data.get('filename','current_dataset')):
                                lts = int(lm.get('timestamp', os.path.getmtime(lf))) if 'timestamp' in lm else int(os.path.getmtime(lf))
                                lmatches.append((lts, lf))
                        except Exception:
                            continue
                    if lmatches:
                        lmatches.sort(key=lambda x: x[0], reverse=True)
                        latents_all = np.load(lmatches[0][1], allow_pickle=True)['data']
                        if latents_all.shape[0] == len(all_data):
                            latent_representations = latents_all[working_indices]
        except Exception:
            latent_representations = None
            latent_2d = None

        # if no saved projection, compute latent representations and UMAP now
        if latent_2d is None:
            with torch.no_grad():
                latent_representations = model.encoder(dataset).numpy()
            if deterministic:
                reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
            else:
                reducer = umap.UMAP(n_components=2, init='spectral', n_neighbors=15, n_jobs=-1, random_state=None)
            latent_2d = reducer.fit_transform(latent_representations)

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

        # Try to return reconstruction from latest saved reconstructions file for this dataset
        reconstruction = None
        try:
            recon_dir = os.path.join('results', 'reconstructions')
            if os.path.exists(recon_dir):
                rfiles = [os.path.join(recon_dir, f) for f in os.listdir(recon_dir) if f.endswith('.npz')]
                rmatches = []
                for rf in rfiles:
                    try:
                        rm = np.load(rf, allow_pickle=True)
                        dsname = str(rm.get('filename',''))
                        if dsname == str(data.get('filename','current_dataset')):
                            rts = int(rm.get('timestamp', os.path.getmtime(rf))) if 'timestamp' in rm else int(os.path.getmtime(rf))
                            rmatches.append((rts, rf))
                    except Exception:
                        continue
                if rmatches:
                    rmatches.sort(key=lambda x: x[0], reverse=True)
                    chosen_recon = rmatches[0][1]
                    try:
                        rm = np.load(chosen_recon, allow_pickle=True)
                        if 'data' in rm and int(global_idx) < rm['data'].shape[0]:
                            reconstruction = rm['data'][int(global_idx)].tolist()
                    except Exception:
                        reconstruction = None
        except Exception:
            reconstruction = None

        # Fallback to model reconstruction if saved reconstruction not found
        if reconstruction is None:
            sample = torch.FloatTensor(all_data[global_idx:global_idx+1])

            # load model state and instantiate model
            state = torch.load('models/current_model.pth')
            encoding_dim = None
            if isinstance(state, dict) and 'encoder.2.weight' in state:
                encoding_dim = state['encoder.2.weight'].shape[0]
            if encoding_dim is None:
                encoding_dim = 8

            # input dim comes from the full dataset
            input_dim = all_data.shape[1]
            model = SimpleAutoencoder(input_dim=input_dim, encoding_dim=encoding_dim)
            model.load_state_dict(state)
            model.eval()

            with torch.no_grad():
                recon_t = model(sample)
                reconstructed = recon_t.detach().cpu().numpy()[0]
            reconstruction = reconstructed.tolist()

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
