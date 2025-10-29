import os
import re
import asyncio
import json
import time

from fastapi import APIRouter, HTTPException, Body

import torch
import numpy as np

from autoencoder_for_labelling.services.training_service import training_metrics, run_training, perform_full_inference
from autoencoder_for_labelling.models.autoencoder import Convolutional_Beta_VAE

router = APIRouter()


@router.get("/model-options")
async def get_model_options():
    """Give saved model options"""
    saved_models = []
    if os.path.exists("models/saved_model"):
        saved_models = os.listdir("models/saved_model")
    return {"saved_models": saved_models}


@router.delete("/delete-model")
async def delete_model(model_filename: str):
    """Delete a specific saved model file"""
    try:
        model_path = f"models/saved_model/{model_filename}"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model file not found")
        
        # If it was the current model, remove that as well
        if os.path.exists("models/current_model.pth"):
            try:
                current_state = torch.load("models/current_model.pth")
                saved_state = torch.load(model_path)

                def state_dicts_equal(a, b):
                    # Both must be dict-like
                    if not isinstance(a, dict) or not isinstance(b, dict):
                        return False
                    if set(a.keys()) != set(b.keys()):
                        return False
                    for k in a.keys():
                        va = a[k]
                        vb = b[k]
                        # If tensors, use torch.equal or allclose
                        if isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor):
                            try:
                                if va.shape != vb.shape:
                                    return False
                                if not torch.equal(va, vb):
                                    # fallback to allclose for floating types
                                    if not torch.allclose(va, vb, atol=1e-6, rtol=1e-5):
                                        return False
                            except Exception:
                                return False
                        else:
                            # fallback to simple equality for non-tensor values
                            if va != vb:
                                return False
                    return True
                
                if state_dicts_equal(current_state, saved_state):
                    try:
                        os.remove("models/current_model.pth")
                    except Exception:
                        # ignore errors removing the current model file
                        pass
            except Exception:
                # If any error occurs while comparing/loading, don't block deletion of the saved model
                pass

        os.remove(model_path)
        # try to remove associated architecture file if present
        try:
            arch_name = os.path.splitext(model_filename)[0] + '.json'
            arch_path = os.path.join('models', 'saved_architechture', arch_name)
            if os.path.exists(arch_path):
                os.remove(arch_path)
        except Exception:
            pass
        
        return {
            "status": "success", 
            "model_filename": model_filename
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current-model")
async def get_current_model():
    """Return info about the currently loaded model (if any)."""
    try:
        if not os.path.exists("models/current_model.pth"):
            return {"loaded": False}
        
        state = torch.load("models/current_model.pth")
        encoding_dim = None
        if isinstance(state, dict) and 'encoder.2.weight' in state:
            encoding_dim = int(state['encoder.2.weight'].shape[0])
        
        # try to infer input_dim from current dataset
        input_dim = None
        if os.path.exists("data/current_dataset.npz"):
            try:
                d = np.load("data/current_dataset.npz")
                if 'data' in d:
                    input_dim = int(d['data'].shape[1])
            except Exception:
                input_dim = None

        return {
            "loaded": True, 
            "encoding_dim": encoding_dim, 
            "input_dim": input_dim
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start-training")
async def start_training(epochs: int = 100, learning_rate: float = 0.001, encoding_dim: int = 8,
                         config: dict = Body(None)):
    """Start training the autoencoder model"""
    if training_metrics["is_training"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    # Check if a dataset is loaded
    if not os.path.exists("data/current_dataset.npz"):
        raise HTTPException(status_code=400, detail="No dataset loaded")
    
    data = np.load("data/current_dataset.npz", allow_pickle=True)
    all_data = data['data']
    # if is_train is present, split accordingly, else assume all data is train and val empty
    if 'is_train' in data:
        is_train = data['is_train']
    else:
        is_train = np.ones(len(all_data), dtype=bool)
    train_idx = np.where(is_train)[0]
    val_idx = np.where(~is_train)[0]
    train_data = torch.FloatTensor(all_data[train_idx])
    val_data = torch.FloatTensor(all_data[val_idx]) if len(val_idx) > 0 else None
    
    training_metrics.update({
        "is_training": True,
        "current_epoch": 0,
        "total_epochs": epochs,
        "train_loss_history": [],
        "val_loss_history": [],
        "current_train_loss": 0.0,
        "current_val_loss": None,
        "epochs_data": []
    })

    # If the client provided a JSON body (from the form), prefer those values over query params
    encoder_layer_sizes = None
    decoder_layer_sizes = None
    conv_layers = 0
    conv_filter_size = 3
    if config:
        try:
            # override simple params if present in body
            if 'epochs' in config:
                try:
                    epochs = int(config.get('epochs'))
                except Exception:
                    pass
            if 'learning_rate' in config:
                try:
                    learning_rate = float(config.get('learning_rate'))
                except Exception:
                    pass
            if 'encoding_dim' in config:
                try:
                    encoding_dim = int(config.get('encoding_dim'))
                except Exception:
                    pass

            # prefer explicit lists if provided
            encoder_layer_sizes = config.get('encoder_layer_sizes', None)
            decoder_layer_sizes = config.get('decoder_layer_sizes', None)
            # convolutional params (optional)
            try:
                conv_layers = int(config.get('conv_layers', conv_layers))
            except Exception:
                conv_layers = conv_layers
            try:
                conv_filter_size = int(config.get('conv_filter_size', conv_filter_size))
            except Exception:
                conv_filter_size = conv_filter_size
        except Exception:
            encoder_layer_sizes = None
            decoder_layer_sizes = None

    asyncio.create_task(run_training(train_data, val_data, epochs, learning_rate, encoding_dim,
                                      encoder_layer_sizes=encoder_layer_sizes,
                                      decoder_layer_sizes=decoder_layer_sizes,
                                      conv_layers=conv_layers,
                                      conv_filter_size=conv_filter_size))
    
    return {"status": "Training started", "total_epochs": epochs}


@router.get("/training-status")
async def get_training_status():
    return training_metrics


@router.post("/load-model")
async def load_model(model_filename: str):
    """Load a specific saved model file as the current model"""
    try:
        model_path = f"models/saved_model/{model_filename}"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found")
        
        if not os.path.exists("data/current_dataset.npz"):
            raise HTTPException(status_code=400, detail="First load a dataset")
        
        # Check if a dataset is loaded to know the dimension
        data = np.load("data/current_dataset.npz")
        input_dim = data['data'].shape[1]
        
        # Load the saved state or checkpoint. The checkpoint may be either:
        # - a plain state_dict (mapping of tensor names -> tensors)
        # - a dict containing {'state_dict': <state_dict>, 'encoder_layer_sizes':..., ...}
        raw = torch.load(model_path)
        encoding_dim = None
        encoder_layer_sizes = None
        decoder_layer_sizes = None
        actual_state = None

        if isinstance(raw, dict) and 'state_dict' in raw:
            # new-style saved dict with metadata
            actual_state = raw['state_dict']
            encoder_layer_sizes = raw.get('encoder_layer_sizes', None)
            decoder_layer_sizes = raw.get('decoder_layer_sizes', None)
            encoding_dim = raw.get('encoding_dim', None)
        elif isinstance(raw, dict):
            # likely a raw state_dict
            actual_state = raw

        # If we still don't have encoding_dim or layer sizes, try to infer from actual_state
        if actual_state is not None:
            try:
                enc_weights = []
                dec_weights = []
                for k, v in actual_state.items():
                    if k.startswith('encoder.') and k.endswith('.weight'):
                        try:
                            idx = int(k.split('.')[1])
                        except Exception:
                            idx = None
                        enc_weights.append((idx, v))
                    if k.startswith('decoder.') and k.endswith('.weight'):
                        try:
                            idx = int(k.split('.')[1])
                        except Exception:
                            idx = None
                        dec_weights.append((idx, v))

                enc_weights.sort(key=lambda x: (x[0] is None, x[0]))
                dec_weights.sort(key=lambda x: (x[0] is None, x[0]))

                enc_outs = [int(t[1].shape[0]) for t in enc_weights if hasattr(t[1], 'shape')]
                dec_outs = [int(t[1].shape[0]) for t in dec_weights if hasattr(t[1], 'shape')]

                if len(enc_outs) > 0:
                    # last encoder out is the encoding dim
                    encoding_dim = int(enc_outs[-1])
                    # intermediate encoder sizes are all but the last
                    if len(enc_outs) > 1:
                        encoder_layer_sizes = [int(x) for x in enc_outs[:-1]]
                # For decoder, decoder_outs typically end with input_dim
                if len(dec_outs) > 0:
                    # if the final decoder out matches input_dim, use preceding outs as decoder_layer_sizes
                    if dec_outs[-1] == input_dim and len(dec_outs) > 1:
                        decoder_layer_sizes = [int(x) for x in dec_outs[:-1]]
                    else:
                        # fallback: use all but last if more than one, else use dec_outs
                        if len(dec_outs) > 1:
                            decoder_layer_sizes = [int(x) for x in dec_outs[:-1]]
                        else:
                            decoder_layer_sizes = [int(x) for x in dec_outs]
            except Exception:
                encoding_dim = None

        # If we don't have architecture metadata, try to detect conv-based checkpoints
        # by looking for conv_encoder / conv_decoder keys in the state_dict and
        # populate arch_meta so we can reconstruct the correct model topology.
        if 'arch_meta' not in locals() or arch_meta is None:
            try:
                arch_meta = {}
                # if original raw checkpoint had metadata, prefer it
                if isinstance(raw, dict):
                    # raw may contain conv_layers/conv_filter_size
                    if 'conv_layers' in raw:
                        arch_meta['conv_layers'] = int(raw.get('conv_layers', 0))
                    if 'conv_filter_size' in raw:
                        arch_meta['conv_filter_size'] = int(raw.get('conv_filter_size', 3))

                # inspect actual_state keys for conv encoder presence
                if actual_state is not None:
                    conv_keys = [k for k in actual_state.keys() if k.startswith('conv_encoder.') and k.endswith('.weight')]
                    if len(conv_keys) > 0 and 'conv_layers' not in arch_meta:
                        idxs = set()
                        for k in conv_keys:
                            parts = k.split('.')
                            if len(parts) >= 3:
                                try:
                                    idx = int(parts[1])
                                    idxs.add(idx)
                                except Exception:
                                    pass
                        # number of conv layers equals number of unique conv indices found
                        arch_meta['conv_layers'] = len(idxs)
                        # default kernel size if not available
                        if 'conv_filter_size' not in arch_meta:
                            arch_meta['conv_filter_size'] = int(raw.get('conv_filter_size', 3)) if isinstance(raw, dict) else 3
            except Exception:
                arch_meta = None

        if encoding_dim is None:
            # try to parse from filename pattern _dim_(\d+)
            m = re.search(r'_dim_(\d+)', model_filename)
            if m:
                encoding_dim = int(m.group(1))
        if encoding_dim is None:
            encoding_dim = 8

        # Create model with inferred encoding_dim and inferred layer sizes (if any) and load weights
        # try to read a saved architecture JSON matching this model filename
        # If we already detected arch_meta earlier (from the checkpoint/state_dict), keep it
        try:
            arch_name = os.path.splitext(model_filename)[0] + '.json'
            arch_path = os.path.join('models', 'saved_architechture', arch_name)
            if os.path.exists(arch_path):
                with open(arch_path, 'r') as f:
                    arch_meta = json.load(f)
        except Exception:
            # preserve any previously-detected arch_meta; if none exists, ensure it's None
            if 'arch_meta' not in locals():
                arch_meta = None

        model = Convolutional_Beta_VAE(input_dim=input_dim, encoding_dim=encoding_dim,
                                       conv_layers=arch_meta.get('conv_layers', 0) if arch_meta else 0,
                                       conv_filter_size=arch_meta.get('conv_filter_size', 3) if arch_meta else 3,
                                       encoder_layer_sizes=encoder_layer_sizes,
                                       decoder_layer_sizes=decoder_layer_sizes)
        try:
            model.load_state_dict(actual_state)
        except Exception as e:
            # If loading the state dict into the reconstructed model fails, raise a helpful error
            raise HTTPException(status_code=500, detail=f"Error(s) in loading state_dict for Convolutional_Beta_VAE: {e}")
        model.eval()

        # Store the model state as current model (keep original checkpoint bytes to preserve metadata)
        # If original was a state_dict, save it as-is; otherwise save the file bytes
        try:
            # preserve original raw checkpoint (may include metadata)
            torch.save(raw, "models/current_model.pth")
            # also write a JSON file describing the current architecture for reproducible reconstruction
            try:
                arch_meta = {
                    'encoding_dim': encoding_dim,
                    'encoder_layer_sizes': encoder_layer_sizes,
                    'decoder_layer_sizes': decoder_layer_sizes,
                    'conv_layers': (arch_meta.get('conv_layers') if arch_meta and 'conv_layers' in arch_meta else raw.get('conv_layers') if isinstance(raw, dict) else None),
                    'conv_filter_size': (arch_meta.get('conv_filter_size') if arch_meta and 'conv_filter_size' in arch_meta else raw.get('conv_filter_size') if isinstance(raw, dict) else None),
                    'filename': model_filename,
                    'timestamp': int(time.time())
                }
                with open(os.path.join('models', 'current_architecture.json'), 'w') as f:
                    json.dump(arch_meta, f)
            except Exception:
                pass
        except Exception:
            # fallback: copy file bytes
            with open(model_path, 'rb') as fr, open('models/current_model.pth', 'wb') as fw:
                fw.write(fr.read())

        return {"status": "success", "model_loaded": model_filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset-training")
async def reset_training():
    training_metrics.update({
        "is_training": False,
        "current_epoch": 0,
        "total_epochs": 0,
        "train_loss_history": [],
        "val_loss_history": [],
        "current_train_loss": 0.0,
        "current_val_loss": None,
        "epochs_data": []
    })
    return {"status": "Training reset"}


@router.post("/run-inference")
async def run_inference(deterministic: bool = True):
    """Trigger inference using the current model and dataset and save latents/reconstructions/projection files.

    Query/JSON param `deterministic` controls deterministic UMAP.
    """
    try:
        if not os.path.exists("data/current_dataset.npz"):
            raise HTTPException(status_code=400, detail="No dataset loaded")
        if not os.path.exists("models/current_model.pth"):
            raise HTTPException(status_code=400, detail="No model loaded")

        result = await perform_full_inference(deterministic=deterministic)
        return {"status": "inference_complete", "files": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/inference-options")
async def inference_options():
    """Return available saved inference artifacts for the currently loaded dataset (latents, projections, reconstructions)."""
    try:
        current_dataset = None
        if os.path.exists("data/current_dataset.npz"):
            try:
                import numpy as _np
                d = _np.load("data/current_dataset.npz", allow_pickle=True)
                current_dataset = str(d.get('filename', 'current_dataset'))
            except Exception:
                current_dataset = None

        def list_matches(dirpath):
            out = []
            if not os.path.exists(dirpath):
                return out
            for f in os.listdir(dirpath):
                if not f.endswith('.npz'):
                    continue
                full = os.path.join(dirpath, f)
                try:
                    import numpy as _np
                    meta = _np.load(full, allow_pickle=True)
                    dsname = str(meta.get('filename', ''))
                    ts = int(meta.get('timestamp', os.path.getmtime(full))) if 'timestamp' in meta else int(os.path.getmtime(full))
                    if current_dataset is None or dsname == current_dataset:
                        out.append({'file': full, 'dataset': dsname, 'timestamp': int(ts)})
                except Exception:
                    # ignore unreadable files
                    pass
            out.sort(key=lambda x: x['timestamp'], reverse=True)
            return out

        latents = list_matches(os.path.join('results','latents'))
        projections = list_matches(os.path.join('results','projections2d'))
        reconstructions = list_matches(os.path.join('results','reconstructions'))

        return { 'latents': latents, 'projections': projections, 'reconstructions': reconstructions }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current-inference")
async def current_inference():
    """Return information about currently loaded inference files (if any).

    Checks for results/current_latents.npz, results/current_projection2d.npz and
    results/current_reconstructions.npz and returns their metadata (dataset filename and timestamp)
    so the frontend can display which inference is currently active.
    """
    try:
        out = {
            'latents': None,
            'projection2d': None,
            'reconstructions': None,
            'loaded': False
        }
        base = os.path.join('results')
        latf = os.path.join(base, 'current_latents.npz')
        projf = os.path.join(base, 'current_projection2d.npz')
        reconf = os.path.join(base, 'current_reconstructions.npz')

        def read_meta(path):
            try:
                if not os.path.exists(path):
                    return None
                import numpy as _np
                m = _np.load(path, allow_pickle=True)
                return {
                    'file': path,
                    'dataset': str(m.get('filename', '')),
                    'timestamp': int(m.get('timestamp', os.path.getmtime(path))) if ('timestamp' in m or True) else int(os.path.getmtime(path))
                }
            except Exception:
                return {'file': path}

        out['latents'] = read_meta(latf)
        out['projection2d'] = read_meta(projf)
        out['reconstructions'] = read_meta(reconf)

        if out['latents'] or out['projection2d'] or out['reconstructions']:
            out['loaded'] = True

        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load-inference")
async def load_inference(latents_file: str = None, projection_file: str = None, reconstructions_file: str = None):
    """Load selected inference artifacts as the current inference.

    The API expects full paths as returned by `/inference-options` (or relative paths under the repo).
    It will copy the files to results/current_latents.npz, results/current_projection2d.npz and
    results/current_reconstructions.npz so the visualization and classification code can reference
    the "current" inference files.
    """
    try:
        # ensure at least one file provided
        if not (latents_file or projection_file or reconstructions_file):
            raise HTTPException(status_code=400, detail="No inference file specified to load")

        os.makedirs(os.path.join('results'), exist_ok=True)
        # destination names
        dest_latents = os.path.join('results', 'current_latents.npz')
        dest_proj = os.path.join('results', 'current_projection2d.npz')
        dest_recon = os.path.join('results', 'current_reconstructions.npz')

        # helper copy function
        def copy_if(src, dst):
            if not src:
                return False
            if not os.path.exists(src):
                # try relative path under workspace
                alt = src
                if os.path.exists(alt):
                    src = alt
                else:
                    raise HTTPException(status_code=404, detail=f"File not found: {src}")
            # perform a simple copy by reading/writing binary
            with open(src, 'rb') as fr, open(dst, 'wb') as fw:
                fw.write(fr.read())
            return True

        copied = {}
        if latents_file:
            copy_if(latents_file, dest_latents)
            copied['latents'] = dest_latents
        if projection_file:
            copy_if(projection_file, dest_proj)
            copied['projection2d'] = dest_proj
        if reconstructions_file:
            copy_if(reconstructions_file, dest_recon)
            copied['reconstructions'] = dest_recon

        return { 'status': 'success', 'loaded': copied }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete-inference")
async def delete_inference(file_path: str):
    """Delete a saved inference artifact file (full path as returned by /inference-options).

    This will remove the file from disk. If the deleted file is currently set as a current_* file,
    the current_* file will not be automatically removed (safer behaviour).
    """
    try:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Try to load metadata from the file being deleted to detect if it corresponds
        # to the currently loaded inference triplet. If so, remove current_* files as well
        try:
            meta_del = np.load(file_path, allow_pickle=True)
            del_dataset = str(meta_del.get('filename', '')) if 'filename' in meta_del or True else None
            del_ts = int(meta_del.get('timestamp', os.path.getmtime(file_path))) if ('timestamp' in meta_del or True) else int(os.path.getmtime(file_path))
        except Exception:
            meta_del = None
            del_dataset = None
            del_ts = None

        # Check current_* files — if any of them originates from the same dataset+timestamp
        # as the file being deleted, we consider the loaded inference invalid and remove all currents.
        currents_removed = []
        try:
            base = os.path.join('results')
            cur_lat = os.path.join(base, 'current_latents.npz')
            cur_proj = os.path.join(base, 'current_projection2d.npz')
            cur_recon = os.path.join(base, 'current_reconstructions.npz')

            should_remove_currents = False
            if meta_del is not None and (del_dataset is not None or del_ts is not None):
                for cur in (cur_lat, cur_proj, cur_recon):
                    if os.path.exists(cur):
                        try:
                            cm = np.load(cur, allow_pickle=True)
                            cur_ds = str(cm.get('filename', '')) if 'filename' in cm or True else None
                            cur_ts = int(cm.get('timestamp', os.path.getmtime(cur))) if ('timestamp' in cm or True) else int(os.path.getmtime(cur))
                            if (del_dataset and cur_ds and del_dataset == cur_ds) or (del_ts and cur_ts and del_ts == cur_ts):
                                should_remove_currents = True
                                break
                        except Exception:
                            # if we can't read current metadata, skip
                            continue

            if should_remove_currents:
                for cur in (cur_lat, cur_proj, cur_recon):
                    if os.path.exists(cur):
                        try:
                            os.remove(cur)
                            currents_removed.append(cur)
                        except Exception:
                            # ignore removal errors
                            pass
        except Exception:
            # ignore any error in current-file handling and continue with deletion
            pass

        # Finally remove the requested saved artifact
        os.remove(file_path)

        resp = { 'status': 'success', 'file': file_path }
        if currents_removed:
            resp['currents_removed'] = currents_removed
        return resp
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
