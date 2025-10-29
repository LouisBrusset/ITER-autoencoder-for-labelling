import asyncio

import torch
import numpy as np
import time
import os
import json
import umap

from autoencoder_for_labelling.models.autoencoder import SimpleAutoencoder
from autoencoder_for_labelling.training.trainer import train_autoencoder

# Shared training metrics state
training_metrics = {
    "is_training": False,
    "current_epoch": 0,
    "total_epochs": 0,
    "train_loss_history": [],
    "val_loss_history": [],
    "current_train_loss": 0.0,
    "current_val_loss": None,
    "epochs_data": []
}


async def run_training(train_data, val_data, epochs, learning_rate, encoding_dim,
                       encoder_layer_sizes=None, decoder_layer_sizes=None, verbose=False):
    try:
        input_dim = train_data.shape[1]
        # Create model with optional custom layer sizes
        model = SimpleAutoencoder(input_dim=input_dim,
                                  encoding_dim=encoding_dim,
                                  encoder_layer_sizes=encoder_layer_sizes,
                                  decoder_layer_sizes=decoder_layer_sizes)
        
        for epoch in range(1, epochs + 1):
            train_loss, val_loss = train_autoencoder(model, train_data, val_data, epoch, learning_rate, verbose=False)

            training_metrics["current_epoch"] = epoch
            training_metrics["current_train_loss"] = float(train_loss)
            training_metrics["current_val_loss"] = float(val_loss) if val_loss is not None else None
            training_metrics["train_loss_history"].append(float(train_loss))
            if val_loss is not None:
                training_metrics["val_loss_history"].append(float(val_loss))
            training_metrics["epochs_data"].append(epoch)
            
            await asyncio.sleep(0.05)
        
        # Save the trained model state dict along with architecture metadata so it can be
        # reconstructed exactly when loading for inference later.
        ts = int(time.time())
        base_name = f"model_epochs_{epochs}_dim_{encoding_dim}_{ts}"
        model_path = os.path.join("models", "saved_model", f"{base_name}.pth")
        arch_path = os.path.join("models", "saved_architechture", f"{base_name}.json")

        save_dict = {
            'state_dict': model.state_dict(),
            'encoder_layer_sizes': encoder_layer_sizes,
            'decoder_layer_sizes': decoder_layer_sizes,
            'encoding_dim': encoding_dim,
            'timestamp': ts,
            'epochs': epochs
        }
        # Save combined checkpoint (state + metadata)
        torch.save(save_dict, model_path)
        # Save a separate human-readable architecture file as JSON as well
        try:
            with open(arch_path, 'w') as f:
                json.dump({k: v for k, v in save_dict.items() if k != 'state_dict'}, f)
        except Exception:
            pass

        # also save as current model and current architecture
        try:
            torch.save(save_dict, "models/current_model.pth")
            with open(os.path.join('models', 'current_architecture.json'), 'w') as f:
                json.dump({k: v for k, v in save_dict.items() if k != 'state_dict'}, f)
        except Exception:
            # fallback: save only state_dict as before
            torch.save(model.state_dict(), "models/current_model.pth")
    
    except Exception as e:
        print(f"Training error: {e}")
    finally:
        training_metrics["is_training"] = False


async def perform_full_inference(deterministic: bool = True) -> dict:
    """Run inference with the current model on the current dataset and save latents,
    reconstructions and 2D projections as .npz files that mirror the dataset structure.

    The saved files will be placed in:
      - results/latents/latents_<ts>_<dataset_filename>.npz  (key 'data' = latents)
      - results/projections2d/projection2d_<ts>_<dataset_filename>.npz (key 'data' = 2D proj)
      - results/reconstructions/reconstructions_<ts>_<dataset_filename>.npz (key 'data' = reconstructions)

    Each file will also contain 'labels' and 'is_train' if present in the original dataset
    and a 'filename' and 'timestamp' metadata field.
    """
    dataset_path = 'data/current_dataset.npz'
    model_path = 'models/current_model.pth'
    if not os.path.exists(dataset_path):
        raise RuntimeError('No current dataset found for inference')
    if not os.path.exists(model_path):
        raise RuntimeError('No current model found for inference')

    d = np.load(dataset_path, allow_pickle=True)
    all_data = d['data']
    dataset_filename = str(d.get('filename', 'current_dataset'))
    labels = d['labels'] if 'labels' in d else None
    is_train = d['is_train'] if 'is_train' in d else None
    n_samples = int(all_data.shape[0])

    # Prefer reading separate current architecture JSON if present
    arch_file = os.path.join('models', 'current_architecture.json')
    arch_meta = None
    if os.path.exists(arch_file):
        try:
            with open(arch_file, 'r') as f:
                arch_meta = json.load(f)
        except Exception:
            arch_meta = None

    # load model checkpoint
    raw = torch.load(model_path)
    actual_state = None
    encoder_layer_sizes = None
    decoder_layer_sizes = None
    encoding_dim = None

    if arch_meta is not None:
        encoder_layer_sizes = arch_meta.get('encoder_layer_sizes', None)
        decoder_layer_sizes = arch_meta.get('decoder_layer_sizes', None)
        encoding_dim = arch_meta.get('encoding_dim', None)

    if isinstance(raw, dict) and 'state_dict' in raw:
        actual_state = raw['state_dict']
        # use metadata from checkpoint if present but prefer arch_meta
        if encoding_dim is None:
            encoding_dim = raw.get('encoding_dim', None)
        if encoder_layer_sizes is None:
            encoder_layer_sizes = raw.get('encoder_layer_sizes', None)
        if decoder_layer_sizes is None:
            decoder_layer_sizes = raw.get('decoder_layer_sizes', None)
    elif isinstance(raw, dict):
        actual_state = raw

    if actual_state is None:
        raise RuntimeError('Loaded model file does not contain a state_dict')

    # If encoding_dim still unknown, try to infer from state keys
    if encoding_dim is None:
        try:
            enc_keys = [k for k in actual_state.keys() if k.startswith('encoder.') and k.endswith('.weight')]
            if enc_keys:
                def key_idx(k):
                    parts = k.split('.')
                    try:
                        return int(parts[1])
                    except Exception:
                        return 999
                enc_keys = sorted(enc_keys, key=key_idx)
                last = enc_keys[-1]
                encoding_dim = int(actual_state[last].shape[0])
        except Exception:
            encoding_dim = encoding_dim or 8

    # Build model with inferred or provided topology
    input_dim = int(all_data.shape[1])
    model = SimpleAutoencoder(input_dim=input_dim, encoding_dim=encoding_dim,
                              encoder_layer_sizes=encoder_layer_sizes,
                              decoder_layer_sizes=decoder_layer_sizes)
    try:
        model.load_state_dict(actual_state)
    except Exception as e:
        raise RuntimeError(f"Error(s) in loading state_dict for SimpleAutoencoder: {e}")
    model.eval()

    # compute latents and reconstructions in batches
    batch = 1024
    latents_list = []
    recons_list = []
    with torch.no_grad():
        for i in range(0, n_samples, batch):
            chunk = torch.FloatTensor(all_data[i:i+batch])
            latent = model.encoder(chunk).cpu().numpy()
            recon = model(chunk).cpu().numpy()
            latents_list.append(latent)
            recons_list.append(recon)
    latents = np.vstack(latents_list)
    reconstructions = np.vstack(recons_list)

    # UMAP projection (deterministic option)
    try:
        if deterministic:
            reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
        else:
            reducer = umap.UMAP(n_components=2, init='spectral', n_neighbors=15, n_jobs=-1, random_state=None)
        projection2d = reducer.fit_transform(latents)
    except Exception:
        # fallback to PCA-like reduction
        X = latents - np.mean(latents, axis=0, keepdims=True)
        u, s, vt = np.linalg.svd(X, full_matrices=False)
        projection2d = (u[:, :2] * s[:2])

    ts = int(time.time())
    latents_dir = os.path.join('results', 'latents')
    proj_dir = os.path.join('results', 'projections2d')
    recon_dir = os.path.join('results', 'reconstructions')
    os.makedirs(latents_dir, exist_ok=True)
    os.makedirs(proj_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)

    latents_fname = os.path.join(latents_dir, f'latents_{ts}_{dataset_filename}.npz')
    proj_fname = os.path.join(proj_dir, f'projection2d_{ts}_{dataset_filename}.npz')
    recon_fname = os.path.join(recon_dir, f'reconstructions_{ts}_{dataset_filename}.npz')

    # Save in dataset-like structure: key 'data' holds the array
    # record indices so that saved files can be unambiguously mapped to dataset global indices
    indices = np.arange(n_samples, dtype=int)
    latents_save = {'data': latents, 'latent_dim': latents.shape[1], 'timestamp': ts, 'filename': dataset_filename, 'indices': indices}
    if labels is not None:
        latents_save['labels'] = labels
    if is_train is not None:
        latents_save['is_train'] = is_train
    np.savez_compressed(latents_fname, **latents_save)

    proj_save = {'data': projection2d, 'timestamp': ts, 'filename': dataset_filename, 'indices': indices}
    if labels is not None:
        proj_save['labels'] = labels
    if is_train is not None:
        proj_save['is_train'] = is_train
    np.savez_compressed(proj_fname, **proj_save)

    recon_save = {'data': reconstructions, 'timestamp': ts, 'filename': dataset_filename, 'indices': indices}
    if labels is not None:
        recon_save['labels'] = labels
    if is_train is not None:
        recon_save['is_train'] = is_train
    np.savez_compressed(recon_fname, **recon_save)

    return {'latents': latents_fname, 'projection2d': proj_fname, 'reconstructions': recon_fname}
