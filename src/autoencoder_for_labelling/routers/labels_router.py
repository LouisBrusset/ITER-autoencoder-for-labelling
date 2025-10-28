import os
import time
import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import numpy as np

router = APIRouter()


class LabelsPayload(BaseModel):
    labels: list
    indices: list = None


@router.post('/save-labels')
async def save_labels(payload: LabelsPayload):
    """Save labels (list) into the current_dataset.npz file.

    This will overwrite or add the 'labels' array in data/current_dataset.npz.
    """
    try:
        if not os.path.exists('data/current_dataset.npz'):
            raise HTTPException(status_code=400, detail='No dataset loaded')
        labels = payload.labels
        indices = payload.indices
        data = np.load('data/current_dataset.npz', allow_pickle=True)
        X = data['data']
        n = X.shape[0]

        # Validate inputs: indices must be provided and in dataset range
        if indices is None:
            raise HTTPException(status_code=400, detail='Indices must be provided')

        # Validate inputs: labels length should match indices length
        if len(labels) != len(indices):
            raise HTTPException(status_code=400, detail='Labels length must match indices length')

        # Ensure all provided indices are within dataset bounds
        for idx in indices:
            if int(idx) < 0 or int(idx) >= n:
                raise HTTPException(status_code=400, detail=f'Index {idx} is out of range (0..{n-1})')

        # Merge with existing labels stored in current_dataset (avoid creating new results files here)
        existing_full = np.zeros(n, dtype=int)
        try:
            if 'labels' in data:
                existing_full = np.array(data['labels'], dtype=int)
        except Exception:
            existing_full = np.zeros(n, dtype=int)

        # apply new labels on top of existing (only indices provided)
        for idx, lab in zip(indices, labels):
            existing_full[int(idx)] = int(lab)

        # Update data/current_dataset.npz in-place to reflect new labels so visualization uses them
        try:
            dataset_path = 'data/current_dataset.npz'
            if os.path.exists(dataset_path):
                ds = np.load(dataset_path, allow_pickle=True)
                save_dict = {k: ds[k] for k in ds.files}
                save_dict['labels'] = existing_full
                np.savez_compressed(dataset_path, **save_dict)
        except Exception:
            # ignore dataset write errors but continue to update current_* files
            pass

        # Update any current_* inference files (current_latents/projection2d/reconstructions) so UI shows labels
        try:
            base = os.path.join('results')
            current_files = [os.path.join(base, 'current_latents.npz'), os.path.join(base, 'current_projection2d.npz'), os.path.join(base, 'current_reconstructions.npz')]
            for cf in current_files:
                if not os.path.exists(cf):
                    continue
                try:
                    meta = np.load(cf, allow_pickle=True)
                    # determine how to map provided global indices into this file's rows
                    if 'indices' in meta:
                        src_idx = np.array(meta['indices'], dtype=int)
                        labels_arr = np.array(meta['labels']) if 'labels' in meta else np.zeros(src_idx.shape[0], dtype=int)
                        # for each provided index, if present in src_idx, update corresponding position
                        pos_map = {int(v): i for i, v in enumerate(src_idx)}
                        for gi, lab in zip(indices, labels):
                            gi = int(gi)
                            if gi in pos_map:
                                labels_arr[pos_map[gi]] = int(lab)
                        # write back
                        save_dict = {k: meta[k] for k in meta.files}
                        save_dict['labels'] = labels_arr
                        np.savez_compressed(cf, **save_dict)
                    else:
                        # if file covers full dataset, update positions directly
                        if 'data' in meta and meta['data'].shape[0] == n:
                            labels_arr = np.array(meta['labels']) if 'labels' in meta else np.zeros(n, dtype=int)
                            for gi, lab in zip(indices, labels):
                                labels_arr[int(gi)] = int(lab)
                            save_dict = {k: meta[k] for k in meta.files}
                            save_dict['labels'] = labels_arr
                            np.savez_compressed(cf, **save_dict)
                        else:
                            # ambiguous mapping: skip updating this current file
                            continue
                except Exception:
                    continue
        except Exception:
            pass

        return {
            'message': 'labels applied to current dataset and current inference artifacts',
            'n_labels': len(labels)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/export-all-labels')
async def export_all_labels():
    """Export all labels grouped by train/validation and save to results/exported_labels_<ts>.json

    Returns the saved filename and the JSON payload.
    """
    try:
        dataset_path = 'data/current_dataset.npz'
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=400, detail='No dataset loaded')

        data = np.load(dataset_path, allow_pickle=True)
        X = data['data']
        n = X.shape[0]

        # determine train/val mask
        if 'is_train' in data:
            is_train = np.array(data['is_train'], dtype=bool)
        else:
            is_train = np.ones(n, dtype=bool)

        train_indices = np.where(is_train)[0]
        val_indices = np.where(~is_train)[0]

        # build full labels: prefer labels in current_dataset if present, else use latest results file
        full_labels = np.zeros(n, dtype=int)
        if 'labels' in data:
            try:
                full_labels = np.array(data['labels'], dtype=int)
            except Exception:
                full_labels = np.zeros(n, dtype=int)
        else:
            # try to load latest labels file
            try:
                results_dir = os.path.join('results', 'labels')
                files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.startswith('labels_') and f.endswith('.json')]
                if files:
                    latest = max(files, key=os.path.getmtime)
                    with open(latest, 'r') as fh:
                        payload = json.load(fh)
                        if 'label' in payload and 'index' in payload:
                            for idx, lab in zip(payload['index'], payload['label']):
                                if 0 <= int(idx) < n:
                                    full_labels[int(idx)] = int(lab)
            except Exception:
                pass

        # build mappings
        train_map = {}
        val_map = {}
        for i in train_indices:
            train_map[str(int(i))] = int(full_labels[int(i)])
        for i in val_indices:
            val_map[str(int(i))] = int(full_labels[int(i)])

        out = {'trainset': train_map, 'validationset': val_map}

        # save to results/labels
        results_dir = os.path.join('results', 'labels')
        os.makedirs(results_dir, exist_ok=True)
        ts = int(time.time())
        filename = os.path.join(results_dir, f'exported_final_labels_{ts}.json')
        with open(filename, 'w') as fh:
            json.dump(out, fh)

        return {'file': filename, 'data': out}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
