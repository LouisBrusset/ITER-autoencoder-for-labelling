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

        # Merge with existing labels so saving labels for one subset doesn't reset others
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)

        # start from existing labels if present
        existing_full = np.zeros(n, dtype=int)
        try:
            files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.startswith('labels_') and f.endswith('.json')]
            if files:
                latest = max(files, key=os.path.getmtime)
                with open(latest, 'r') as fh:
                    existing_payload = json.load(fh)
                    if 'label' in existing_payload and 'index' in existing_payload:
                        for idx, lab in zip(existing_payload['index'], existing_payload['label']):
                            if 0 <= int(idx) < n:
                                existing_full[int(idx)] = int(lab)
        except Exception:
            # ignore issues reading existing labels
            pass

        # apply new labels on top of existing
        for idx, lab in zip(indices, labels):
            existing_full[int(idx)] = int(lab)

        # prepare compact arrays to save: only store non-zero labels to keep files small
        to_save_indices = [int(i) for i, v in enumerate(existing_full) if int(v) != 0]
        to_save_labels = [int(existing_full[i]) for i in to_save_indices]

        ts = int(time.time())
        filename = os.path.join(results_dir, f'labels_{ts}.json')
        payload_out = {'label': to_save_labels, 'index': to_save_indices, 'n_labels': len(to_save_labels), 'timestamp': ts}
        with open(filename, 'w') as fh:
            json.dump(payload_out, fh)

        # Also write the merged labels back into data/current_dataset.npz so the UI
        # (visualization <-> classification) keeps the labels in memory when switching
        try:
            dataset_path = 'data/current_dataset.npz'
            if os.path.exists(dataset_path):
                ds = np.load(dataset_path, allow_pickle=True)
                # build a dict of arrays to preserve other fields
                save_dict = {k: ds[k] for k in ds.files}
                # write the full labels array (length n) so other code can read it
                save_dict['labels'] = existing_full
                # overwrite the current_dataset.npz with the updated labels
                np.savez_compressed(dataset_path, **save_dict)
        except Exception:
            # don't break saving if dataset update fails; labels file was already written
            pass

        return {
            'message': 'labels saved',
            'n_labels': len(labels),
            'file': filename
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
                results_dir = 'results'
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

        # save to results
        results_dir = 'results'
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
