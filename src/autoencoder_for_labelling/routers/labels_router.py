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
        if len(labels) != n:
            raise HTTPException(status_code=400, detail=f'Labels length {len(labels)} does not match dataset size {n}')
        if indices is None:
            indices = list(range(n))
        if len(indices) != n:
            raise HTTPException(status_code=400, detail=f'Indices length {len(indices)} does not match dataset size {n}')
        # ensure results dir exists
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        ts = int(time.time())
        filename = os.path.join(results_dir, f'labels_{ts}.json')
        # save under keys 'label' and 'index' as requested
        payload_out = {'label': labels, 'index': indices, 'n_labels': len(labels), 'timestamp': ts}
        with open(filename, 'w') as fh:
            json.dump(payload_out, fh)

        # remove labels key from current_dataset.npz if present, preserving other arrays
        try:
            new_kwargs = {}
            for k in data.files:
                if k == 'labels':
                    continue
                new_kwargs[k] = data[k]
            # overwrite npz without labels
            np.savez('data/current_dataset.npz', **new_kwargs)
        except Exception:
            # if anything goes wrong, continue — labels were saved to results
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
