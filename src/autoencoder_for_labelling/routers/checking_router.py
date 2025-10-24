import os
import io

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

router = APIRouter()


@router.get('/check-sample-plot')
async def check_sample_plot(index: int = 0, subset: str | None = None):
    """Return a PNG image plotting original for a dataset sample index.

    Parameters
    - index: integer index. By default this is interpreted as a global index into the
      dataset (0..n-1). If `subset='train'` is provided, the index is interpreted as
      an index inside the training subset (0..n_train-1).
    - subset: optional string 'train' to interpret index as training-subset index.
    """
    try:
        # Ensure dataset exists
        if not os.path.exists('data/current_dataset.npz'):
            raise HTTPException(status_code=400, detail='No dataset loaded')

        data = np.load('data/current_dataset.npz', allow_pickle=True)
        X = data['data']

        # Determine train mask; legacy support: if no is_train provided assume all train
        if 'is_train' in data:
            is_train = np.array(data['is_train'], dtype=bool)
        else:
            is_train = np.ones(len(X), dtype=bool)

        # Interpret index: either global index (default) or index within training subset
        if subset == 'train':
            train_indices = np.where(is_train)[0]
            n_train = len(train_indices)
            if n_train == 0:
                raise HTTPException(status_code=400, detail='No training samples available in dataset')
            if index < 0 or index >= n_train:
                raise HTTPException(status_code=400, detail=f'Index out of range (0..{n_train-1}) for training set')
            global_idx = int(train_indices[index])
        else:
            # global index into full dataset
            n = X.shape[0]
            if index < 0 or index >= n:
                raise HTTPException(status_code=400, detail=f'Index out of range (0..{n-1})')
            global_idx = int(index)

        orig = X[global_idx]
        fig, ax = plt.subplots(figsize=(8,3))

        # plot using matplotlib
        ax.plot(orig, label='Original', color='tab:blue')
        ax.set_xlabel('Feature index')
        ax.set_ylabel('Value')

        # Determine whether the sample is in train or validation set
        subset_label = 'train' if is_train[global_idx] else 'validation'
        if subset == 'train':
            title = f'Train sample {index} (global idx {global_idx}) - {subset_label.capitalize()}'
        else:
            title = f'Sample {index} (global idx {global_idx}) - {subset_label.capitalize()}'
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        return Response(content=buf.read(), media_type='image/png')
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
