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
async def check_sample_plot(index: int = 0):
    """Return a PNG image plotting original for a dataset sample index."""
    try:
        # Ensure dataset exists
        if not os.path.exists('data/current_dataset.npz'):
            raise HTTPException(status_code=400, detail='No dataset loaded')
        
        data = np.load('data/current_dataset.npz', allow_pickle=True)
        X = data['data']
        n = X.shape[0]
        if index < 0 or index >= n:
            raise HTTPException(status_code=400, detail=f'Index out of range (0..{n-1})')
        
        orig = X[index]
        fig, ax = plt.subplots(figsize=(8,3))

        # plot using matplotlib
        ax.plot(orig, label='Original', color='tab:blue')
        ax.set_xlabel('Feature index')
        ax.set_ylabel('Value')
        title = f'Sample {index} - Original'
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
