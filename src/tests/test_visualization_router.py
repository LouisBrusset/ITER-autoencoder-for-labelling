from pathlib import Path
import os
import json
import numpy as np
import torch

import pytest
from fastapi.testclient import TestClient

from autoencoder_for_labelling.main import app
from autoencoder_for_labelling.models.autoencoder import Convolutional_Beta_VAE

client = TestClient(app)

