from pathlib import Path
import os
import time
import numpy as np

import pytest
from fastapi.testclient import TestClient
from autoencoder_for_labelling.main import app

client = TestClient(app)

