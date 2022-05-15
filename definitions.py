from pathlib import Path
import os
import numpy as np

# dirs
ROOT_DIR = Path(os.path.dirname(__file__))
DATA_DIR = ROOT_DIR / "data"

# dtypes
calibration_image_dtype = np.float32
image_dtype = np.uint16

