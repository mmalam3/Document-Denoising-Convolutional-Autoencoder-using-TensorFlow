# import necessary libraries
import os
from pathlib import Path

# set the root directory
ROOT_DIR = Path(__file__).parent.parent.parent

# set data directories
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')


