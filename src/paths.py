from pathlib import Path
import os

PARENT_DIR = Path(__file__).parent.resolve().parent
DATA_DIR = PARENT_DIR / 'data'
RAW_DATA_DIR = PARENT_DIR / 'data' / 'raw'
TRASNFORMED_DATA_DIR = PARENT_DIR / 'data' / 'transformed'
 
if not Path(DATA_DIR).exists():
    os.mkdir(DATA_DIR)

if not Path(RAW_DATA_DIR).exists():
    os.mkdir(RAW_DATA_DIR)

if not Path(TRASNFORMED_DATA_DIR).exists():
    os.mkdir(TRASNFORMED_DATA_DIR)