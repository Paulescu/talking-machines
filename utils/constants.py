import os
from pathlib import Path

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = Path(THIS_FILE_DIR).resolve().parent / 'data'
ARTIFACTS_DIR = Path(THIS_FILE_DIR).resolve().parent / 'artifacts'

# print(DATA_DIR)
# BINARIES_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'binaries')
# DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, 'data')
# os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
# os.makedirs(BINARIES_PATH, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# print(DATA_DIR)
# print(os.path.dirname(os.path.abspath(__file__)))
# THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
