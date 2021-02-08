from pathlib import Path
from typing import Union
import pickle

def save_vocab(vocab, path: Union[Path, str]):
    """Saves vocab to disk"""
    with open(path, "wb") as f:
        pickle.dump(vocab, f)

def load_vocab(path: Union[str, Path]):
    """Loads vocab from disk and returns it"""
    with open(path, "rb") as f:
        return pickle.load(f)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
