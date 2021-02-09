from pathlib import Path
from typing import Union
import pickle
import random

import pandas as pd

from .data import load_raw_data


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

def pprint(x):
    import json
    print(json.dumps(x, indent=4, sort_keys=True))


def print_random_example(data_dir: Path, is_train: bool = True):

    print('Generated sentence pairs:')
    processed_data = pd.read_csv(data_dir / 'train.csv', header=None) if is_train \
        else pd.read_csv(data_dir / 'test.csv', header=None)
    processed_data.columns = ['id', 'src', 'tgt']

    row = random.randint(0, len(processed_data))
    id = processed_data['id'][row]
    print('id: ', id)
    print('src: ', processed_data['src'][row])
    print('tgt: ', processed_data['tgt'][row])

    print('\nOriginal data:')
    train, test = load_raw_data(data_dir / 'personachat_self_original.json')
    original_data = train if is_train else test
    pprint(original_data[id]['utterances'][-1])
