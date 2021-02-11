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


import matplotlib.pyplot as plt
from utils.tokenizer import tokenizer
def plot_sentence_lengths(
        train_file: Path,
        test_file: Path,
) -> 'something':
    figure, ax = plt.subplots(2, 2, figsize=(12, 8))

    # train
    train = pd.read_csv(train_file, header=None)
    train.columns = ['id', 'src', 'tgt']
    train_src_len = train['src'].apply(lambda x: len(tokenizer(x)))
    train_tgt_len = train['tgt'].apply(lambda x: len(tokenizer(x)))

    ax[0, 0].hist(train_src_len, bins=100)
    ax[0, 0].set_title('Train, src_len')
    ax[0, 1].hist(train_tgt_len, bins=100)
    ax[0, 1].set_title('Train, tgt_len')

    # test
    test = pd.read_csv(test_file, header=None)
    test.columns = ['id', 'src', 'tgt']
    test_src_len = test['src'].apply(lambda x: len(tokenizer(x)))
    test_tgt_len = test['tgt'].apply(lambda x: len(tokenizer(x)))

    ax[1, 0].hist(test_src_len, bins=100)
    ax[1, 0].set_title('Test, src_len')
    ax[1, 1].hist(test_tgt_len, bins=100)
    ax[1, 1].set_title('Test, tgt_len')

    # return ax