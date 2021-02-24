from pathlib import Path
from typing import Union
import pickle
import random
import uuid

import pandas as pd
import matplotlib.pyplot as plt

from .data import tokenizer

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


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pprint(x):
    import json
    print(json.dumps(x, indent=4, sort_keys=True))


def plot_sentence_lengths(
    train_file: Path,
    test_file: Path,
) -> 'something':
    figure, ax = plt.subplots(2, 2, figsize=(12, 8))

    # train
    train = pd.read_csv(train_file, header=None)
    train.columns = ['src', 'tgt']
    train_src_len = train['src'].apply(lambda x: len(tokenizer(x)))
    train_tgt_len = train['tgt'].apply(lambda x: len(tokenizer(x)))

    ax[0, 0].hist(train_src_len, bins=100)
    ax[0, 0].set_title('Train, src_len')
    ax[0, 1].hist(train_tgt_len, bins=100)
    ax[0, 1].set_title('Train, tgt_len')

    # test
    test = pd.read_csv(test_file, header=None)
    test.columns = ['src', 'tgt']
    test_src_len = test['src'].apply(lambda x: len(tokenizer(x)))
    test_tgt_len = test['tgt'].apply(lambda x: len(tokenizer(x)))

    ax[1, 0].hist(test_src_len, bins=100)
    ax[1, 0].set_title('Test, src_len')
    ax[1, 1].hist(test_tgt_len, bins=100)
    ax[1, 1].set_title('Test, tgt_len')


def get_random_id():
    return str(uuid.uuid1())


def get_dataset_size(dataloader):
    """Returns size of the underlying data behind the 'dataloader'"""
    batch = next(iter(dataloader))
    return len(batch.dataset)