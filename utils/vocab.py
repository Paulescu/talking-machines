import os
from typing import (
    Tuple,
    List,
    Union
)
from pathlib import Path
import pickle
from collections import Counter
import io
import re
import pdb

import pandas as pd
import spacy
import torch
from torch.utils.data import Dataset
from torchtext.data import (
    Field,
    TabularDataset,
)
from torchtext.vocab import Vocab

from .tokenizer import tokenizer
from .constants import *

def get_vocab(csv_file: Union[str, Path],
              use_glove: bool = False) -> Vocab:

    # create Vocab object
    counter = Counter()
    file_path = os.path.join(DATA_DIR, csv_file)
    with io.open(file_path, encoding="utf8") as f:
        for string_ in f:
            counter.update(tokenizer(string_))
    vocab = Vocab(counter,specials=[UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN])

    if use_glove:
        # from torchtext.vocab import Glove
        # GloVe(name='6B', dim=100, **kwargs)
        vocab.load_vectors('glove.6B.100d')

    return vocab


class WordVocab(Vocab):

    def __init__(self, corpus_path):

        counter = Counter()
        # file_path = os.path.join(DATA_DIR, csv_file)
        with io.open(corpus_path, encoding="utf8") as f:
            for string_ in f:
                counter.update(tokenizer(string_))
        
        super().__init__(counter,
                         specials=[UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN])
    
    # def build_from_file(self, csv_file):
    
    #     counter = Counter()
    #     file_path = os.path.join(DATA_DIR, csv_file)
    #     with io.open(file_path, encoding="utf8") as f:
    #         for string_ in f:
    #             counter.update(tokenizer(string_))
    #     self.vocab = Vocab(counter,
    #                        specials=[UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN])

    def load_glove_vectors(self):
        self.load_vectors('glove.6B.100d')

    def save(self, path: Union[Path, str]):
        """Saves vocab to disk"""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Union[str, Path]):
        """Loads vocab from disk and returns it"""
        with open(path, "rb") as f:
            return pickle.load(f)

    @property
    def size(self):
        return len(self)

    @property
    def vectors_dim(self):
        return self.vectors.shape[1]

    @property
    def pad_token_id(self):
        """Returns the integer representation of the PAD_TOKEN"""
        return self.stoi[PAD_TOKEN]