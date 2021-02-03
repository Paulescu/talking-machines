import json
import os
from typing import (
    Tuple,
    List,
    Union
)
from pathlib import Path
import pickle

import pandas as pd
import spacy
import torch
# from torch.utils.data import (
#     Dataset,
#     DataLoader
# )
from torchtext.data import (
    Field,
    Dataset,
    TabularDataset,
    BucketIterator
)
from torchtext.data.utils import get_tokenizer
from torchtext.data.utils import interleave_keys
from autocorrect import Speller
from tqdm.auto import tqdm

from .vocab import WordVocab
from .tokenizer import tokenizer
from .constants import *

# spacy_en = spacy.load('en_core_web_sm')


def generate_train_validation_test_files(autocorrect=False):
    """
    Generates train, validation, and test conversations from the
    raw data files.
    """
    # load raw data from json file
    train_data, test_data = load_raw_data()

    # build sentence pairs (context, next_utterance) from raw data
    train_pairs = build_sentence_pairs_from_raw_data(
        train_data,
        autocorrect=autocorrect
    )
    print(f'Train set {len(train_pairs):,}')
    
    test_pairs = build_sentence_pairs_from_raw_data(
        test_data,
        autocorrect=autocorrect
    )
    print(f'Test set {len(test_pairs):,}')

    # save sentence pairs into train, validation and test CSV files
    pd.DataFrame(train_pairs[:-10000]).to_csv(
        os.path.join(DATA_DIR, 'train.csv'),
        index=False,
        header=False)
    pd.DataFrame(train_pairs[-10000:]).to_csv(
        os.path.join(DATA_DIR, 'val.csv'),
        index=False,
        header=False)
    pd.DataFrame(test_pairs).to_csv(
        os.path.join(DATA_DIR, 'test.csv'),
        index=False,
        header=False)


def load_raw_data() -> Tuple[List, List]:
    """
    Returns training data and test data
    """
    file_path = os.path.join(DATA_DIR, 'personachat_self_original.json')
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data['train'], data['valid']


def build_sentence_pairs_from_raw_data(conversations, autocorrect=False):
    """
    Returns list of pairs (context, next_utterance) from the raw json file
    'conversations'
    """
    pairs = []
    removed = 0
    spell = Speller(fast=True)

    for conversation in tqdm(conversations):
        for utterance in conversation['utterances']:
            next_utterance = utterance['candidates'][-1]
            
            # if utterance['history'][0] == '__ SILENCE __':
            #     # import pdb
            #     # pdb.set_trace()
            #     history = '.'.join(utterance['history'][1:])
            # else:
            history = '.'.join(utterance['history'])

            # if not history:
            #     continue

            if autocorrect:
                next_utterance = spell(next_utterance)
                history = spell(history)

            pairs.append([history, next_utterance])

    print(f'{removed:,} lines removed')
    return pairs

# spacy_en = spacy.load('en_core_web_sm')
# def tokenizer_fn(text):
#     return [tok.text for tok in spacy_en.tokenizer(text)]

def get_datasets(vocab: WordVocab,
                 train_size: int = None,
                 val_size: int = None) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load and return PyTorch Datasets train, validation, test sets with tokenized
    and numericalized inputs.

    By using torchtext APIs: Field(), TabularDataset.split() we avoid
    writing a lot of boilerplate code.
    """
    if train_size:
        # generate a temporal smaller version of the train set
        new_train_file = os.path.join(DATA_DIR, f'train_{train_size}.csv')
        pd.read_csv(
            os.path.join(DATA_DIR, 'train.csv'),
            header=None
        ).head(train_size).to_csv(new_train_file, index=False, header=None)

    if val_size:
        # generate a temporal smaller version of the validation set
        new_validation_file = os.path.join(DATA_DIR, f'val_{val_size}.csv')
        pd.read_csv(
            os.path.join(DATA_DIR, 'val.csv'),
            header=None
        ).head(val_size).to_csv(new_validation_file, index=False, header=None)

    # we tell torchtext we want to lowercase text and tokenize it using
    # the given 'tokenizer_fn'
    tokenizer = get_tokenizer('basic_english', language='en')
    sentence_processor = Field(
        tokenize=tokenizer,
        init_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
        batch_first=True,
        include_lengths=True,
        lower=True,
    )
    fields = [('src', sentence_processor), ('tgt', sentence_processor)]

    # assign the previously constructed torchtext.vocab.Vocab
    sentence_processor.vocab = vocab

    # we tell torchtext the files in disk to look for, and how the text
    # data is organized in these files.
    # In this case, each file has 2 columns 'src' and 'tgt' 
    train, val, test = TabularDataset.splits(
        path='',
        train=new_train_file,
        validation=new_validation_file,
        test=os.path.join(DATA_DIR, 'test.csv'),
        format='csv',
        skip_header=False,
        fields=fields,
    )
    
    if train_size:
        # delete temporary file
        os.remove(new_train_file)
    
    if val_size:
        # delete temporary file
        os.remove(new_validation_file)

    return train, val, test

def get_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 2400,
    device = None
) -> Tuple[BucketIterator, BucketIterator, BucketIterator]:

    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    longest_src_sentence = 0
    longest_tgt_sentence = 0

    def sort_key(ex):
        """
        Heuristic that helps the BucketIterator group examples that have
        similar length, and minimize padding
        """
        return interleave_keys(len(ex.src), len(ex.tgt))

    def batch_size_fn(new_example, count, sofar):
        """
        Auxilliary function that returns the maximum number of tokens
        in the current batch.
        """
        global longest_src_sentence, longest_tgt_sentence

        if count == 1:
            longest_src_sentence = 0
            longest_tgt_sentence = 0

        longest_src_sentence = max(longest_src_sentence,
                                    len(new_example.src))
        # +2 because of start/end of sentence tokens (<s> and </s>)
        longest_tgt_sentence = max(longest_tgt_sentence,
                                    len(new_example.tgt) + 2)

        num_of_tokens_in_src_tensor = count * longest_src_sentence
        num_of_tokens_in_tgt_tensor = count * longest_tgt_sentence

        return max(num_of_tokens_in_src_tensor, num_of_tokens_in_tgt_tensor)

    train_iter = BucketIterator(
        train_dataset,
        batch_size=batch_size,
        device=device,
        sort_key=sort_key,
        sort_within_batch=True,
        batch_size_fn=batch_size_fn,
        train=True
    )

    val_iter, test_iter = BucketIterator.splits(
        (val_dataset, test_dataset),
        batch_size=batch_size,
        device=device,
        sort_key=sort_key,
        sort_within_batch=True,
        batch_size_fn=batch_size_fn,
    )

    return train_iter, val_iter, test_iter


from torchtext.vocab import Vocab
def preprocess_sentence(self, sentence: str, vocab: Vocab):
    """
    TODO
    """
    # lowercase sentence
    sentence = sentence.lower()

    # tokenize
    tokens = self.tokenizer_fn(sentence)

    # numericalize
    output = [BOS_TOKEN]
    output += [self.vocab.stoi[t] for t in tokens]
    output += [EOS_TOKEN]

    return output