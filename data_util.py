import json
import os
from typing import (
    Tuple,
    List
)

import pandas as pd
import spacy
import torch
from torch.utils.data import Dataset
import pandas as pd
from torchtext.data import (
    Field,
    TabularDataset,
    BucketIterator
)
from torchtext.data.utils import interleave_keys
from autocorrect import spell

DATA_DIR = './data'

# 
# Data pre-processing, from raw dataset to train, validation, test sets
#

def generate_train_validation_test_files():
    """Generates train, validation, and test conversations from the
    raw data files.
    """
    # load raw data from json file
    train_data, test_data = load_raw_data()

    # build sentence pairs (context, next_utterance) from raw data
    train_pairs = build_sentence_pairs_from_raw_data(train_data)
    print(f'Train set {len(train_pairs):,}')
    test_pairs = build_sentence_pairs_from_raw_data(test_data)
    print(f'Test set {len(test_pairs):,}')

    # save sentences pairs into separate train, validation and test CSV files
    pd.DataFrame(train_pairs[:-10000]).to_csv(
        os.path.join(DATA_DIR, 'train.csv'),
        index=False,
        header=False)
    pd.DataFrame(train_pairs[-10000:]).to_csv(
        os.path.join(DATA_DIR, 'validation.csv'),
        index=False,
        header=False)
    pd.DataFrame(test_pairs).to_csv(
        os.path.join(DATA_DIR, 'test.csv'),
        index=False,
        header=False)

def load_raw_data() -> Tuple[List, List]:
    """Returns training data and test data
    """
    file_path = os.path.join(DATA_DIR, 'personachat_self_original.json')
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data['train'], data['valid']


def build_sentence_pairs_from_raw_data(conversations, autocorrect=False):
    """Returns list of pairs (context, next_utterance) from the raw json file
    'conversations'
    """
    pairs = []
    removed = 0
    for conversation in conversations:
        for utterance in conversation['utterances']:
            next_utterance = utterance['candidates'][-1]
            history = '.'.join(utterance['history'])

            if autocorrect:
                next_utterance = spell(next_utterance)
                history = spell(history)

            pairs.append([history, next_utterance])

    print(f'{removed:,} lines removed')
    return pairs

#
# Data loading
#
spacy_en = spacy.load('en_core_web_sm')

class DataWrapper:
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    PAD_TOKEN = "<pad>"

    @staticmethod
    def tokenizer_fn(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def embedding_dim(self):
        return self.vocab.vectors.shape[1]

    def __init__(self):

        self.vocab = None
        self.embeddings = None

    def get_datasets(self,
                     train_size, 
                     val_size,
                     use_glove=False) -> Tuple[Dataset, Dataset, Dataset]:
        
        # generate a temporal smaller version of the train set
        original_file = os.path.join(DATA_DIR, 'train.csv')
        new_train_file = os.path.join(DATA_DIR, f'train_{train_size}.csv')
        pd.read_csv(original_file, header=None). \
            head(train_size). \
            to_csv(new_train_file, index=False, header=None)

        # generate a temporal smaller version of the validation set
        original_file = os.path.join(DATA_DIR, 'validation.csv')
        new_validation_file = os.path.join(DATA_DIR, f'validation_{val_size}.csv')
        pd.read_csv(original_file, header=None) \
            .head(val_size) \
            .to_csv(new_validation_file, index=False, header=None)

        sentence_processor = Field(
            tokenize=self.tokenizer_fn,
            init_token=self.BOS_TOKEN,
            eos_token=self.EOS_TOKEN,
            pad_token=self.PAD_TOKEN,
            batch_first=True,
            include_lengths=True,
            lower=True,
        )
        fields = [('src', sentence_processor), ('tgt', sentence_processor)]

        train, validation, test = TabularDataset.splits(
            path='',
            train=new_train_file,
            validation=new_validation_file,
            test=os.path.join(DATA_DIR, 'test.csv'),
            format='csv',
            skip_header=False,
            fields=fields,
        )

        # build vocabulary using train set only
        if use_glove:
            # vocabulary from GloVe
            sentence_processor.build_vocab(train,
                                           min_freq=3,
                                           vectors='glove.6B.100d')
            self.embeddings = sentence_processor.vocab.vectors
        else:
            # new vocabulary from scratch
            sentence_processor.build_vocab(train, min_freq=3)
        
        self.vocab = sentence_processor.vocab
        
        # delete temporary files generated at the start of this function
        os.remove(new_train_file)
        os.remove(new_validation_file)

        return train, validation, test

    def get_dataloaders(self,
                        train, validation, test,
                        batch_size=2400,
                        device=None):
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

        train_iter, validation_iter, test_iter = BucketIterator.splits(
            (train, validation, test),
            batch_size=batch_size,
            device=device,
            sort_key=sort_key,
            sort_within_batch=True,
            batch_size_fn=batch_size_fn,
        )

        return train_iter, validation_iter, test_iter

    @property
    def pad_token_id(self):
        """Returns the integer representation of the PAD_TOKEN"""
        return self.vocab.stoi[self.PAD_TOKEN]

    def tokenStr2Int(self, token_str):
        return self.vocab.stoi[token_str]

def get_dataset_size(dataloader):
    """Returns size of the underlying data behind the 'dataloader'"""
    batch = next(iter(dataloader))
    return len(batch.dataset)
