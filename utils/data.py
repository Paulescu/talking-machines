import json
import sys
import os
from typing import (
    Tuple,
    List,
    Union
)
from pathlib import Path
import pickle
import wget
import pdb

import pandas as pd
import spacy
import torch
from torchtext.data import (
    Field,
    Dataset,
    TabularDataset,
    BucketIterator
)
# from torchtext.data.utils import get_tokenizer
from torchtext.data.utils import interleave_keys
from torchtext.vocab import Vocab
from autocorrect import Speller
from tqdm.auto import tqdm


# from .vocab import WordVocab
# from .tokenizer import tokenizer
# from .constants import *

# special tokens
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

DATASETS = {
    'convai': 'https://raw.githubusercontent.com/DeepPavlov/convai/master/data/export_2018-07-07_train.json',
    'personachat': 'https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json',
}

# NEW stuff start
import codecs

def generate_file_with_sentence_pairs(
    input_file: Path,
    output_file: Path,
    n_past_utterances: int = 0,
    autocorrect: bool = False,
):

    pairs = list()
    past_utterances = []

    with codecs.open(input_file, 'r', 'utf-8', errors='ignore') as f:

        for line in f:

            # each line starts as an integer followed by a space.
            # e.g. 1 bla bla bla
            #      2 be be be
            #      3 yo yo yo
            # we store line_id and keep only the characters after the space.
            position_first_space = line.find(' ')
            line_id = int(line[:position_first_space])
            line = line[position_first_space + 1:]

            # extract source, target sentences
            src_sentence, tgt_sentence, _, _ = line.split('\t')

            if line_id == 1:
                # no history
                past_utterances = [src_sentence]
            else:
                # append history
                past_utterances.append(src_sentence)

            # if previous_target_line is not None:
            #
            #     pdb.set_trace()
            #
            #     # store pair (source, target) from previous iteration
            #     pairs.append([previous_target_line, source_line])

            # store pair (source, target) from this iteration
            # past_utterances.append(source_line)
            conversation_history = ' '.join(past_utterances[-n_past_utterances:])
            pairs.append([conversation_history, tgt_sentence])

            # update 'past_utterances' for the next loop iteration
            past_utterances.append(tgt_sentence)

            # if line_id == 7:
            #     break

    pd.DataFrame(pairs).to_csv(output_file, index=False, header=False)
    print(f'Generated {output_file}')

    # df =  pd.DataFrame(pairs)
    # pdb.set_trace()


from nltk.tokenize.treebank import TreebankWordTokenizer
word_tokenizer = TreebankWordTokenizer()

def tokenizer(sentence):
    return word_tokenizer.tokenize(sentence)

def get_sentence_processor(
        train_file: Path,
        min_word_freq : int = 3,
        max_vocab_size: int = 10000,
        use_glove_vectors: bool = False,
        vectors_cache: Path = None,
) -> Field:
    """"""
    sentence_processor = Field(
        tokenize=tokenizer,
        init_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        batch_first=True,
        include_lengths=True,
        lower=True,
    )
    fields = [('src', sentence_processor), ('tgt', sentence_processor)]

    train_ds = TabularDataset(
        path=train_file,
        format='csv',
        skip_header=False,
        fields=fields,
    )

    if use_glove_vectors:
        # vocabulary from GloVe
        sentence_processor.build_vocab(
            train_ds,
            min_freq=min_word_freq,
            max_size=max_vocab_size,
            vectors='glove.6B.300d',
            vectors_cache=vectors_cache,
        )
    else:
        # new vocabulary from scratch
        sentence_processor.build_vocab(
            train_ds,
            min_freq=min_word_freq,
            max_size=max_vocab_size,
        )

    return sentence_processor

# NEW stuff end

def download_data(dataset: str, destination_dir: str):
    """
    Downloads the initial dataset from remote URL
    """
    def bar_progress(current, total, width=80):
        """Auxiliary function to print progress bar while downloading"""
        progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
        # Don't use print() as it will print in new line every time.
        sys.stdout.write("\r" + progress_message)
        sys.stdout.flush()

    url = DATASETS[dataset]
    print(f'Downloading from {url}...')
    destination_file = str(Path(destination_dir) / url.split('/')[-1])
    wget.download(url, destination_file, bar=bar_progress)


def load_raw_data(file: Union[str, Path]) -> Tuple[List, List]:
    """
    Returns training data and test data lists
    """
    with open(file) as json_file:
        data = json.load(json_file)
    return data['train'], data['valid']

def get_vocab(
        train_file: Path,
        min_word_freq : int = 3,
        use_glove_vectors: bool = False
) -> Vocab:
    """"""
    sentence_processor = Field(
        tokenize=tokenizer,
        init_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        batch_first=True,
        include_lengths=True,
        lower=True,
    )
    fields = [('id', None),
              ('src', sentence_processor), ('tgt', sentence_processor)]
    train_ds = TabularDataset(
        path=train_file,
        format='csv',
        skip_header=False,
        fields=fields,
    )

    if use_glove_vectors:
        # vocabulary from GloVe
        sentence_processor.build_vocab(
            train_ds,
            min_freq=min_word_freq,
            vectors='glove.6B.100d'
        )
    else:
        # new vocabulary from scratch
        sentence_processor.build_vocab(
            train_ds, min_freq=min_word_freq)

    return sentence_processor.vocab


from torchtext.data import Field

def get_torchtext_field(
        train_file: Path,
        min_word_freq : int = 3,
        max_vocab_size: int = 10000,
        use_glove_vectors: bool = False,
        vectors_cache: Path = None,
) -> Field:
    """"""
    sentence_processor = Field(
        tokenize=tokenizer,
        init_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        batch_first=True,
        include_lengths=True,
        lower=True,
    )
    fields = [('id', None),
              ('src', sentence_processor),
              ('tgt', sentence_processor)]

    train_ds = TabularDataset(
        path=train_file,
        format='csv',
        skip_header=False,
        fields=fields,
    )

    if use_glove_vectors:
        # vocabulary from GloVe
        sentence_processor.build_vocab(
            train_ds,
            min_freq=min_word_freq,
            max_size=max_vocab_size,
            vectors='glove.6B.100d',
            vectors_cache=vectors_cache,
        )
    else:
        # new vocabulary from scratch
        sentence_processor.build_vocab(
            train_ds,
            min_freq=min_word_freq,
            max_size=max_vocab_size,
        )

    return sentence_processor

def generate_train_val_test_files(
        raw_data_file: Path,
        names: Tuple[str] = ('train.csv', 'val.csv', 'test.csv'),
        autocorrect=False,
        n_utterances_history=99999
):
    """
    Generates 3 CSV files: train, validation, test.
    """
    train_data, test_data = load_raw_data(raw_data_file)

    # build sentence pairs (context, next_utterance) from raw data
    train_pairs = build_sentence_pairs_from_raw_data(
        train_data,
        autocorrect=autocorrect,
        n_utterances_history=n_utterances_history
    )
    print(f'Train set {len(train_pairs):,}')
    
    test_pairs = build_sentence_pairs_from_raw_data(
        test_data,
        autocorrect=autocorrect,
        n_utterances_history=n_utterances_history
    )
    print(f'Test set {len(test_pairs):,}')

    # save sentence pairs into train, validation and test CSV files
    train_file = Path(raw_data_file).resolve().parent / names[0]
    pd.DataFrame(train_pairs[:-10000]).to_csv(train_file, index=False,
                                              header=False) # .sample(frac=1)
    print(f'Saved {train_file}')

    val_file = Path(raw_data_file).resolve().parent / names[1]
    pd.DataFrame(train_pairs[-10000:]).to_csv(val_file, index=False,
                                              header=False)
    print(f'Saved {val_file}')

    test_file = Path(raw_data_file).resolve().parent / names[2]
    pd.DataFrame(test_pairs).to_csv(test_file, index=False, header=False)   
    print(f'Saved {test_file}')


def build_sentence_pairs_from_raw_data(
        conversations: List,
        autocorrect=False,
        n_utterances_history=99999
):
    """
    Returns list of pairs (context, next_utterance) from the raw json file
    'conversations'
    """
    pairs = []
    removed = 0
    spell = Speller(fast=True)

    for id, conversation in enumerate(tqdm(conversations)):

        for utterance in conversation['utterances']:
            next_utterance = utterance['candidates'][-1]
            past_utterances = utterance['history'][-n_utterances_history:]

            if not past_utterances:
                # no past utterances, skip
                continue

            if past_utterances[0] == '__ SILENCE __':
                history = ' '.join(past_utterances[1:])
            else:
                history = ' '.join(past_utterances)

            # if id == 13980:
            #     pdb.set_trace()
            if history == '':
                continue

            if autocorrect:
                next_utterance = spell(next_utterance)
                history = spell(history)

            pairs.append([id, history, next_utterance])
            # pairs.append([history, next_utterance])

    print(f'{removed:,} lines removed')
    return pairs


def get_datasets_and_vocab(
        path_to_files,
        train,
        validation,
        test,
        train_size: int = None,
        validation_size: int = None,
        use_glove_vectors: bool = False,
        vectors_cache: str = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load and return PyTorch Datasets train, validation, test sets.

    By using torchtext APIs: Field(), TabularDataset.split() we avoid
    writing a lot of boilerplate code.
    """
    train_path = os.path.join(path_to_files, train)
    validation_path = os.path.join(path_to_files, validation)
    test_path = os.path.join(path_to_files, test)

    if train_size:
        # generate a temporal smaller version of the train set
        new_train_path = os.path.join(path_to_files, f'train_{train_size}.csv')
        pd.read_csv(train_path, header=None) \
            .head(train_size) \
            .to_csv(new_train_path, index=False, header=None)
        # train_path = new_train_path
    else:
        new_train_path = train_path

    if validation_size:
        # generate a temporal smaller version of the validation set
        new_validation_path = os.path.join(path_to_files,
                                           f'validation_{validation_size}.csv')
        pd.read_csv(validation_path, header=None) \
            .head(validation_size) \
            .to_csv(new_validation_path, index=False, header=None)
        validation_path = new_validation_path

    sentence_processor = Field(
        tokenize=tokenizer,
        init_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        # unk_token=UNK_TOKEN,
        batch_first=True,
        include_lengths=True,
        lower=True,
    )
    fields = [('id', None), ('src', sentence_processor), ('tgt', sentence_processor)]
    train_dataset = TabularDataset.splits(
        path='',
        train=train_path,
        format='csv',
        skip_header=False,
        fields=fields,
    )

    if use_glove_vectors:
        # vocabulary from GloVe
        sentence_processor.build_vocab(train_dataset,
                                       min_freq=3,
                                       vectors='glove.6B.100d',
                                       vectors_cache=vectors_cache)
    else:
        # new vocabulary from scratch
        sentence_processor.build_vocab(train_dataset, min_freq=3)

    # TODO: here

    # generate potentially smaller datasets
    train_dataset, validation_dataset, test_dataset = TabularDataset.splits(
        path='',
        train=new_train_path,
        validation=validation_path,
        test=test_path,
        format='csv',
        skip_header=False,
        fields=fields,
    )

    if train_size:
        # delete temporary file
        os.remove(train_path)
    
    if validation_size:
        # delete temporary file
        os.remove(validation_path)

    return train_dataset, validation_dataset, test_dataset, sentence_processor.vocab


def get_datasets(
        path: Path,
        train: str,
        val: str,
        test: str,
        sentence_processor: Field,
        train_size: int = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load and return PyTorch Datasets train, validation, test sets.

    By using torchtext APIs: Field(), TabularDataset.split() we avoid
    writing a lot of boilerplate code.
    """
    train_path = os.path.join(path, train)
    val_path = os.path.join(path, val)
    test_path = os.path.join(path, test)

    if train_size:
        # generate a temporal smaller version of the train set
        new_train_path = os.path.join(path, f'train_{train_size}.csv')
        pd.read_csv(train_path, header=None) \
            .head(train_size) \
            .to_csv(new_train_path, index=False, header=None)

        train_path = new_train_path

    fields = [('src', sentence_processor), ('tgt', sentence_processor)]

    # generate potentially smaller datasets
    train_ds, val_ds, test_ds = TabularDataset.splits(
        path='',
        train=train_path,
        validation=val_path,
        test=test_path,
        format='csv',
        skip_header=False,
        fields=fields,
    )

    if train_size:
        # delete temporary file
        os.remove(train_path)

    return train_ds, val_ds, test_ds

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

    train_iter, val_iter, test_iter = BucketIterator.splits(
        (train_dataset, val_dataset, test_dataset),
        batch_size=batch_size,
        device=device,
        sort_key=sort_key,
        sort_within_batch=True,
        # batch_size_fn=batch_size_fn,
    )

    # pdb.set_trace()

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


