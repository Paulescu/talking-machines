import json
import os
from typing import (
    Tuple,
    List,
)
from pathlib import Path
import codecs
import pdb

# from nltk.tokenize.treebank import TreebankWordTokenizer
# from tqdm.auto import tqdm
import pandas as pd
import torch
from torchtext.data import (
    Field,
    Dataset,
    TabularDataset,
    BucketIterator
)
from torchtext.data.utils import interleave_keys
from torchtext.data import Field
# from torchtext.vocab import Vocab
# from autocorrect import Speller

# from .util import pprint
from .tokenizer import tokenizer

# special tokens
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def generate_file_with_training_examples(
    input_file: Path,
    output_file: Path,
    n_past_utterances: int = 0,
    # include_persona: bool = False,
    include_partner: bool = False,
    autocorrect: bool = False,
):
    """Generates a json file with examples we can use to train ML models.

    Each example is a dictionary with keys:
    {
        'persona': ...,
        'src': ...,
        'target': ...
    }

    Args:
        input_file:
        output_file:
        n_past_utterances:
        include_persona:
        include_partner:
        include_other_speaker:
        autocorrect:

    """
    YOUR_PERSONA_STR = "your persona: "
    PARTNERS_PERSONA_STR = "partner's persona: "

    your_persona = []
    partners_persona = []
    your_history = []
    partners_history = []
    examples = []

    with codecs.open(input_file, 'r', 'utf-8', errors='ignore') as f:

        for idx, line in enumerate(f):

            # each line starts as an integer followed by a space.
            # e.g. 1 bla bla bla
            #      2 be be be
            #      3 yo yo yo
            # we store line_id and keep only the characters after the space.
            position_first_space = line.find(' ')
            line_id = int(line[:position_first_space])
            line = line[position_first_space + 1:]

            if line_id == 1:
                your_persona = []
                partners_persona = []
                your_history = []
                partners_history = []
                prev_partners_utterance = None

            if line.startswith(YOUR_PERSONA_STR):
                # your persona description
                your_persona.append(line[len(YOUR_PERSONA_STR):])
                # continue

            elif line.startswith(PARTNERS_PERSONA_STR):
                # partners persona description
                partners_persona.append(line[len(PARTNERS_PERSONA_STR):])
                # continue

            else:
                # extract source, target sentences
                partners_utterance, your_reply, _, _ = line.split('\t')
                your_history.append(partners_utterance)

                examples.append({
                    'persona': ' '.join(your_persona),
                    'src': ' '.join(your_history[-n_past_utterances:]),
                    'tgt': your_reply,
                })
                your_history.append(your_reply)

                if prev_partners_utterance and include_partner:
                    partners_history.append(prev_partners_utterance)
                    partners_history.append(prev_your_reply)

                    examples.append({
                        'persona': ' '.join(partners_persona),
                        'src': ' '.join(partners_history[-n_past_utterances:]),
                        'tgt': partners_utterance,
                    })

                prev_partners_utterance, prev_your_reply = partners_utterance, your_reply

            # if (idx > 0) and (line_id == 1):
            #     pprint(examples)
            #     break

    load_list_dicts_from_file()


    print(f'Generated {output_file}, {len(examples):,} examples')


def load_list_dicts_from_file(
    file: str,
    n_elements: int = None,
) -> List[dict]:

    list_dicts = []
    counter = 0
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            list_dicts.append(json.loads(line))
            counter += 1
            if n_elements and (counter >= n_elements):
                break

    return list_dicts

def save_list_dicts_to_file(
    list_dicts: List[dict],
    file: str,
):
    with open(file, 'w', encoding='utf-8') as f:

        # json.dump(examples, f)
        for ex in list_dicts:
            json.dump(ex, f)
            f.write('\n')


def get_sentence_processor(
    train_file: Path,
    min_word_freq : int = 1,
    max_vocab_size: int = 20000,
    use_persona_info: bool = False,
    glove_vectors: str = 'glove.6B.300d',
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

    if use_persona_info:
        # fields = [('persona', sentence_processor),
        #           ('src', sentence_processor), ('tgt', sentence_processor)]
        fields = {
            'persona': ('persona', sentence_processor),
            'src': ('src', sentence_processor),
            'tgt': ('tgt', sentence_processor),
        }
    else:
        # fields = [('src', sentence_processor), ('tgt', sentence_processor)]
        fields = {
            'src': ('src', sentence_processor),
            'tgt': ('tgt', sentence_processor),
        }

    # pdb.set_trace()
    train_ds = TabularDataset(
        path=train_file,
        format='json',
        fields=fields,
    )
    # pdb.set_trace()
    if glove_vectors:
        # vocabulary from GloVe
        sentence_processor.build_vocab(
            train_ds,
            min_freq=min_word_freq,
            max_size=max_vocab_size,
            vectors=glove_vectors,
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
        new_train_path = os.path.join(path_to_files, f'train_{train_size}.json')


        pdb.set_trace()

        pd.read_csv(train_path, header=None) \
            .head(train_size) \
            .to_csv(new_train_path, index=False, header=None)
        # train_path = new_train_path

        # TODO




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
    use_persona_info: bool = False,
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
        new_train_path = os.path.join(path, f'train_{train_size}.json')

        save_list_dicts_to_file(
            load_list_dicts_from_file(train_path, n_elements=train_size),
            file=new_train_path
        )

        train_path = new_train_path

    if use_persona_info:
        fields = {
            'persona': ('persona', sentence_processor),
            'src': ('src', sentence_processor),
            'tgt': ('tgt', sentence_processor),
        }
    else:
        # fields = [('src', sentence_processor), ('tgt', sentence_processor)]
        fields = {
            'src': ('src', sentence_processor),
            'tgt': ('tgt', sentence_processor),
        }

    # generate potentially smaller datasets
    train_ds, val_ds, test_ds = TabularDataset.splits(
        path='',
        train=train_path,
        validation=val_path,
        test=test_path,
        format='json',
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
    n_examples_per_batch: int = None,
    n_tokens_per_batch: int = None,
    device = None
) -> Tuple[BucketIterator, BucketIterator, BucketIterator]:

    # validate inputs
    if n_examples_per_batch and n_tokens_per_batch:
        raise Exception('You cannot set both n_examples_per_batch and n_tokens_per_batch')
    if not n_examples_per_batch and not n_tokens_per_batch:
        raise Exception('You have to set either n_examples_per_batch or n_tokens_per_batch')

    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    longest_src_sentence = 0
    longest_tgt_sentence = 0

    def sort_key(ex):
        """
        Heuristic that helps the BucketIterator group examples that have
        similar length, and minimize padding
        """
        # TODO: take into account ex.persona
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

    if n_examples_per_batch:
        train_iter, val_iter, test_iter = BucketIterator.splits(
            (train_dataset, val_dataset, test_dataset),
            batch_size=n_examples_per_batch,
            device=device,
            sort_key=sort_key,
            sort_within_batch=True)

    elif n_tokens_per_batch:
        train_iter, val_iter, test_iter = BucketIterator.splits(
            (train_dataset, val_dataset, test_dataset),
            batch_size=n_tokens_per_batch,
            device=device,
            sort_key=sort_key,
            sort_within_batch=True,
            batch_size_fn=batch_size_fn)
    else:
        raise Exception('')

    return train_iter, val_iter, test_iter