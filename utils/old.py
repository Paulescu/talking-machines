# class TrainingDataWrapper:   
#     """
#     Encapsulates functionality to generate PyTorch datasets and dataloaders
#     at training time
#     """

#     def __init__(self):

#         self.vocab = None
#         self.embeddings = None

#     def get_datasets(self,
#                      train_size: Optional[int],
#                      val_size: Optional[int],
#                      use_glove: Optional[bool] = False) -> Tuple[Dataset, Dataset, Dataset]:
#         """
#         Load and return train, validation, test sets with tokenized and
#         numericalized inputs.

#         By using torchtext APIs: Field(), TabularDataset.split() we avoid
#         writing a lot of boilerplate code.
#         """

#         # generate a temporal smaller version of the train set
#         original_file = os.path.join(DATA_DIR, 'train.csv')
#         new_train_file = os.path.join(DATA_DIR, f'train_{train_size}.csv')
#         pd.read_csv(original_file, header=None). \
#             head(train_size). \
#             to_csv(new_train_file, index=False, header=None)

#         # generate a temporal smaller version of the validation set
#         original_file = os.path.join(DATA_DIR, 'val.csv')
#         new_validation_file = os.path.join(DATA_DIR, f'val_{val_size}.csv')
#         pd.read_csv(original_file, header=None) \
#             .head(val_size) \
#             .to_csv(new_validation_file, index=False, header=None)

#         # we tell torchtext we want to lowercase text and tokenize it using
#         # the given 'tokenizer_fn'
#         sentence_processor = Field(
#             tokenize=tokenizer_fn,
#             init_token=BOS_TOKEN,
#             eos_token=EOS_TOKEN,
#             pad_token=PAD_TOKEN,
#             batch_first=True,
#             include_lengths=True,
#             lower=True,
#         )
#         fields = [('src', sentence_processor), ('tgt', sentence_processor)]

#         # we tell torchtext the files in disk to look for, and how the text
#         # data is organized in these files.
#         # In this case, each file has 2 columns 'src' and 'tgt' 
#         train, validation, test = TabularDataset.splits(
#             path='',
#             train=new_train_file,
#             validation=new_validation_file,
#             test=os.path.join(DATA_DIR, 'test.csv'),
#             format='csv',
#             skip_header=False,
#             fields=fields,
#         )

#         # build vocabulary using train set only
#         if use_glove:
#             # vocabulary from GloVe
#             sentence_processor.build_vocab(train,
#                                            min_freq=3,
#                                            vectors='glove.6B.100d')
#             self.embeddings = sentence_processor.vocab.vectors
#         else:
#             # new vocabulary from scratch
#             sentence_processor.build_vocab(train, min_freq=3)
        
#         # store the vocabulary, very important! We need to use the same
#         # at inference time.
#         self.vocab = sentence_processor.vocab
        
#         # delete temporary files generated at the start of this function
#         os.remove(new_train_file)
#         os.remove(new_validation_file)

#         return train, validation, test

#     def get_dataloaders(self,
#                         train, validation, test,
#                         batch_size=2400,
#                         device=None):
#         if not device:
#             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         longest_src_sentence = 0
#         longest_tgt_sentence = 0

#         def sort_key(ex):
#             """
#             Heuristic that helps the BucketIterator group examples that have
#             similar length, and minimize padding
#             """
#             return interleave_keys(len(ex.src), len(ex.tgt))

#         def batch_size_fn(new_example, count, sofar):
#             """
#             Auxilliary function that returns the maximum number of tokens
#             in the current batch.
#             """
#             global longest_src_sentence, longest_tgt_sentence

#             if count == 1:
#                 longest_src_sentence = 0
#                 longest_tgt_sentence = 0

#             longest_src_sentence = max(longest_src_sentence,
#                                        len(new_example.src))
#             # +2 because of start/end of sentence tokens (<s> and </s>)
#             longest_tgt_sentence = max(longest_tgt_sentence,
#                                        len(new_example.tgt) + 2)

#             num_of_tokens_in_src_tensor = count * longest_src_sentence
#             num_of_tokens_in_tgt_tensor = count * longest_tgt_sentence

#             return max(num_of_tokens_in_src_tensor, num_of_tokens_in_tgt_tensor)

#         train_iter, validation_iter, test_iter = BucketIterator.splits(
#             (train, validation, test),
#             batch_size=batch_size,
#             device=device,
#             sort_key=sort_key,
#             sort_within_batch=True,
#             batch_size_fn=batch_size_fn,
#         )

#         return train_iter, validation_iter, test_iter

#     @property
#     def vocab_size(self):
#         return len(self.vocab)

#     @property
#     def embedding_dim(self):
#         return self.vocab.vectors.shape[1]

#     @property
#     def pad_token_id(self):
#         """Returns the integer representation of the PAD_TOKEN"""
#         return self.vocab.stoi[PAD_TOKEN]