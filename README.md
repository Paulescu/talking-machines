# Next steps

- [x] Save model checkpoints during training.
- [ ] Get a decent model.
- [ ] Load model from checkpoint.
- [ ] Implement beam-search decoding.
- [ ] Add Luong's attention
- [ ] Save checkpoints to gdrive, not to the instance.

- [ ] Start transformer based chatbot.
- [ ] Memory network to attend over personality?

# Setup

```
$ python -m spacy download en_core_web_sm
```

```
$ conda env export > environment.yml
```

```
from data_util import DataWrapper

# Dataset objects
dw = DataWrapper()
train_ds, val_ds, test_ds = dw.get_datasets(
    train_size=132000,
    val_size=7801,  # 7801
    use_glove=True
)
print(f'Train set size: {len(train_ds):,}')
print(f'Validation set size: {len(val_ds):,}')
print('Vocab size: ', dw.vocab_size)
```

# Theory

- [From RNNs to Transformers](https://dzone.com/articles/rnn-seq2seq-transformers-introduction-to-neural-ar)

- https://github.com/pytorch/examples/blob/master/word_language_model/model.py

# References
- [Example of trainer class to decrease complexity of training loop code](https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/trainer/pretrain.py)

https://charon.me/posts/pytorch/pytorch_seq2seq_4/