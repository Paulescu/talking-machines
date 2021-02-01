# Next steps

- [x] Save model checkpoints during training.
- [ ] Get a decent model.
- [ ] Load model from checkpoint.
- [ ] Allow inference.
- [ ] Implement dynamic chat.
- [ ] Implement beam-search decoding.
- [ ] Fine-tune hyperparameters

# Setup

```
$ python -m spacy download en_core_web_sm
```

```
$ conda env export > environment.yml
```

# Theory

- [From RNNs to Transformers](https://dzone.com/articles/rnn-seq2seq-transformers-introduction-to-neural-ar)

- https://github.com/pytorch/examples/blob/master/word_language_model/model.py

# References
- [Example of trainer class to decrease complexity of training loop code](https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/trainer/pretrain.py)

https://charon.me/posts/pytorch/pytorch_seq2seq_4/