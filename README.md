# Talking machines

This repo contains several PyTorch implementations of chatbots using deep learning. You can deploy the trained models as Facebook bots if you follow
the instructions I provide.

# Table of contents

* [What is a chatbot?](#what-is-a-chatbot?)
* [Setup](#setup)
* [Model architectures](#model-architectures)
* [Deployment to FB Messenger](#deployment-to-fb-messenger)

# What is a chatbot?

# Setup

```
$ python -m spacy download en_core_web_sm
```

```
$ conda env export > environment.yml
```

# Typical errors when building NLP models

## Vocabulary full of weird tokens.
Check your tokenization.


# Model architectures

# Deployment to FB Messenger

# WIP

- [ ] Save whole torchtet field object.
- [ ] Fit model using only last sentence from context.
- [ ] Fit model without glove (decrease complexity and error sources).
- [ ] Fit model for short targets only.

- [ ] Get a decent model.
    * [ ] + attention and context vector
    * [ ] + feed context vector at next step

Tricks & learnings:
- [ ] Triple check the vocabulary you generated from the corpus.
- [ ] Switch on gradient clipping.
- [ ] Reduce LR on plateau (see [Reduce learning rate on plateau](https://github.com/marumalo/pytorch-seq2seq/blob/master/train.py))
- [ ] Adjust teacher-forcing
- [ ] Tie weights as in (here)[https://github.com/marumalo/pytorch-seq2seq/blob/master/model.py]
- [ ] Implement beam-search decoding.


Transformer:
- [ ] Start transformer based chatbot.
- [ ] Memory network to attend over personality?


# References

- [From RNNs to Transformers](https://dzone.com/articles/rnn-seq2seq-transformers-introduction-to-neural-ar)

- https://github.com/pytorch/examples/blob/master/word_language_model/model.py

- [Example of trainer class to decrease complexity of training loop code](https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/trainer/pretrain.py)

https://charon.me/posts/pytorch/pytorch_seq2seq_4/


- [Tricks to normalize text before tokenization](https://pytorch.org/text/_modules/torchtext/data/utils.html)

- [How to build vocab, datasets and dataloaders with torchtext](https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html)

- [Reduce learning rate on plateau](https://github.com/marumalo/pytorch-seq2seq/blob/master/train.py)