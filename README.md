# Talking machines

This repo contains several PyTorch implementations of open-domain chatbots
with consistent personality.

You can:
* train the bots
* talk to the bots
* deploy the bots to Facebook Messenger

# Table of contents

* [What is an open-domain chatbot with consistent personality?](#what-is-a-chatbot?)
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

- [x] Save whole torchtet field object.
- [x] Fit model using only last sentence from context.
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

- [Vocab size PersonaChat dataset and perplexities](https://arxiv.org/pdf/2008.05640v1.pdf)
- [Seq2seq model hyper-parameters](https://www.aclweb.org/anthology/P19-1004.pdf)
- [Working code with Persona Chata data](https://github.com/urikz/ChatBot/blob/master/ShaLab/models/model.py)

- [From RNNs to Transformers](https://dzone.com/articles/rnn-seq2seq-transformers-introduction-to-neural-ar)
- https://github.com/pytorch/examples/blob/master/word_language_model/model.py

- [Example of trainer class to decrease complexity of training loop code](https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/trainer/pretrain.py)

https://charon.me/posts/pytorch/pytorch_seq2seq_4/

- [Tricks to normalize text before tokenization](https://pytorch.org/text/_modules/torchtext/data/utils.html)

- [How to build vocab, datasets and dataloaders with torchtext](https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html)

- [Reduce learning rate on plateau](https://github.com/marumalo/pytorch-seq2seq/blob/master/train.py)
- [Disciplined approach to train neural nets](https://arxiv.org/pdf/1803.09820.pdf)
- [Tips for training seq2seq models with attention](https://awni.github.io/train-sequence-models/)