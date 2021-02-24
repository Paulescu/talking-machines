import os
import json
import math
from pathlib import Path
from typing import Union, List
import pdb

import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch import optim

from src.utils.util import (
    count_trainable_parameters,
    get_random_id,
    get_dataset_size
)


class Seq2seqRNNTrainer:

    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        vocab,
        optimizer: str = 'adam',
        learning_rate: float = 3e-4,
        sgd_momentum: float = 0.9,
        lr_schedule_factor: float = 0.9999,
        pad_token_id = None,
        gradient_clip: float = 99999,
        teacher_forcing: float = 0.0,
        validation_freq: int = 1,
        with_cuda: bool = True,
        checkpoint_dir: Union[str, Path] = Path('./')
    ):
        # store all input parameters
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.gradient_clip = gradient_clip
        self.teacher_forcing = teacher_forcing
        self.pad_token_id = pad_token_id
        self.vocab = vocab
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.sgd_momentum = sgd_momentum
        self.lr_schedule_factor = lr_schedule_factor
        self.validation_freq = validation_freq

        # set device for training
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device('cuda' if cuda_condition else 'cpu')

        self.loss_fn = self._get_loss_fn()
        self.optimizer = self._get_optimizer()
        self.learning_rate_scheduler = self._get_lr_scheduler()

        self.train_size = get_dataset_size(self.train_dataloader)
        self.val_size = get_dataset_size(self.val_dataloader)

        # state variables
        self.min_test_loss = float('inf')
        self.min_test_perplexity = float('inf')
        self.epochs = 0

        # directory to store checkpoints
        self.checkpoint_dir = Path(checkpoint_dir)
        if not self.checkpoint_dir.exists():
            os.mkdir(self.checkpoint_dir)

        # unique identifier for each run
        self.run_id = get_random_id()

    def train_test_loop(self, n_epochs):
        """"""
        for epoch in range(n_epochs):

            train_loss, train_ppl = self.train_epoch()
            test_loss, test_ppl = self.test_epoch()

            # print metrics to console
            log = 'Epoch: {:03d}, Train loss: {:.4f}, Val loss: {:.4f}, Train ppl: {:.1f}, Val ppl: {:.1f}'
            print(log.format(epoch, train_loss, test_loss, train_ppl, test_ppl))

            # save checkpoint if test loss is lower than self.min_test_loss
            if test_loss < self.min_test_loss:
                self.min_test_loss = test_loss
                self.save()

            self.update_lr(test_loss)

            # update internal variable
            self.epochs += 1

    def train_epoch(self):

        self.model.train()
        epoch_loss = 0
        epoch_tgt_tokens = 0

        print(f'Learning rate: {self.get_learning_rate():.4f}')
        print(f'Teacher forcing rate: {self.teacher_forcing:.2f}')

        with tqdm(total=self.train_size) as pbar:

            for batch in self.train_dataloader:

                # forward step
                src, src_len = batch.src
                tgt_input, _ = batch.tgt
                scores, predictions = self.model(
                    src,
                    src_len,
                    tgt_input,
                    teacher_forcing=self.teacher_forcing
                )
                scores = scores.to(self.device)

                # loss
                vocab_size = scores.shape[-1]
                loss = self.loss_fn(scores.reshape(-1, vocab_size),
                                    tgt_input[:, 1:].reshape(-1))
                batch_tgt_tokens = (tgt_input[:, 1:] != self.pad_token_id).sum().item()
                batch_loss = loss.item() / batch_tgt_tokens
                batch_ppl = np.exp(batch_loss)

                epoch_loss += loss.item()
                epoch_tgt_tokens += batch_tgt_tokens

                # backward step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.gradient_clip)
                self.optimizer.step()

                # update progress bar
                pbar.update(batch.batch_size)
                pbar.set_postfix({'Loss': batch_loss, 'Perplexity': batch_ppl})

        loss = epoch_loss / epoch_tgt_tokens
        perplexity = np.exp(loss)

        return loss, perplexity

    @torch.no_grad()
    def test_epoch(self):
        
        self.model.eval()
        
        epoch_loss = 0
        epoch_tgt_tokens = 0

        with tqdm(total=self.val_size) as pbar:

            for batch_idx, batch in enumerate(self.val_dataloader):

                # forward step
                src, src_len = batch.src
                tgt_input, _ = batch.tgt

                # forward pass seq2seq model
                scores, predictions = self.model(
                    src,
                    src_len,
                    tgt_input,
                    teacher_forcing=0.0
                )
                scores = scores.to(self.device)

                # loss
                vocab_size = scores.shape[-1]
                loss = self.loss_fn(scores.reshape(-1, vocab_size),
                                    tgt_input[:, 1:].reshape(-1))

                epoch_loss += loss.item()
                epoch_tgt_tokens += (tgt_input[:, 1:] != self.pad_token_id).sum().item()

                # print a few examples of actual model output
                if batch_idx == 0:
                    self.print_examples(src, tgt_input, predictions, n_examples=10)

                # update progress bar
                pbar.update(batch.batch_size)

        loss = epoch_loss / epoch_tgt_tokens
        perplexity = np.exp(loss)

        return loss, perplexity

    def save(self):
        
        # save trainer state
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'n_epochs': self.n_epochs,
            'min_test_loss': self.min_test_loss,
            'min_test_perplexity': self.min_test_perplexity,
        }
        dir = self.checkpoint_dir / f'{self.run_id}'
        if not dir.exists():
            os.mkdir(dir)
        file = dir / f'{self.n_epochs}.ckpt'
        torch.save(state, file)
        print(f'{file} was saved')

        # save model hyperparameters
        json_file = dir / f'params.json'
        with open(json_file, 'w') as f:
            json.dump(self.model.hyperparams, f)
        print(f'{json_file} file was saved')

    def load(self, run_id, epoch=None):
        
        checkpoint_dir = self.checkpoint_dir / run_id
        if epoch:
            checkpoint_file = checkpoint_dir / f'{epoch}.ckpt'
        else:
            # load latest checkpoint
            raise Exception('Not implemented')
            checkpoint_file = get_latest_checkpoint_file(checkpoint_dir)
        
        # set state
        state = torch.load(checkpoint_file)
        self.model.load_state_dict(state['model_state_dict'])
        self.model.to(self.device)
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.n_epochs = int(state['n_epochs'])
        self.min_test_loss = float(state['min_test_loss'])
        self.min_test_perplexity = float(state['min_test_perplexity'])
        self.run_id = run_id

    #
    # Internal methods
    #
    def _update_lr(self, val_loss):
        self.lr_scheduler.step(val_loss, self.n_epochs)

    def _print_examples(self, src, tgt, predictions, n_examples=3):
        """
        Print a few examples of models inputs and outputs during training.
        Sanity check for the win!
        """
        for i in range(n_examples):
            print(f'\nExample {i}: ')
            # source text
            ids = src[i, :].cpu().detach().numpy()
            words = [self.vocab.itos[x] for x in ids]
            print('SRC: ', ' '.join(words))

            # actual response
            ids = tgt[i, :].cpu().detach().numpy()
            words = [self.vocab.itos[x] for x in ids]
            print('TGT: ', ' '.join(words))

            # model response
            ids = predictions[i, :].cpu().detach().numpy()
            words = [self.vocab.itos[x] for x in ids]
            print('MODEL: ', ' '.join(words))

    def _get_loss_fn(self):
        """Returns a cross-entropy loss function that ignores padding tokens"""
        return nn.CrossEntropyLoss(ignore_index=self.pad_token_id, reduction='sum')

    def _get_optimizer(self):
        """Returns the optimizer specified in 'self.optimizer'"""
        assert self.optimizer in {'adam', 'sgd'}
        if self.optimizer == 'adam':
            return Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'sgd':
            return SGD(self.model.parameters(),
                       lr=self.learning_rate, momentum=self.sgd_momentum)
        else:
            raise Exception(f'Optimizer {self.optimizer} is not supported')

    def _get_lr_scheduler(self):
        """Returns a learning rate scheduler based on reduction when plateauing
        model performance"""
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode='min',
            factor=self.lr_schedule_factor,
            patience=0,
            cooldown=1,
            verbose=True,
        )

    def _get_learning_rate(self):
        return self.optimizer.state_dict()['param_groups'][0]['lr']


if __name__ == '__main__':
    pass


