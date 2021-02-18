# Train script
import time
import os
import json
import math
from pathlib import Path
import pdb
from typing import Union, List

import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch import optim

from util import (
    count_trainable_parameters,
    get_random_id,
    get_dataset_size
)

def cross_entropy_loss_fn(pad_token_id = None):
    """Returns a cross-entropy loss function that ignores positions with
    padding tokens"""
    return nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='sum')
    
class Seq2seqRNNTrainer:

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        learning_rate: float = 3e-4,
        pad_token_id: int = None,
        gradient_clip: float = 99999,
        teacher_forcing: float = 0.0,
        with_cuda: bool = True,

        log_every_n_steps: int = 10,
        validate_every_n_steps: int = 30,
        checkpoint_dir: Union[str, Path] = Path('./')
    ):

        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device('cuda' if cuda_condition else 'cpu')
        
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.gradient_clip = gradient_clip
        self.teacher_forcing = teacher_forcing
        self.pad_token_id = pad_token_id

        self.train_size = get_dataset_size(self.train_dataloader)
        self.val_size = get_dataset_size(self.val_dataloader)

        self.loss_fn = cross_entropy_loss_fn(pad_token_id)
        self.optimizer = Adam(model.parameters(), lr=learning_rate)

        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode='min',
            factor=0.25,
            patience=0,
            cooldown=1,
            verbose=True,
        )

        # state variables
        self.min_test_loss = float('inf')
        self.min_test_perplexity = float('inf')
        self.epoch = 0
        self.global_steps = 0

        self.checkpoint_dir = Path(checkpoint_dir)
        if not self.checkpoint_dir.exists():
            os.mkdir(self.checkpoint_dir)

        self.log_every_n_steps = log_every_n_steps
        # self.validate_every_n_sec = validate_every_n_sec
        self.validate_every_n_steps = validate_every_n_steps

        self.run_id = get_random_id()

    def get_learning_rate(self):
        return self.optimizer.state_dict()['param_groups'][0]['lr']

    def update_lr(self, val_loss):
        self.lr_scheduler.step(val_loss, self.n_epochs)


    def train(self, epochs):
        """Adjusts model parameters using one batch of data from self.train_dataloader"""

        # self.ts = time.time()

        self.model.train()

        for epoch in range(epochs):

            epoch_loss = 0
            epoch_tgt_tokens = 0
            n_batches = len(self.train_dataloader)
            lr = self.get_learning_rate()
            epoch_ts = time.time()

            for batch_idx, batch in enumerate(self.train_dataloader, 1):

                self.global_steps += 1
                
                # forward step
                src, src_len = batch.src
                tgt_input, _ = batch.tgt
                tgt_output_scores = self.model(
                    src, src_len, tgt_input, teacher_forcing=self.teacher_forcing
                ).to(self.device)

                # compute loss
                vocab_size = tgt_output_scores.shape[-1]
                loss = self.loss_fn(tgt_output_scores.reshape(-1, vocab_size),
                                    tgt_input[:, 1:].reshape(-1))

                # count how many tokens are not the pad token
                batch_tgt_tokens = (tgt_input[:, 1:] != self.pad_token_id).sum().item()
                # loss normalized by the amount of tokens
                batch_loss = loss.item() / batch_tgt_tokens
                # definition of perplexity
                batch_ppl = np.exp(batch_loss)
                # print out batch metrics
                if (self.global_steps % self.log_every_n_steps) == 0:
                    print(f'Epoch {self.epoch} Batch: {batch_idx}/{n_batches} '
                          f'Loss: {batch_loss:.4f} Perplexity: {batch_ppl:.2f} '
                          f'lr: {lr:.4f} teacher_forcing: {self.teacher_forcing:.2f}')

                # update epoch level metrics
                epoch_loss += loss.item()
                epoch_tgt_tokens += batch_tgt_tokens

                # backward step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.gradient_clip)
                self.optimizer.step()

                # check if we need to run a validation loop
                # if (time.time() - self.ts) > self.validate_every_n_sec:
                if (self.global_steps % self.validate_every_n_steps) == 0:
                    # validation loop
                    self.validate()
                    # # update clock
                    # self.ts = time.time()
                    # set train mode
                    self.model.train()

            epoch_loss = epoch_loss / epoch_tgt_tokens
            ppl = np.exp(epoch_loss)

            self.epoch += 1

            # update learning rate
            # TODO.

            # pdb.set_trace()

    @torch.no_grad()
    def validate(self):
        """Evaluates the model performance on the validation data
        self.val_dataloader

        It also prints examples of model responses, to get a qualitative measure
        of how good the model is at talking.
        """
        print('Validating model performance...')

        self.model.eval()

        total_loss = 0
        total_tgt_tokens = 0

        for batch_idx, batch in enumerate(self.val_dataloader, 1):

            # forward step
            src, src_len = batch.src
            tgt_input, _ = batch.tgt
            tgt_output_scores = self.model(
                src, src_len, tgt_input, teacher_forcing=0.0).to(self.device)

            # compute loss
            vocab_size = tgt_output_scores.shape[-1]
            loss = self.loss_fn(
                tgt_output_scores.reshape(-1, vocab_size),
                tgt_input[:, 1:].reshape(-1)
            )

            # update loss and total number of tokens (excluding padding)
            total_loss += loss.item()
            total_tgt_tokens += (tgt_input[:, 1:] != self.pad_token_id).sum().item()

        total_loss = total_loss / total_tgt_tokens
        ppl = np.exp(total_loss)

        print(f'Val loss: {total_loss:.4f}  Val perplexity: {ppl:.4f}')



    def train_test_loop(self, n_epochs):
        """

        :param n_epochs:
        :return:
        """
        for epoch in range(n_epochs):

            current_lr = self.get_learning_rate()
            print(f'Current LR: {current_lr}')

            # adjust model parameters
            train_loss, test_loss = self.train()

            # check peformance on test data
            test_loss, test_ppl = self.test()



            # print metrics to console
            log = 'Epoch: {:03d}, Train loss: {:.4f}, Val loss: {:.4f}, Train ppl: {:.1f}, Val ppl: {:.1f} \n'
            print(log.format(epoch, train_loss, test_loss, train_ppl, test_ppl))

            # save checkpoint if test loss is lower than self.min_test_loss
            if test_loss < self.min_test_loss:
                self.min_test_loss = test_loss
            
            # save always checkpoint
            self.save()

            self.update_lr(test_loss)

            # update internal variable
            self.n_epochs += 1

    # def train(self):
    #     return self.iteration(self.train_dataloader, self.train_size)
    #
    # def test(self):
    #     with torch.no_grad():
    #         return self.iteration(self.val_dataloader, self.val_size, train=False)

    def iteration(self, dataloader, dataset_size, train=True):
        
        if train:
            self.model.train()
        else:
            self.model.eval()
        
        total_loss = 0
        total_tgt_tokens = 0

        # teacher_forcing = self.teacher_forcing if train else 0.0
        # teacher_forcing = self.teacher_forcing

        with tqdm(total=dataset_size) as pbar:

            for batch in dataloader:

                # forward step
                src, src_len = batch.src
                tgt_input, _ = batch.tgt
                tgt_output_scores = self.model(
                    src,
                    src_len,
                    tgt_input,
                    teacher_forcing=0.0  # No teacher forcing.
                ).to(self.device)

                # loss
                vocab_size = tgt_output_scores.shape[-1]
                loss = self.loss_fn(
                    tgt_output_scores.reshape(-1, vocab_size),
                    tgt_input[:, 1:].reshape(-1)
                )

                # pdb.set_trace()

                total_loss += loss.item()
                total_tgt_tokens += (tgt_input[:, 1:] != self.pad_token_id).sum().item()

                # pdb.set_trace()

                if train:
                    # backward step
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.gradient_clip)
                    self.optimizer.step()

                pbar.update(batch.batch_size)

        loss = total_loss / total_tgt_tokens
        import numpy as np
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


if __name__ == '__main__':

    import torch
    from data_util import DataWrapper
    
    # Datasets & Dataloaders
    dw = DataWrapper()
    train_ds, val_ds, test_ds = dw.get_datasets(
        train_size=999,
        val_size=999,
        use_glove=True
    )
    train_iter, val_iter, test_iter = dw.get_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=2400,
        device=torch.device("cpu")
    )

    # model architecture    
    from model import Seq2seqRNN
    vocab_size = dw.vocab_size
    embedding_dim = dw.embedding_dim
    hidden_dim = 256
    n_layers = 3
    n_directions_encoder = 2
    model = Seq2seqRNN(vocab_size,
                    embedding_dim,
                    hidden_dim,
                    n_layers,
                    n_directions_encoder,
                    dropout=0.2,
                    pretrained_embeddings=dw.embeddings,
                    freeze_embeddings=False)

    # TODO: double-check if NLL or CEL?
    # loss_fn = cross_entropy_loss_fn(dw.pad_token_id)

    trainer = Seq2seqRNNTrainer(model,
                                train_iter,
                                val_iter,
                                learning_rate=3e-4,
                                pad_token_id=dw.pad_token_id,
                                gradient_clip=99999,
                                teacher_forcing=0.5)

    n_epochs = 10
    trainer.train(n_epochs)
