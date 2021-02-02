# Train script
import os
import json
import math
from pathlib import Path
import pdb
from typing import Union, List

from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from util import (
    count_trainable_parameters,
    get_random_id,
    download_artifacts
)
from data_util import get_dataset_size

def cross_entropy_loss_fn(pad_token_id = None):
    """Returns a cross-entropy loss function that ignores positions with
    padding tokens"""
    return nn.CrossEntropyLoss(ignore_index=pad_token_id)
    
class Seq2seqRNNTrainer:

    def __init__(self,
                 model,
                 train_dataloader,
                 val_dataloader,
                 learning_rate: float = 3e-4,
                 pad_token_id = None,
                 gradient_clip: float = 99999,
                 teacher_forcing: float = 0.0,
                 with_cuda: bool = True,
                 checkpoint_dir: Union[str, Path] = Path('./')):
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device('cuda' if cuda_condition else 'cpu')
        
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.gradient_clip = gradient_clip
        self.teacher_forcing = teacher_forcing

        self.train_size = get_dataset_size(self.train_dataloader)
        self.val_size = get_dataset_size(self.val_dataloader)

        self.loss_fn = cross_entropy_loss_fn(pad_token_id)
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        
        # state variables
        self.min_test_loss = float('inf')
        self.min_test_perplexity = float('inf')
        self.n_epochs = 0
        
        if isinstance(checkpoint_dir, str):
            self.checkpoint_dir = Path(checkpoint_dir)
        else:
            self.checkpoint_dir = checkpoint_dir
        if not self.checkpoint_dir.exists():
            os.mkdir(self.checkpoint_dir)

        self.run_id = get_random_id()

    def train_test_loop(self, n_epochs):

        for epoch in range(n_epochs):
            train_loss, train_ppl = self.train(epoch)
            test_loss, test_ppl = self.test(epoch)

            # print metrics to console
            log = 'Epoch: {:03d}, Train loss: {:.4f}, Val loss: {:.4f}, Train ppl: {:.1f}, Val ppl: {:.1f}'
            print(log.format(epoch, train_loss, test_loss, train_ppl, test_ppl))

            # save checkpoint if test loss is lower than self.min_test_loss
            if test_loss < self.min_test_loss:
                self.min_test_loss = test_loss
                self.save()

            # update internal variable
            self.n_epochs += 1

    def train(self, epoch):
        return self.iteration(epoch, self.train_dataloader, self.train_size)
    
    def test(self, epoch):
        with torch.no_grad():
            return self.iteration(epoch, self.val_dataloader, self.val_size,
                                  train=False)

    def iteration(self, epoch, dataloader, dataset_size, train=True):
        
        if train:
            self.model.train()
        else:
            self.model.eval()
        
        epoch_loss = 0
        epoch_accuracy = 0
        with tqdm(total=dataset_size) as pbar:
            for batch in dataloader:
                # forward step
                src, src_len = batch.src
                tgt_input, _ = batch.tgt
                tgt_output_scores = self.model(
                    src,
                    src_len,
                    tgt_input, teacher_forcing=self.teacher_forcing
                ).to(self.device)

                # compute batch loss after a bit of reshaping
                vocab_size = tgt_output_scores.shape[-1]
                tgt_output_scores = tgt_output_scores.reshape(-1, vocab_size)               
                tgt_output = tgt_input[:, 1:].reshape(-1)
                loss = self.loss_fn(tgt_output_scores, tgt_output)

                if train:
                    # backward step
                    self.optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    self.optimizer.step()

                epoch_loss += batch.batch_size * loss.item()

                pbar.update(batch.batch_size)

        epoch_loss = epoch_loss / dataset_size
        perplexity = math.exp(epoch_loss)
        return epoch_loss, perplexity

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
