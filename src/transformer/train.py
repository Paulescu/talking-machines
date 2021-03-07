import os
import json
from pathlib import Path
import pdb
from typing import Union

from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn

from src.util import get_random_id, get_dataset_size
from .utils import (
    get_src_and_trg_batches,
    get_masks_and_count_tokens,
    LabelSmoothingDistribution,
    CustomLRAdamOptimizer
)


class TransformerTrainer:
    """
    This class handles all the complexities of training a Transformer model.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader,
        val_dataloader,
        learning_rate: float,
        vocab,
        pad_token_id: int,
        checkpoint_dir: Union[Path, str],
        validation_freq: int,
        validation_n_examples: int,
        loss_fn: str,
        use_label_smoothing: bool = True,
        with_cuda: bool = True,
        debug: bool = False,
        tensorboard_dir: Union[Path, str] = None,
    ):
        # store all input parameters as object members
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.vocab = vocab
        self.pad_token_id = pad_token_id
        self.checkpoint_dir = checkpoint_dir
        self.validation_freq = validation_freq
        self.validation_n_examples = validation_n_examples
        self.with_cuda = with_cuda
        self.debug = debug       
        assert loss_fn in {'kl', 'cross-entropy'}
        self.loss_fn_name = loss_fn
        if self.loss_fn_name == 'cross-entropy':
            self.use_label_smoothing = False
        else:    
            self.use_label_smoothing = use_label_smoothing

        # set device for training
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device('cuda' if cuda_condition else 'cpu')

        self.loss_fn = self._get_loss_fn(loss_fn)
        self.optimizer = self._get_optimizer()
        # self.learning_rate_scheduler = self._get_lr_scheduler()
        self.label_smoothing = self._get_label_smoothing()

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

        # tensorboard
        self.global_train_step = 0
        self.global_val_step = 0
        self.tensorboard_dir = tensorboard_dir
        if self.tensorboard_dir:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(tensorboard_dir)

    def train_test_loop(self, n_epochs):
        """

        """
        for epoch in range(n_epochs):

            print(f'Epoch: {self.epochs:03d}')

            train_metrics = self.train()
            test_metrics = self.test()

            # print train/test losses
            log = '\nEpoch: {:03d}, Train loss: {:.4f}, Val loss: {:.4f}'
            print(log.format(epoch, train_metrics['loss'], test_metrics['loss']))
            if self.loss_fn_name == 'cross-entropy':
                log = '\nEpoch: {:03d}, Train ppl: {:.4f}, Val pll: {:.4f}'
                print(log.format(epoch, train_metrics['ppl'], test_metrics['ppl']))

            # save checkpoint if test loss is lower than self.min_test_loss
            if test_metrics['loss'] < self.min_test_loss:
                self.min_test_loss = test_metrics['loss']
                self.save()

            self.epochs += 1

            print(''.join(['-'] * 80))

    def train(self):
        """One-epoch training"""
        self.model.train()
        epoch_loss = 0
        epoch_tgt_tokens = 0

        with tqdm(total=self.train_size) as pbar:

            for batch_idx, batch in enumerate(self.train_dataloader):

                self.global_train_step += 1

                # get src and tgt tokens
                src_token_ids, trg_token_ids_input, trg_token_ids_batch_gt = \
                    get_src_and_trg_batches(batch)

                # get masks
                src_mask, trg_mask, num_src_tokens, num_trg_tokens = \
                    get_masks_and_count_tokens(src_token_ids,
                                               trg_token_ids_input,
                                               self.pad_token_id,
                                               self.device)

                if self.debug:
                    print('src_token_ids: ', src_token_ids.shape)
                    print('trg_token_ids_input: ', trg_token_ids_input.shape)
                    print('src_mask: ', src_mask.shape)
                    print('trg_mask: ', trg_mask.shape)

                # log because the KL loss expects log probabilities
                # (just an implementation detail)
                predicted_log_distributions, _ = self.model(
                    src_token_ids,
                    trg_token_ids_input,
                    src_mask,
                    trg_mask
                )

                # compute the loss               
                if self.use_label_smoothing:
                    smooth_target_distributions = self.label_smoothing(trg_token_ids_batch_gt)
                    loss = self.loss_fn(predicted_log_distributions, smooth_target_distributions)
                else:
                    loss = self.loss_fn(predicted_log_distributions, trg_token_ids_batch_gt.squeeze(-1))

                # if self.debug:
                #     print('predicted_log_distributions: ', predicted_log_distributions.shape)
                #     print('smooth_target_distributions: ', smooth_target_distributions.shape)
                self.optimizer.zero_grad()  # clean the trainable weights gradients in the computational graph
                loss.backward()  # compute the gradients for every trainable weight in the computational graph
                self.optimizer.step()  # apply the gradients to weights
                
                epoch_loss += loss.item() * batch.batch_size
                # epoch_tgt_tokens += num_trg_tokens

                # update progress bar
                pbar.update(batch.batch_size)
                pbar.set_postfix({'Loss': loss.item()})

                # save logs to tensorboard
                if self.tensorboard_dir:
                    writer.add_scalar('training_loss', loss.item(), self.global_train_step)

        metrics = dict()
        metrics['loss'] = epoch_loss / self.train_size
        if self.loss_fn_name == 'cross-entropy':
            perplexity = np.exp(metrics['loss'])
            metrics['ppl'] = perplexity

        return metrics

    @torch.no_grad()
    def test(self):
        """Validation of model performance"""
        self.model.eval()
        epoch_loss = 0
        epoch_tgt_tokens = 0
        n_batches = 0
        n_examples = 0

        with tqdm(total=self.val_size) as pbar:

            for batch_idx, batch in enumerate(self.val_dataloader):
                
                global_val_step += 1

                # get src and tgt tokens
                src_token_ids, trg_token_ids_input, trg_token_ids_batch_gt = \
                    get_src_and_trg_batches(batch)

                # get masks
                src_mask, trg_mask, num_src_tokens, num_trg_tokens = \
                    get_masks_and_count_tokens(src_token_ids,
                                               trg_token_ids_input,
                                               self.pad_token_id,
                                               self.device)

                if self.debug:
                    print('src_token_ids: ', src_token_ids.shape)
                    print('trg_token_ids_input: ', trg_token_ids_input.shape)
                    print('src_mask: ', src_mask.shape)
                    print('trg_mask: ', trg_mask.shape)

                # log because the KL loss expects log probabilities
                # (just an implementation detail)
                predicted_log_distributions, predicted_token_ids = self.model(
                    src_token_ids,
                    trg_token_ids_input,
                    src_mask,
                    trg_mask
                )

                # compute the loss               
                if self.use_label_smoothing:
                    smooth_target_distributions = self.label_smoothing(trg_token_ids_batch_gt)
                    loss = self.loss_fn(predicted_log_distributions, smooth_target_distributions)
                else:
                    loss = self.loss_fn(predicted_log_distributions, trg_token_ids_batch_gt.squeeze(-1))

                # if self.debug:
                #     print('predicted_log_distributions: ',
                #           predicted_log_distributions.shape)
                #     print('smooth_target_distributions: ',
                #           smooth_target_distributions.shape)

                epoch_loss += loss.item()
                epoch_tgt_tokens += num_trg_tokens

                # update progress bar
                pbar.update(batch.batch_size)
                pbar.set_postfix({'Loss': loss.item()})

                if batch_idx == 0:
                    # print a few examples of actual input text and model output
                    # text
                    self._print_examples(src_token_ids,
                                         trg_token_ids_input,
                                         predicted_token_ids,
                                         n_examples=3)

                if self.tensorboard_dir:
                    writer.add_scalar('validation_loss', loss.item(), self.global_val_step)

                n_batches += 1
                n_examples += batch.batch_size
                if n_examples > self.validation_n_examples:
                    break

        metrics = dict()
        metrics['loss'] = epoch_loss / n_batches
        if self.loss_fn_name == 'cross-entropy':
            perplexity = np.exp(metrics['loss'])
            metrics['ppl'] = perplexity

        return metrics

    def save(self):

        # save trainer state
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epochs': self.epochs,
            'min_test_loss': self.min_test_loss,
            'min_test_perplexity': self.min_test_perplexity,
        }
        dir = self.checkpoint_dir / f'{self.run_id}'
        if not dir.exists():
            os.mkdir(dir)
        file = dir / f'{self.epochs}.ckpt'
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
        self.epochs = int(state['epochs'])
        self.min_test_loss = float(state['min_test_loss'])
        # self.min_test_perplexity = float(state['min_test_perplexity'])
        self.run_id = run_id

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

    def _get_loss_fn(self, fn: str):
        """Returns a KL divergence loss function"""
        if fn == 'kl':
            # KL divergence
            return nn.KLDivLoss(reduction='batchmean')
        elif fn == 'cross-entropy':
            # cross-entropy
            return nn.CrossEntropyLoss(ignore_index=self.pad_token_id, reduction='mean')
        else:
            raise Exception('Not implemented')

    def _get_optimizer(self):
        """Returns the optimizer specified in 'self.optimizer'"""
        # return CustomLRAdamOptimizer()
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _get_label_smoothing(self):
        return LabelSmoothingDistribution(
            0.1, self.pad_token_id, len(self.vocab), self.device)

    def _get_learning_rate(self):
        return self.optimizer.state_dict()['param_groups'][0]['lr']


