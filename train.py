# Train script
import math
import pdb

from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

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
                 with_cuda: boolean = True):
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

        self.n_epochs_trained = 0

    def train(n_epochs):
        for epoch in range(n_epochs):
            train_metrics = self._train_iteration(epoch)
            test_metrics = self._test_iteration(epoch)
            self._print_iteration_metrics(train_metrics, test_metrics)

            # update internal variable
            self.n_epochs_trained += 1

    def _train_iteration(self, epoch):
        self._iteration(epoch, self.train_dataloader, self.train_size)
    
    def _test_iteration(self, epoch):
        with torch.no_grad():
            self._iteration(epoch, self.val_dataloader, self.val_size,
                            train=False)

    def _iteration(self, epoch, dataloader, dataset_size, train=True):
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
                # pbar.write('epoch_loss')               
                # pdb.set_trace()

        epoch_loss = epoch_loss / dataset_size
        perplexity = math.exp(epoch_loss)
        mode = 'train' if train else 'test'
        print(f'Epoch {epoch}: {mode} | '
               f'Loss: {epoch_loss:7.3f} | '
               f'Perplexity: {perplexity:7.3f}')
        
        # return epoch_loss

def minutes_seconds_elapsed(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

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
