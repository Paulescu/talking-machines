from typing import Optional, List, Tuple
import random
import pdb

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Seq2seqRNN(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_dim,
                 n_layers,
                 n_directions_encoder,
                 dropout=0.0,
                 pretrained_embeddings: Optional[torch.Tensor] = None,
                 freeze_embeddings=False,
                 ):
        super(Seq2seqRNN, self).__init__()

        self.vocab_size = vocab_size

        # We use the same embedding layer in the encoder and in the decoder.
        # We let the user choose between using pre-trained GloVe embeddings or
        # learning from scratch 
        if isinstance(pretrained_embeddings, torch.Tensor):
            assert (vocab_size, embedding_dim) == pretrained_embeddings.shape, \
                'pretrained_embeddings shape must be (vocab_size, embedding_dim)'
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=freeze_embeddings)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # encoder network
        self.encoder = EncoderRNN(self.embedding,
                                  embedding_dim,
                                  hidden_dim,
                                  n_directions_encoder,
                                  n_layers,
                                  dropout=dropout)
        
        # decoder network
        self.decoder = DecoderRNN(self.embedding,
                                  embedding_dim,
                                  vocab_size,
                                  hidden_dim,
                                  n_layers,
                                  dropout=dropout)

    def forward(self,
                src,
                src_len,
                tgt_input,
                teacher_forcing=0.0):

        # Dimensions of the output tensors
        #   hidden_states:              [n_layers, batch_size, hidden_dim]
        #   cell_states:                [n_layers, batch_size, hidden_dim]
        #   last_layer_hidden_states:   [seq_len, batch_size, hidden_dim]
        hidden_states, cell_states, last_layer_hidden_states = \
            self.encoder(src, src_len)

        tgt_output_logits = self.decoder(
            tgt_input,  # it is here because we need it for teacher forcing
            hidden_states,
            cell_states,
            attention_vectors=last_layer_hidden_states,
            teacher_forcing=teacher_forcing
        )

        return tgt_output_logits
    
class EncoderRNN(nn.Module):

    def __init__(self,
                 embedding,
                 embedding_dim,
                 hidden_dim,
                 n_directions,
                 n_layers,
                 dropout=0.0):
        super(EncoderRNN, self).__init__()

        self.n_layers = n_layers
        self.n_directions = n_directions
        self.hidden_dim = hidden_dim

        self.embedding = embedding
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           n_layers,
                           batch_first=True,
                           dropout=(0 if n_layers == 1 else dropout),
                           bidirectional=(True if n_directions == 2 else False))
        

    def forward(self,
                src,
                src_len) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        """
        embedded = self.dropout(self.embedding(src))
        
        # we transform the input tensor into a PackedSequence object because
        # we want the LSTM outputs 'last_step_hidden_state' and
        # 'last_step_cell_state' to correspond to the last non-padding token in
        # the input sequence, NOT the last token.
        packed_embedded = pack_padded_sequence(embedded,
                                               src_len.cpu(),
                                               batch_first=True,
                                               enforce_sorted=False)
        
        last_layer_hidden_states, (hidden_states, cell_states) = \
            self.rnn(packed_embedded)

        # unpack PackedSequence into a usual torch.Tensor object
        last_layer_hidden_states, _ = pad_packed_sequence(last_layer_hidden_states)

        # aggregate hidden states across the 'directions' axis.       
        hidden_states = hidden_states.view(
            self.n_layers, self.n_directions, -1, self.hidden_dim)
        hidden_states = torch.mean(hidden_states, dim=1, keepdim=False)
        
        # aggregate cell states across the 'directions' axis.
        cell_states = cell_states.view(
            self.n_layers, self.n_directions, -1, self.hidden_dim)
        cell_states = torch.mean(cell_states, dim=1, keepdim=False)

        return hidden_states, cell_states, last_layer_hidden_states
        

class DecoderRNN(nn.Module):

    def __init__(self,
                 embedding,
                 embedding_dim,
                 vocab_size,
                 hidden_dim,
                 n_layers,
                 dropout=0.0,
                 attention_mechanism=None):
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size

        self.embedding = embedding
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           n_layers,
                           batch_first=True,
                           dropout=(0 if n_layers == 1 else dropout),
                           bidirectional=False)
        self.fc_output = nn.Linear(hidden_dim, vocab_size)

        self.attention_mechanism = attention_mechanism

    def forward(self,
                tgt_input,
                hidden_state,
                cell_state,
                attention_vectors=None,
                teacher_forcing=0.0):
        """
        Outputs the logits (i.e. a probability distribution) for each word
        in tgt. The output is a Tensor with dimensions
        [batch_size, tgt_len, vocab_size].
        """
        # define shape of the output tensor
        batch_size, tgt_input_len = tgt_input.shape
        
        # as we do not predict the first token in tgt_input <BOS>,
        # the tgt_output has length equal to tgt_input - 1
        tgt_output_len = tgt_input_len - 1
        tgt_output_logits = torch.zeros(batch_size,
                                        tgt_output_len,
                                        self.vocab_size)
        
        # tgt_input[:, 0] is the <BOS> token
        input = tgt_input[:, 0]

        for step in range(0, tgt_output_len):

            logits, hidden_state, cell_state = \
                self.decode_step(input, hidden_state, cell_state)

            # store logits in the output tensor
            tgt_output_logits[:, step, :] = logits

            # we use teacher forcing, with probability 'teacher_forcing', to
            # decide if the next tgt_step is the correct one, or the one
            # the model assigns the highest probability (logit) to.
            if random.random() < teacher_forcing:
                input = tgt_input[:, step + 1]
            else:
                input = logits.argmax(1)
            
        return tgt_output_logits

    def decode_step(self, input, hidden_state, cell_state):
        """Outputs logits for one target word"""        
        embedded_input = self.dropout(self.embedding(input))
        
        # we add an extra dimension to the tensor to have the shape
        # that self.rnn expects, i.e. [batch_size, 1, vocab_size]
        embedded_input = embedded_input.unsqueeze(1)

        rnn_output, (next_hidden_state, next_cell_state) = \
            self.rnn(embedded_input, (hidden_state, cell_state))
        
        logits = self.fc_output(rnn_output)

        # we remove the extra dimension we previously added
        logits = logits.squeeze(1)

        return logits, next_hidden_state, next_cell_state

if __name__ == '__main__':

    import torch
    from data_util import DataWrapper
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Datasets
    dw = DataWrapper()
    train_ds, validation_ds, test_ds = dw.get_datasets(
        train_size=999,
        val_size=999,
        use_glove=True
    )

    # Dataloaders
    train_iter, validation_iter, test_iter = dw.get_dataloaders(
        train_ds, validation_ds, test_ds,
        batch_size=2400,
        device=device
    )

    # model architecture
    vocab_size = dw.vocab_size
    embedding_dim = dw.embedding_dim
    hidden_dim = 256
    n_layers = 1
    n_directions_encoder = 2
    model = Seq2seqRNN(vocab_size,
                    embedding_dim,
                    hidden_dim,
                    n_layers,
                    n_directions_encoder,
                    dropout=0.2,
                    pretrained_embeddings=dw.embeddings,
                    freeze_embeddings=False)

    




        
