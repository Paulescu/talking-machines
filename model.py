from typing import Optional, List, Tuple
import random
import pdb

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F



class EncoderRNN(nn.Module):

    def __init__(
        self,
        embedding: nn.Embedding,
        embedding_dim,
        hidden_dim: int,
        n_directions: int,
        n_layers: int,
        dropout=0.0
    ):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.n_directions = n_directions
        self.hidden_dim = hidden_dim

        self.embedding = embedding
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            dropout=(0 if n_layers == 1 else dropout),
            bidirectional=(True if n_directions == 2 else False)
        )

    def forward(self, src, src_lengths) -> Tuple[Tensor, Tensor, Tensor]:
        """
        """
        embedded = self.dropout(self.embedding(src))

        # TODO: not sure about this piece
        # Tensor -> PackedSequence
        # We do this transformation so that 'hidden_states' and
        # 'cell_states' below come from the last non-padding token, NOT the last
        # token in the src sequence. #finesse
        packed_embedded = pack_padded_sequence(
            embedded,
            src_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        outputs, (hidden_states, cell_states) = self.rnn(packed_embedded)
        
        # PackedSequence -> Tensor: [batch_size, max(src_lengths), n_directions * hidden_dim]
        outputs, _ = pad_packed_sequence(outputs)

        # outputs, (hidden_states, cell_states) = self.rnn(embedded)

        # outputs:
        # [batch_size, max(src_lengths), n_directions * hidden_dim] -> 
        # [batch_size, max(src_lengths), hidden_dim]
        batch_size, max_src_len = src.shape[:2]
        outputs = outputs.view(
            batch_size, max_src_len, self.n_directions, self.hidden_dim
        ).sum(dim=-2)

        # Sum hidden_states, cell_states along 'directions' axis
        # [n_layers * n_directions, batch_size, hidden_dim] -> 
        # [n_layers, batch_size, hidden_dim]
        hidden_states = hidden_states.view(
            self.n_layers, self.n_directions, batch_size, self.hidden_dim
        ).sum(dim=1)
        cell_states = cell_states.view(
            self.n_layers, self.n_directions, batch_size, self.hidden_dim
        ).sum(dim=1)

        return outputs, hidden_states, cell_states


class DecoderRNN(nn.Module):

    def __init__(
        self,
        embedding: nn.Embedding,
        embedding_dim,
        vocab_size: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float = 0.0,
        attention_type: str = None,
        padding_idx: int = -1,
    ):
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        
        self.embedding = embedding
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            dropout=(0 if n_layers == 1 else dropout),
            bidirectional=False
        )
        self.fc_output = nn.Linear(hidden_dim, vocab_size)

        # self.use_attention = attention_type is not None
        # if self.use_attention:
        #     self.attention = Attention(hidden_dim, attention_type)
        #     self.w = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self,
        input: Tensor,
        hidden_state: Tensor,
        cell_state: Tensor,
        encoder_outputs: Optional[Tensor] = {}
    ):
        """
        Decodes one tgt word, NOT the whole tgt sentence.
        """
        embedded_input = self.dropout(self.embedding(input))

        # [batch_size, vocab_size] -> [batch_size, 1, vocab_size]
        embedded_input = embedded_input.unsqueeze(1)

        # RNN stuff
        decoder_output, (next_hidden_state, next_cell_state) = \
            self.rnn(embedded_input, (hidden_state, cell_state))

        # if self.use_attention:

        #     # attention weights: [batch_size, max_src_len]
        #     attn_weights = self.attention(decoder_output, encoder_outputs)

        #     # context vector: [batch_size, 1, hidden_dim]
        #     context = torch.bmm(
        #         attn_weights.unsqueeze(1),  # [batch_size, 1, max_src_len]
        #         encoder_outputs  # [batch_size, max_src_len, hidden_dim]
        #     )

        #     # concatenate context_vector and decoder_output: [batch_size, 1, hidden_dim*2]
        #     concat = torch.cat((decoder_output, context), dim=2)

        #     # Linear layers + tanh(): [batch_size, 1, hidden_dim]
        #     decoder_output = self.w(concat).tanh()

        # [batch_size, 1, vocab_size]
        scores = self.fc_output(decoder_output)

        if self.padding_idx >= 0:
            NEAR_INF = 1e20
            scores[:, :, self.padding_idx] = -NEAR_INF

        # [batch_size, vocab_size]
        scores = scores.squeeze(1)

        return scores, next_hidden_state, next_cell_state


class Attention(nn.Module):
    """
    Based on Luong's attention https://arxiv.org/pdf/1508.04025.pdf
    PyTorch implementation inspired by https://github.com/marumalo/pytorch-seq2seq/blob/master/model.py
    """

    def __init__(self,
                 hidden_dim: int,
                 method: str):
        super(Attention, self).__init__()
        assert method in {'dot', 'general', 'concat'}, \
            'method should either be dot, general or concat'
        self.method = method
        self.hidden_dim = hidden_dim

        # dot score_function does not require extra parameters.
        # TODO: add nn.Parameter's for the other two scoring_function: concat, general

        # self.attn = nn.Linear((hidden_dim * multiplier) + hidden_dim,
        #                       hidden_dim)
        # self.v = nn.Linear(hidden_dim, 1, bias=False)

    def dot(self, decoder_hidden: Tensor, encoder_outputs: Tensor) -> Tensor:
        """
        Inputs:
            decoder_hidden shape    -> [batch_size, 1, hidden_dim]
            encoder_outputs shape   -> [batch_size, src_len, hidden_dim]

        Output:
            attn_weights shape      -> [batch_size, src_len]
        """
        # Thanks to broadcasting we can do element-wise multiplication
        # between the 2 input tensors without copying 'decoder_hidden'
        # 'src_len' times along dim=1 to match 'encoder_outputs'. :-)
        attn_weights = torch.sum(decoder_hidden * encoder_outputs, dim=2)

        # It can also can be written as
        # attn_weights = decoder_hidden.dot(encoder_outputs)

        return attn_weights

    def forward(self, decoder_hidden: Tensor,  encoder_outputs: Tensor) -> Tensor:
        """
        Inputs:
            decoder_hidden shape    -> [batch_size, 1, hidden_dim]
            encoder_outputs shape   -> [batch_size, src_len, hidden_dim]

        Output: [batch_size, src_len]
        """
        if self.method == 'dot':
            attn_energies = self.dot(decoder_hidden, encoder_outputs)
        else:
            raise Exception("Not implemented")

        return F.softmax(attn_energies, dim=1)


class Seq2seqRNN(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        n_layers: int,
        n_directions_encoder: int,
        padding_idx: int,
        device: torch.device,
        dropout: Optional[float] = 0.0,
        pretrained_embeddings: Optional[Tensor] = None,
        attention_type: str = None
    ):

        super(Seq2seqRNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_directions_encoder = n_directions_encoder
        
        # We use the same embedding layer in the encoder and in the decoder.
        # We let the user choose between
        # - pre-trained vs random initial embeddings
        # - fine-tune embeddings vs freeze them.
        if isinstance(pretrained_embeddings, Tensor):
            assert (vocab_size, embedding_dim) == pretrained_embeddings.shape, \
                'shape != (vocab_size, embedding_dim)'
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, padding_idx=padding_idx, freeze=False)
            
            # pdb.set_trace()
            
            self.pretrained_embeddings = True
        else:
            self.embedding = nn.Embedding(
                vocab_size, embedding_dim, padding_idx=padding_idx)
            self.pretrained_embeddings = False

        # encoder network
        self.encoder = EncoderRNN(
            self.embedding,
            embedding_dim,
            hidden_dim,
            n_directions_encoder,
            n_layers,
            dropout=dropout
        )
        
        # decoder network
        self.decoder = DecoderRNN(
            self.embedding,
            embedding_dim,
            vocab_size,
            hidden_dim,
            n_layers,
            dropout=dropout,
            attention_type=attention_type
        )

    def forward(self,
                src,
                src_len,
                tgt_input,
                teacher_forcing=0.0) -> Tensor:
        #
        # Forward pass through the Encoder
        #

        #   encoder_outputs:    [batch_size, max_src_len, hidden_dim]
        #   hidden_state:       [batch_size, n_layers, hidden_dim]
        #   cell_state:         [batch_size, n_layers, hidden_dim]
        encoder_outputs, hidden_state, cell_state = self.encoder(src, src_len)

        #
        # Forward pass through the decoder
        #

        # Allocate memory to store the output tensor
        batch_size, tgt_input_len = tgt_input.shape
        tgt_output_len = tgt_input_len - 1 # we do not need to predict the 1st token of tgt
        tgt_output_scores = torch.zeros(
            batch_size, tgt_output_len, self.vocab_size)
        
        # <BOS> token is always the 1st token
        decoder_input = tgt_input[:, 0]

        # iterate over target sequence
        for step in range(0, tgt_output_len):
            
            # decode current step
            scores, hidden_state, cell_state = self.decoder(
                decoder_input,
                hidden_state,
                cell_state,
                encoder_outputs=encoder_outputs
            )

            # store scores in the output tensor
            tgt_output_scores[:, step, :] = scores

            # we use teacher forcing, with probability 'teacher_forcing', to
            # decide if the next tgt_step is the correct one, or the one
            # the model assigns the highest score to.
            if random.random() < teacher_forcing:
                decoder_input = tgt_input[:, step + 1]
            else:
                decoder_input = scores.argmax(1)
            
        return tgt_output_scores
    
    def beam_search_decode(self,
                           ):
        """Beam-search decoding to use at inference time"""
        pass

    @property
    def hyperparams(self):
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'n_layers': self.n_layers,
            'n_directions_encoder': self.n_directions_encoder,
            'pretrained_embeddings': self.pretrained_embeddings,
        }

    @property
    def id(self):
        """str that uniquely identifies the hyperparameters of the model"""
        return '_'.join([str(self.vocab_size),
                         str(self.embedding_dim),
                         str(self.hidden_dim),
                         str(self.n_layers),
                         str(self.n_directions_encoder)])

    

        


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

    




        
