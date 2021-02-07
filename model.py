from typing import Optional, List, Tuple
import random
import pdb

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

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
                 attention_type: str = None
                 ):
        super(Seq2seqRNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_directions_encoder = n_directions_encoder

        # We use the same embedding layer in the encoder and in the decoder.
        # We let the user choose between using pre-trained GloVe embeddings or
        # learning from scratch 
        if isinstance(pretrained_embeddings, torch.Tensor):
            assert (vocab_size, embedding_dim) == pretrained_embeddings.shape, \
                'pretrained_embeddings shape must be (vocab_size, embedding_dim)'
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=freeze_embeddings)
            self.pretrained_embeddings = True
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.pretrained_embeddings = False

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
                                  dropout=dropout,
                                  attention_type=attention_type)

    def forward(self,
                src,
                src_len,
                tgt_input,
                teacher_forcing=0.0):

        # Dimensions of the output tensors
        #   hidden_states:              [n_layers, batch_size, hidden_dim]
        #   cell_states:                [n_layers, batch_size, hidden_dim]
        #   last_layer_hidden_states:   [seq_len, batch_size, hidden_dim]
        encoder_outputs, hidden_states, cell_states = \
            self.encoder(src, src_len)
        
        tgt_output_scores = self.decoder(
            tgt_input,  # it is here because we need it for teacher forcing
            hidden_states,
            cell_states,
            teacher_forcing=teacher_forcing,
            encoder_outputs=encoder_outputs,
        )

        return tgt_output_scores
    
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
        
        # Tensor -> PackedSequence
        # We do this transformation so that 'hidden_states' and
        # 'cell_states' below come from the last non-padding token, NOT the
        # last token in the src sequence. #finesse
        packed_embedded = pack_padded_sequence(embedded,
                                               src_len.cpu(),
                                               batch_first=True,
                                               enforce_sorted=False)
        # RNN stuff
        outputs, (hidden_states, cell_states) = \
            self.rnn(packed_embedded)

        # PackedSequence -> Tensor
        #                   [max_src_len, batch_size, n_directions * hidden_dim]
        outputs, _ = pad_packed_sequence(outputs)
        
        # Reshape outputs
        # [max_src_len, batch_size, n_directions * hidden_dim]
        # -> [max_src_len, batch_size, hidden_dim]
        # -> [batch_size, max_src_len, hidden_dim]
        batch_size, max_src_len = src.shape[:2]
        outputs = outputs.view(
            batch_size, max_src_len, self.n_directions, self.hidden_dim
        ).mean(dim=-2)
        
        # Mean-reduce hidden_states, cell_states along 'directions' axis
        # [n_layers * n_directions, batch_size, hidden_dim]
        # -> [n_layers, batch_size, hidden_dim]     
        hidden_states = hidden_states.view(
            self.n_layers, self.n_directions, batch_size, self.hidden_dim
        ).mean(dim=1)       
        cell_states = cell_states.view(
            self.n_layers, self.n_directions, batch_size, self.hidden_dim
        ).mean(dim=1)

        return outputs, hidden_states, cell_states
        
class DecoderRNN(nn.Module):

    def __init__(self,
                 embedding: nn.Module,
                 embedding_dim: int,
                 vocab_size: int,
                 hidden_dim: int,
                 n_layers: int,
                 dropout: float = 0.0,
                 attention_type: str = None):
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

        # Luong's attention
        self.use_attention = attention_type is not None
        if self.use_attention:
            self.attention = Attention(hidden_dim, attention_type)
            self.w = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self,
                tgt_input: Tensor,
                hidden_state: Tensor,
                cell_state: Tensor,
                teacher_forcing: float = 0.0,
                encoder_outputs: Tensor = {}):
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
                self.decode_step(input, hidden_state, cell_state,
                                 encoder_outputs=encoder_outputs)

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

    def decode_step(self, input, hidden_state, cell_state, encoder_outputs):
        """Outputs logits for one target word"""        
        embedded_input = self.dropout(self.embedding(input))
        
        # Add an extra dimension to the tensor to have the shape
        # that self.rnn expects
        # [batch_size, vocab_size] -> [batch_size, 1, vocab_size]
        embedded_input = embedded_input.unsqueeze(1)

        # RNN stuff
        decoder_output, (next_hidden_state, next_cell_state) = \
            self.rnn(embedded_input, (hidden_state, cell_state))
        
        # Attention over encoder_outputs
        if self.use_attention:
            # attention weights: [batch_size, max_src_len]
            attn_weights = self.attention(decoder_output, encoder_outputs) #.transpose(0, 1)
        
            # context vector: [batch_size, 1, hidden_dim]
            context = torch.bmm(
                attn_weights.unsqueeze(1), # [batch_size, 1, max_src_len]
                encoder_outputs            # [batch_size, max_src_len, hidden_dim]
            )
            
            # concat 'decoder_output' and 'context': [batch_size, 1, hidden_dim*2]
            concat = self.w(torch.cat((decoder_output, context), dim=2))
            
            # [batch_size, 1, hidden_dim]
            decoder_output = concat.tanh()

        # [batch_size, 1, vocab_size]
        logits = self.fc_output(decoder_output)

        # we remove the extra dimension we previously added
        # [batch_size, vocab_size]
        logits = logits.squeeze(1)

        return logits, next_hidden_state, next_cell_state


class Attention(nn.Module):
    """
    Luong's attention
    https://arxiv.org/pdf/1508.04025.pdf

    Based on
    https://github.com/marumalo/pytorch-seq2seq/blob/master/model.py
    https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb
    """
    def __init__(self,
                 hidden_dim: int,
                 method: str):
        super(Attention, self).__init__()
        assert method in {'dot', 'general', 'concat'}, \
            'method should either be dot, general or concat'
        
        # self.attn = nn.Linear((hidden_dim * multiplier) + hidden_dim,
        #                       hidden_dim)
        # self.v = nn.Linear(hidden_dim, 1, bias=False)

        self.method = method
        self.hidden_dim = hidden_dim

        # dot score_function does not require extra parameters.
        # TODO: add nn.Parameter's for the other two scoring_function: concat, general  

    def dot(self, decoder_hidden, encoder_outputs):
        """
        Input tensors:
            decoder_hidden shape    -> [batch_size, 1, hidden_dim]
            encoder_outputs shape   -> [batch_size, src_len, hidden_dim]
        
            Thanks to broadcasting we can do element-wise multiplication
            between the 2 input tensors without copying 'decoder_hidden'
            'src_len' times along dim=1 to match 'encoder_outputs'. :-)
        
        Output tensor:
            attn_weights shape      -> [batch_size, src_len]
        """
        # print(encoder_outputs.shape)
        # print(decoder_hidden.shape)
        attn_weights = torch.sum(decoder_hidden * encoder_outputs, dim=2)

        # also can be written as
        # attn_weights = decoder_hidden.dot(encoder_outputs)
        
        return attn_weights

    def forward(self,
                decoder_hidden: Tensor, # [batch_size, 1, hidden_dim]
                encoder_outputs: Tensor # [batch_size, src_len, hidden_dim]
        ) -> Tensor:
        # print(encoder_outputs.shape)
        if self.method == 'dot':
            attn_energies = self.dot(decoder_hidden, encoder_outputs)
        else:
            raise Error("Not implemented")
        
        return F.softmax(attn_energies, dim=1)

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

    




        
