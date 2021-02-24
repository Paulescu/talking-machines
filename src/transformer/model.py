"""
Transformer based Encoder-Decoder
"""
import math
import copy
import pdb
from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn

class Transformer(nn.Module):

    def __init__(
        self,
        model_dimension,
        src_vocab_size,
        trg_vocab_size,
        number_of_heads,
        number_of_layers,
        dropout_probability,
        log_attention_weights=False
    ):
        super().__init__()
        self.trg_vocab_size = trg_vocab_size

        # Embeds source/target token ids into embedding vectors
        self.src_embedding = Embedding(src_vocab_size, model_dimension)
        self.trg_embedding = Embedding(trg_vocab_size, model_dimension)

        # Adds positional information to source/target token's embedding vector
        # (otherwise we'd lose the positional information which is important in human languages)
        self.src_pos_embedding = PositionalEncoding(model_dimension, dropout_probability)
        self.trg_pos_embedding = PositionalEncoding(model_dimension, dropout_probability)

        # All of these will get deep-copied multiple times internally
        mha = MultiHeadedAttention(model_dimension, number_of_heads,
                                   dropout_probability, log_attention_weights)
        pwn = PositionwiseFeedForwardNet(model_dimension, dropout_probability)
        encoder_layer = EncoderLayer(model_dimension, dropout_probability, mha, pwn)
        decoder_layer = DecoderLayer(model_dimension, dropout_probability, mha, pwn)

        self.encoder = Encoder(encoder_layer, number_of_layers)
        self.decoder = Decoder(decoder_layer, number_of_layers)

        # Converts final target token representations into log probabilities
        # vectors of the target vocab size
        self.decoder_generator = DecoderGenerator(model_dimension, trg_vocab_size)

        self.init_params()

    def init_params(self, default_initialization=False):
        # Not mentioned in the paper, but other implementations used xavier.
        # I tested both PyTorch's default initialization and this, and xavier
        # has tremendous impact! I didn't expect
        # a model's perf, with normalization layers, to be so much dependent
        # on the choice of weight initialization.
        if not default_initialization:
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(
        self,
        src_token_ids_batch: Tensor,
        trg_token_ids_batch: Tensor,
        src_mask: Tensor,
        trg_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """"""
        src_representations_batch = self.encode(src_token_ids_batch, src_mask)
        trg_log_probs = self.decode(trg_token_ids_batch,
                                    src_representations_batch,
                                    trg_mask, src_mask)

        batch_size = trg_token_ids_batch.shape[0]
        reshaped = trg_log_probs.view(batch_size, -1, self.trg_vocab_size)
        _, preds = reshaped.max(dim=-1)

        return trg_log_probs, preds

    # Modularized into encode/decode functions for optimizing the
    # decoding/translation process (see translation script)
    def encode(self, src_token_ids_batch, src_mask):

        # get embedding vectors for src token ids
        src_embeddings_batch = self.src_embedding(src_token_ids_batch)

        # add positional embedding
        src_embeddings_batch = self.src_pos_embedding(src_embeddings_batch)

        # forward pass through the encoder
        src_representations_batch = self.encoder(src_embeddings_batch, src_mask)

        return src_representations_batch

    def decode(self, trg_token_ids_batch, src_representations_batch, trg_mask, src_mask):
        # get embedding vectors for trg token ids
        trg_embeddings_batch = self.trg_embedding(trg_token_ids_batch)

        # add positional embedding
        trg_embeddings_batch = self.trg_pos_embedding(trg_embeddings_batch)

        # Shape (B, T, D), where
        #   B - batch size
        #   T - longest target token-sequence length
        #   D - model dimension
        trg_representations_batch = self.decoder(trg_embeddings_batch, src_representations_batch, trg_mask, src_mask)

        # After this line we'll have a shape (B, T, V),
        # where V - target vocab size, decoder generator does a simple
        # linear projection followed by log softmax
        trg_log_probs = self.decoder_generator(trg_representations_batch)

        # Reshape into (B*T, V) as that's a suitable format for passing it into
        # KL div loss
        trg_log_probs = trg_log_probs.reshape(-1, trg_log_probs.shape[-1])

        # the reason I use log here is that PyTorch's nn.KLDivLoss expects
        # log probabilities
        return trg_log_probs

#
# Encoder architecture
#

class Encoder(nn.Module):

    def __init__(self, encoder_layer, number_of_layers):
        super().__init__()
        assert isinstance(encoder_layer, EncoderLayer), f'Expected EncoderLayer got {type(encoder_layer)}.'

        self.encoder_layers = get_clones(encoder_layer, number_of_layers)
        self.norm = nn.LayerNorm(encoder_layer.model_dimension)

    def forward(self, src_embeddings_batch, src_mask):
        # Just update the naming so as to reflect the semantics of what this var will become (the initial encoder layer
        # has embedding vectors as input but later layers have richer token representations)
        src_representations_batch = src_embeddings_batch

        # Forward pass through the encoder stack
        for encoder_layer in self.encoder_layers:
            # src_mask's role is to mask/ignore padded token representations in the multi-headed self-attention module
            src_representations_batch = encoder_layer(src_representations_batch, src_mask)

        # Not mentioned explicitly in the paper (a consequence of using LayerNorm before instead of after the sublayer
        # check out the SublayerLogic module)
        return self.norm(src_representations_batch)


class EncoderLayer(nn.Module):

    def __init__(self, model_dimension, dropout_probability, multi_headed_attention, pointwise_net):
        super().__init__()
        num_of_sublayers_encoder = 2
        self.sublayers = get_clones(SublayerLogic(model_dimension, dropout_probability), num_of_sublayers_encoder)

        self.multi_headed_attention = multi_headed_attention
        self.pointwise_net = pointwise_net

        self.model_dimension = model_dimension

    def forward(self, src_representations_batch, src_mask):
        # Define anonymous (lambda) function which only takes src_representations_batch (srb) as input,
        # this way we have a uniform interface for the sublayer logic.
        encoder_self_attention = lambda srb: self.multi_headed_attention(query=srb, key=srb, value=srb, mask=src_mask)

        # Self-attention MHA sublayer followed by point-wise feed forward net sublayer
        src_representations_batch = self.sublayers[0](src_representations_batch, encoder_self_attention)
        src_representations_batch = self.sublayers[1](src_representations_batch, self.pointwise_net)

        return src_representations_batch


#
# Decoder architecture
#


class Decoder(nn.Module):

    def __init__(self, decoder_layer, number_of_layers):
        super().__init__()
        assert isinstance(decoder_layer, DecoderLayer), f'Expected DecoderLayer got {type(decoder_layer)}.'

        self.decoder_layers = get_clones(decoder_layer, number_of_layers)
        self.norm = nn.LayerNorm(decoder_layer.model_dimension)

    def forward(self, trg_embeddings_batch, src_representations_batch, trg_mask, src_mask):
        # Just update the naming so as to reflect the semantics of what this var will become
        trg_representations_batch = trg_embeddings_batch

        # Forward pass through the decoder stack
        for decoder_layer in self.decoder_layers:
            # Target mask masks pad tokens as well as future tokens (current target token can't look forward)
            trg_representations_batch = decoder_layer(trg_representations_batch, src_representations_batch, trg_mask, src_mask)

        # Not mentioned explicitly in the paper (a consequence of using LayerNorm before instead of after the sublayer
        # check out the SublayerLogic module)
        return self.norm(trg_representations_batch)


class DecoderLayer(nn.Module):

    def __init__(self, model_dimension, dropout_probability, multi_headed_attention, pointwise_net):
        super().__init__()
        num_of_sublayers_decoder = 3
        self.sublayers = get_clones(SublayerLogic(model_dimension, dropout_probability), num_of_sublayers_decoder)

        self.trg_multi_headed_attention = copy.deepcopy(multi_headed_attention)
        self.src_multi_headed_attention = copy.deepcopy(multi_headed_attention)
        self.pointwise_net = pointwise_net

        self.model_dimension = model_dimension

    def forward(self, trg_representations_batch, src_representations_batch, trg_mask, src_mask):
        # Define anonymous (lambda) function which only takes trg_representations_batch (trb - funny name I know)
        # as input - this way we have a uniform interface for the sublayer logic.
        # The inputs which are not passed into lambdas are "cached" here that's why the thing works.
        srb = src_representations_batch  # simple/short alias
        decoder_trg_self_attention = lambda trb: self.trg_multi_headed_attention(query=trb, key=trb, value=trb, mask=trg_mask)
        decoder_src_attention = lambda trb: self.src_multi_headed_attention(query=trb, key=srb, value=srb, mask=src_mask)

        # Self-attention MHA sublayer followed by a source-attending MHA and point-wise feed forward net sublayer
        trg_representations_batch = self.sublayers[0](trg_representations_batch, decoder_trg_self_attention)
        trg_representations_batch = self.sublayers[1](trg_representations_batch, decoder_src_attention)
        trg_representations_batch = self.sublayers[2](trg_representations_batch, self.pointwise_net)

        return trg_representations_batch


#
# Helper modules (designed with modularity in mind) and organized top to bottom.
#


# Note: the original paper had LayerNorm AFTER the residual connection and addition operation
# multiple experiments I found showed that it's more effective to do it BEFORE, how did they figure out which one is
# better? Experiments! There is a similar thing in DCGAN and elsewhere.
class SublayerLogic(nn.Module):
    def __init__(self, model_dimension, dropout_probability):
        super().__init__()
        self.norm = nn.LayerNorm(model_dimension)
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, representations_batch, sublayer_module):
        # Residual connection between input and sublayer output, details: Page 7, Chapter 5.4 "Regularization",
        return representations_batch + self.dropout(sublayer_module(self.norm(representations_batch)))


class DecoderGenerator(nn.Module):
    def __init__(self, model_dimension, vocab_size):
        super().__init__()

        self.linear = nn.Linear(model_dimension, vocab_size)

        # -1 stands for apply the log-softmax along the last dimension i.e. over the vocab dimension as the output from
        # the linear layer has shape (B, T, V), B - batch size, T - max target token-sequence, V - target vocab size
        # again using log softmax as PyTorch's nn.KLDivLoss expects log probabilities (just a technical detail)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, trg_representations_batch):
        # Project from D (model dimension) into V (target vocab size) and apply the log softmax along V dimension
        return self.log_softmax(self.linear(trg_representations_batch))


class MultiHeadedAttention(nn.Module):
    """
    """

    def __init__(self, model_dimension, number_of_heads, dropout_probability, log_attention_weights):
        super().__init__()
        assert model_dimension % number_of_heads == 0, f'Model dimension must be divisible by the number of heads.'

        self.head_dimension = int(model_dimension / number_of_heads)
        self.number_of_heads = number_of_heads

        self.qkv_nets = get_clones(nn.Linear(model_dimension, model_dimension), 3)
        self.out_projection_net = nn.Linear(model_dimension, model_dimension)

        self.attention_dropout = nn.Dropout(p=dropout_probability)
        self.softmax = nn.Softmax(dim=-1)

        self.log_attention_weights = log_attention_weights
        # for visualization purposes, I cache the weights here (translation_script.py)
        self.attention_weights = None

    def attention(self, query, key, value, mask):
        # Step 1: Scaled dot-product attention,
        # Page 4, Chapter 3.2.1 "Scaled Dot-Product Attention"
        # Notation:
        #   B - batch size,
        #   S/T max src/trg token-sequence length,
        #   NH - number of heads,
        #   HD - head dimension
        # query/key/value shape = (B, NH, S/T, HD),
        # scores shape = (B, NH, S, S), (B, NH, T, T) or (B, NH, T, S)
        # scores have different shapes as MHA is used in 3 contexts:
        # self attention for src/trg and source attending MHA
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dimension)

        # Step 2: Optionally mask tokens whose representations we want to ignore
        # by setting a big negative number to locations corresponding to those
        # tokens (force softmax to output 0 probability on those locations).
        # mask shape = (B, 1, 1, S) or (B, 1, T, T) will get broad-casted (copied)
        # as needed to match scores shape
        if mask is not None:
            scores.masked_fill_(mask == torch.tensor(False), float("-inf"))

        # Step 3: Calculate the attention weights - how much should we attend to
        # surrounding token representations
        attention_weights = self.softmax(scores)

        # Step 4: Not defined in the original paper apply dropout to attention
        # weights as well
        attention_weights = self.attention_dropout(attention_weights)

        # Step 5: based on attention weights calculate new token representations
        # attention_weights shape = (B, NH, S, S)/(B, NH, T, T) or (B, NH, T, S),
        # value shape = (B, NH, S/T, HD)
        # Final shape (B, NH, S, HD) for source MHAs or (B, NH, T, HD) target MHAs (again MHAs are used in 3 contexts)
        intermediate_token_representations = torch.matmul(attention_weights, value)

        return intermediate_token_representations, attention_weights  # attention weights for visualization purposes

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]

        # Step 1: Input linear projection
        # Notation: B - batch size, NH - number of heads, S/T - max src/trg token-sequence length, HD - head dimension
        # Shape goes from (B, S/T, NH*HD) over (B, S/T, NH, HD) to (B, NH, S/T, HD) (NH*HD=D where D is model dimension)
        query, key, value = [
            net(x).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
            for net, x in zip(self.qkv_nets, (query, key, value))
        ]

        # Step 2: Apply attention - compare query with key and use that to combine values (see the function for details)
        intermediate_token_representations, attention_weights = self.attention(query, key, value, mask)

        # Potentially, for visualization purposes, log the attention weights, turn off during training though!
        # I had memory problems when I leave this on by default
        if self.log_attention_weights:
            self.attention_weights = attention_weights

        # Step 3: Reshape from (B, NH, S/T, HD) over (B, S/T, NH, HD) (via transpose) into (B, S/T, NHxHD) which is
        # the same shape as in the beginning of this forward function i.e. input to MHA (multi-head attention) module
        reshaped = intermediate_token_representations.transpose(1, 2).reshape(batch_size, -1, self.number_of_heads * self.head_dimension)

        # Step 4: Output linear projection
        token_representations = self.out_projection_net(reshaped)

        return token_representations


class PositionwiseFeedForwardNet(nn.Module):
    """
        It's position-wise because this feed forward net will be independently applied to every token's representation.
        Representations batch is of the shape (batch size, max token sequence length, model dimension).
        This net will basically be applied independently to every token's representation (you can think of it as if
        there was a nested for-loop going over the batch size and max token sequence length dimensions
        and applied this net to token representations. PyTorch does this auto-magically behind the scenes.
    """
    def __init__(self, model_dimension, dropout_probability, width_mult=4):
        super().__init__()

        self.linear1 = nn.Linear(model_dimension, width_mult * model_dimension)
        self.linear2 = nn.Linear(width_mult * model_dimension, model_dimension)

        # This dropout layer is not explicitly mentioned in the paper but it's common to use to avoid over-fitting
        self.dropout = nn.Dropout(p=dropout_probability)
        self.relu = nn.ReLU()

    def forward(self, representations_batch):
        return self.linear2(self.dropout(self.relu(self.linear1(representations_batch))))


class Embedding(nn.Module):

    def __init__(self, vocab_size, model_dimension):
        super().__init__()

        self.embeddings_table = nn.Embedding(vocab_size, model_dimension)
        self.model_dimension = model_dimension

    def forward(self, token_ids_batch):
        assert token_ids_batch.ndim == 2, \
            f'Expected: (batch size, max token sequence length), got {token_ids_batch.shape}'

        # token_ids_batch has shape (B, S/T), where B - batch size, S/T max src/trg token-sequence length
        # Final shape will be (B, S/T, D) where D is the model dimension, every token id has associated vector
        embeddings = self.embeddings_table(token_ids_batch)

        # (stated in the paper) multiply the embedding weights by the square
        # root of model dimension
        # Page 5, Chapter 3.4 "Embeddings and Softmax"
        return embeddings * math.sqrt(self.model_dimension)


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        model_dimension,
        dropout_probability,
        expected_max_sequence_length=5000
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_probability)

        # (stated in the paper) Use sine functions whose frequencies form a geometric
        # progression as position encodings,
        # (learning encodings will also work so feel free to change it!).
        # Page 6, Chapter 3.5 "Positional Encoding"
        position_id = torch.arange(0, expected_max_sequence_length).unsqueeze(1)
        frequencies = torch.pow(10000., -torch.arange(0, model_dimension, 2,
                                                      dtype=torch.float) / model_dimension)

        # Checkout playground.py for visualization of how these look like
        # (it's super simple don't get scared)
        positional_encodings_table = torch.zeros(expected_max_sequence_length,
                                                 model_dimension)
        # sine on even positions
        positional_encodings_table[:, 0::2] = torch.sin(position_id * frequencies)
        # cosine on odd positions
        positional_encodings_table[:, 1::2] = torch.cos(position_id * frequencies)

        # Register buffer because we want to save the positional encodings table
        # inside state_dict even though
        # these are not trainable (not model's parameters) so they otherwise would
        # be excluded from the state_dict
        self.register_buffer('positional_encodings_table', positional_encodings_table)

    def forward(self, embeddings_batch):
        assert embeddings_batch.ndim == 3 and embeddings_batch.shape[-1] == self.positional_encodings_table.shape[1], \
            f'Expected (batch size, max token sequence length, model dimension) got {embeddings_batch.shape}'

        # embedding_batch's shape = (B, S/T, D), where
        #   S/T max src/trg token-sequence length,
        #   D - model dimension
        # So here we get (S/T, D) shape which will get broad-casted to (B, S/T, D)
        # when we try and add it to embeddings
        positional_encodings = self.positional_encodings_table[:embeddings_batch.shape[1]]

        # (stated in the paper) Applying dropout to the sum of positional encodings
        # and token embeddings
        # Page 7, Chapter 5.4 "Regularization"
        return self.dropout(embeddings_batch + positional_encodings)


def get_clones(module, num_of_deep_copies):
    # Create deep copies so that we can tweak each module's weights independently
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_deep_copies)])

def analyze_state_dict_shapes_and_names(model):
    # This part helped me figure out that I don't have positional encodings saved in the state dict
    print(model.state_dict().keys())

    # This part helped me see that src MHA was missing in the decoder since both it and trg MHA were referencing
    # the same MHA object in memory - stupid mistake, happens all the time, embrace the suck!
    for name, param in model.named_parameters():
        print(name, param.shape)
        if not param.requires_grad:
            raise Exception('Expected all of the params to be trainable - no param freezing used.')

# Count how many trainable weights the model has <- just for having a feeling for how big the model is
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    use_big_transformer = False

    BIG_MODEL_DIMENSION = 512
    BIG_MODEL_NUMBER_OF_HEADS = 8
    BIG_MODEL_NUMBER_OF_LAYERS = 6
    BIG_MODEL_DROPOUT_PROB = 0.2

    BASELINE_MODEL_DIMENSION = 256
    BASELINE_MODEL_NUMBER_OF_HEADS = 8
    BASELINE_MODEL_NUMBER_OF_LAYERS = 6
    BASELINE_MODEL_DROPOUT_PROB = 0.2

    # Dummy data
    src_vocab_size = 11
    trg_vocab_size = 11
    src_token_ids_batch = torch.randint(1, 10, size=(3, 2))
    trg_token_ids_batch = torch.randint(1, 10, size=(3, 2))

    transformer = Transformer(
        model_dimension=BIG_MODEL_DIMENSION if use_big_transformer else BASELINE_MODEL_DIMENSION,
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        number_of_heads=BIG_MODEL_NUMBER_OF_HEADS if use_big_transformer else BASELINE_MODEL_NUMBER_OF_HEADS,
        number_of_layers=BIG_MODEL_NUMBER_OF_LAYERS if use_big_transformer else BASELINE_MODEL_NUMBER_OF_LAYERS,
        dropout_probability=BIG_MODEL_DROPOUT_PROB if use_big_transformer else BASELINE_MODEL_DROPOUT_PROB
    )

    # These 2 functions helped me figure out the 2 bugs I had:
    # 1) I did not register positional encodings and thus they wouldn't be saved and later model-loading would fail
    # 2) I had a bug with MHA (attention) in decoder, where both src and trg were referencing the same MHA object in mem
    # It's a good practice to see whether the names, shapes and number of params make sense.
    # e.g. I knew that the big transformer had ~175 M params and I verified that here.
    analyze_state_dict_shapes_and_names(transformer)
    print(f'Size of the {"big" if use_big_transformer else "baseline"} transformer = {count_parameters(transformer):,}')

    out = transformer(src_token_ids_batch, trg_token_ids_batch, src_mask=None, trg_mask=None)