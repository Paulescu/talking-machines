import os
import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F

# from .utils_v0 import Beam
# from .dict_v0 import DictionaryAgent

"""
dict_lower=True,
dict_include_valid=False,
dict_maxexs=-1,
datatype='train',
batchsize=64,
hiddensize=1024,
embeddingsize=256,
# attention='general',
numlayers=2,
rnn_class='lstm',
learningrate=3,
dropout=0.1,
gradient_clip=0.1,
lookuptable='enc_dec',
optimizer='sgd',
# embedding_type='glove',
momentum=0.9,
bidirectional=False,
context_length=-1,
validation_every_n_secs=90,
validation_metric='ppl',
validation_metric_mode='min',
validation_patience=12,
log_every_n_secs=10,
dict_tokenizer='split',
"""
opt = {
    'rnn_class': 'lstm',
    'embeddingsize': 300,
    'hiddensize': 1024,
    'numlayers': 2,
    'dropout': 0.1,

    'lookuptable': 'enc_dec',
    'bidirectional': False,


}

class Seq2seq(nn.Module):

    RNN_OPTS = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}

    def __init__(
        self,
        opt,
        num_features,
        padding_idx=0, start_idx=1, end_idx=2,
        longest_label=1
    ):
        super().__init__()
        self.opt = opt

        # self.rank = opt['rank_candidates']
        self.attn_type = opt['attention']

        self.NULL_IDX = padding_idx
        self.END_IDX = end_idx
        self.register_buffer('START', torch.LongTensor([start_idx]))
        self.longest_label = longest_label

        rnn_class = Seq2seq.RNN_OPTS[opt['rnn_class']]
        self.decoder = Decoder(
            num_features,
            padding_idx=self.NULL_IDX,
            rnn_class=rnn_class,
            emb_size=opt['embeddingsize'],
            hidden_size=opt['hiddensize'],
            num_layers=opt['numlayers'],
            dropout=opt['dropout'],
            share_output=opt['lookuptable'] in ['dec_out', 'all'],
            attn_type=opt['attention'],
            attn_length=opt['attention_length'],
            attn_time=opt.get('attention_time'),
            bidir_input=opt['bidirectional'],
            numsoftmax=opt.get('numsoftmax', 1),
            softmax_layer_bias=opt.get('softmax_layer_bias', False),
        )

        shared_lt = (
            self.decoder.lt if opt['lookuptable'] in ['enc_dec', 'all'] else None
        )
        shared_rnn = self.decoder.rnn if opt['decoder'] == 'shared' else None
        self.encoder = Encoder(
            num_features,
            padding_idx=self.NULL_IDX,
            rnn_class=rnn_class,
            emb_size=opt['embeddingsize'],
            hidden_size=opt['hiddensize'],
            num_layers=opt['numlayers'],
            dropout=opt['dropout'],
            bidirectional=opt['bidirectional'],
            shared_lt=shared_lt,
            shared_rnn=shared_rnn,
        )

        # if self.rank:
        #     self.ranker = Ranker(
        #         self.decoder, padding_idx=self.NULL_IDX, attn_type=opt['attention']
        #     )

        self.beam_log_freq = opt.get('beam_log_freq', 0.0)
        if self.beam_log_freq > 0.0:
            self.dict = DictionaryAgent(opt)
            self.beam_dump_filecnt = 0
            self.beam_dump_path = opt['model_file'] + '.beam_dump'
            if not os.path.exists(self.beam_dump_path):
                os.makedirs(self.beam_dump_path)

    def unbeamize_hidden(self, hidden, beam_size, batch_size):
        """
        Creates a view of the hidden where batch axis is collapsed with beam axis, we
        need to do this for batched beam search, i.e. we emulate bigger mini-batch.

        :param hidden: hidden state of the decoder
        :param beam_size: beam size, i.e. num of hypothesis
        :param batch_size: number of samples in the mini batch
        :return: view of the hidden
        """
        if isinstance(hidden, tuple):
            num_layers = hidden[0].size(0)
            hidden_size = hidden[0].size(-1)
            return (
                hidden[0].view(num_layers, batch_size * beam_size, hidden_size),
                hidden[1].view(num_layers, batch_size * beam_size, hidden_size),
            )
        else:  # GRU
            num_layers = hidden.size(0)
            hidden_size = hidden.size(-1)
            return hidden.view(num_layers, batch_size * beam_size, hidden_size)

    def unbeamize_enc_out(self, enc_out, beam_size, batch_size):
        hidden_size = enc_out.size(-1)
        return enc_out.view(batch_size * beam_size, -1, hidden_size)

    def forward(
        self,
        xs,
        ys=None,
        cands=None,
        valid_cands=None,
        prev_enc=None,
        rank_during_training=False,
        beam_size=1,
        topk=1,
    ):
        """
        Get output predictions from the model.

        Arguments:
        xs -- input to the encoder
        ys -- expected output from the decoder
        cands -- set of candidates to rank, if applicable
        valid_cands -- indices to match candidates with their appropriate xs
        prev_enc -- if you know you'll pass in the same xs multiple times and
            the model is in eval mode, you can pass in the encoder output from
            the last forward pass to skip recalcuating the same encoder output
        rank_during_training -- (default False) if set, ranks any available
            cands during training as well
        """
        input_xs = xs
        nbest_beam_preds, nbest_beam_scores = None, None
        bsz = len(xs)
        if ys is not None:
            # keep track of longest label we've ever seen
            # we'll never produce longer ones than that during prediction
            self.longest_label = max(self.longest_label, ys.size(1))

        if prev_enc is not None:
            enc_out, hidden, attn_mask = prev_enc
        else:
            enc_out, hidden = self.encoder(xs)
            attn_mask = xs.ne(0).float() if self.attn_type != 'none' else None
        encoder_states = (enc_out, hidden, attn_mask)
        start = self.START.detach()
        starts = start.expand(bsz, 1)

        predictions = []
        scores = []
        cand_preds, cand_scores = None, None
        # if self.rank and cands is not None:
        #     decode_params = (start, hidden, enc_out, attn_mask)
        #     if self.training:
        #         if rank_during_training:
        #             cand_preds, cand_scores = self.ranker.forward(
        #                 cands, valid_cands, decode_params=decode_params
        #             )
        #     else:
        #         cand_preds, cand_scores = self.ranker.forward(
        #             cands, valid_cands, decode_params=decode_params
        #         )

        if ys is not None:
            y_in = ys.narrow(1, 0, ys.size(1) - 1)
            xs = torch.cat([starts, y_in], 1)
            if self.attn_type == 'none':
                preds, score, hidden = self.decoder(xs, hidden, enc_out, attn_mask)
                predictions.append(preds)
                scores.append(score)
            else:
                for i in range(ys.size(1)):
                    xi = xs.select(1, i)
                    preds, score, hidden = self.decoder(xi, hidden, enc_out, attn_mask)
                    predictions.append(preds)
                    scores.append(score)
        else:
            # here we do search: supported search types: greedy, beam search
            if beam_size == 1:
                done = [False for _ in range(bsz)]
                total_done = 0
                xs = starts

                for _ in range(self.longest_label):
                    # generate at most longest_label tokens
                    preds, score, hidden = self.decoder(
                        xs, hidden, enc_out, attn_mask, topk
                    )
                    scores.append(score)
                    xs = preds
                    predictions.append(preds)

                    # check if we've produced the end token
                    for b in range(bsz):
                        if not done[b]:
                            # only add more tokens for examples that aren't done
                            if preds.data[b][0] == self.END_IDX:
                                # if we produced END, we're done
                                done[b] = True
                                total_done += 1
                    if total_done == bsz:
                        # no need to generate any more
                        break

            elif beam_size > 1:
                enc_out, hidden = (
                    encoder_states[0],
                    encoder_states[1],
                )  # take it from encoder
                enc_out = enc_out.unsqueeze(1).repeat(1, beam_size, 1, 1)
                # create batch size num of beams
                data_device = enc_out.device
                beams = [
                    Beam(
                        beam_size,
                        3,
                        0,
                        1,
                        2,
                        min_n_best=beam_size / 2,
                        cuda=data_device,
                    )
                    for _ in range(bsz)
                ]
                # init the input with start token
                xs = starts
                # repeat tensors to support batched beam
                xs = xs.repeat(1, beam_size)
                attn_mask = input_xs.ne(0).float()
                attn_mask = attn_mask.unsqueeze(1).repeat(1, beam_size, 1)
                repeated_hidden = []

                if isinstance(hidden, tuple):
                    for i in range(len(hidden)):
                        repeated_hidden.append(
                            hidden[i].unsqueeze(2).repeat(1, 1, beam_size, 1)
                        )
                    hidden = self.unbeamize_hidden(
                        tuple(repeated_hidden), beam_size, bsz
                    )
                else:  # GRU
                    repeated_hidden = hidden.unsqueeze(2).repeat(1, 1, beam_size, 1)
                    hidden = self.unbeamize_hidden(repeated_hidden, beam_size, bsz)
                enc_out = self.unbeamize_enc_out(enc_out, beam_size, bsz)
                xs = xs.view(bsz * beam_size, -1)
                for step in range(self.longest_label):
                    if all((b.done() for b in beams)):
                        break
                    out = self.decoder(xs, hidden, enc_out)
                    scores = out[1]
                    scores = scores.view(bsz, beam_size, -1)  # -1 is a vocab size
                    for i, b in enumerate(beams):
                        b.advance(F.log_softmax(scores[i, :], dim=-1))
                    xs = torch.cat(
                        [b.get_output_from_current_step() for b in beams]
                    ).unsqueeze(-1)
                    permute_hidden_idx = torch.cat(
                        [
                            beam_size * i + b.get_backtrack_from_current_step()
                            for i, b in enumerate(beams)
                        ]
                    )
                    new_hidden = out[2]
                    if isinstance(hidden, tuple):
                        for i in range(len(hidden)):
                            hidden[i].data.copy_(
                                new_hidden[i].data.index_select(
                                    dim=1, index=permute_hidden_idx
                                )
                            )
                    else:  # GRU
                        hidden.data.copy_(
                            new_hidden.data.index_select(
                                dim=1, index=permute_hidden_idx
                            )
                        )

                for b in beams:
                    b.check_finished()
                beam_pred = [
                    b.get_pretty_hypothesis(b.get_top_hyp()[0])[1:] for b in beams
                ]
                # these beam scores are rescored with length penalty!
                beam_scores = torch.stack([b.get_top_hyp()[1] for b in beams])
                pad_length = max([t.size(0) for t in beam_pred])
                beam_pred = torch.stack(
                    [pad(t, length=pad_length, dim=0) for t in beam_pred], dim=0
                )

                #  prepare n best list for each beam
                n_best_beam_tails = [
                    b.get_rescored_finished(n_best=len(b.finished)) for b in beams
                ]
                nbest_beam_scores = []
                nbest_beam_preds = []
                for i, beamtails in enumerate(n_best_beam_tails):
                    perbeam_preds = []
                    perbeam_scores = []
                    for tail in beamtails:
                        perbeam_preds.append(
                            beams[i].get_pretty_hypothesis(
                                beams[i].get_hyp_from_finished(tail)
                            )
                        )
                        perbeam_scores.append(tail.score)
                    nbest_beam_scores.append(perbeam_scores)
                    nbest_beam_preds.append(perbeam_preds)

                if self.beam_log_freq > 0.0:
                    num_dump = round(bsz * self.beam_log_freq)
                    for i in range(num_dump):
                        dot_graph = beams[i].get_beam_dot(dictionary=self.dict)
                        dot_graph.write_png(
                            os.path.join(
                                self.beam_dump_path,
                                "{}.png".format(self.beam_dump_filecnt),
                            )
                        )
                        self.beam_dump_filecnt += 1

                predictions = beam_pred
                scores = beam_scores

        if isinstance(predictions, list):
            predictions = torch.cat(predictions, 1)
        if isinstance(scores, list):
            scores = torch.cat(scores, 1)

        return (
            predictions,
            scores,
            cand_preds,
            cand_scores,
            encoder_states,
            nbest_beam_preds,
            nbest_beam_scores,
        )


class Encoder(nn.Module):
    def __init__(
        self,
        num_features,
        padding_idx=0,
        rnn_class='lstm',
        emb_size=128,
        hidden_size=128,
        num_layers=2,
        dropout=0.1,
        bidirectional=False,
        shared_lt=None,
        shared_rnn=None,
        sparse=False,
    ):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.layers = num_layers
        self.dirs = 2 if bidirectional else 1
        self.hsz = hidden_size

        if shared_lt is None:
            self.lt = nn.Embedding(
                num_features, emb_size, padding_idx=padding_idx, sparse=sparse
            )
        else:
            self.lt = shared_lt

        if shared_rnn is None:
            self.rnn = rnn_class(
                emb_size,
                hidden_size,
                num_layers,
                dropout=dropout,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif bidirectional:
            raise RuntimeError('Cannot share decoder with bidir encoder.')
        else:
            self.rnn = shared_rnn

    def forward(self, xs):
        bsz = len(xs)

        # embed input tokens
        xes = self.dropout(self.lt(xs))
        try:
            x_lens = [x for x in torch.sum((xs > 0).int(), dim=1).data]
            xes = pack_padded_sequence(xes, x_lens, batch_first=True)
            packed = True
        except ValueError:
            # packing failed, don't pack then
            packed = False

        encoder_output, hidden = self.rnn(xes)
        if packed:
            encoder_output, _ = pad_packed_sequence(encoder_output, batch_first=True)
        if self.dirs > 1:
            # take elementwise max between forward and backward hidden states
            # NOTE: currently using max, but maybe should use Linear
            if isinstance(self.rnn, nn.LSTM):
                hidden = (
                    hidden[0].view(-1, self.dirs, bsz, self.hsz).max(1)[0],
                    hidden[1].view(-1, self.dirs, bsz, self.hsz).max(1)[0],
                )
            else:
                hidden = hidden.view(-1, self.dirs, bsz, self.hsz).max(1)[0]

        return encoder_output, hidden


class Decoder(nn.Module):
    def __init__(
        self,
        num_features,
        padding_idx=0,
        rnn_class='lstm',
        emb_size=128,
        hidden_size=128,
        num_layers=2,
        dropout=0.1,
        bidir_input=False,
        share_output=True,
        attn_type='none',
        attn_length=-1,
        attn_time='pre',
        sparse=False,
        numsoftmax=1,
        softmax_layer_bias=False,
    ):
        super().__init__()

        if padding_idx != 0:
            raise RuntimeError(
                'This module\'s output layer needs to be fixed '
                'if you want a padding_idx other than zero.'
            )

        self.dropout = nn.Dropout(p=dropout)
        self.layers = num_layers
        self.hsz = hidden_size
        self.esz = emb_size

        self.lt = nn.Embedding(
            num_features, emb_size, padding_idx=padding_idx, sparse=sparse
        )
        self.rnn = rnn_class(
            emb_size, hidden_size, num_layers, dropout=dropout, batch_first=True
        )

        # rnn output to embedding
        if hidden_size != emb_size and numsoftmax == 1:
            # self.o2e = RandomProjection(hidden_size, emb_size)
            # other option here is to learn these weights
            self.o2e = nn.Linear(hidden_size, emb_size, bias=False)
        else:
            # no need for any transformation here
            self.o2e = lambda x: x
        # embedding to scores, use custom linear to possibly share weights
        shared_weight = self.lt.weight if share_output else None
        self.e2s = Linear(
            emb_size, num_features, bias=softmax_layer_bias, shared_weight=shared_weight
        )
        self.shared = shared_weight is not None

        self.attn_type = attn_type
        self.attn_time = attn_time
        self.attention = AttentionLayer(
            attn_type=attn_type,
            hidden_size=hidden_size,
            emb_size=emb_size,
            bidirectional=bidir_input,
            attn_length=attn_length,
            attn_time=attn_time,
        )

        self.numsoftmax = numsoftmax
        if numsoftmax > 1:
            self.softmax = nn.Softmax(dim=1)
            self.prior = nn.Linear(hidden_size, numsoftmax, bias=False)
            self.latent = nn.Linear(hidden_size, numsoftmax * emb_size)
            self.activation = nn.Tanh()

    def forward(self, xs, hidden, encoder_output, attn_mask=None, topk=1):
        xes = self.dropout(self.lt(xs))
        if self.attn_time == 'pre':
            xes = self.attention(xes, hidden, encoder_output, attn_mask)
        if xes.dim() == 2:
            # if only one token inputted, sometimes needs unsquezing
            xes.unsqueeze_(1)
        output, new_hidden = self.rnn(xes, hidden)
        if self.attn_time == 'post':
            output = self.attention(output, new_hidden, encoder_output, attn_mask)

        if self.numsoftmax > 1:
            bsz = xs.size(0)
            seqlen = xs.size(1) if xs.dim() > 1 else 1
            latent = self.latent(output)
            active = self.dropout(self.activation(latent))
            logit = self.e2s(active.view(-1, self.esz))

            prior_logit = self.prior(output).view(-1, self.numsoftmax)
            prior = self.softmax(prior_logit)  # softmax over numsoftmax's

            prob = self.softmax(logit).view(bsz * seqlen, self.numsoftmax, -1)
            probs = (prob * prior.unsqueeze(2)).sum(1).view(bsz, seqlen, -1)
            scores = probs.log()
        else:
            e = self.dropout(self.o2e(output))
            scores = self.e2s(e)

        # select top scoring index, excluding the padding symbol (at idx zero)
        # we can do topk sampling from renoramlized softmax here, default topk=1 is greedy
        if topk == 1:
            _max_score, idx = scores.narrow(2, 1, scores.size(2) - 1).max(2)
        elif topk > 1:
            max_score, idx = torch.topk(
                F.softmax(scores.narrow(2, 1, scores.size(2) - 1), 2),
                topk,
                dim=2,
                sorted=False,
            )
            probs = F.softmax(
                scores.narrow(2, 1, scores.size(2) - 1).gather(2, idx), 2
            ).squeeze(1)
            dist = torch.distributions.categorical.Categorical(probs)
            samples = dist.sample()
            idx = idx.gather(-1, samples.unsqueeze(1).unsqueeze(-1)).squeeze(-1)
        preds = idx.add_(1)

        return preds, scores, new_hidden


class Linear(nn.Module):
    """
    Custom Linear layer which allows for sharing weights (e.g. with an nn.Embedding
    layer).
    """

    def __init__(self, in_features, out_features, bias=True, shared_weight=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.shared = shared_weight is not None

        # init weight
        if not self.shared:
            self.weight = Parameter(torch.Tensor(out_features, in_features))
        else:
            if (
                shared_weight.size(0) != out_features
                or shared_weight.size(1) != in_features
            ):
                raise RuntimeError('wrong dimensions for shared weights')
            self.weight = shared_weight

        # init bias
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        if not self.shared:
            # weight is shared so don't overwrite it
            self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        weight = self.weight
        if self.shared:
            # detach weight to prevent gradients from changing weight
            # (but need to detach every time so weights are up to date)
            weight = weight.detach()
        return F.linear(input, weight, self.bias)

    def __repr__(self):
        return (
            self.__class__.__name__
            + ' ('
            + str(self.in_features)
            + ' -> '
            + str(self.out_features)
            + ')'
        )


class RandomProjection(nn.Module):
    """
    Randomly project input to different dimensionality.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.Tensor(out_features, in_features), requires_grad=False
        )  # fix weights
        self.reset_parameters()

    def reset_parameters(self):
        # experimentally: std=1 appears to affect scale too much, so using 0.1
        self.weight.data.normal_(std=0.1)
        # other init option: set randomly to 1 or -1
        # self.weight.data.bernoulli_(self.weight.fill_(0.5)).mul_(2).sub_(1)

    def forward(self, input):
        return F.linear(input, self.weight)

