import torch
import torch.nn as nn


class CustomLRAdamOptimizer:
    """
        Linear ramp learning rate for the warm-up number of steps and then start decaying
        according to the inverse square root law of the current training step number.

        Check out playground.py for visualization of the learning rate (visualize_custom_lr_adam).
    """

    def __init__(self, optimizer, model_dimension, num_of_warmup_steps):
        self.optimizer = optimizer
        self.model_size = model_dimension
        self.num_of_warmup_steps = num_of_warmup_steps

        self.current_step_number = 0

    def step(self):
        self.current_step_number += 1
        current_learning_rate = self.get_current_learning_rate()

        for p in self.optimizer.param_groups:
            p['lr'] = current_learning_rate

        self.optimizer.step()  # apply gradients

    # Check out the formula at Page 7, Chapter 5.3 "Optimizer" and playground.py for visualization
    def get_current_learning_rate(self):
        # For readability purpose
        step = self.current_step_number
        warmup = self.num_of_warmup_steps

        return self.model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))

    def zero_grad(self):
        self.optimizer.zero_grad()


class LabelSmoothingDistribution(nn.Module):
    """
        Instead of one-hot target distribution set the target word's probability to "confidence_value" (usually 0.9)
        and distribute the rest of the "smoothing_value" mass (usually 0.1) over the rest of the vocab.

        Check out playground.py for visualization of how the smooth target distribution looks like compared to one-hot.
    """

    def __init__(self, smoothing_value, pad_token_id, trg_vocab_size, device):
        assert 0.0 <= smoothing_value <= 1.0

        super(LabelSmoothingDistribution, self).__init__()

        self.confidence_value = 1.0 - smoothing_value
        self.smoothing_value = smoothing_value

        self.pad_token_id = pad_token_id
        self.trg_vocab_size = trg_vocab_size
        self.device = device

    def forward(self, trg_token_ids_batch):

        batch_size = trg_token_ids_batch.shape[0]
        smooth_target_distributions = torch.zeros((batch_size, self.trg_vocab_size), device=self.device)

        # -2 because we are not distributing the smoothing mass over the pad token index and over the ground truth index
        # those 2 values will be overwritten by the following 2 lines with confidence_value and 0 (for pad token index)
        smooth_target_distributions.fill_(self.smoothing_value / (self.trg_vocab_size - 2))

        smooth_target_distributions.scatter_(1, trg_token_ids_batch, self.confidence_value)
        smooth_target_distributions[:, self.pad_token_id] = 0.

        # If we had a pad token as a target we set the distribution to all 0s instead of smooth labeled distribution
        smooth_target_distributions.masked_fill_(trg_token_ids_batch == self.pad_token_id, 0.)

        return smooth_target_distributions


# def get_src_and_trg_batches(token_ids_batch):


#     src_token_ids_batch, _ = token_ids_batch.src
#     trg_token_ids_batch, _ = token_ids_batch.tgt

#     # Target input should be shifted by 1 compared to the target output tokens
#     # Example: if we had a sentence like: [<s>,what,is,up,</s>] then to train the NMT model what we do is we pass
#     # [<s>,what,is,up] to the input as set [what,is,up,</s>] as the expected output.
#     trg_token_ids_batch_input = trg_token_ids_batch[:, :-1]

#     # We reshape from (B, S) into (BxS, 1) as that's the the shape expected by LabelSmoothing which will produce
#     # the shape (BxS, V) where V is the target vocab size which is the same shape as the one that comes out
#     # from the transformer so we can directly pass them into the KL divergence loss
#     trg_token_ids_batch_gt = trg_token_ids_batch[:, 1:].reshape(-1, 1)

#     return src_token_ids_batch, trg_token_ids_batch_input, trg_token_ids_batch_gt

def get_src_and_trg_batches(batch):

    batch_fields_are_tuples = True if isinstance(batch.src, tuple) else False

    if batch_fields_are_tuples:       
        history_token_ids, _ = batch.src
        tgt_token_ids, _ = batch.tgt
    else:
        history_token_ids = batch.src
        tgt_token_ids = batch.tgt

    # check if batch contains 'persona' field. It it does, we concatenate
    # the 'persona' and 'history' tensors.
    if hasattr(batch, 'persona'):
        if batch_fields_are_tuples:       
            persona_token_ids, _ = batch.persona
        else:
            persona_token_ids = batch.persona
        # concatenate tensors        
        src_token_ids = torch.cat([persona_token_ids, history_token_ids], dim=1)
    
    else:
        src_token_ids = history_token_ids

    # src_token_ids_batch, _ = batch.src
    # trg_token_ids_batch, _ = batch.tgt

    # Target input should be shifted by 1 compared to the target output tokens
    # Example: if we had a sentence like: [<s>,what,is,up,</s>] then to train the NMT model what we do is we pass
    # [<s>,what,is,up] to the input as set [what,is,up,</s>] as the expected output.
    tgt_token_ids_input = tgt_token_ids[:, :-1]

    # We reshape from (B, S) into (BxS, 1) as that's the the shape expected by LabelSmoothing which will produce
    # the shape (BxS, V) where V is the target vocab size which is the same shape as the one that comes out
    # from the transformer so we can directly pass them into the KL divergence loss
    tgt_token_ids_output = tgt_token_ids[:, 1:].reshape(-1, 1)

    return src_token_ids, tgt_token_ids_input, tgt_token_ids_output


def get_masks_and_count_tokens_src(src_token_ids_batch, pad_token_id):
    batch_size = src_token_ids_batch.shape[0]

    # src_mask shape = (B, 1, 1, S) check out attention function in transformer_model.py where masks are applied
    # src_mask only masks pad tokens as we want to ignore their representations (no information in there...)
    src_mask = (src_token_ids_batch != pad_token_id).view(batch_size, 1, 1, -1)
    num_src_tokens = torch.sum(src_mask.long())

    return src_mask, num_src_tokens


def get_masks_and_count_tokens_trg(trg_token_ids_batch, pad_token_id):
    batch_size = trg_token_ids_batch.shape[0]
    device = trg_token_ids_batch.device

    # Same as src_mask but we additionally want to mask tokens from looking forward into the future tokens
    # Note: wherever the mask value is true we want to attend to that token, otherwise we mask (ignore) it.
    sequence_length = trg_token_ids_batch.shape[1]  # trg_token_ids shape = (B, T) where T max trg token-sequence length
    trg_padding_mask = (trg_token_ids_batch != pad_token_id).view(batch_size, 1, 1, -1)  # shape = (B, 1, 1, T)
    trg_no_look_forward_mask = torch.triu(torch.ones((1, 1, sequence_length, sequence_length), device=device) == 1).transpose(2, 3)

    # logic AND operation (both padding mask and no-look-forward must be true to attend to a certain target token)
    trg_mask = trg_padding_mask & trg_no_look_forward_mask  # final shape = (B, 1, T, T)
    num_trg_tokens = torch.sum(trg_padding_mask.long())

    return trg_mask, num_trg_tokens


def get_masks_and_count_tokens(src_token_ids_batch, trg_token_ids_batch, pad_token_id, device):
    src_mask, num_src_tokens = get_masks_and_count_tokens_src(src_token_ids_batch, pad_token_id)
    trg_mask, num_trg_tokens = get_masks_and_count_tokens_trg(trg_token_ids_batch, pad_token_id)

    return src_mask, trg_mask, num_src_tokens, num_trg_tokens