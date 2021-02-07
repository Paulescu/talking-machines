import torch.nn as nn
from torch import Tensor

def beam_search_decode(model: nn.Module,
                       
                       prev_decoder_hidden: Tensor,
                        
                       n_beams: int):
    """Beam-search decoding"""