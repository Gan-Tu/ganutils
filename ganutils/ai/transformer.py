"""transformer
Transformer Architecture (https://arxiv.org/pdf/1706.03762.pdf)
"""

import torch
import torch.nn as nn
import numpy as np
from .self_attention import MultiHeadSelfAttention


class Transformer(nn.Module):

    def __init__(self):
        super(Transformer, self).__init__()

    def forward(self, X):
        raise NotImplementedError

    def score(self, a, b):
        raise NotImplementedError
