"""attention.attention
PyTorch Core Attention Modules
"""

import torch
import torch.nn as nn
import numpy as np


class _BaseAttention(nn.Module):

    def __init__(self):
        super(_BaseAttention, self).__init__()

    def init_linear(self, input_linear):
        """
        Initialize linear transformation
        """
        bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
        nn.init.uniform_(input_linear.weight, -bias, bias)
        if input_linear.bias is not None:
            input_linear.bias.data.zero_()

    def forward(self, X):
        raise NotImplementedError

