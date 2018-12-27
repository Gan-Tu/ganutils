"""attention
PyTorch Core Attention Modules
"""

import torch
import torch.nn as nn
import numpy as np

#####################################################################
# Base Class for Attention Mechanisms
# ===================================
#

class BaseAttention(nn.Module):

    def __init__(self):
        super(BaseAttention, self).__init__()

    
    def init_linear(self, input_linear):
        """Initialize linear transformation"""
        bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
        nn.init.uniform_(input_linear.weight, -bias, bias)
        if input_linear.bias is not None:
            input_linear.bias.data.zero_()


    def initialize_layers(self):
        raise NotImplementedError

    
    def forward(self, X):
        raise NotImplementedError

    
    def score(self, a, b):
        raise NotImplementedError


#####################################################################
# Bahdanau Attention
# ====================
#
# Paper: https://arxiv.org/pdf/1409.0473.pdf
#
# Similar to Luong Attention, except:
#     - use the **concatenation** of the forward and backward source 
#       hidden states in the bi-directional encoder
#     - target hidden states in their non-stacking uni-directional decoder.
#     - result is concatenated with the hidden state h_{t-1} of the decoder
#

class BahdanauAttention(BaseAttention):

    def __init__(self):
        super(BahdanauAttention, self).__init__()

    
    def forward(self, X):
        raise NotImplementedError


    def score(self, a, b):
        raise NotImplementedError


######################################################################
# Luong Attention
# ====================
#
# Paper: https://arxiv.org/pdf/1508.04025.pdf
#
# score(h_t, h_s_bar)
#      = h_t^T * h_s_bar (dot)
#      = h_t^T * W_a * h_s_bar (general)
#      = v_a^T * W_a * [h_t; h_s_bar] (concat)
#
# Global Attention: 
#      - simllar to Bahdanau Attention, except:
#          - use hidden state of the **top** LSTM layer (encoder & decoder)
#          - use decoder state at **time t**
#      - result is concatenated with the hidden state h_t of the decoder
#
# Local Attention:
#      - faster than global attention, a blend between hard & soft attention
#      - weighted average of [p_t - D, p_t + D] where p_t is predicted
#          - p_t = S * sigmoid(v_p^T * tanh(W_p * h_t) )
#          - a_t(s) = align(h_t, h_s_bar) * exp(- (s-p_t)^2 / (2 * sigma^2) )
#              - idea is to favor p_t and have a Gaussian over the rest
#              - empirically the original paper set sigma = D/2
#              - s is an **integer** while p_t is a **real number**
#

class LuongAttention(BaseAttention):

    def __init__(self, hidden_dim, scoring="dot", local=False):
        super(BaseAttention, self).__init__()


    def forward(self, X):
        raise NotImplementedError


    def score(self, a, b):
        raise NotImplementedError


######################################################################
# Key, Value, and Query Attention
# ===============================
# 
# Paper: https://arxiv.org/pdf/1706.03762.pdf
#
# This method uses the KVQ attention technique from 
# the Transformer architecture's mult-head self-attention
#

class KVQAttention(nn.Module):

    def __init__(self):
        super(KVQAttention, self).__init__()


    def forward(self, X):
        raise NotImplementedError


    def score(self, a, b):
        raise NotImplementedError



