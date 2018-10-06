"""attention.self_attention
PyTorch Specialized Self Attention Modules
"""

import torch
import torch.nn as nn
import numpy as np

class _BaseSelfAttention(nn.Module):

    def __init__(self):
        super(_BaseSelfAttention, self).__init__()

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

    def score(self, a, b):
        raise NotImplementedError


class SentenceEmbedding(_BaseSelfAttention):

    def __init__(self, embedding_dim, hidden_dim, num_annotations):
        """ Structured Self-Attentive Sentence Embedding (https://arxiv.org/pdf/1703.03130.pdf)

        Given a sentence of words, turn the sequence of word embeddings into a single sentence embedding that attends to important part of the sentence. The number of output sentence embeddings is determined by the number of annotations specified at module initialization. 
        """
        super(SentenceEmbedding, self).__init__()
        # Save Parameters for reference
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_annotations = num_annotations
        # Define layers
        self.Ws1 = nn.Linear(embedding_dim, hidden_dim)
        self.Ws2 = nn.Linear(hidden_dim, num_annotations)
        # Initialize Layers
        self.init_linear(self.Ws1)
        self.init_linear(self.Ws2)

    def forward(self, word_embeddings):
        """
        Args:
            word_embeddings: 
                (batch_size, doc_maxlen, embedding_dim)
        Output:
            sentence_embedding: 
                (batch_size, num_annotations, embedding_dim)
        """
        # dim: (batch_size, doc_maxlen, embedding_dim)
        hidden = F.tanh(self.Ws1(word_embeddings))
        # dim: (batch_size, doc_maxlen, hidden_dim)
        atten_weights = F.softmax(self.Ws2(hidden))
        # dim: (batch_size, doc_maxlen, num_annotations)
        atten_weights = atten_weights.transpose(1, 2)
        # dim: (batch_size, num_annotations, doc_maxlen)
        sentence_embedding = atten_weights.bmm(word_embeddings)
        # dim: (batch_size, num_annotations, embedding_dim)
        return sentence_embedding

class MultiHeadSelfAttention(nn.Module):

    def __init__(self):
        """
        Multi-head Self Attention mechanism from the Transformer architecture (https://arxiv.org/pdf/1706.03762.pdf).
        """
        super(MultiHeadSelfAttention, self).__init__()

    def forward(self, X):
        raise NotImplementedError

    def score(self, a, b):
        raise NotImplementedError

