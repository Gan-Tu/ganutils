"""self_attention
PyTorch Specialized Self Attention Modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


######################################################################
# Base Class for Self Attention Mechanisms
# =========================================
#

class BaseSelfAttention(nn.Module):

    def __init__(self):
        super(BaseSelfAttention, self).__init__()

    
    def init_linear(self, input_linear):
        """Initialize linear transformation"""
        bias = np.sqrt(6.0 / (input_linear.weight.size(0) + \
            input_linear.weight.size(1)))
        nn.init.uniform_(input_linear.weight, -bias, bias)
        if input_linear.bias is not None:
            input_linear.bias.data.zero_()


    def initialize_layers(self):
        raise NotImplementedError

    
    def forward(self, X):
        raise NotImplementedError

    
    def score(self, a, b):
        raise NotImplementedError


######################################################################
# Structured Self-Attentive Sentence Embedding
# =============================================
#
# Paper: https://arxiv.org/pdf/1703.03130.pdf
#
# Given a sentence of words, turn the sequence of word embeddings into 
# a single sentence embedding that attends to important part of the 
# sentence. The number of output sentence embeddings is determined by 
# the number of annotations specified at module initialization. 
#

class SentenceEmbedding(BaseSelfAttention):


    def __init__(self, embedding_dim, hidden_dim, num_annotations):
        super(SentenceEmbedding, self).__init__()
        
        # Save Parameters for reference
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_annotations = num_annotations

        # Define layers
        self.initialize_layers()


    def initialize_layers(self):
        self.Ws1 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.Ws2 = nn.Linear(self.hidden_dim, self.num_annotations)
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
        atten_weights = F.softmax(self.Ws2(hidden), dim=2)
        # dim: (batch_size, doc_maxlen, num_annotations)
        atten_weights = atten_weights.transpose(1, 2)
        # dim: (batch_size, num_annotations, doc_maxlen)
        sentence_embedding = atten_weights.bmm(word_embeddings)
        # dim: (batch_size, num_annotations, embedding_dim)
        return sentence_embedding


######################################################################
# Self Attention using Luong Attention
# =====================================
#
# Paper: https://arxiv.org/pdf/1508.04025.pdf
#
# Luong Attention Scoring Function
# 
#   score(h_t, h_s_bar)
#      = h_t^T * h_s_bar (dot)
#      = h_t^T * W_a * h_s_bar (general)
#      = v_a^T * W_a * [h_t; h_s_bar] (concat)
#
# This does not implement the local attention proposed by the original paper.
#
# ADVICE
# =======
# It is not recommended to use `concat` scoring for self attention
# when document length is too long (>20-100), because it will VERY VERY SLOW and 
# memory intensive, because of the amount of individual scoring we need to do
#

class SelfAttention(BaseSelfAttention):


    def __init__(self, hidden_dim, scoring="general"):
        super(SelfAttention, self).__init__()
        
        # Save parameters for reference
        self.scoring = scoring
        self.hidden_dim = hidden_dim

        ### Define Layers ###
        self.initialize_layers()


    def initialize_layers(self):
        if self.scoring == 'general':
            self.W = nn.Linear(self.hidden_dim, self.hidden_dim)
            # initialization
            self.init_linear(self.W)
        elif self.scoring == 'concat':
            self.W = nn.Linear(2*self.hidden_dim, self.hidden_dim)
            self.v = nn.Linear(self.hidden_dim, 1)
            # initialization
            self.init_linear(self.W)
            self.init_linear(self.v)
        elif self.scoring == "dot":
            pass # don't need to do anything
        else:
            raise RuntimeError("Unrecognized attention scoring method: %s" % self.scoring)


    def forward(self, hidden_outputs):
        # hidden_outputs: (batch_size, doc_maxlen, hidden_dim)
        scores = self.score(hidden_outputs)
        # scores: (batch_size, doc_maxlen, doc_maxlen)
        context = scores.bmm(hidden_outputs)
        # context: (batch_size, doc_maxlen, hidden_dim)
        return context

    
    def score(self, hidden_outputs):
        # hidden_outputs: (batch_size, doc_maxlen, hidden_dim)
        if self.scoring == "dot":
            H = hidden_outputs.transpose(1,2)
            # H: (batch_size, hidden_dim, doc_maxlen)
            attention_energies = hidden_outputs.bmm(H)
            # (batch_size, doc_maxlen, doc_maxlen)
            scores = F.softmax(attention_energies, dim=2)
            return scores
        elif self.scoring == "general":
            H = self.W(hidden_outputs)
            # H: (batch_size, doc_maxlen, hidden_dim) with new hidden values
            H = H.transpose(1, 2)
            # H: (batch_size, hidden_dim, doc_maxlen)
            attention_energies = hidden_outputs.bmm(H)
            # (batch_size, doc_maxlen, doc_maxlen)
            scores = F.softmax(attention_energies, dim=2)
            return scores
        elif self.scoring == "concat":
            # hidden_outputs: (batch_size, doc_maxlen, hidden_dim)
            H = hidden_outputs.transpose(1,2)
            # H: (batch_size, hidden_dim, doc_maxlen)
            scores = []
            batch_size, doc_maxlen, hidden_dim = hidden_outputs.shape
            for doc_idx in range(H.shape[-1]):
                h_t = hidden_outputs[:,doc_idx, :]
                # h_t: (batch_size, hidden_dim)
                h_t = h_t.unsqueeze(1)
                # h_t: (batch_size, 1, hidden_dim)
                h_t = h_t.repeat(1, doc_maxlen, 1)
                # h_t: (batch_size, doc_maxlen, hidden_dim)
                # hidden_outputs: (batch_size, doc_maxlen, hidden_dim)
                H_t = torch.cat((h_t, hidden_outputs), dim=2)
                # H_t: (batch_size, doc_maxlen, 2 * hidden_dim)
                H_t = self.W(H_t)
                # H_t: (batch_size, doc_maxlen, hidden_dim) with new hidden values
                H_t = torch.nn.functional.tanh(H_t)
                # H_t: (batch_size, doc_maxlen, hidden_dim)
                H_t = self.v(H_t)
                # H_t: (batch_size, doc_maxlen, 1)
                H_t = H_t.view(batch_size, doc_maxlen)
                # H_t: (batch_size, doc_maxlen)
                scores.append(H_t)
            scores = torch.stack(scores)
            # scores: (doc_maxlen, batch_size, doc_maxlen)
            scores = scores.transpose(0, 1)
            # scaling trick: scaling 1/sqrt(d)
            scores = scores / torch.sqrt(torch.Tensor([hidden_dim]))
            # scores: (batch_size, doc_maxlen, doc_maxlen)
            scores = F.softmax(scores, dim=2)
            return scores
        else:
            raise RuntimeError("Unrecognized scoring method: %s" % self.scoring)


######################################################################
# Self Attention using Key, Value, and Query 
# ==========================================
# 
# Paper: https://arxiv.org/pdf/1706.03762.pdf
#
# This method uses the KVQ attention technique from 
# the Transformer architecture's multi-head self-attention
#

class KVQSelfAttention(SelfAttention):


    def __init__(self, hidden_dim):
        super(KVQSelfAttention, self).__init__(hidden_dim, "KVQ")


    def initialize_layers(self):
        self.Wk = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.Wv = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.Wq = nn.Linear(self.hidden_dim, self.hidden_dim)
        # initialization
        self.init_linear(self.Wk)
        self.init_linear(self.Wv)
        self.init_linear(self.Wq)


    def forward(self, hidden_outputs):
        # hidden_outputs: (batch_size, doc_maxlen, hidden_dim)
        scores = self.score(hidden_outputs)
        # scores: (batch_size, doc_maxlen, doc_maxlen)
        hidden_outputs = self.Wv(hidden_outputs)
        # hidden_outputs: (batch_size, doc_maxlen, hidden_dim) 
        context = scores.bmm(hidden_outputs)
        # context: (batch_size, doc_maxlen, hidden_dim)
        return context


    def score(self, hidden_outputs):
        K = self.Wk(hidden_outputs)
        Q = self.Wv(hidden_outputs)
        # K: (batch_size, doc_maxlen, hidden_dim) with new hidden values #1
        # Q: (batch_size, doc_maxlen, hidden_dim) with new hidden values #2
        Q = Q.transpose(1, 2)
        # K: (batch_size, doc_maxlen, hidden_dim) with new hidden values #1
        # Q: (batch_size, hidden_dim, doc_maxlen) with new hidden values #2
        attention_energies = K.bmm(Q)
        # (batch_size, doc_maxlen, doc_maxlen)
        scores = F.softmax(attention_energies, dim=2)
        # (batch_size, doc_maxlen, doc_maxlen)
        return scores


######################################################################
#  Multi-head Self Attention Mechanisms
# =============================================
# 
# Paper: https://arxiv.org/pdf/1706.03762.pdf
#
# This method uses the multi-head self attention technique from 
# the Transformer architecture.
#


class MultiHeadSelfAttention(SelfAttention):


    def __init__(self, hidden_dim, num_heads):
        self.num_heads = num_heads
        super(MultiHeadSelfAttention, self).__init__(hidden_dim, "multi-head")


    def initialize_layers(self):
        self.Wk = nn.ModuleList()
        self.Wv = nn.ModuleList()
        self.Wq = nn.ModuleList()
        for i in range(self.num_heads):
            self.Wk.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.Wv.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.Wq.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.linear = nn.Linear(self.num_heads*self.hidden_dim, self.hidden_dim)
        # initialization
        for i in range(self.num_heads):
            self.init_linear(self.Wk[i])
            self.init_linear(self.Wv[i])
            self.init_linear(self.Wq[i])
        self.init_linear(self.linear)


    def forward(self, hidden_outputs):
        # hidden_outputs: (batch_size, doc_maxlen, hidden_dim)
        scores = self.score(hidden_outputs)
        # scores: (num_heads, batch_size, doc_maxlen, doc_maxlen)
        attention_outputs = []
        for i in range(self.num_heads):
            outputs_i = self.Wv[i](hidden_outputs)
            # outputs_i: (batch_size, doc_maxlen, hidden_dim) 
            context_i = scores[i].bmm(outputs_i)
            # context_i: (batch_size, doc_maxlen, hidden_dim)
            attention_outputs.append(context_i)
        # (num_heads, batch_size, doc_maxlen, hidden_dim)
        attention_outputs = torch.cat(attention_outputs, dim=2)
        # (batch_size, doc_maxlen, hidden_dim * num_heads)
        context = self.linear(attention_outputs)
        # (batch_size, doc_maxlen, hidden_dim)
        return context


    def score(self, hidden_outputs):
        scores = []
        # technique from paper: scaled KVQ attention
        scale = torch.sqrt(torch.Tensor([hidden_outputs.shape[2]]))
        # move new tensor to the correct device, in case GPU is used
        scale = scale.to(hidden_outputs.device)
        # scoring for each heads
        for i in range(self.num_heads):
            K = self.Wk[i](hidden_outputs)
            # K: (batch_size, doc_maxlen, hidden_dim) with new hidden values #1
            Q = self.Wv[i](hidden_outputs)
            # Q: (batch_size, doc_maxlen, hidden_dim) with new hidden values #2
            Q = Q.transpose(1, 2)
            # K: (batch_size, doc_maxlen, hidden_dim) with new hidden values #1
            # Q: (batch_size, hidden_dim, doc_maxlen) with new hidden values #2
            attention_energies = K.bmm(Q)
            # (batch_size, doc_maxlen, doc_maxlen)
            scores.append(F.softmax(attention_energies / scale, dim=2))
            # (batch_size, doc_maxlen, doc_maxlen)
        return scores # (num_heads, batch_size, doc_maxlen, doc_maxlen)


##########################################################
# Basic Test
# ===========
#
# This only test if a model runs without any exceptions.
# It does NOT test for accuracy of the implementation.
#


def test_modules(batch_size=64, doc_maxlen=1000, hidden_dim=256, 
                    num_heads=5, num_annotations=10):
    # dummy data
    hidden_outputs = torch.randn(size=(batch_size, doc_maxlen, hidden_dim))
    # test 0
    print("Testing Sentence Self Embedding")
    model = SentenceEmbedding(hidden_dim, hidden_dim, num_annotations)
    _ = model(hidden_outputs)
    # test 1
    print("Testing Self Attention with `general` scoring")
    model = SelfAttention(hidden_dim, scoring="general")
    _ = model(hidden_outputs)
    # test 2
    print("Testing Self Attention with `dot` scoring")
    model = SelfAttention(hidden_dim, scoring="dot")
    _ = model(hidden_outputs)
    # test 3
    print("Testing Self Attention with `concat` scoring")
    x = doc_maxlen
    if x >= 20:
        x = np.random.randint(2, 10)
        print("MODIFICATION: doc_maxlen is too large!")
        print("-- to make the test slower, a new doc_maxlen %d is used" % x)
    tmp = torch.randn(size=(batch_size, x, hidden_dim))
    model = SelfAttention(hidden_dim, scoring="concat")
    _ = model(tmp)
    # test 4
    print("Testing Self Attention with `other` scoring (expects exception)")
    exception_raised = False
    try:
        model = SelfAttention(hidden_dim, scoring="other")
    except RuntimeError as e:
        exception_raised = True
    assert exception_raised, "no exception is raised for unknown scoring"
    # test 5
    print("Testing KVQ Self Attention")
    model = KVQSelfAttention(hidden_dim)
    _ = model(hidden_outputs)
    # test 5
    print("Testing Multi-Head Self Attention")
    model = MultiHeadSelfAttention(hidden_dim, num_heads)
    _ = model(hidden_outputs)
    # end
    print("Test passed.")
 
 