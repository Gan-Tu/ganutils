"""crf
Conditional Random Field in PyTorch
"""

import torch
import torch.nn as nn

class CRF(nn.Module):

    def __init__(self, config, start_tag_idx, pad_tag_idx, end_tag_idx):
        super(CRF, self).__init__()
        
        # Save parameters for reference
        self.config = config
        self.start_tag = start_tag_idx
        self.pad_tag = pad_tag_idx
        self.end_tag = end_tag_idx
    
        ### Define Layers ###
        self.label_size = config.label_size

        self.transitions = nn.Parameter(torch.randn(self.label_size, self.label_size))
        self.transitions.data[:, self.start_tag] = -10000 # cannot transition to start
        self.transitions.data[self.end_tag, :] = -10000 # cannot transition from end
        self.transitions.data[self.pad_tag, :] = -10000 # cannot transition from pad
        self.transitions.data[self.pad_tag, self.end_tag] = 0 # unless it's end
        self.transitions.data[self.pad_tag, self.pad_tag] = 0 # or itself
        self.transitions.data[self.start_tag, :] = 0 # can transit from start to anything

    def log_sum_exp(self, vec, m_size):
        """
        calculate log of exp sum

        args:
            vec (batch_size, vanishing_dim, hidden_dim) : input tensor
            m_size : hidden_dim
        return:
            (batch_size, hidden_dim)
        """
        _, idx = torch.max(vec, 1)  # B * 1 * M
        max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M

        return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)  # B * M

    def get_crf_scores(self, scores):
        """
        Args: 
            scores: (batch_size, doc_maxlen, label_size)
                values returned by RNN
        Return: 
            crf_scores: (batch_size, doc_maxlen+1, label_size(from), label_size(to))
                for each batch, the score of taking tag_from at current timestamp (based from emission scores) and transitioning to tag_to in the next timestamp, for every possible such sequences
        """
        batch_size = scores.size(0)
        doc_maxlen = scores.size(1)
        # reshape
        scores = scores.view(-1, self.label_size, 1)
        # dim: (batch_size * doc_maxlen, label_size, 1)
        n = scores.size(0)
        emissions = scores.expand(n, self.label_size, self.label_size)
        transitions = self.transitions.view(1, self.label_size, self.label_size)
        transitions = transitions.expand(n, self.label_size, self.label_size)
        crf_scores = emissions + transitions
        # dim: (batch_size * doc_maxlen, label_size, label_size)
        crf_scores = crf_scores.view(batch_size, doc_maxlen, self.label_size, self.label_size)
        # dim: (batch_size, doc_maxlen, label_size, label_size)
        
        start_scores = self.transitions.expand(batch_size, 1, self.label_size, self.label_size)
        crf_scores = torch.cat([start_scores, crf_scores], dim=1)
        # dim: (batch_size, doc_maxlen+1, label_size(from), label_size(to))
        return crf_scores

    def get_loss(self, crf_scores, gold_target, mask=None, average_loss=True):
        """
        Args:
            crf_scores: (batch_size, doc_maxlen+1, label_size, label_size)
                obtained from get_crf_scores function, where first timestamp is START
            gold_target: (batch_size, doc_maxlen)
                the gold tag sequences, in which we calculate the loss of crf_scores against
            mask: (batch_size, doc_maxlen)
                masks for padding. If none, no masks is used.
            average_loss: boolean
                Return the loss over all batches, if True; otherwise, return the average loss
        return:
            loss: the loss 
        """

        ### calculate batch size and seq len
        batch_size = crf_scores.size(0) # note, this may differ from config.batch_size
        doc_maxlen = crf_scores.size(1)-1

        ### dim fix
        crf_scores = crf_scores.transpose(0, 1)
        # dim: (doc_maxlen+1, batch_size, label_size, label_size)
        
        gold_target = gold_target.transpose(0, 1).unsqueeze(2)
        # dim: (doc_maxlen, batch_size, 1)
        start_broadcast = torch.full((1, batch_size, 1), self.start_tag, device=crf_scores.device)
        gold_target = torch.cat([start_broadcast, gold_target.float()], dim=0).long()
        # dim: (doc_maxlen+1, batch_size, 1)

        if mask is not None:
            mask = mask.transpose(0, 1).detach()
            # dim: (doc_maxlen, batch_size)
            mask_broadcast = torch.ones(size=(1, batch_size), device=mask.device).byte()
            mask = torch.cat([mask_broadcast, mask], dim=0)
            # dim: (doc_maxlen+1, batch_size)
        else:
            mask = torch.ones(size=(doc_maxlen+1, batch_size)).byte()
            mask = mask.to(crf_scores.device)
            # dim: (doc_maxlen+1, batch_size)


        ### calculate sentence score
        tg_energy = torch.gather(crf_scores.view(doc_maxlen+1, batch_size, -1), 2, gold_target)
        tg_energy = tg_energy.view(doc_maxlen+1, batch_size)
        # dim: (doc_maxlen+1, batch_size)
        tg_energy = tg_energy.masked_select(mask).sum()
        
        ### calculate forward partition score

        seq_iter = enumerate(crf_scores)
        # the first score should start with <start>
        _, inivalues = seq_iter.__next__()  # (batch_size, label_size(from), label_size(to))
        # only need start from start_tag
        forscores = inivalues[:, self.start_tag, :].clone()  # (batch_size, label_size)

        for idx, transition_score in seq_iter:
            # dim: (batch_size, label_size(to))
            broadcast_forscores = forscores.contiguous().view(batch_size, self.label_size, 1)
            # dim: (batch_size, label_size(to), 1)
            broadcast_forscores = broadcast_forscores.expand(batch_size, self.label_size, self.label_size)
            # dim: (batch_size, label_size(from), label_size(to))
            score_of_all_current_to_next = broadcast_forscores + transition_score
            # dim: (batch_size, label_size(from), label_size(to))
            cur_partition = self.log_sum_exp(score_of_all_current_to_next, self.label_size)
            # dim: (batch_size, label_size(to))
            mask_idx = mask[idx, :].view(batch_size, 1).expand(batch_size, self.label_size)
            # dim: (batch_size, label_size(to))
            forscores.masked_scatter_(mask_idx, cur_partition.masked_select(mask_idx)) 
            # dim: (batch_size, label_size(to))

        # only need end at end_tag
        forscores = forscores[:, self.end_tag].sum()

        # average_batch
        if average_loss:
            loss = (forscores - tg_energy) / batch_size
        else:
            loss = (forscores - tg_energy)
        return loss

    def decode(self, crf_scores, mask=None):
        """
        Find the optimal path with viterbe decode

        Args:
            crf_scores: (batch_size, doc_maxlen+1, label_size(from), label_size(to))
                obtained from get_crf_scores function; for each batch, the score of taking tag_from at current timestamp (based from emission scores) and transitioning to tag_to in the next timestamp, for every possible such sequences
            mask: (batch_size, doc_maxlen)
                masks for padding. If none, no masks is used.
        return:
            decoded_sequence: (batch_size, doc_maxlen)
        """
        ### calculate batch size and seq len
        batch_size = crf_scores.size(0) # note, this may differ from config.batch_size
        doc_maxlen = crf_scores.size(1)-1

        ### dim fix
        crf_scores = crf_scores.transpose(0, 1).detach()
        # dim: (doc_maxlen+1, batch_size, label_size(from), label_size(to))
        
        if mask is not None:
            mask = mask.transpose(0, 1).detach()
            # dim: (doc_maxlen, batch_size)
            mask_broadcast = torch.ones(size=(1, batch_size), device=mask.device).byte()
            mask = torch.cat([mask_broadcast, mask], dim=0)
            # dim: (doc_maxlen+1, batch_size)
        else:
            mask = torch.ones(size=(doc_maxlen+1, batch_size)).byte()
            mask = mask.to(crf_scores.device)
            # dim: (doc_maxlen+1, batch_size)


        mask = 1 - mask
        decode_idx = torch.LongTensor(doc_maxlen, batch_size)

        ### calculate forward score and checkpoint

        # Induction: 
        # if the optimal sequence needs to use tag_i of this current stamp, then tag_i must be the optimal sequence ending at tag_i. We call this the forward score, because we only need to compute it once and carry it forward.

        seq_iter = enumerate(crf_scores)
        # the first score should start with <start>
        _, inivalues = seq_iter.__next__()  # (batch_size, label_size(from), label_size(to))
        # only need start from start_tag
        forscores = inivalues[:, self.start_tag, :].clone()  # (batch_size, label_size)

        back_points = list()
        # iter over last scores
        for idx, transition_score in seq_iter:
            # BEFORE:

            # forscores:
            #     the scores of optimal paths ending at each possible current_tag, tag_i
            # transition_score:
            #     the score of taking each possible current_tag, tag_i, at current timestamp 
            #     and transitioning to any of the possible tags, tag_{i+1}, in next timstamp
            #     for every possible such sequences

            forscores = forscores.contiguous().view(batch_size, self.label_size, 1)
            forscores = forscores.expand(batch_size, self.label_size, self.label_size)
            # dim: (batch_size, label_size(from), label_size(to))
            score_of_all_current_to_next = forscores + transition_score
            forscores, cur_bp = torch.max(score_of_all_current_to_next, 1)

            # AFTER:

            # forscores: (batch_size, label_size(to))
            #    contains the scores of optimal paths ending at each possible next_tag, tag_{i+1}, which will be used as that of the "current" score in next timestamp
            
            # cur_bp: (batch_size, label_size(to))
            #    will then contain the pointers from each possible next_tag to its best tag to transition from at this current timestamp. This will then maksed, and append to back_points, so back_points will be (doc_maxlen-1, batch_size, label_size(to))
            #   NOTE, dimension 1 is of size label_size(to) but takes values of label_size(from)
            
            # if any of the timestamp is padded (e.g. masked), then no matter where it goes in the next timestamp, it should transition from padding; by induction, it SHOULD continue to be this case, until all of paddings fill doc_maxlen
            cur_bp.masked_fill_(mask[idx].view(batch_size, 1).expand(batch_size, self.label_size), self.pad_tag)
            
            # sanity check: 
            #   once it's padded, it should continue to be padded
            #   in future, because once padding started, it should
            #   continue to the end of document sequence

            back_points.append(cur_bp)

        ### backtracking

        # now, last back_points should contain the pointers from each possible next_tag (after the end of doc_maxlen sequence, if we were to continue generating) to its best tag to transition from at the last timestamp; dim: (batch_size, label_size(to))


        pointer = back_points[-1][:, self.end_tag]  # the best last tag to transition to PAD
        decode_idx[-1] = pointer # save it as the last

        # back_points[len(back_points)-2] will be effectively, back_points[-2]
        for idx in range(len(back_points)-2, -1, -1):
            back_point = back_points[idx] # (batch_size)
            index = pointer.contiguous().view(-1,1)
            pointer = torch.gather(back_point, 1, index).view(-1) # (batch_size)
            decode_idx[idx] = pointer
        decoded_sequence = decode_idx.transpose(0, 1)
        # the size of decoded_sequence is restricted by the size of back_points

        return decoded_sequence

    def forward(self, scores, gold_target, mask=None, average_loss=True):
        crf_scores = self.get_crf_scores(scores)
        return self.get_loss(crf_scores, gold_target, mask=mask, average_loss=average_loss)
