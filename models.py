"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import new_layers
import torch
import torch.nn as nn


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    '''
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()


        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)
        '''
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()

        self.emb = new_layers.EmbeddingWithChar(word_vectors=word_vectors,
                                    char_vectors=char_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        #self.att = layers.BiDAFAttention(hidden_size=hidden_size,
        #                                 drop_prob=drop_prob)

        self.pre_satt = nn.Linear(8 * hidden_size, hidden_size)

        self.post_satt = nn.Linear(hidden_size, 8 * hidden_size)


        self.self_att1 = new_layers.MultiheadSelfAttention(n_embd=hidden_size,
                                                          n_head=2,
                                                          drop_prob=drop_prob)

        self.self_att2 = new_layers.MultiheadSelfAttention(n_embd=hidden_size,
                                                          n_head=2,
                                                          drop_prob=drop_prob)

        self.self_att3 = new_layers.MultiheadSelfAttention(n_embd=hidden_size,
                                                          n_head=2,
                                                          drop_prob=drop_prob)

        self.self_attn_blocks = nn.ModuleList([
            new_layers.MultiheadSelfAttention(n_embd=hidden_size, n_head=2, drop_prob=drop_prob)
            for block_index in range(3)])

        self.mod = layers.RNNEncoder(input_size = 8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    #def forward(self, cw_idxs, qw_idxs):
    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs

        c_is_pad = torch.zeros_like(cw_idxs) == cw_idxs ## shape (batch_size, c_len)
        q_is_pad = torch.zeros_like(qw_idxs) == qw_idxs ## shape (batch_size, q_len)

        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        #c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        #q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)
        
        #print("input cw_idxs")
        #print(cw_idxs)
        #print("input qu_idxs")
        #print(qw_idxs)


        c_emb = self.emb(cw_idxs, cc_idxs)   # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)  # (batch_size, q_len, hidden_size)
        

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        #print("c_emb is")
        #print(c_emb)
        #print("q_emb is")
        #print(q_emb)

        #c_enc = self.self_att(c_emb, c_mask)
        #q_enc = self.self_att(q_emb, q_mask)

        #print(c_enc)
        #print(q_enc)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)
        
        #print("att is")
        #print(att)

        self_attn = self.pre_satt(att)    # (batch_size, c_len, hidden_size)

        '''
        self_att = self.self_att1(att_prelim, c_is_pad)

        
        self_att2 = self.self_att2(self_att, c_is_pad)

        
        self_att3 = self.self_att3(self_att2, c_is_pad)
        '''


        for self_att in self.self_attn_blocks:
            self_attn = self_att(x, c_is_pad)

        self_attn = self.post_satt(self_attn)       # (batch_size, c_len, 8 * hidden_size)
        #print("size of self att3")
        #print(self_att3.size())

        mod = self.mod(self_attn, c_len)        # (batch_size, c_len, 2 * hidden_size)
        #print("size of mod")
        #print(mod.size())
        
        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)
        #print("out is")
        #print(out)
        return out
