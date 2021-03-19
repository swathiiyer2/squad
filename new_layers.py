"""Assortment of layers for use in my_models.py.
Author:
    Chris Chute (chute@stanford.edu)
"""

from layers import Embedding as WordEmbeddings

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from util import masked_softmax

class FForward(nn.Module):

    def __init__(self, embed_dim, dropout_rate):
        super(FForward, self).__init__()
        
        #added layernorm for compatability across our last dimension  
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.enter_linear = nn.Linear(embed_dim, embed_dim)
        self.exit_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        
    def forward(self, x):

        x_copy = x # shape: (batch_size, text_len, input_dim)
        
        x = self.enter_linear(self.layer_norm(x)) # shape: (batch_size, text_len, input_dim)
        x = self.exit_linear(F.relu(x)) # shape: (batch_size, text_len, input_dim)
        
        return x_copy + self.dropout(x) # shape: (batch_size, text_len, input_dim)


class MultiheadSelfAttention(nn.Module):
    """
    Ref Assignment 5
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling (??????) my own here.
    """

    def __init__(self, n_embd, n_head, drop_prob=0.1, num_convs=4, n_kernel=7, block_index=1, num_blocks=1):
        #print(num_blocks)
        '''
        super().__init__()
        # key, query, value projections for all heads
        print("making new layer")
        self.key = nn.Linear(n_embd, n_embd)
        print("made key layer")
        self.query = nn.Linear(n_embd, n_embd)
        print("made query layer")
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(drop_prob)
        self.resid_drop = nn.Dropout(drop_prob)
        print("finished reg")

        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        block_size = 128
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))
        self.n_head = n_head
        print("made whole self-attn")
'''

        super(MultiheadSelfAttention, self).__init__()

        self.n_embd = n_embd
        numerator = (3 + num_convs) * block_index + 1
        denominator = (3 + num_convs) * num_blocks

        # Each layer of our separatable convolution
        self.convs = nn.Sequential(*[DSCNN(n_embd, n_kernel, (0.1 * (numerator + conv_step)) / denominator) for conv_step in range(1, 1 + num_convs)])
        self.resize = nn.Linear(n_embd, 2 * n_embd)

        self.leveled_dropout = nn.Dropout((0.1 * numerator) / denominator)
        
        # self.convs = DSCNN(n_embd, kernel_size, 0.1 )
                         
        self.mh_attention = nn.MultiheadAttention(n_embd, n_head, (0.1 * (numerator + 1 + num_convs)) / denominator)
        self.dropout = nn.Dropout(drop_prob)

        self.layernorm = nn.LayerNorm(n_embd)
        self.ffwd = FForward(n_embd, (0.1 * (numerator + 1 + num_convs)) / denominator)

    def forward(self, x, x_pe, is_pad):

#    def forward(self, x, layer_past=None):
        '''
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, -1e10) # todo: just use float('-inf') instead?
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
        '''
        x = self.leveled_dropout(x + x_pe)
        x_copy = self.convs(x) ## shape (batch_size, text_len, input_dim)
        x = self.layernorm(x_copy) ## shape (batch_size, text_len, input_dim)

        ## REMINDER: Dimension requirments for MH Attention mean we have to use transpose, ew to their documentation        
        x = x.transpose(0,1)        
        # print(is_pad.size())
        x, _ = self.mh_attention(x, x, x, key_padding_mask = is_pad, need_weights = False) 
        x = self.dropout(x.transpose(0,1)) + x_copy ## shape (batch_size, text_len, input_dim)
        #ll = nn.Linear(self.n_embd, 2 * self.n_embd)

        #x =  self.resize(x)
        #print(x.size())
        x = self.ffwd(x)
        return x


class WordAndCharEmbedding(nn.Module):

    def __init__(self, char_vectors, word_vectors, hidden_size, drop_prob):
        super(WordAndCharEmbedding, self).__init__()

        em_sz = hidden_size//2
        self.word_embedding = WordEmbeddings(word_vectors, em_sz, drop_prob)
        self.char_embedding = CharEmbeddings(char_vectors, em_sz, drop_prob)

    def forward(self, w_idxs, c_idxs):

        out = torch.cat([self.word_embedding(w_idxs), self.char_embedding(c_idxs)], dim=2)

        return out


class CharEmbeddings(nn.Module):
    # Apparently this is from THE OLD HOMEWORK 5 damn I was SOOO confused - used starter code from the assignment for
    # draft of structure.

    def __init__(self, char_vectors, embed_dim, dropout_prob=0.2, char_embed_size=64, largest_char_size=16, kernel_size=5):

        super(CharEmbeddings, self).__init__()

        self.cnn = CNN(char_embed_size, embed_dim, largest_char_size=largest_char_size, kernel_size=kernel_size)
        self.char_embedding = nn.Embedding.from_pretrained(char_vectors)
        self.gate = nn.Linear(embed_dim, embed_dim, bias=True)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
 
        x = self.char_embedding(x) #shape: (sent_len, batch_size, max_word_len, char_embed_size)

        sent_len = x.size()[0]
        batch_size = x.size()[1]
        max_word_len = x.size()[2]
        char_embed_dim = x.size()[3]

        x = x.view(sent_len * batch_size, max_word_len, char_embed_dim) # shape: (sent_len, batch_size, max_word_len, char_embed_size)
        x = x.permute(0, 2, 1) # shape: (sent_len, batch_size, max_word_len, char_embed_size)

        t = self.cnn(x) #shape: (batch_size * sent_len, char_embed_size, max_word_len)

        g = torch.sigmoid(self.gate(t))
        x = g * t + (1 - g) * t # shape: (sent_len * batch_size, word_embed_size)

        x = self.dropout(x) #shape:(sent_len * batch_size, word_embed_size)
        out = x.view(sent_len, batch_size, -1) #shape:(sent_len * batch_size, word_embed_size)

        return out


class CNN(nn.Module):

    def __init__(self, char_embed_size, word_embed_size, largest_char_size=21, kernel_size=5): 
        super(CNN, self).__init__()

        self.max_pooling = nn.MaxPool1d((largest_char_size - kernel_size) + 1)

        self.conv1d = nn.Conv1d(kernel_size=kernel_size, in_channels=char_embed_size, out_channels=word_embed_size, bias=True)

    def forward(self, x):

        return self.max_pooling(torch.relu(self.conv1d(x))).squeeze()

class DSCNN(nn.Module):
    def __init__(self, embed_dim, kernel_dim, dropout_prob):
        super(DSCNN, self).__init__()
                
        ## Layer normalization across the features, i.e. across the last dimension that is equal to input_dim
        self.layernorm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_prob)
        
        ## In order to enable skip connection we have the input and output dimensions be equal. 
        self.conv_point = nn.Linear(embed_dim, embed_dim)

        #Padding is set to half of the kernel size to standardize output and input text. 

        padding_size = kernel_dim // 2
        #REMINDER: Kernel size needs to be an odd number
        self.conv_depthwise = nn.Conv1d(embed_dim, embed_dim, kernel_dim, padding = padding_size, groups = embed_dim, bias = False)
        
    def forward(self, x):
        x_copy = x # shape: (batch_size, text_len, input_dim)
        x = self.layernorm(x) # shape: (batch_size, text_len, input_dim)
        x = self.conv_depthwise(x.transpose(1,2)).transpose(1,2) # shape: (batch_size, text_len, input_dim)
        return x_copy + self.dropout(F.leaky_relu(self.conv_point(x))) # shape: (batch_size, text_len, input_dim)



class ModOutput(nn.Module):
    def __init__(self, hidden_size):
        super(ModOutput, self).__init__()
        

        self.left_bias = nn.Parameter(torch.zeros(1))
        self.right_bias = nn.Parameter(torch.zeros(1))

        self.proj1 = nn.Parameter(torch.empty(hidden_size, 1))
        self.proj3 = nn.Parameter(torch.empty(hidden_size, 1))
        self.proj2 = nn.Parameter(torch.empty(hidden_size, 1))
        self.proj4 = nn.Parameter(torch.empty(hidden_size, 1))

        nn.init.xavier_uniform_(self.proj1)
        nn.init.xavier_uniform_(self.proj3)
        nn.init.xavier_uniform_(self.proj2)
        nn.init.xavier_uniform_(self.proj4)
        
        
    def forward(self, satt1, satt2, satt3, is_pad):

        mult1 = torch.matmul(satt1, self.proj1)
        mult2 = torch.matmul(satt2, self.proj3)

        sum1 = self.left_bias + mult1 + mult2   # shape: (batch_size, text_len, 1)
        log_p1 = masked_softmax(sum1.squeeze(dim=2), is_pad, log_softmax=True)  # shape: (batch_size, text_len)

        mult1 = torch.matmul(satt1, self.proj2)
        mult2 = torch.matmul(satt3, self.proj4)

        sum2 = self.right_bias + mult1 + mult2  # shape: (batch_size, text_len, 1)
        log_p2 = masked_softmax(sum2.squeeze(dim=2), is_pad, log_softmax=True)  # shape: (batch_size, text_len)
        print(log_p1.size())
        print(log_p2.size())
        return log_p1, log_p2

class PositionalEncoding(nn.Module):
    # Reference: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
