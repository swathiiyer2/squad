"""Assortment of layers for use in my_models.py.
Author:
    Chris Chute (chute@stanford.edu)
"""

from layers import Embedding as WordEmbedding

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadSelfAttention(nn.Module):
    """
    Ref Assignment 5
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
    """

    def __init__(self, n_embd, n_head, drop_prob=0.1):
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


        self.attention = nn.MultiheadAttention(n_embd, n_head)
        self.dropout = nn.Dropout(drop_prob)

        ## Layer normalization across the features, i.e. across the last dimension that is equal to input_dim
        self.layernorm = nn.LayerNorm(n_embd)
    def forward(self, x, is_pad):
    '''
    def forward(self, x, layer_past=None):
        
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

        """
        x: input tensor of shape (batch_size, text_len, input_dim).
            Here text_len is the length of the context/question.
        is_pad: tensor of shape(batch_size, text_len). Hold value TRUE for pad tokens. 
        Output: tensor of the same shape as the input, (batch_size, text_len, input_dim)
        """
        skip_connection = x

        x = self.layernorm(x) ## shape (batch_size, text_len, input_dim)

        ## shape (text_len, batch_size, input_dim).
        ## Here transpose() is needed because of the convention of nn.MultiheadAttention.
        x = x.transpose(0,1)		
        x, _ = self.attention(x, x, x, key_padding_mask = is_pad, need_weights=False) 

        x = x.transpose(0,1) ## shape (batch_size, text_len, input_dim)		
        return self.dropout(x) + skip_connection



class EmbeddingWithChar(nn.Module):
    """Embedding layer used by BiDAF, with the character-level component.
    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).
    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        char_vectors (torch.Tensor): Initial char vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob):
        super(EmbeddingWithChar, self).__init__()
        self.word_embed = WordEmbedding(word_vectors, hidden_size//2, drop_prob)
        self.char_embed = CharEmbeddings(char_vectors, hidden_size//2, drop_prob)

    def forward(self, w_idxs, c_idxs):
        word_emb = self.word_embed(w_idxs)   # (batch_size, seq_len, hidden_size//2)
        char_emb = self.char_embed(c_idxs)   # (batch_size, seq_len, hidden_size//2)

        emb = torch.cat([word_emb, char_emb], dim=2)

        return emb


class CharEmbeddings(nn.Module):
    """
    Class that converts input words to their embeddings from convex neural networks.
    """
    def __init__(self, char_vectors, embed_size, drop_prob=0.2,
                       char_embed_size=64, char_limit=16, kernel_size=5):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output
        @param vocab_size (int): Vocabulary size (i.e. Character tokens numbers).
        """
        super(CharEmbeddings, self).__init__()

        self.embed_size = embed_size

        self.max_word_len = char_limit
        self.dropout_rate = drop_prob
        self.kernel_size = kernel_size

        self.char_embedding = nn.Embedding.from_pretrained(char_vectors)
        self.char_embed_size = self.char_embedding.embedding_dim

        self.cnn = CNN(
            char_embed_dim=self.char_embed_size,
            word_embed_dim=self.embed_size,
            max_word_length=self.max_word_len,
            kernel_size=self.kernel_size
        )

        self.highway = HighwayEncoderChar(
            embed_dim=self.embed_size
        )

        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param x: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary
        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        # (sentence_length, batch_size, max_word_length)
        x_emb = self.char_embedding(x) # look up char embedding
        sentence_length, batch_size, max_word_length, char_embed_size = x_emb.size()
        # (sentence_length, batch_size, max_word_length, char_embed_size)
        x_reshaped = x_emb.view(sentence_length*batch_size, max_word_length, char_embed_size).permute(0, 2, 1)
        # (sentence_length * batch_size, char_embed_size, max_word_length)
        x_conv = self.cnn(x_reshaped)
        # (sentence_length * batch_size, word_embed_size)
        x_highway = self.highway(x_conv)
        # (sentence_length * batch_size, word_embed_size)
        x_word_emb = self.dropout(x_highway)
        #  (sentence_length * batch_size, word_embed_size)
        output = x_word_emb.view(sentence_length, batch_size, -1)
        # (sentence_length, batch_size, word_embed_size)

        return output

class CNN(nn.Module):
    """ Uses convex neural network to combine the initial character embeddings """

    def __init__(self, char_embed_dim: int, # e_char
                       word_embed_dim: int, # e_word (set filter number to be equal to e_word)
                       max_word_length: int=21, # (m_word) max word length
                       kernel_size: int=5): # window size
        super(CNN, self).__init__()

        self.conv1d = nn.Conv1d(
            in_channels=char_embed_dim,
            out_channels=word_embed_dim, # number of filter, output feature
            kernel_size=kernel_size,
            bias=True)

        # MaxPool simply takes the maximum across the second dimension
        self.maxpool = nn.MaxPool1d(max_word_length - kernel_size + 1)

    def forward(self, x):
        # (batch size, char embedding size, max word length)
        x_conv = self.conv1d(x)
        # (batch size, word embedding size, max_word_length - kernel_size + 1)
        x_conv_out = self.maxpool(torch.relu(x_conv)).squeeze()
        # (batch size, word embedding size)

        return x_conv_out

class HighwayEncoderChar(nn.Module):
    """ Highway Networks6 have a skip-connection controlled by a dynamic gate """

    def __init__(self, embed_dim: int): # word embedding dimension
        super(HighwayEncoderChar, self).__init__()

        self.conv_out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.gate = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x_conv_out):
        x_proj = torch.relu(self.conv_out_proj(x_conv_out))
        g = torch.sigmoid(self.gate(x_conv_out))

        x = g * x_conv_out + (1 - g) * x_conv_out

        return x
