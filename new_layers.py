"""Assortment of layers for use in my_models.py.
Author:
    Chris Chute (chute@stanford.edu)
"""

from layers import Embedding as WordEmbedding


import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # (sentence_length * batch_size, word_embed_size)
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