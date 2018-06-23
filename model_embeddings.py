#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""
import torch
import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway
import torch.nn as nn

# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()
        ### YOUR CODE HERE for part 1h

        self.word_embed_size = word_embed_size
        self.vocab = vocab
        self.char_embedding = nn.Embedding(len(self.vocab.char2id), 50)
        self.cnn = CNN(50, self.word_embed_size, 5, 1)
        self.highway = Highway(self.word_embed_size)
        self.dropout = nn.Dropout(p=0.3)
        
        ### END YOUR CODE

    def forward(self, input):
        
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1h
        batch_size = input.size(1)
        sentence_length = input.size(0)
        max_word_length = input.size(2)
        embedded = self.char_embedding(input) # sentence_length, batch_size, max_word_length
        embedded = embedded.view(sentence_length*batch_size, max_word_length, -1) # sentence_length*batch_size, max_word_length, embedding
        embedded = embedded.transpose(1,2) # sentence_length*batch_size, embedding, max_word_length
        conv_out = self.cnn(embedded)
        word_embed = self.highway(conv_out)
        word_embed = self.dropout(word_embed)
        word_embed = word_embed.view(sentence_length, batch_size, -1)
        return word_embed
    
        ### END YOUR CODE