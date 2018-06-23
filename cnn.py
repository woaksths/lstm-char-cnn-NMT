#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class CNN(nn.Module):
    
    def __init__(self, inputs_dim, filters_num, kernel_size=5, padding_size=1):
        super(CNN, self).__init__()
        self.inputs_dim = inputs_dim
        self.filters_num = filters_num
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.conv1d = nn.Conv1d(in_channels=inputs_dim, out_channels=filters_num, kernel_size=kernel_size, padding=padding_size)
    
    
    def forward(self, inputs):
        
        '''
        @params inputs -> (max_sentence_length, batch, max_word_length)
        @return out -> (max_sentence_length, batch, hidden) 
        '''

        conv_out = self.conv1d(inputs)
        return conv_out
    

