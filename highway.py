#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class Highway(nn.Module):
    
    def __init__(self, input_dim):
        super(Highway, self).__init__()
        self.input_dim = input_dim
        self.transform_layer = nn.Linear(self.input_dim, self.input_dim)
        self.highway_layer = nn.Linear(self.input_dim, self.input_dim)
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        '''
        @params inputs: inputs is an output of convolution step #(batch, input_dim)
        @return out: #(batch, input_dim)
        '''
        t = self.transform_layer(inputs)
        t = torch.sigmoid(t)
        
        transform_gate = t
        carray_gate = 1 - t
        
        projection = self.highway_layer(inputs)
        projection = self.relu(projection)
        out = (transform_gate* projection) + (carray_gate*inputs)
        return out
    
    