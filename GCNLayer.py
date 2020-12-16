# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 18:30:15 2020

@author: Patrice CHANOL
"""

import math
import torch
import torch.nn as nn


class GraphConvolutionnalLayer(nn.Module):
    """"Graph Convolutionnal Layer"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = torch.nn.Parameter(torch.FloatTensor(self.in_features, self.out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, input, adj_matrix):
        y = input.shape[-1]
        if y != self.in_features:
            print(f'Wrong Input Features. Please use tensor with {self.in_features} Input Features')
            return 0
        output = torch.matmul(input, self.weight)
        output = torch.bmm(adj_matrix.float(), output)
        if self.bias is not None:
            output += self.bias
        ret = output
        return ret
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
