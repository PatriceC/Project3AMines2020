# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 23:41:47 2020

@author: Patrice CHANOL
"""

import torch.nn as nn
import torch.nn.functional as F
from GCNLayer import GraphConvolutionnalLayer


class GCN(nn.Module):
    def __init__(self, in_features, hidden_dim, classes, dropout=0.1):
        super(GCN, self).__init__()

        self.gcl1 = GraphConvolutionnalLayer(in_features, hidden_dim)
        self.gcl2 = GraphConvolutionnalLayer(hidden_dim, classes)
        self.dropout = dropout

    def forward(self, x, adj):
        out = self.gc1(x, adj)
        out = out.relu()
        out = F.dropout(out, self.dropout, training=self.training)
        out = self.gc2(out, adj)
        return F.log_softmax(out, dim=0)
