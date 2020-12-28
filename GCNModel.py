# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 23:41:47 2020

@author: Patrice CHANOL
"""

import torch
import torch.nn as nn
from GCNLayer import GraphConvolutionnalLayer


class GCN_Class(nn.Module):
    def __init__(self, in_features, hidden_dim, classes, dropout=0.1):
        super(GCN_Class, self).__init__()

        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.classes = classes
        self.name_model = "GCN_Class"

        self.gcl1 = GraphConvolutionnalLayer(in_features, hidden_dim)
        self.gcl2 = GraphConvolutionnalLayer(hidden_dim, classes)
        self.dropout = dropout

    def forward(self, x, adj):
        out = self.gcl1(x, adj)
        out = out.relu()
        out = torch.nn.functional.dropout(out, self.dropout, training=self.training)
        out = self.gcl2(out, adj)
        return torch.nn.functional.log_softmax(out, dim=2).transpose(1,2)

    def save(self):
        """Enregistre le modèle pour inférence dans le futur."""
        torch.save(self.state_dict(), './models/model_' + self.name_model + '_' + str(self.in_features) + 'in_' + str(self.classes) + 'out.gcn')

    def load(self):
        """Récupère un modèle déjà entrainé pour inférer."""
        self.load_state_dict(torch.load('./models/model_' + self.name_model + '_' + str(self.in_features) + 'in_' + str(self.classes) + 'out.gcn'))


class GCN_Reg(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, dropout=0.1):
        super(GCN_Reg, self).__init__()

        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        self.name_model = "GCN_Reg"

        self.gcl1 = GraphConvolutionnalLayer(in_features, hidden_dim)
        self.gcl2 = GraphConvolutionnalLayer(hidden_dim, out_features)
        self.dropout = dropout

    def forward(self, x, adj):
        out = self.gcl1(x, adj)
        out = out.relu()
        out = torch.nn.functional.dropout(out, self.dropout, training=self.training)
        out = self.gcl2(out, adj)
        return out.relu()

    def save(self):
        """Enregistre le modèle pour inférence dans le futur."""
        torch.save(self.state_dict(), './models/model_' + self.name_model + '_' + str(self.in_features) + 'in_' + str(self.out_features) + 'out.gcn')

    def load(self):
        """Récupère un modèle déjà entrainé pour inférer."""
        self.load_state_dict(torch.load('./models/model_' + self.name_model + '_' + str(self.in_features) + 'in_' + str(self.out_features) + 'out.gcn'))
