# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 23:41:47 2020

@author: Patrice CHANOL
"""

import torch
import torch.nn as nn
from GCNLayer import GraphConvolutionnalLayer


class GCN(nn.Module):
    def __init__(self, in_features, hidden_dim, classes, dropout=0.1):
        super(GCN, self).__init__()

        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.classes = classes
        self.name_model = "GCN"

        self.gcl1 = GraphConvolutionnalLayer(in_features, hidden_dim)
        self.gcl2 = GraphConvolutionnalLayer(hidden_dim, classes)
        self.dropout = dropout

    def forward(self, x, adj):
        out = self.gcl1(x, adj)
        out = out.relu()
        out = torch.nn.functional.dropout(out, self.dropout, training=self.training)
        out = self.gcl2(out, adj)
        return torch.nn.functional.log_softmax(out, dim=out.shape[-1])

    def save(self):
        """Enregistre le modèle pour inférence dans le futur."""
        torch.save(self.state_dict(), './models/model_' + self.name_model + '_' + str(self.in_features) + '_souches_' + str(self.classes) + 'classes.pt')

    def load(self):
        """Récupère un modèle déjà entrainé pour inférer."""
        self.load_state_dict(torch.load('./models/model_' + self.name_model + '_' + str(self.in_features) + '_souches_' + str(self.classes) + 'classes.pt'))
