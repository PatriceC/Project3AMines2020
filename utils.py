# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 00:47:01 2020

@author: Patrice CHANOL
"""

import numpy as np
import random
import torch


def load_data(souches_file='data/souches.pt', adj_mats_file='data/adj_mats.pt', pops_file='data/pops.pt', tr=0.8, batch_size=16):
    souches = torch.load(souches_file)
    adj_mats = torch.load(adj_mats_file)
    pops = torch.load(pops_file)
    data = list(zip(list(zip(souches, adj_mats)), pops))
    random.shuffle(data)
    train = torch.utils.data.DataLoader(data[:int(len(data)*tr)], batch_size=batch_size)
    test = torch.utils.data.DataLoader(data[int(len(data)*tr):], batch_size=batch_size)
    return train, test


def convert(M):
    """
    input: M is Scipy sparse matrix
    output: pytorch sparse tensor 
    """
    M = M.tocoo()
    indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    Ms = torch.sparse_coo_tensor(indices, values, shape)
    return Ms

def accuracy(output, labels):
    preds = output.max(2)[1].type_as(labels)
    correct = preds.eq(labels).double()
    return (correct.sum(), len(correct.view(-1)))