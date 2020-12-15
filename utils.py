# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 00:47:01 2020

@author: Patrice CHANOL
"""

import scipy as sp
import numpy as np
import math
import torch


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
