# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 00:47:01 2020

@author: Patrice CHANOL
"""

import numpy as np
import random
import torch

def load_data(x_file=None, adj_mats_file='data/adj_mats.pt', targets_file='data/labels.pt', tr=0.8, batch_size=128, perm=False):
    adj_mats = torch.load(adj_mats_file)
    if x_file is not None:
        x = torch.load(x_file)
    else:
        x = torch.ones((adj_mats.size(0), adj_mats.size(1)))
    targets = torch.load(targets_file)
    if perm:
        x, adj_mats, targets = permutations([x, adj_mats, targets], r=8, batch=True)
    x = x.unsqueeze(2)
    data = list(zip(zip(x, adj_mats.to_dense()), targets))
    random.shuffle(data)
    train = torch.utils.data.DataLoader(data[:int(len(data)*tr)], batch_size=batch_size, shuffle=True)
    test = torch.utils.data.DataLoader(data[int(len(data)*tr):], batch_size=batch_size, shuffle=True)
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
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    return (correct.sum(), len(correct.view(-1)))


def adj_normalize(adj):
    if adj.dim() == 2:
        adj_c = adj + torch.eye(adj.size(0), adj.size(1))
    else:
        i_adj = torch.eye(adj.size(-2), adj.size(-1))
        i_adj = i_adj.reshape([1 for i in range(adj.dim() - 2)] + [adj.size(-2), adj.size(-1)])
        i_adj = i_adj.repeat([adj.size(i) for i in range(adj.dim() - 2)] + [1, 1])
        adj_c = adj + i_adj.to_sparse()
    adj_c = adj_c.to_dense()
    D = torch.diag_embed(1/torch.sqrt(adj_c.sum(adj_c.dim()-1)))
    return torch.bmm(torch.bmm(D, adj_c), D).to_sparse()


def permutations(list_tenseurs, r=np.inf, batch=False):
    """
    Génère une même permutation de tenseur dans une liste de tenseurs.
    
    Chaque tenseur de la liste se verra permutter de façon identique.
    r est le nombre de permuttation différente générer (attention, avec batch,
    on explose la puissance de calcul)
    Si batch est Vrai alors il peut y avoir des doublons
    """
    n = list_tenseurs[0].shape[-1]
    l = len(list_tenseurs)
    perm_list = list([] for k in range(l))
    already_perm = list((i,i) for i in range(n))
    if r >= n:
        r = n-1
    for _ in range(r):
        (i,j) = (0, 0)
        while (i,j) in already_perm:
            (i,j) = (np.random.randint(n), np.random.randint(n))
        already_perm.append((i,j))
        for tenseur in range(l):
            new_tenseur = list_tenseurs[tenseur].clone()
            dim = new_tenseur.dim()
            sparse_t = False
            if new_tenseur.layout != torch.strided:
                new_tenseur = new_tenseur.to_dense()
                sparse_t = True
            if dim == 1:
                a, b = new_tenseur[j].clone(), new_tenseur[i].clone()
                new_tenseur[i], new_tenseur[j] = a, b
            elif dim == 2:
                if batch:
                    a, b = new_tenseur[:, j].clone(), new_tenseur[:, i].clone()
                    new_tenseur[:, i], new_tenseur[:, j] = a, b
                else:
                    a_col, b_col = new_tenseur[:, i].clone(), new_tenseur[:, j].clone()
                    new_tenseur[:, j], new_tenseur[:, i] = a_col, b_col
                    a_line, b_line = new_tenseur[i, :].clone(), new_tenseur[j, :].clone()
                    new_tenseur[j, :], new_tenseur[i, :] = a_line, b_line
            elif dim == 3:
                a_col, b_col = new_tenseur[:, :, i].clone(), new_tenseur[:, :, j].clone()
                new_tenseur[:, :, j], new_tenseur[:, :, i] = a_col, b_col
                a_line, b_line = new_tenseur[:, i, :].clone(), new_tenseur[:, j, :].clone()
                new_tenseur[:, j, :], new_tenseur[:, i, :] = a_line, b_line
            else:
                break
            if batch:
                if sparse_t:
                    new_tenseur = torch.cat((new_tenseur, list_tenseurs[tenseur].to_dense()), dim=0)
                    new_tenseur = new_tenseur.to_sparse()
                else:
                    new_tenseur = torch.cat((new_tenseur, list_tenseurs[tenseur]), dim=0)
                list_tenseurs[tenseur] = new_tenseur
                perm_list[tenseur] = list_tenseurs[tenseur]
            else:
                if sparse_t:
                    new_tenseur = new_tenseur.to_sparse()
                perm_list[tenseur].append(new_tenseur)
    return perm_list