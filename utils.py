# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 00:47:01 2020

@author: Patrice CHANOL
"""

import numpy as np
import random
import torch
from sklearn.metrics import confusion_matrix
from GCNLayer import GraphConvolutionnalLayer

def load_data(x_file: str = None,
              adj_mats_file: str = 'data/adj_mats.pt',
              targets_file: str = 'data/labels.pt',
              tr: float = 0.9,
              cv: int = 0,
              batch_size: int = 128,
              diversity: bool = True,
              equi: bool = True,
              label: int = 2,
              perm: bool = False,
              r: float = np.inf,
              load: bool = False):
    """
    Load data.

    Parameters
    ----------
    x_file : str, optional
        Input file path. The default is None.
    adj_mats_file : str, optional
        Adj_mats file path. The default is 'data/adj_mats.pt'.
    targets_file : str, optional
        Target file path. The default is 'data/labels.pt'.
    tr : float, optional
        Training size. The default is 0.8.
    cv : int, optional
        Number of fold for cross validation. The default is 0.
    batch_size : int, optional
        Batch size. The default is 128.
    diversity : bool, optional
        If you want the labels distribution. The default is True.
    equi : bool, optional
        If you want to have equal size labels. The default is True.
    label : int, optional
        The label you want to limit the size if equi is True. The default is 2.
    perm : bool, optional
        Dataset increase by permutation. The default is False.
    r : int, optional
        Number of permutation. The default is np.inf.
    load : bool, optional
        If you want only one set. The default is False.

    Returns
    -------
    out_features : int
        Number of out features of the model.
    dataset : list
        List containing the train and test set.

    """
    adj_mats = torch.load(adj_mats_file).to_dense()
    targets = torch.load(targets_file)
    if x_file is not None:
        x = torch.load(x_file)
    else:
        x = torch.ones((adj_mats.size(0), adj_mats.size(1)))
    if equi:
        [targets, x, adj_mats] = equilibrage(targets, label,
                                             [x, adj_mats])
    if perm:
        if load:
            x, adj_mats, targets = permutations([x, adj_mats, targets],
                                                r=8, batch=True)
        else:
            x, adj_mats, targets = permutations([x, adj_mats, targets],
                                                r=r, batch=True)
    if x_file is not None:
        out_features = 1
        targets = targets.unsqueeze(2)

    x = x.unsqueeze(2)
    if diversity:
        class_repr = dataset_diversity(targets)
        print('Répartition des bactéries dans les classes')
        for i, classes in enumerate(class_repr):
            print('Label {} : {}'.format(i, classes))
        out_features = len(class_repr)
    n = len(x)
    print('Taille du dataset', n)


    if load:
        test = [[x, adj_mats, targets]]
        train = 0
        dataset = [[train, test]]
    else:
        shuf = np.arange(n)
        random.shuffle(shuf)
        x, adj_mats, targets = x[shuf], adj_mats[shuf], targets[shuf]
    
        if cv > 0:
            dataset = list()
            for k in range(cv):
                ind = np.arange(int(1 * n//cv), int((1 + 1) * n//cv))
                ind_test = np.array([i for i in range(n) if i not in ind])
                test = [[x[ind], adj_mats[ind], targets[ind]]]
                train = torch.utils.data.TensorDataset(x[ind_test],
                                                      adj_mats[ind_test],
                                                      targets[ind_test])
                train = torch.utils.data.DataLoader(train,
                                                    batch_size=batch_size,
                                                    shuffle=True)
                dataset.append([train, test])
    
        else:
            dataset_train = torch.utils.data.TensorDataset(x[:int(n*tr)],
                                                           adj_mats[:int(n*tr)],
                                                           targets[:int(n*tr)])
            test = [[x[int(n*tr):], adj_mats[int(n*tr):], targets[int(n*tr):]]]
            train = torch.utils.data.DataLoader(dataset_train,
                                                batch_size=batch_size,
                                                shuffle=True)
            dataset = [[train, test]]
    return out_features, dataset


def weights_init(m):
    """Initialize weights."""
    if isinstance(m, GraphConvolutionnalLayer):
        m.reset_parameters()
    elif isinstance(m, torch.nn.Linear):
        m.reset_parameters()


def convert(M: any) -> torch.Tensor:
    """
    Convert Scipy sparse matrix to pytorch sparse tensor.

    Parameters
    ----------
    M : any
        Scipy sparse matrix.

    Returns
    -------
    Ms : torch.Tensor
        pytorch sparse tensor.

    """
    M = M.tocoo()
    indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    Ms = torch.sparse_coo_tensor(indices, values, shape)
    return Ms


def accuracy(output: torch.Tensor, labels: torch.Tensor, p: bool = False):
    """Compute tensor accuracy."""
    device = output.device
    preds = output.max(1)[1].type_as(labels)
    confmat = confusion_matrix(labels.cpu().flatten(), preds.cpu().flatten())
    correct = preds.eq(labels).double()
    correct_sim = (correct.sum(dim=1) == correct.size(1) *
                   torch.ones(correct.size(0)).to(device))
    correct_sim = correct_sim.double()
    correct_1 = correct.sum(dim=1) / correct.size(1)
    if p:
        print(preds)
        print(labels)
    return ([correct_sim.sum(), len(correct_sim.view(-1))],
            [correct_1.sum(), len(correct_1 .view(-1))],
            confmat)


def equilibrage(tenseur_to_eq: torch.Tensor,
                label, list_tenseurs: list = [], perc: float = 0.41):
    """
    Keep elements which have less than perc of "label" in tenseur_to_eq.

    Parameters
    ----------
    tenseur_to_eq : torch.Tensor
        Tensor use for the selection.
    label : TYPE
        The label we want to limit.
    list_tenseurs : list, optional
        Tensors with the same indexation as tenseur_to_eq. The default is [].
    perc : float, optional
        The threshold. The default is 0.41.

    Returns
    -------
    new_list_tenseurs : TYPE
        DESCRIPTION.

    """
    indices = ((tenseur_to_eq == label).sum(1) / tenseur_to_eq.size(1)) <= perc
    new_list_tenseurs = [tenseur_to_eq[indices]]
    for tenseur in list_tenseurs:
        new_list_tenseurs.append(tenseur[indices])
    return new_list_tenseurs


def adj_normalize(adj):
    """Normalize adj_mat."""
    if adj.dim() == 2:
        adj_c = adj + torch.eye(adj.size(0), adj.size(1))
    else:
        i_adj = torch.eye(adj.size(-2), adj.size(-1))
        i_adj = i_adj.reshape(
            [1 for i in range(adj.dim() - 2)] + [adj.size(-2), adj.size(-1)])
        i_adj = i_adj.repeat(
            [adj.size(i) for i in range(adj.dim() - 2)] + [1, 1])
        adj_c = (adj + i_adj.to_sparse()).to_dense()
    D = torch.diag_embed(1/torch.sqrt(adj_c.sum(adj_c.dim()-1)))
    return torch.bmm(torch.bmm(D, adj_c), D).to_sparse()


def normalize(x: torch.Tensor):
    """Locally row normalize a tensor."""
    x_max = x.max(dim=x.dim() - 2)
    x_min = x.min(dim=x.dim() - 2)
    x_norm = torch.zeros((x.shape))
    for i in range(len(x)):
        x_norm[i] = 2*(x[i] - x_min[0][i])/(x_max[0][i] - x_min[0][i]) - 1
    return x_norm


def dataset_diversity(targets):
    """Compute dataset class representation."""
    L = targets.view(-1).size().numel()
    C = int(targets.max())
    R = list()
    for i in range(C+1):
        R.append(round(float((targets == i).long().sum()/L), 2))
    return R


def permutations(list_tenseurs, r=np.inf, batch=False):
    """
    Génère une même permutation de tenseur dans une liste de tenseurs.

    Chaque tenseur de la liste se verra permutter de façon identique.
    r est le nombre de permuttation différente générer (attention, avec batch,
    on explose la puissance de calcul)
    Si batch est Vrai alors il peut y avoir des doublons
    """
    n = list_tenseurs[0].shape[-1]
    L = len(list_tenseurs)
    perm_list = list([] for k in range(L))
    already_perm = list((i, i) for i in range(n))
    if r >= n:
        r = n-1
    for _ in range(r):
        (i, j) = (0, 0)
        while (i, j) in already_perm:
            (i, j) = (np.random.randint(n), np.random.randint(n))
        already_perm.append((i, j))
        for tenseur in range(L):
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
                    a_col, b_col = new_tenseur[:, i].clone(),new_tenseur[:, j].clone()
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
