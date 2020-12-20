# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 18:52:28 2020

@author: Patrice CHANOL
"""

import torch
import torch.nn as nn
from datetime import datetime
import time
import numpy as np
import matplotlib.pyplot as plt

from GCNModel import GCN_Class, GCN_Reg
import utils

# %% Data
classification = True

if classification:
    train, test = utils.load_data(adj_mats_file='data/adj_mats_old.pt', targets_file='data/labels_old.pt')
    out_features = 3
else:
    train, test = utils.load_data(x_file='data/init_pops.pt', targets_file='data/last_pops.pt')
    out_features = 1

exemple = next(iter(train))
in_features = exemple[0][0].shape[-1]

# %% Model

if classification:
    model = GCN_Class(in_features, 64, out_features)
    criterion = nn.NLLLoss()
else:
    model = GCN_Reg(in_features, 64, out_features)
    criterion = nn.MSELoss()

print(model)
learning_rate = 0.001
weight_decay = 0.0001
num_epochs = 20
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.50)

dateTimeObj = datetime.now()
print('Début Entrainement : ', dateTimeObj.hour, 'H', dateTimeObj.minute)
test_loss_list = []
train_loss_list = []
n_batches = len(train)
# On va entrainer le modèle num_epochs fois
for epoch in range(1, num_epochs + 1):

    # Temps epoch
    epoch_start_time = time.time()
    dateTimeObj = datetime.now()
    print('Début epoch', epoch, ':', dateTimeObj.hour, 'H', dateTimeObj.minute)
    # Modèle en mode entrainement
    model.train()
    # Pourcentage du Dataset réaliser
    pourcentage = 0.
    # Loss du batch en cours
    test_loss_batch = []
    train_loss_batch = []

    # Temps pour réaliser 10%
    start_time = time.time()

    for batch, ((x, adj_mat), label) in enumerate(train):

        label = label.long()
        adj_mat = utils.adj_normalize(adj_mat)
        # Initializing a gradient as 0 so there is no mixing of gradient among the batches
        optimizer.zero_grad()
        # Forward pass
        output = model.forward(x, adj_mat)
        loss = criterion(output, label)
        # Propagating the error backward
        loss.backward()

        # Optimizing the parameters
        optimizer.step()
        train_loss_batch.append(loss.item())

        # Pourcentage réel réaliser
        count_pourcentage = batch / n_batches
        # Si on a réalisé 10% nouveau du Dataset, on test
        if count_pourcentage >= pourcentage:
            # Temps des 10%
            T = time.time() - start_time
            # Evaluation du modèel
            model.eval()
            acc, cor, lcor = 0, 0, 0
            with torch.no_grad():
                for ((x_t, adj_mat_t), label_t) in test:
                    label_t = label_t.long()
                    adj_mat_t = utils.adj_normalize(adj_mat_t)
                    output_t = model.forward(x_t, adj_mat_t)
                    loss_t = criterion(output_t, label_t)
                    test_loss_batch.append(loss_t.item())
                    acc = utils.accuracy(output_t, label_t)
                    cor += acc[0]
                    lcor += acc[1]
            test_loss = np.mean(test_loss_batch)
            test_loss_list.append(test_loss)

            print('-'*10)
            print("Pourcentage: {}%, Test Loss : {}, Accuracy: {}, Epoch: {}, Temps : {}s".format(round(100*pourcentage), test_loss, cor/lcor, epoch, round(T)))
            print('-'*10)

            pourcentage += 0.2
            start_time = time.time()

    print('Fin epoch : {}, Temps de l\'epoch : {}s'.format(epoch, round(time.time() - epoch_start_time)))
    train_loss_list.append(np.mean(train_loss_batch)) 
    scheduler.step()

model.save()

plt.figure(0)
plt.plot(test_loss_list)
plt.title(model.name_model +': Test Loss')
plt.show()

plt.figure(1)
plt.plot(train_loss_list)
plt.title(model.name_model +': Train Loss')
plt.show()
