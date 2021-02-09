# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 18:52:28 2020

@author: Patrice CHANOL
"""

# Program that computes PI with MPI
# Execution : direct invocation in terminal or via python3

# System and maths libs
import torch
import torch.nn as nn
from datetime import datetime
import time
import numpy as np
import matplotlib.pyplot as plt

# NN Import
from GCNModel import GCN_Class, GCN_Reg
import utils


# %% Data

def data_processing(classification: bool = True,
                    equi: bool = True,
                    perm: bool = True,
                    cv: int = 0,
                    load: bool = False):
    """
    Choose between classification or regression for Data Processing.

    Parameters
    ----------
    classification : bool, optional
        True: Classification else Regression. The default is True.

    Returns
    -------
    train : torch.utils.data.dataloader.DataLoader
        Train dataset.
    test : torch.utils.data.dataloader.DataLoader
        Test dataset.
    in_features : int
        Number of in features of the model.
    out_features : int
        Number of out features of the model.

    """
    if classification:
        if cv > 0:
            out_features, dataset = utils.load_data(targets_file='data/labels.pt',
                                                cv=cv,
                                                batch_size=64,
                                                diversity=classification,
                                                equi=equi,
                                                perm=perm)
        else:
            out_features, dataset = utils.load_data(targets_file='data/labels.pt',
                                                batch_size=64,
                                                diversity=classification,
                                                equi=equi,
                                                perm=perm,
                                                load=load)
    else:
        if cv > 0:
            out_features, dataset = utils.load_data(x_file='data/init_pops.pt',
                                                    targets_file='data/last_pops.pt',
                                                    cv=cv,
                                                    batch_size=64,
                                                    diversity=classification,
                                                    equi=equi,
                                                    perm=perm, r=8)
        else:
            out_features, dataset = utils.load_data(x_file='data/init_pops.pt',
                                                    targets_file='data/last_pops.pt',
                                                    batch_size=64,
                                                    diversity=classification,
                                                    equi=equi,
                                                    perm=perm, r=8,
                                                    load=load)
    exemple = dataset[0][1][0]
    in_features = exemple[0].shape[-1]

    return dataset, in_features, out_features


# %% Model

def model_prep(in_features: int, out_features: int,
               classification: bool = True):
    """
    Model preparation.

    Parameters
    ----------
    in_features : int
        Number of in features of the model.
    out_features : int
        Number of out features of the model.
    classification : bool, optional
        True: Classification else Regression. The default is True.

    Returns
    -------
    model : GCN_Class or GCN_Reg
        GCN Model.
    criterion : torch.nn.modules.loss.NLLLoss or torch.nn.modules.loss.MSELoss
        Loss function.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if classification:
        model = GCN_Class(in_features, 64, out_features).to(device)
        criterion = nn.NLLLoss()
    else:
        model = GCN_Reg(in_features, 64, out_features).to(device)
        criterion = nn.MSELoss()

    print(model)
    return model, criterion


# %% Trainning and/or Testing

def trainning(model: GCN_Class or GCN_Reg,
              criterion: torch.nn.modules.loss.NLLLoss or torch.nn.modules.loss.MSELoss,
              train: torch.utils.data.dataloader.DataLoader,
              test: torch.utils.data.dataloader.DataLoader,
              classification: bool = True,
              learning_rate: float = 0.01,
              weight_decay: float = 0.0001,
              num_epochs: float = 5):
    """
    Train a model.

    Parameters
    ----------
    model : GCN_Class or GCN_Reg
        Model to train.
    criterion : torch.nn.modules.loss.NLLLoss or torch.nn.modules.loss.MSELoss
        Loss function.
    train : torch.utils.data.dataloader.DataLoader
        Train dataset.
    test : torch.utils.data.dataloader.DataLoader
        Test dataset.
    classification : bool, optional
        True: Classification else Regression. The default is True.

    Returns
    -------
    model : GCN_Class or GCN_Reg
        Model trained.
    test_loss_list : list
        Test loss.
    train_loss_list : list
        Train loss.
    accuracy_list_sim : list
        Test Simulation accuracy.
    accuracy_list_1 : list
        Test Classification accuracy.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=learning_rate,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.50)

    dateTimeObj_start = datetime.now()
    print('Début Entrainement :',
          dateTimeObj_start.hour, 'H', dateTimeObj_start.minute, 'min')
    test_loss_list = []
    train_loss_list = []
    accuracy_list_sim = []
    accuracy_list_1 = []
    n_batches = len(train)

    # On va entrainer le modèle num_epochs fois
    for epoch in range(1, num_epochs + 1):

        # Temps epoch
        epoch_start_time = time.time()
        dateTimeObj = datetime.now()
        print('Début epoch', epoch, ':',
              dateTimeObj.hour, 'H', dateTimeObj.minute, 'min')
        # Modèle en mode entrainement
        model.train()
        # Pourcentage du Dataset réaliser
        pourcentage = 0.
        flag = True
        # Loss du batch en cours
        test_loss_batch = []
        train_loss_batch = []

        # Temps pour réaliser 10%
        start_time = time.time()

        for batch, (x, adj_mat, target) in enumerate(train):

            # adj_mat = utils.adj_normalize(adj_mat.to_sparse())
            if not classification:
                x = utils.normalize(x)
                target = utils.normalize(target)
            # Initializing a gradient as 0
            optimizer.zero_grad()
            # Forward pass
            output = model.forward(x.to(device), adj_mat.to_sparse().to(device))
            loss = criterion(output, target.to(device))
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
                if classification:
                    acc, cor_sim, lcor_sim, cor_1, lcor_1 = 0, 0, 0, 0, 0
                    confmat = np.zeros((model.classes, model.classes))
                with torch.no_grad():
                    for (x_t, adj_mat_t, target_t) in test:
                        # adj_mat_t = utils.adj_normalize(adj_mat_t.to_sparse())
                        if not classification:
                            x_t = utils.normalize(x_t)
                            target_t = utils.normalize(target_t)
                        output_t = model.forward(x_t.to(device),
                                                 adj_mat_t.to_sparse().to(device))
                        loss_t = criterion(output_t, target_t.to(device))
                        test_loss_batch.append(loss_t.item())
                        if classification:
                            if pourcentage < 0.2 and flag:
                                acc = utils.accuracy(output_t, target_t.to(device), p=True)
                                flag = False
                            else:
                                acc = utils.accuracy(output_t, target_t.to(device), p=False)
                            cor_sim += acc[0][0]
                            lcor_sim += acc[0][1]
                            cor_1 += acc[1][0]
                            lcor_1 += acc[1][1]
                            confmat += acc[2]
                test_loss = np.mean(test_loss_batch)
                test_loss_list.append(test_loss)
                if classification:
                    accuracy_list_sim.append(cor_sim/lcor_sim)
                    accuracy_list_1.append(cor_1/lcor_1)
                print('-'*10)
                if classification:
                    print("Pourcentage: {}%, Epoch: {}, Temps : {}s".format(
                        round(100*pourcentage), epoch, round(T)))
                    print("Test Loss : {:.6f}, Accuracy Simulation: {:.3f}, Accuracy Classification: {:.3f}".format(
                        test_loss, cor_sim/lcor_sim, cor_1/lcor_1))
                    print('Confusion Matrix :\n', confmat,
                          '\n Labels :', np.arange(model.classes))
                else:
                    print("Pourcentage: {}%, Epoch: {}, Temps : {}s".format(
                        round(100*pourcentage), epoch, round(T)))
                    print("Test Loss : {:.6f}".format(test_loss))
                print('-'*10)

                pourcentage += 0.2
                flag = True
                start_time = time.time()

        print('Fin epoch : {}, Temps de l\'epoch : {}s'.format(
            epoch, round(time.time() - epoch_start_time)))
        train_loss_list.append(np.mean(train_loss_batch))
        scheduler.step()

    dateTimeObj_end = datetime.now()
    print('Fin Entrainement :',
          dateTimeObj_end.hour, 'H', dateTimeObj_end.minute,
          'min, Durée :', dateTimeObj_end-dateTimeObj_start)
    model.save()

    return model, test_loss_list, train_loss_list, accuracy_list_sim, accuracy_list_1


def cross_validation(model: GCN_Class or GCN_Reg,
              criterion: torch.nn.modules.loss.NLLLoss or torch.nn.modules.loss.MSELoss,
              dataset: list,
              cv: int,
              classification: bool = True,
              learning_rate: float = 0.01,
              weight_decay: float = 0.0001,
              num_epochs: float = 5):
    """
    Train a model with cross validation.

    Parameters
    ----------
    model : GCN_Class or GCN_Reg
        Model to train.
    criterion : torch.nn.modules.loss.NLLLoss or torch.nn.modules.loss.MSELoss
        Loss function.
    train : torch.utils.data.dataloader.DataLoader
        Train dataset.
    test : torch.utils.data.dataloader.DataLoader
        Test dataset.
    cv : int,
        Number of fold
    classification : bool, optional
        True: Classification else Regression. The default is True.

    Returns
    -------
    model : GCN_Class or GCN_Reg
        Model trained.
    test_loss_list : list
        Test loss.
    train_loss_list : list
        Train loss.
    accuracy_list_sim : list
        Test Simulation accuracy.
    accuracy_list_1 : list
        Test Classification accuracy.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dateTimeObj_start = datetime.now()
    print('Début Entrainement :',
          dateTimeObj_start.hour, 'H', dateTimeObj_start.minute, 'min')
    test_loss_list_fold = []
    train_loss_list_fold = []
    accuracy_list_sim_fold = []
    accuracy_list_1_fold = []
    name = model.name_model
    # On va faire les k folds
    for k in range(cv):
        [train, test] = dataset[k]
        model_k = model
        model_k.apply(utils.weights_init)
        model_k.name_model = name + ' ' + str(k + 1)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=learning_rate,
                                    weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.50)
        n_batches = len(train)
        test_loss_list = []
        train_loss_list = []
        accuracy_list_sim = []
        accuracy_list_1 = []
        # On va entrainer le modèle num_epochs fois
        for epoch in range(1, num_epochs + 1):
    
            # Temps epoch
            epoch_start_time = time.time()
            dateTimeObj = datetime.now()
            print('Fold', k + 1, 'Début epoch', epoch, ':',
                  dateTimeObj.hour, 'H', dateTimeObj.minute, 'min')
            # Modèle en mode entrainement
            model_k.train()
            # Pourcentage du Dataset réaliser
            pourcentage = 0.
            flag = True
            # Loss du batch en cours
            test_loss_batch = []
            train_loss_batch = []
    
            # Temps pour réaliser 10%
            start_time = time.time()
    
            for batch, (x, adj_mat, target) in enumerate(train):
    
                # adj_mat = utils.adj_normalize(adj_mat.to_sparse())
                if not classification:
                    x = utils.normalize(x)
                    target = utils.normalize(target)
                # Initializing a gradient as 0
                optimizer.zero_grad()
                # Forward pass
                output = model_k.forward(x.to(device), adj_mat.to_sparse().to(device))
                loss = criterion(output, target.to(device))
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
                    model_k.eval()
                    if classification:
                        acc, cor_sim, lcor_sim, cor_1, lcor_1 = 0, 0, 0, 0, 0
                        confmat = np.zeros((model.classes, model.classes))
                    with torch.no_grad():
                        for (x_t, adj_mat_t, target_t) in test:
                            # adj_mat_t = utils.adj_normalize(adj_mat_t.to_sparse())
                            if not classification:
                                x_t = utils.normalize(x_t)
                                target_t = utils.normalize(target_t)
                            output_t = model_k.forward(x_t.to(device),
                                                     adj_mat_t.to_sparse().to(device))
                            loss_t = criterion(output_t, target_t.to(device))
                            test_loss_batch.append(loss_t.item())
                            if classification:
                                if pourcentage < 0.2 and flag:
                                    acc = utils.accuracy(output_t, target_t.to(device), p=True)
                                    flag = False
                                else:
                                    acc = utils.accuracy(output_t, target_t.to(device), p=False)
                                cor_sim += acc[0][0]
                                lcor_sim += acc[0][1]
                                cor_1 += acc[1][0]
                                lcor_1 += acc[1][1]
                                confmat += acc[2]
                    test_loss = np.mean(test_loss_batch)
                    test_loss_list.append(test_loss)
                    if classification:
                        accuracy_list_sim.append(cor_sim/lcor_sim)
                        accuracy_list_1.append(cor_1/lcor_1)
                    print('-'*10)
                    if classification:
                        print("Fold: {}, Epoch: {}, Pourcentage: {}%,  Temps : {}s".format(
                            k + 1, epoch, round(100*pourcentage), round(T)))
                        print("Test Loss : {:.6f}, Accuracy Simulation: {:.3f}, Accuracy Classification: {:.3f}".format(
                            test_loss, cor_sim/lcor_sim, cor_1/lcor_1))
                        print('Confusion Matrix :\n', confmat,
                              '\n Labels :', np.arange(model.classes))
                    else:
                        print("Fold: {}, Epoch: {}, Pourcentage: {}%, Temps : {}s".format(
                            k +1, epoch, round(100*pourcentage), round(T)))
                        print("Test Loss : {:.6f}".format(test_loss))
                    print('-'*10)
    
                    pourcentage += 0.2
                    flag = True
                    start_time = time.time()
    
            print('Fold: {}, Fin epoch : {}, Temps de l\'epoch : {}s'.format(
                k + 1, epoch, round(time.time() - epoch_start_time)))
            train_loss_list.append(np.mean(train_loss_batch))
            scheduler.step()
        graph(model_k, test_loss_list, train_loss_list, accuracy_list_sim,
              accuracy_list_1, True)
        test_loss_list_fold.append(test_loss_list)
        train_loss_list_fold.append(train_loss_list)
        accuracy_list_sim_fold.append(accuracy_list_sim)
        accuracy_list_1_fold.append(accuracy_list_1)
        model_k.save()

    dateTimeObj_end = datetime.now()
    print('Fin Entrainement :',
          dateTimeObj_end.hour, 'H', dateTimeObj_end.minute,
          'min, Durée :', dateTimeObj_end-dateTimeObj_start)

    return model, test_loss_list_fold, train_loss_list_fold, accuracy_list_sim_fold, accuracy_list_1_fold


def testing(model: GCN_Class or GCN_Reg,
            criterion: torch.nn.modules.loss.NLLLoss or torch.nn.modules.loss.MSELoss,
            test: torch.utils.data.dataloader.DataLoader,
            classification: bool = True):
    """
    Test a model.

    Parameters
    ----------
    model : GCN_Class or GCN_Reg
        Model to test.
    criterion : torch.nn.modules.loss.NLLLoss or torch.nn.modules.loss.MSELoss
        Loss function.
    test : torch.utils.data.dataloader.DataLoader
        Test dataset.
    remain : torch.utils.data.dataloader.DataLoader
        Other part of dataset.
    classification : bool, optional
        True: Classification else Regression. The default is True.
    entireDataset : bool, optional
        True: Test entire dataset. The default is True.

    Returns
    -------
    model : GCN_Class or GCN_Reg
        Model tested.
    test_loss_list : list
        Test loss.
    accuracy_list_sim : list
        Test Simulation accuracy.
    accuracy_list_1 : list
        Test Classification accuracy.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loss_list = []
    accuracy_list_sim = []
    accuracy_list_1 = []
    acc, cor_sim, lcor_sim, cor_1, lcor_1 = 0, 0, 0, 0, 0
    confmat =  np.zeros((model.classes, model.classes))

    try:
        model.load()
    except Exception as e:
        raise Exception(e, 'No model to load')
    model.eval()
    with torch.no_grad():
        for (x_t, adj_mat_t, target_t) in test:
            # adj_mat_t = utils.adj_normalize(adj_mat_t.to_sparse())
            if not classification:
                x_t = utils.normalize(x_t)
                target_t = utils.normalize(target_t)
            output_t = model.forward(x_t.to(device),
                                     adj_mat_t.to_sparse().to(device))
            loss_t = criterion(output_t, target_t.to(device))
            test_loss_list.append(loss_t.item())
            if classification:
                acc = utils.accuracy(output_t, target_t.to(device))
                cor_sim += acc[0][0]
                lcor_sim += acc[0][1]
                cor_1 += acc[1][0]
                lcor_1 += acc[1][1]
                confmat += acc[2]
                accuracy_list_sim.append(cor_sim/lcor_sim)
                accuracy_list_1.append(cor_1/lcor_1)
    return model, test_loss_list, accuracy_list_sim, accuracy_list_1, confmat


# %% Plot

def graph(model: GCN_Class or GCN_Reg,
          test_loss_list: list, train_loss_list: list,
          accuracy_list_sim: list, accuracy_list_1: list,
          isTrained: bool = True):
    """
    Plot.

    Parameters
    ----------
    model : GCN_Class or GCN_Reg
        Model trained.
    test_loss_list : list
        Test loss.
    train_loss_list : list
        Train loss.
    accuracy_list : list
        Test accuracy.

    Returns
    -------
    None.

    """
    plt.figure(0)
    plt.plot(test_loss_list)
    plt.title(model.name_model + ': Test Loss')
    plt.xlabel('5 * Number of epochs')
    plt.ylabel('Loss')
    plt.show()

    if isTrained:
        plt.figure(1)
        plt.plot(train_loss_list)
        plt.title(model.name_model + ': Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

    if classification:
        plt.figure(2)
        plt.plot(accuracy_list_sim)
        plt.title(model.name_model + ': Simulation Accuracy')
        plt.ylim((0, 1))
        plt.xlabel('5 * Number of epochs')
        plt.ylabel('Ratio')
        plt.show()
        plt.figure(3)
        plt.plot(accuracy_list_1)
        plt.title(model.name_model + ': Classification Accuracy')
        plt.ylim((0, 1))
        plt.xlabel('5 * Number of epochs')
        plt.ylabel('Ratio')
        plt.show()


# %%

if __name__ == "__main__":

    classification = input("Do you want a regression model ? Y/[N] ") != "Y"
    load = input("Do you want to load an existing model ? Y/[N] ") == "Y"
    crossval = False
    if not load:
        crossval = input("Do you want k fold cross-validation ? Y/[N] ") == "Y"
        if crossval:
            try:
                cv = int(input('k fold value: [5] '))
            except Exception:
                cv = 5
        try:
            num_epochs = int(input('How many epochs of training ? [10] '))
        except Exception:
            num_epochs = 10
    equi = input("Do you want to readjust the dataset ? [Y]/N ") != "N"
    perm = input("Do you want to create permutation of the dataset ? [Y]/N ") != "N"

    learning_rate = 0.01
    weight_decay = 0.0001

    if crossval:
        dataset, in_features, out_features = data_processing(
            classification=classification, equi=equi, perm=perm, cv=cv)
        model, criterion = model_prep(in_features=in_features,
                                      out_features=out_features,
                                      classification=classification)
        model, test_loss_list_fold, train_loss_list_fold, accuracy_list_sim_fold, accuracy_list_1_fold = cross_validation(
            model=model, criterion=criterion, dataset=dataset, cv=cv,
            classification=classification, learning_rate=learning_rate,
            weight_decay=weight_decay, num_epochs=num_epochs)
        isTrained = True

    else:
        dataset, in_features, out_features = data_processing(
            classification=classification, equi=equi, perm=perm, load=load)
        train = dataset[0][0]
        test = dataset[0][1]
        model, criterion = model_prep(in_features=in_features,
                                      out_features=out_features,
                                      classification=classification)
    
        if load:
            try:
                model, test_loss_list, accuracy_list_sim, accuracy_list_1, confmat = testing(
                    model=model, criterion=criterion, test=test,
                    classification=classification)
                print('Loss :', round(test_loss_list[0], 6),
                      'Simulation Accuracy : {:.2f}%'.format(accuracy_list_sim[0]*100),
                      'Classification Accuracy : {:.2f}%'.format(accuracy_list_1[0]*100))
                print('Confusion Matrix :\n', confmat,
                      '\n Labels :', np.arange(model.classes))
            except Exception as e:
                print(e)
        else:
            model, test_loss_list, train_loss_list, accuracy_list_sim, accuracy_list_1 = trainning(
                model=model, criterion=criterion, train=train,
                test=test, classification=classification,
                learning_rate=learning_rate, weight_decay=weight_decay,
                num_epochs=num_epochs)
            isTrained = True
            graph(model=model, test_loss_list=test_loss_list,
                      train_loss_list=train_loss_list,
                      accuracy_list_sim=accuracy_list_sim,
                      accuracy_list_1=accuracy_list_1, isTrained=isTrained)
