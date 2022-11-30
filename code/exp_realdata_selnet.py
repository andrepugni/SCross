#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for experiments using SelNet
"""
import pandas as pd
import numpy as np
from attributes import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from model import *
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from time import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_datasets_selnet(file, atts_dict, data_fold='data/', hold=False):
    """
    

    Parameters
    ----------
    file : str
        the name of the dataset.
    atts_dict : dict
        A dictionary containing all the features names for a specific dataset.
    data_fold : str, optional
        The name of folder containing the dataset. The default is 'data/'.

    Returns
    -------
    X_train : pd.DataFrame
        The training set features.
    X_test : pd.DataFrame
        The test set features.
    X_hold : pd.DataFrame
        The validation set features.
    y_train : pd.Series
        The training set target variable.
    y_test : pd.Series
        The test set target variable.
    y_hold : pd.Series
        The validation set target variable.

    """
    train = pd.read_pickle('{}{}_train.pkl'.format(data_fold,file))
    test = pd.read_pickle('{}{}_test.pkl'.format(data_fold,file))
    if (file=='adultNMNOH')|(file=='adultNM'):
        train.age = train.age.astype(float)
        test.age = test.age.astype(float)
        for col in ['education-num', 'capital-gain', 'capital-loss', 'hours-per-week','age']:
            train[col] = train[col].astype(float)
            test[col] = test[col].astype(float)
    if file =='CSDS2':
            col = 'TARGET'
            train[col] = train[col].astype(np.int64)
            test[col] = test[col].astype(np.int64)
        
    if file =='CSDS3':
            col = 'TARGET'
            train[col] = train[col].astype(np.int64)
            test[col] = test[col].astype(np.int64)
            
    if file =='GiveMe':
            cols = ['age', 'NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfOpenCreditLinesAndLoans',
                     'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
                     'RevolvingLines']
            for col in cols:
                train[col] = train[col].astype(float)
                test[col] = test[col].astype(float)
                
            train['TARGET'] = train['TARGET'].astype(np.int64)
            test['TARGET'] = test['TARGET'].astype(np.int64)
            
            
    if file=='UCI_credit':
        train.AGE = train.AGE.astype(float) 
        test.AGE = test.AGE.astype(float) 
    train, hold = train_test_split(train, stratify=train['TARGET'], test_size=0.1)  
    X_train = train[atts_dict[file]]
    X_test = test[atts_dict[file]]
    X_hold = hold[atts_dict[file]]
    y_train = train['TARGET'] 
    y_test = test['TARGET']
    y_hold = hold['TARGET']
    return X_train, X_test, X_hold, y_train, y_test, y_hold

def cross_entropy_vectorized(y_true, y_pred):
    """
    

    Parameters
    ----------
    y_true : Pytorch Tensor
        The tensor with actual labaels.
    y_pred : Pytorch Tensor
        The predicted values from auxiliary head.

    Returns
    -------
    loss : Pytorch tensor
        The cross entropy loss for auxiliary head.

    """
    n_batch, n_class = y_pred.shape
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    if len(y_true.shape)==1:
        y_true = torch.unsqueeze(y_true, dim=-1)
    y_true = y_true.to(torch.int64)
    log_ypred = torch.log(torch.gather(y_pred,1,y_true)+0.00000001)
    loss = -(torch.sum(log_ypred))/n_batch
    return loss

def cross_entropy_selection_vectorized(y_true, hg, theta=.5, lamda=32, c=0.9):
    """
    

    Parameters
    ----------
    y_true : Pytorch Tensor
        The tensor with actual labels.
    hg : Pytorch Tensor
        The outputs of predictive head and selecticve head.
    theta : float, optional
        The threshold to make g(x)=1. The default is .5.
    lamda : float, optional
        Parameter to weigh the importance of constraint for coverage. The default is 32.
    c : float, optional
        The desired coverage. The default is 0.9.

    Returns
    -------
    loss : Pytorch Tensor
        The selective loss from Geifman et al. (2019).

    """
    n_batch, n_class = hg[:,:-1].shape
    if len(y_true.shape)==1:
        y_true = torch.unsqueeze(y_true, dim=-1)
    if c==1:
        selected = n_batch
    else:
        selected = torch.sum(hg[:,-1])+0.00000001
    selection = torch.unsqueeze(hg[:,-1],dim=-1)
    y_true = y_true.to(torch.int64)
    log_ypred = torch.log(torch.gather(hg[:,:-1],1,y_true)+0.00000001)*selection
    loss = -((torch.sum(log_ypred))/(selected))+lamda*(max(0,c-(selected/n_batch)))**2
    
    return loss
    

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


def traindata(device, model, epochs, optimizer, train_dl, valid_dl,
              patience=2, coverage=0.9, criterion = 'loss', lamda=32):
    """
    

    Parameters
    ----------
    device : torch.device
        The device over which training will be performed.
    model : torch.module
        The network architecture to train.
    epochs : int
        The number of epochs.
    optimizer : torch.optimizer
        The optimizer to perform network training.
    train_dl : torch.dataset
        The training dataset.
    valid_dl : torch.dataset
        The validation dataset.
    patience : int, optional
        The number of epochs without improvements 
        that will be considered before stopping. The default is 2.
    coverage : float, optional
        The desired coverage for selective loss. The default is 0.9.

    Returns
    -------
    model : torch.module
        The trained network.

    """
    # Early stopping
    early_loss = 100
    last_auc = 0.5
    trigger_times = 0
    model.to(device)
    running_loss = 0
    for epoch in range(1, epochs+1):
        model.train()
        with tqdm(train_dl, unit="batch") as tepoch:
            for i,batch in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")
                X_cont, X_cat, X_one, y = batch           
                X_cont, X_cat, X_one, y  = X_cont.to(device), X_cat.to(device), X_one.to(device), y.to(device)
                optimizer.zero_grad()
                hg, aux = model.forward(X_cont, X_cat, X_one)
                loss1 = 0.5*cross_entropy_selection_vectorized(y,hg, c=coverage, lamda=lamda)
                loss2 = 0.5*cross_entropy_vectorized(y,aux)
                loss = loss1+loss2
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 1000 == 10:
                    last_loss = running_loss / 1000 # loss per batch
    #                 print('  batch {} loss: {}'.format(i + 1, last_loss))
                    tb_x = epoch * len(train_dl) + i + 1
                    running_loss = 0.
                tepoch.set_postfix(loss=loss.item())
            # Early stopping
            current_loss, current_auc = validation(model, device, valid_dl)
            print("Current Epoch:", epoch, 'The Current Loss:', current_loss, 'Current AUC:', current_auc)
            if criterion == 'loss':
                if current_loss > early_loss:
                    trigger_times += 1
                    print("trigger times: {}".format(trigger_times))
                    if trigger_times >= patience:
                        print('Early stopping!\nStart to test process.')
                        return model
                else:
                    print('trigger times: 0')
                    trigger_times = 0  
            elif criterion =='auc':
                if current_auc < last_auc:
                    trigger_times += 1
                    print('Trigger Times:', trigger_times)

                    if trigger_times >= patience:
                        print('Early stopping!\nStart to test process.')
                        return model

                else:
                    print('trigger times: 0')
                    trigger_times = 0    
            if current_auc> last_auc:
                last_auc = current_auc
            if current_loss < early_loss:
                early_loss = current_loss

    return model


def train_epoch(device, model, optimizer, train_dl, valid_dl, epoch_num=1, coverage=0.9, lamda=32):
    """
    

    Parameters
    ----------
    device : torch.device
        The device over which training will be performed.
    model : torch.module
        The network architecture to train.
    epochs : int
        The number of epochs.
    optimizer : torch.optimizer
        The optimizer to perform network training.
    train_dl : torch.dataset
        The training dataset.
    valid_dl : torch.dataset
        The validation dataset.
    patience : int, optional
        The number of epochs without improvements 
        that will be considered before stopping. The default is 2.
    coverage : float, optional
        The desired coverage for selective loss. The default is 0.9.

    Returns
    -------
    model : torch.module
        The trained network.

    """
    # Early stopping
    model.to(device)
    model.train()
    running_loss = 0
    with tqdm(train_dl, unit="batch") as tepoch:
        for i,batch in enumerate(tepoch):
            tepoch.set_description("Epoch:{}".format(epoch_num))
            X_cont, X_cat, X_one, y = batch           
            X_cont, X_cat, X_one, y  = X_cont.to(device), X_cat.to(device), X_one.to(device), y.to(device)
            optimizer.zero_grad()
            hg, aux = model.forward(X_cont, X_cat, X_one)
            loss1 = 0.5*cross_entropy_selection_vectorized(y,hg, c=coverage, lamda=lamda)
            loss2 = 0.5*cross_entropy_vectorized(y,aux)
            loss = loss1+loss2
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 1000 == 10:
                last_loss = running_loss / 1000 # loss per batch
    #                 print('  batch {} loss: {}'.format(i + 1, last_loss))
                #tb_x = epoch * len(train_dl) + i + 1
                running_loss = 0.
            tepoch.set_postfix(loss=loss.item())
        scores_hold = []
        confs_hold = []
        y_hold = []
        model.eval()
        model.to(device)
        for x_cont,x_cat,x_one,y in valid_dl:
            x_cont, x_cat, x_one, y  = x_cont.to(device), x_cat.to(device), x_one.to(device), y.to(device)
            hg, aux = model(x_cont,x_cat, x_one)
            score = hg[:,1].detach().cpu().numpy().reshape(-1,1)
            conf = hg[:,-1].detach().cpu().numpy().reshape(-1,1)
            scores_hold.append(score)
            confs_hold.append(conf)
            y_hold.append(y.detach().cpu().numpy().reshape(-1,1))
        select_hold = np.vstack(confs_hold).flatten()
        y_hat_hold = np.vstack(scores_hold).flatten()
        y_hold = np.vstack(y_hold).flatten()
        select = np.where(select_hold>.5,1,0)
        preds = np.where(y_hat_hold>.5,1,0)
        accu = accuracy_score(y_hold[select>0],preds[select>0])
        print("\n validation accuracy: {}n".format(accu))
        # Early stopping


    
def train(device, model, optimizer, train_dl, 
          valid_dl, max_epochs, lamda=32,
          td=True, gamma=.5, coverage=0.9,
          epochs_lr = [24,49,74,99,124,149,174,199,224,249,274,299]):
    model.to(device)
    model.train()
    print(lamda)
    print(td)
    for epoch in range(max_epochs):
        if td:
            if epoch in epochs_lr:
                try:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= gamma
                        print("\n: lr now is: {}\n".format(param_group['lr']))
                except:
                    import pdb; pdb.set_trace()
        train_epoch(device, model, optimizer, train_dl, valid_dl, epoch_num=epoch,coverage=coverage, lamda=lamda)
    return model
            

def validation(model, device, valid_dl):
    """
    

    Parameters
    ----------
    model : torch.module
        The network that is used to predict.
    device : torch.device
        The device that is used to train the model.
    valid_dl : torch.dataset
        The validation dataset.

    Returns
    -------
    loss_total/len(valid_dl) float
        The validation loss.
    auc : 
        The validation AUC.

    """
    model.to(device)
    model.eval()
    loss_total = 0

    # Test validation data
    with torch.no_grad():
        y_true = []
        y_scores = []
        y_sel = []
        for i,batch in enumerate(valid_dl):
            X_cont, X_cat, X_one, y = batch
            X_cont, X_cat, X_one, y  = X_cont.to(device), X_cat.to(device), X_one.to(device), y.to(device)
            hg, aux = model.forward(X_cont, X_cat, X_one)
            loss1 = 0.5*cross_entropy_selection_vectorized(y,hg, c=1)
            loss2 = 0.5*cross_entropy_vectorized(y,aux)
            loss = loss1+loss2
            loss_total += loss.item()
            y_true.append(y.detach().cpu().numpy().reshape(-1,1))
            y_scores.append(hg[:,1].detach().cpu().numpy().reshape(-1,1))
            y_sel.append(hg[:,-1].detach().cpu().numpy().reshape(-1,1))
        y_true = np.vstack(y_true)
        y_scores = np.vstack(y_scores)
        y_sel = np.vstack(y_sel)
        y_sel = y_sel>.5
        if (np.sum(y_true[y_sel])==0)|(np.sum(y_true[y_sel])==y_true[y_sel].shape[0]):
            auc=0
        else:
            auc = roc_auc_score(y_true[y_sel], y_scores[y_sel])
    return loss_total / len(valid_dl), auc

def get_predictions(model, device, valid_dl, test_dl, q):
    """
    

    Parameters
    ----------
    model : torch.module
        The network that is used to predict.
    device : torch.device
        The device that is used to train the model.
    valid_dl : torch.dataset
        The validation dataset.
    test_dl : torch.dataset
        The test dataset.
    q : float
        The desired quantile.

    Returns
    -------
    calibrated_theta : float
        A theta to determine g=1 that is calibrated over holdout set.
    y_hat : np.array
        The values of scores over test set.
    select : np.array
        The values of selection function over test set.

    """
    scores = []
    confs = []
    scores_hold = []
    confs_hold = []
    model.eval()
    model.to(device)
    for x_cont,x_cat,x_one,y in valid_dl:
        x_cont, x_cat, x_one, y  = x_cont.to(device), x_cat.to(device), x_one.to(device), y.to(device)
        hg, aux = model(x_cont,x_cat, x_one)
        score = hg[:,1].detach().cpu().numpy().reshape(-1,1)
        conf = hg[:,-1].detach().cpu().numpy().reshape(-1,1)
        scores_hold.append(score)
        confs_hold.append(conf)
    select_hold = np.vstack(confs_hold).flatten()
    y_hat_hold = np.vstack(scores_hold).flatten()
    calibrated_theta = np.quantile(select_hold, 1-q,interpolation='nearest')
    for x_cont, x_cat,x_one,y in test_dl:
        x_cont, x_cat, x_one, y  = x_cont.to(device), x_cat.to(device), x_one.to(device), y.to(device)
        hg, aux = model(x_cont, x_cat, x_one)
        score = hg[:,1].detach().cpu().numpy().reshape(-1,1)
        conf = hg[:,-1].detach().cpu().numpy().reshape(-1,1)
        scores.append(score)
        confs.append(conf)

    select = np.vstack(confs).flatten()
    y_hat = np.vstack(scores).flatten()
    return calibrated_theta, y_hat, select 




def experiment(file_name, file_model_params='default', meta='selnet', classifier_string = 'resnet', 
               coverages = [0.99, 0.95, 0.9, 0.85, 0.8, 0.75], boot_iter=1000, max_epochs=300, lamda=32, td=True):
    """
    

    Parameters
    ----------
    file_name : str
        The dataset name.
    file_model_params : str, optional
       The name of the file containing the parameters of the model.
       If 'default', the network is isntantiated with default parameters.
       The default is 'default'.
    meta : str, optional
        Used for uniformity with AUCross. The default is 'selnet'.
    classifier_string : str, optional
        The name of the base structure of SelNet. The default is 'resnet'.
    coverages : list, optional
        A list of desired coverages. The default is [0.99, 0.95, 0.9, 0.85, 0.8, 0.75].
    boot_iter : int, optional
        The number of bootstrap iterations over test set. The default is 1000.
    max_epochs : The number of maximum epochs for training, optional
        DESCRIPTION. The default is 300.

    Returns
    -------
    results : pd.DataFrame
        A dataframe containing results for SelNet.

    """
    if file_name in ['adultNM', 'adultNMNOH', 'UCI_credit']:
        n_batch = 32
    elif file_name in ['lendingNOH']:
        n_batch = 512
    else:
        n_batch = 128
    res_cols = ['dataset', 'desired_coverage', 'coverage','accuracy', 'auc', 'perc_pos','meta', 'boot_iter', 'model', 'time_to_fit']
    results = pd.DataFrame(columns = res_cols)
    X_train, X_test, X_hold, y_train, y_test, y_hold = read_datasets_selnet(file, atts_dict)
    print(X_train.shape)
    print(X_train.info())
    emb_cols = [col for col in atts_dict[file] if X_train[col].dtype!=float]
    not_emb_cols = [col for col in atts_dict[file] if (X_train[col].dtype==float)]
    embedded_cols = {col: len(X_train[col].unique()) for col in emb_cols if len(X_train[col].unique()) > 2}
    embedded_col_names = list(embedded_cols.keys())
    one_hot_cols = {col: len(X_train[col].unique()) for col in emb_cols if len(X_train[col].unique()) == 2}
    one_hot_col_names = list(one_hot_cols.keys())
    embedding_sizes = [(n_categories, min(50, (n_categories+1)//2)) for _,n_categories in embedded_cols.items()]
    print(embedding_sizes)
    not_embedded_cols = [col for col in atts_dict[file] if col not in embedded_col_names]
    if file_name in ['adultNM', 'adultNMNOH', 'lending', 'lendingNMNOH', 'UCI_credit', 'GiveMe']:
        print("scaling data")
        scaler = StandardScaler()
        scaler.fit(X_train[not_emb_cols])
        X_train.loc[:,not_emb_cols] = scaler.transform(X_train[not_emb_cols])
        X_test.loc[:,not_emb_cols] = scaler.transform(X_test[not_emb_cols])
        X_hold.loc[:,not_emb_cols] = scaler.transform(X_hold[not_emb_cols])
    train_ds = ToTorchDataset(X_train, y_train, embedded_col_names, one_hot_col_names)
    train_dl = DataLoader(train_ds, n_batch, shuffle=False)
    #here we create the torch tensors for holdout
    valid_ds = ToTorchDataset(X_hold, y_hold, embedded_col_names, one_hot_col_names)
    valid_dl = DataLoader(valid_ds, n_batch, shuffle=False)
    #here we create the torch tensors for test set
    test_ds = ToTorchDataset(X_test, y_test, embedded_col_names, one_hot_col_names)
    test_dl = DataLoader(test_ds, n_batch, shuffle=False)
    if file_model_params!='default':
        model_params = pd.read_csv(file_model_params)
        dict_params = model_params[(model_params['model']=='resnet') & (model_params['file']==file)].iloc[0].to_dict()
        dict_params.pop('model')
        dict_params.pop('file')
        dict_params.pop('Unnamed: 0')
        dict_params.pop('max_epochs')
        grid_params = dict(output_dim = 2, n_cont=len(not_emb_cols), n_one_hot=len(one_hot_col_names), embedding_sizes=embedding_sizes,
                           n_blocks=dict_params['n_blocks'], d_main = dict_params['d_main'], d_hidden=(dict_params['d_hidden']),
                           dropout_first=dict_params['dropout_first'], dropout_second = dict_params['dropout_second'], coverage=.99 )
    else:
        grid_params = dict(output_dim = 2, n_cont=len(not_emb_cols),
                           n_one_hot=len(one_hot_col_names), embedding_sizes=embedding_sizes, coverage=.99 )
    for cov in tqdm(coverages):
        if classifier_string=='resnet':
            model_structure = SelectiveResNetOne
        if classifier_string=='linnet':
            model_structure = SelectiveLinearNetOne
        grid_params['coverage'] = cov
        if classifier_string=='linnet':
            model = SelectiveLinearNetOne(**grid_params)
        elif classifier_string=='linnet2':
            model = SelectiveLinearNetTwo(**grid_params)
            
        else:
            if file_name in ['CSDS1', 'lending', 'GiveMe']:
                model = SelectiveResNetOneV2(**grid_params)
            else:
                model = SelectiveResNetOne(**grid_params)
        print(model.coverage)
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        print("td is {}".format(td))
        print("lamda is {}".format(lamda))
        if td == True:
            opt = torch.optim.SGD(model.parameters(),0.1, weight_decay = 5e-4, momentum=0.9, nesterov=True)
        else:
            opt = torch.optim.SGD(model.parameters(),0.0001, momentum=0.9, nesterov=True)
        start_time = time()
        model = train(device, model, opt, train_dl, valid_dl, max_epochs, td=td, lamda=lamda)
        end_time = time()
        path_model = 'models/{}_{}_{}_{}_{}_{}.pt'.format(classifier_string, file_name, cov, lamda, td, max_epochs)
        torch.save(model.state_dict(), path_model)
        time_to_fit = (end_time - start_time)
        model.eval()
        theta, scores, select = get_predictions(model, device, valid_dl, test_dl, cov)
        bands = (select>theta)
        preds = np.where(scores>=.5,1,0)
        yy = pd.DataFrame(np.c_[y_test,scores, preds, select], columns = ['true', 'scores','preds', 'select'])
        if boot_iter ==0:
            auc = roc_auc_score(y_test[select>theta], scores[select>theta])
            acc = accuracy_score(y_test[select>theta], preds[select>theta])
            perc_pos = np.sum(y_test[select>theta])/len(y_test[select>theta])
            coverage = len(y_test[select>theta])/len(y_test)
            res_cols = ['dataset', 'desired_coverage', 'coverage','accuracy', 'auc', 'perc_pos']
            tmp = pd.DataFrame([[file_name, cov, coverage, acc, auc, perc_pos]], columns =res_cols )
            tmp['meta'] = meta
            tmp['model'] = "{}_{}".format(meta, classifier_string)
            tmp['time_to_fit'] = time_to_fit
            results = pd.concat([results, tmp], axis=0)
            results.to_csv("res_{}_selnet_{}.csv".format(file_name, boot_iter))
        else:
            for b in range(boot_iter+1):
                if b==0:
                    db = yy.copy()
                else:
                    db = yy.sample(len(y_test), random_state=b, replace=True)
                db = db.reset_index()
                if (np.sum(db['select']>theta)==0):
                    auc = 0
                    acc = 0
                    perc_pos = 0
                    coverage = 0
                elif (np.sum(db['true'][db['select']>theta])==0)|(np.sum(db['true'][db['select']>theta])==len(y_test)):
                    auc = 0
                    acc = 0
                    perc_pos = 0
                    coverage = 0
                else:
                    auc = roc_auc_score(db['true'][db['select']>theta], db['scores'][db['select']>theta])
                    acc = accuracy_score(db['true'][db['select']>theta], db['preds'][db['select']>theta])
                    perc_pos = np.sum(db['true'][db['select']>theta])/len(db['true'][db['select']>theta])
                    coverage = len(db['true'][db['select']>theta])/len(db['true'])
                res_cols = ['dataset', 'desired_coverage', 'coverage','accuracy', 'auc', 'perc_pos']
                tmp = pd.DataFrame([[file_name, cov, coverage, acc, auc, perc_pos]], columns =res_cols )
                tmp['meta'] = meta
                tmp['boot_iter'] = b
                tmp['model'] = "{}_{}".format(meta, classifier_string)
                tmp['time_to_fit'] = time_to_fit
                results = pd.concat([results, tmp], axis=0)
                results.to_csv("results_{}_selnet_{}.csv".format(file_name, boot_iter))
            
    return results

if __name__=='__main__':
    
    np.random.seed(42)
    torch.manual_seed(14)
    res = pd.DataFrame()
    atts_dict = {'lending': atts_lending, 'lendingNOH': atts_lendingNOH, 'CSDS1': atts_CSDS1, 'CSDS2':atts_CSDS2, 'CSDS3':atts_CSDS3, 'GiveMe':atts_giveme, 
             'adultNM':atts_adult, 'adultNMNOH':atts_adultNOH, 'UCI_credit' : atts_credit}
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--model', type=str, required=False, default='resnet')
    parser.add_argument('--max_epochs', type=int, required=False, default=300)
    parser.add_argument('--file', type=list, nargs="+", required=False, default=['UCI_credit' ,'lendingNOH','GiveMe','CSDS1', 'CSDS2', 'CSDS3','adultNMNOH',])
    parser.add_argument('--boot_iter', type=int, default=1000)
    parser.add_argument('--params', type=str, required=False, default='default')
    parser.add_argument('--lamda', type=int, default=32)
    parser.add_argument('--td', type=str, default='True')
    
    # Parse the argument
    args = parser.parse_args()
    classifier_string = args.model
    print(classifier_string)
    max_epochs = args.max_epochs
    filelist = args.file
    params = args.params
    boot_iter = args.boot_iter
    lamda = args.lamda
    td = eval(args.td)
    print(td)
    for file in tqdm(filelist):
        tmp = experiment(file, classifier_string = classifier_string, boot_iter=boot_iter, max_epochs=max_epochs, lamda=lamda, td=td)
        tmp.to_csv('best_model_{}_{}_{}epochs_lamda{}_td{}.csv'.format(classifier_string, file, max_epochs, lamda, td))
        res = pd.concat([res, tmp], 0)
    res.to_csv('all_models_{}_{}_{}epochs_lamda{}_td{}.csv'.format(classifier_string, file, max_epochs, lamda, td))


