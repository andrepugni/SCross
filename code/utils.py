import pandas as pd
import numpy as np
from attributes import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from model import *
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from time import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ToTorchDataset(Dataset):
    def __init__(self, X, Y, embedded_col_names, one_hot_cols_names):
        X = X.copy()
        self.X1 = X.loc[:,embedded_col_names].copy().values.astype(np.int64) #categorical columns
        self.X3 = X.loc[:,one_hot_cols_names].copy().values.astype(np.int64) #categorical columns
        self.X2 = X.drop(columns=embedded_col_names+one_hot_cols_names).copy().values.astype(np.float32) #numerical columns
        if type(Y)==np.ndarray:
            self.y = Y
        else:
            self.y = Y.values
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X2[idx], self.X1[idx], self.X3[idx], self.y[idx]

class ToTorchDatasetId(Dataset):
    def __init__(self, X, Y, embedded_col_names, one_hot_cols_names):
        X = X.copy()
        self.X1 = X.loc[:,embedded_col_names].copy().values.astype(np.int64) #categorical columns
        self.X3 = X.loc[:,one_hot_cols_names].copy().values.astype(np.int64) #categorical columns
        self.X2 = X.drop(columns=embedded_col_names+one_hot_cols_names).copy().values.astype(np.float32) #numerical columns
        if type(Y)==np.ndarray:
            self.y = Y
        else:
            self.y = Y.values
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X2[idx], self.X1[idx], self.X3[idx], self.y[idx], idx
        
class ToTorchDatasetNP(Dataset):
    def __init__(self, X, Y):
        X = X.copy()
        self.X = X
        if type(Y)==np.ndarray:
            self.y = Y
        else:
            self.y = Y.values
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ToTorchDatasetNPId(Dataset):
    def __init__(self, X, Y):
        X = X.copy()
        self.X = X
        if type(Y)==np.ndarray:
            self.y = Y
        else:
            self.y = Y.values
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], idx
    
class ToTorchDatasetPred(Dataset):
    def __init__(self, X, embedded_col_names, one_hot_cols_names):
        X = X.copy()
        self.X1 = X.loc[:,embedded_col_names].copy().values.astype(np.int64) #categorical columns
        self.X3 = X.loc[:,one_hot_cols_names].copy().values.astype(np.int64) #categorical columns
        self.X2 = X.drop(columns=embedded_col_names+one_hot_cols_names).copy().values.astype(np.float32) #numerical columns
        
    def __len__(self):
        return len(self.X1)
    
    def __getitem__(self, idx):
        return self.X2[idx], self.X1[idx], self.X3[idx]
        
class ToTorchDatasetNPPred(Dataset):
    def __init__(self, X):
        X = X.copy()
        self.X = X

        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx]  
        
        


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


def deep_gambler_loss(outputs, targets, reward):
    outputs = F.softmax(outputs, dim=1)
    outputs, reservation = outputs[:,:-1], outputs[:,-1]
    # gain = torch.gather(outputs, dim=1, index=targets.unsqueeze(1)).squeeze()
    gain = outputs[torch.arange(targets.shape[0]), targets]
    doubling_rate = (gain.add(reservation.div(reward))).log()
    return -doubling_rate.mean()


class SelfAdativeTraining():
    def __init__(self, num_examples=50000, num_classes=10, mom=0.9):
        self.prob_history = torch.zeros(num_examples, num_classes)
        self.updated = torch.zeros(num_examples, dtype=torch.int)
        self.mom = mom
        self.num_classes = num_classes

    def _update_prob(self, prob, index, y):
        onehot = torch.zeros_like(prob)
        onehot[torch.arange(y.shape[0]), y] = 1
        prob_history = self.prob_history[index].clone().to(prob.device)

        # if not inited, use onehot label to initialize runnning vector
        cond = (self.updated[index] == 1).to(prob.device).unsqueeze(-1).expand_as(prob)
        prob_mom = torch.where(cond, prob_history, onehot)

        # momentum update
        prob_mom = self.mom * prob_mom + (1 - self.mom) * prob

        self.updated[index] = 1
        self.prob_history[index] = prob_mom.to(self.prob_history.device)

        return prob_mom

    def __call__(self, logits, y, index):
        prob = F.softmax(logits.detach()[:, :self.num_classes], dim=1)
        prob = self._update_prob(prob, index, y)

        soft_label = torch.zeros_like(logits)
        soft_label[torch.arange(y.shape[0]), y] = prob[torch.arange(y.shape[0]), y]
        soft_label[:, -1] = 1 - prob[torch.arange(y.shape[0]), y]
        soft_label = F.normalize(soft_label, dim=1, p=1)
        loss = torch.sum(-F.log_softmax(logits, dim=1) * soft_label, dim=1)
        return torch.mean(loss)






def traindata_sat(device, model, epochs, optimizer, train_dl,
              td=True, gamma=.5, verbose=True,
              epochs_lr = [24,49,74,99,124,149,174,199,224,249,274,299], crit='sat', num_examples=50000,
              pretrain=0, num_classes=3, sat_momentum=.99):
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
    td : boolean, optional
        The use of time decay. The default is True.
    gamma : float, optional.
        The factor used to scale the learning rate. The default is .5.
    verbose : boolean, optional.
        Boolean to print epochs loss. The default is True.
    epochs_lr : list, optional.
        A list of epochs when the learning rate is decreased by gamma. The default is [24,49,74,99,124,149,174,199,224,249,274,299].
    crit : str, optional.
        A string for the criterion used to train the model.
        Possible values are:
            - 'sat' : for Self Adaptive Training
            - 'ce' : for  Cross Entropy Loss
        The default is 'sat'.
    num_examples : int, optional.
        An integer that is used by SAT loss. It should equal the training set size. The default is 50,000.
    pretrain : int, optional.
        The epochs after which the loss is switched to SAT if 'sat' is chosen as a crit. The default is 0 (as in the selective classification implementation of SAT).
    num_classes : int, optional
        The desired number of classes to predict. The default is 3.
    sat_momentum : float, optional
        The value alpha of SAT loss. The default is .99.

    Returns
    -------
    model : torch.module
        The trained network.

    """
    # No Early stopping
    model.to(device)
    running_loss = 0
    for epoch in range(1, epochs+1):
        model.train()
        if td:
          if epoch in epochs_lr:
              for param_group in optimizer.param_groups:
                  param_group['lr'] *= gamma
                  print("\n: lr now is: {}\n".format(param_group['lr']))
        if verbose:
            with tqdm(train_dl, unit="batch") as tepoch:
                for i,batch in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")
                    X_cont, X_cat, X_one, y, indices = batch           
                    X_cont, X_cat, X_one, y, indices = X_cont.to(device), X_cat.to(device), X_one.to(device), y.to(device), indices.to(device)
                    if len(y)==1:
                        pass
                    else:
                        optimizer.zero_grad()
                        outputs = model.forward(X_cont, X_cat, X_one)
                        if crit=='sat':
                            if (epoch==1)&(i==0):
                                print("\n criterion is {} \n".format(crit))
                            if epoch>pretrain:
                                if (epoch==pretrain+1)&(i==0):
                                    print("switching to Adaptive")
                                criterion = SelfAdativeTraining(num_examples=num_examples, num_classes=num_classes, mom=.99)
                                loss = criterion(outputs, y, indices)
                            else:
                                loss = torch.nn.functional.cross_entropy(outputs[:, :-1], y)
                        elif crit=='ce':
                            if (epoch==1) & (i==0):
                                print("\n criterion is {} \n".format(crit)) 
                            loss = torch.nn.functional.cross_entropy(outputs, y)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                        if i % 1000 == 10:
                            last_loss = running_loss / 1000 # loss per batch
            #                 print('  batch {} loss: {}'.format(i + 1, last_loss))
                            tb_x = epoch * len(train_dl) + i + 1
                            running_loss = 0.
                        tepoch.set_postfix(loss=loss.item())
        else:
            for i,batch in enumerate(train_dl):
                    #tepoch.set_description(f"Epoch {epoch}")
                    X_cont, X_cat, X_one, y, indices = batch           
                    X_cont, X_cat, X_one, y, indices = X_cont.to(device), X_cat.to(device), X_one.to(device), y.to(device), indices.to(device)
                    if len(y)==1:
                        pass
                    else:
                        opt.zero_grad()
                        outputs = model.forward(X_cont, X_cat, X_one)
                        if crit=='sat':
                            if epoch==1:
                                print("\n criterion is {} \n".format(crit))
                            if epoch>pretrain:
                                if (epoch==pretrain+1)&(i==0):
                                    print("switching to Adaptive")
                                criterion = SelfAdativeTraining(num_examples=num_examp, num_classes=num_classes, mom=.99)
                                loss = criterion(outputs, y, indices)
                            else:
                                loss = torch.nn.functional.cross_entropy(outputs[:, :-1], y)
                        elif crit=='ce':
                            if (epoch==1) & (i==0):
                                print("\n criterion is {} \n".format(crit)) 
                            loss = torch.nn.functional.cross_entropy(outputs, y)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                        if i % 1000 == 10:
                            last_loss = running_loss / 1000 # loss per batch
            #                 print('  batch {} loss: {}'.format(i + 1, last_loss))
                            tb_x = epoch * len(train_dl) + i + 1
                            running_loss = 0.
    return model

def validation(model, device, valid_dl, alpha=.5):
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
            loss1 = alpha*cross_entropy_selection_vectorized(y,hg, c=1)
            loss2 = (1-alpha)*cross_entropy_vectorized(y,aux)
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