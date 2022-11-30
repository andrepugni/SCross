import time
import random
import math
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models
import dataset_utils
import numpy as np
import torch.nn.functional as F
from model2 import *
from time import time
from tqdm import tqdm
import pandas as pd
from utils2 import SelfAdativeTraining
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss


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

class ImageLoaderSAT(Dataset):
    def __init__(self, dataset, transform=None, resize=None, start=None, end=None):
        self.data = [] # some images are CMYK, Grayscale, check only RGB 
        self.transform = transform
        if start == None:
            start = 0
        if end == None:
            end = dataset.__len__()
        if resize==None:
            for i in range(start, end):
                self.data.append((dataset.__getitem__(i)))
        else:
            for i in range(start, end):
                item=dataset.__getitem__(i)
                self.data.append((transforms.functional.center_crop(
                    transforms.functional.resize(item[0],resize,Image.BILINEAR),resize),item[1]))
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.data[idx][0]), self.data[idx][1], idx
        else:
            return self.data[idx][0], self.data[idx][1], idx 
        
def coverage_and_res(level, y, y_scores, y_pred, bands, perc_train = 0.2):
    """
    

    Parameters
    ----------
    classifier : sklearn.BaseEstimator
        A sklearn style classifier.
    level : int 
        The level for which we want to obtain the coverage.
    X : pd.DataFrame or np.array
        Test set features.
    y : pd.DataFrame or np.array
        Test set target.
    y_scores : np.array
        The scores predicted by classifier.
    y_pred : np.array
        The predicted class by classifier.
    perc_train : float, optional
        The percentage of positives in the train set. The default is 0.2.

    Returns
    -------
    coverage : float
        The empirical coverage.
    auc : float
        The empirical AUC.
    accuracy : float
        The selective accuracy.
    brier : float
        Brier score.
    bss : float
        BSS.
    perc_pos : float
        The positive rate.
    median_score_rej : float
        The median score for rejected instances.

    """
    covered = bands>=level
    coverage = len(y[covered])/len(y)
    y = y.astype(np.int64)
    if (np.sum(y[covered])>0) & (np.sum(y[covered])<len(y[covered])):
      try:
        auc = roc_auc_score(y[covered],y_scores[covered])
      except:
        import pdb; pdb.set_trace()
      accuracy = accuracy_score(y[covered], y_pred[covered])
      brier = brier_score_loss(y[covered], y_scores[covered])
      bss_denom = brier_score_loss(y[covered], np.repeat(perc_train, len(y))[covered])
      bss = 1-brier/bss_denom
      perc_pos = np.sum(y[covered])/len(y[covered])
    else:
      auc = 0
      accuracy = 0
      brier = -1
      bss = -1
      perc_pos = 0
    return coverage, auc, accuracy, brier, bss, perc_pos
        
def train_sat(model, device, trainloader, opt, max_epochs, pretrain, num_examp, td=True, epochs_lr=[24,49,74,99,124,149,174,199,224,249,274]):
    model.to(device)
    running_loss = 0
    for epoch in range(1, max_epochs+1):
        model.train()
        if td:
            if epoch in epochs_lr:
                for param_group in opt.param_groups:
                    param_group['lr'] *= gamma
                    print("\n: lr now is: {}\n".format(param_group['lr']))
            with tqdm(trainloader, unit="batch") as tepoch:
                for i,batch in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")
                    X_cont, y, indices = batch           
                    X_cont, y, indices = X_cont.to(device), y.to(device), indices.to(device)
                    if len(y)==1:
                        pass
                    else:
                        opt.zero_grad()
                        outputs = model.forward(X_cont)
                        if epoch>pretrain:
                            if (epoch==pretrain+1)&(i==0):
                                print("switching to Adaptive")
                            criterion = SelfAdativeTraining(num_examples=num_examp, num_classes=model.output_dim, mom=.99)
                            loss = criterion(outputs, y, indices)
                        else:
                            loss = torch.nn.functional.cross_entropy(outputs[:, :-1], y)
                        loss.backward()
                        opt.step()
                        running_loss += loss.item()
                        if i % 1000 == 10:
                            last_loss = running_loss / 1000 # loss per batch
            #                 print('  batch {} loss: {}'.format(i + 1, last_loss))
                            tb_x = epoch * len(trainloader) + i + 1
                            running_loss = 0.
                        tepoch.set_postfix(loss=loss.item())
def train_sel(model, device, trainloader, opt, max_epochs, coverage, alpha, lamda=32, td=True, epochs_lr=[24,49,74,99,124,149,174,199,224,249,274]):
    model.to(device)
    running_loss = 0
    for epoch in range(1, max_epochs+1):
        model.train()
        if td:
            if epoch in epochs_lr:
                for param_group in opt.param_groups:
                    param_group['lr'] *= gamma
                    print("\n: lr now is: {}\n".format(param_group['lr']))
            with tqdm(trainloader, unit="batch") as tepoch:
                for i,batch in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")
                    X_cont, y, indices = batch           
                    X_cont, y, indices = X_cont.to(device), y.to(device), indices.to(device)
                    if len(y)==1:
                        pass
                    else:
                        opt.zero_grad()
                        hg, aux = model.forward(X_cont)
                        loss1 = cross_entropy_selection_vectorized(y,hg, lamda=lamda, c=coverage)
                        loss2 = cross_entropy_vectorized(y,aux)
                        if coverage ==1:
                            loss = loss2
                        else:
                            loss = (alpha*loss1)+((1-alpha)*loss2)
                        loss.backward()
                        opt.step()
                        running_loss += loss.item()
                        if i % 1000 == 10:
                            last_loss = running_loss / 1000 # loss per batch
            #                 print('  batch {} loss: {}'.format(i + 1, last_loss))
                            tb_x = epoch * len(trainloader) + i + 1
                            running_loss = 0.
                        tepoch.set_postfix(loss=loss.item())
                        
def get_scores(model, device, test_dl, coverage=.9):
    model.eval()
    scores = []
    model.to(device)
    if model.selective==False:
        for batch in test_dl:
            x_cont, y, indices = batch[0], batch[1], batch[2]
            x_cont  = x_cont.to(device)
            outputs = model(x_cont)
            outputs = torch.nn.functional.softmax(outputs[:,:-1], dim=1)
            score = outputs.detach().cpu().numpy()
            scores.append(score)
        y_hat = np.vstack(scores)
    else:
        for batch in test_dl:
            x_cont, y, indices = batch[0], batch[1], batch[2]
            x_cont  = x_cont.to(device)
            hg, aux = model(x_cont)
            if coverage==1:
                score = aux.detach().cpu().numpy()
            else:
                score = hg[:,:-1].detach().cpu().numpy()
            scores.append(score)
        y_hat = np.vstack(scores)
    return y_hat

def get_preds(model, device, test_dl, coverage=.9):
    model.eval()
    scores = []
    model.to(device)
    if model.selective==False:
        for batch in test_dl:
            x_cont, y, indices = batch[0], batch[1], batch[2]
            x_cont  = x_cont.to(device)
            outputs = model(x_cont)
            outputs = torch.nn.functional.softmax(outputs[:,:-1], dim=1)
            score = outputs.detach().cpu().numpy()
            scores.append(score)
        y_hat = np.vstack(scores)
    else:
        for batch in test_dl:
            x_cont, y, indices = batch[0], batch[1], batch[2]
            x_cont  = x_cont.to(device)
            hg, aux = model(x_cont)
            if coverage==1:
                score = aux.detach().cpu().numpy()
            else:
                score = hg[:,:-1].detach().cpu().numpy()
            scores.append(score)
        y_hat = np.vstack(scores)
    preds = np.argmax(y_hat, axis=1)
    return preds

def get_confs(model, device, test_dl, coverage=.9):
    model.eval()
    scores = []
    model.to(device)
    if model.selective==False:
        for batch in test_dl:
            x_cont, y, indices = batch[0], batch[1], batch[2]
            x_cont  = x_cont.to(device)
            outputs = model(x_cont)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            conf = outputs[:,-1].detach().cpu().numpy().reshape(-1,1)
            scores.append(conf)
        y_hat = np.vstack(scores).flatten()
    else:
        for batch in test_dl:
            x_cont, y, indices = batch[0], batch[1], batch[2]
            x_cont  = x_cont.to(device)
            hg, aux = model(x_cont)
            score = hg[:,-1].detach().cpu().numpy().reshape(-1,1)
            scores.append(score)
        y_hat = np.vstack(scores).flatten()
    return y_hat
                        
def get_theta(model, device, validloader, meta, coverage=.9, quantiles = [.01,.05,.1,.15,.2,.25]):
    if meta=='selnet':
        confs = get_confs(model,device,validloader)
        theta = np.quantile(confs, 1-coverage, method='nearest')
        return theta
    elif meta=='sat':
        confs = get_confs(model, device, validloader)
        thetas = [np.quantile(confs, 1-cov,method='nearest') for cov in sorted(quantiles, reverse=True)]
        return thetas
    
def get_true(testloader):
    y_true = []
    for batch in testloader:
        y = batch[1].detach().cpu().numpy().reshape(-1,1)
        y_true.append(y)
    y_true = np.vstack(y_true).flatten()
    return y_true

def qband(model, device, testloader, meta, thetas):
    if len(thetas)==1:
        band = get_confs(model, device, testloader)
        return np.where(band>theta, 1,0)
    else:
        if meta=='sat':
            band = np.digitize(get_confs(model, device, testloader), sorted(thetas, reverse=True), right=True)
            return band
