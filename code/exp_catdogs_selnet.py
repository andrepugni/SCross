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
from utils_catsdogs_ import *
from sklearn.model_selection import StratifiedKFold
from model2 import *
import pandas as pd
from time import time
from tqdm import tqdm
import pandas as pd
from utils2 import SelfAdativeTraining
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

def main(meta):
    torch.manual_seed(42)
    num_classes = 2
    input_size = 64
    max_epochs = 300
    print(max_epochs)
    lr = .1
    boot_iter = 1000
    print(boot_iter)
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=6),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    # use the "ImageFolder" datasets
    assert os.path.exists("./cats_dogs/train") and os.path.exists("./cats_dogs/test"), "Please download and put the 'cats vs dogs' dataset to paths 'data/cats_dogs/train' and 'cats_dogs/test'"
    # trainset = CatsDogs(root='./cats_dogs', split='train', transform=transform_train, resize=64)
    # testset = CatsDogs(root='./cats_dogs', split='test', transform=transform_test, resize=64)
    from sklearn.model_selection import train_test_split, StratifiedKFold
    trainset =  datasets.ImageFolder('cats_dogs/train')
    testset =  datasets.ImageFolder('cats_dogs/test') 
    if meta == 'selnet':
        trainset, validset = train_test_split(trainset, test_size=.1, random_state=42)
        trs = ImageLoaderSAT(trainset, transform=transform_train, resize=64)
        vls = ImageLoaderSAT(validset, transform=transform_test, resize=64)
        tes = ImageLoaderSAT(testset, transform=transform_test, resize=64)
    elif meta=='sat':
        trainset, validset = train_test_split(trainset, test_size=2000, random_state=42)
        trs = ImageLoaderSAT(trainset, transform=transform_train, resize=64)
        vls = ImageLoaderSAT(validset, transform=transform_test, resize=64)
        tes = ImageLoaderSAT(testset, transform=transform_test, resize=64)
    trainloader = torch.utils.data.DataLoader(trs, batch_size=128, shuffle=True)
    validloader = torch.utils.data.DataLoader(vls, batch_size=128, shuffle=False)
    testloader = torch.utils.data.DataLoader(tes, batch_size=128, shuffle=False)
    td = True
    if meta=='selnet':
        model = sel_vgg16_bn(selective=True, output_dim = 2, input_size=64)
    elif meta=='sat':
        model = sel_vgg16_bn(selective=False, output_dim = 3, input_size=64)
    optimizer = torch.optim.SGD
    lr = .1
    if td == True:
        opt = optimizer(model.parameters(), lr, weight_decay = 5e-4, momentum=0.9, nesterov=True)
    else:
        opt = optimizer(model.parameters(), lr, momentum=0.9, nesterov=True)
    num_examp = len(trainset)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    pretrain = 0
    epochs_lr = [24,49,74,99,124,149,174,199,224,249,274]
    coverages = [.99, .95, .9, .85, .8, .75]
    y_train = get_true(trainloader)
    results = pd.DataFrame()
    if meta=='selnet':
        model_dict = {k:sel_vgg16_bn(selective=True, output_dim = 2, input_size=64) for k in coverages}
        thetas_dict = {k:.5 for k in coverages}
        for k in coverages:
            optimizer = torch.optim.SGD
            if td == True:
                opt = optimizer(model_dict[k].parameters(), lr, weight_decay = 5e-4, momentum=0.9, nesterov=True)
            else:
                opt = optimizer(model_dict[k].parameters(), lr, momentum=0.9, nesterov=True)
            start_time = time()
            train_sel(model_dict[k], device=device, trainloader=trainloader, max_epochs=max_epochs, opt=opt,
                                          td=td, coverage=k, alpha=.5, lamda=32)
            end_time = time()
            time_to_fit = end_time - start_time
            path_model = 'SelNetVGG_coverage{}_{}_catsdogs.pt'.format(k, max_epochs)
            torch.save(model_dict[k].state_dict(), path_model)
            theta = get_theta(model_dict[k],device,validloader,'selnet', coverage=k)
            thetas_dict[k] = theta
            scores = get_scores(model_dict[k], device, testloader)[:,1]
            preds = get_preds(model_dict[k], device, testloader)
            confs = get_confs(model_dict[k], device, testloader)
            y_test = get_true(testloader)
            covered = confs>theta
            y_test = y_test.astype(np.int64)
            yy = pd.DataFrame(np.c_[y_test, scores, preds, covered], columns = ['true', 'scores','preds', 'select'])
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
                    bss = -1
                    brier = 1
                elif (np.sum(db['true'][db['select']>theta])==0)|(np.sum(db['true'][db['select']>theta])==len(y_test)):
                    auc = 0
                    acc = 0
                    perc_pos = 0
                    coverage = 0
                    bss = -1
                    brier = 1
                else:
                    auc = roc_auc_score(db['true'][db['select']>theta], db['scores'][db['select']>theta])
                    acc = accuracy_score(db['true'][db['select']>theta], db['preds'][db['select']>theta])
                    perc_pos = np.sum(db['true'][db['select']>theta])/len(db['true'][db['select']>theta])
                    coverage = len(db['true'][db['select']>theta])/len(db['true'])
                    ### compute brier
                    brier = brier_score_loss(db['true'][covered], db['scores'][covered])
                    ### compute brier score adj
                    perc_train = np.mean(db['true'])
                    bss_denom = brier_score_loss(db['true'][covered], np.repeat(perc_train, len(db['true']))[covered])
                    bss = 1-brier/bss_denom
                res_cols = ['dataset', 'desired_coverage', 'coverage','accuracy', 'auc', 'perc_pos', 'brier', 'bss']
                tmp = pd.DataFrame([['catsdogs', k, coverage, acc, auc, perc_pos, brier, bss]], columns =res_cols )
                tmp['meta'] = meta
                tmp['boot_iter'] = b
                tmp['model'] = "{}_{}".format(meta, 'vgg')
                tmp['time_to_fit'] = time_to_fit
                tmp['thetas'] = theta
                results = pd.concat([results, tmp], axis=0)
                
                results.to_csv("results_catsdogs_{}_{}.csv".format(meta, boot_iter))
            
    elif meta=='sat':
        start_time = time()
        train_sat(model=model, device=device, trainloader=trainloader, opt=opt, max_epochs=max_epochs,
                  pretrain=0, num_examp=len(trainset))
        end_time = time()
        path_model = 'SATVGG_{}_catsdogs.pt'.format(max_epochs)
        torch.save(model.state_dict(), path_model)
        scores = get_scores(model, device, testloader)[:,1]
        preds = get_preds(model, device, testloader)
        confs = get_confs(model, device, testloader)
        thetas = get_theta(model,device,validloader, meta)
        y_test = get_true(testloader)
        bands = qband(model, device, testloader, meta,thetas=thetas)
        time_to_fit = (end_time - start_time)
        yy = pd.DataFrame(np.c_[y_test,scores, preds, bands], columns = ['true', 'scores','preds', 'bands'])
        for b in range(boot_iter+1):
            if b==0:
                db = yy.copy()
            else:
                db = yy.sample(len(y_test), random_state=b, replace=True)
            db = db.reset_index()
            res = [coverage_and_res(level, db['true'].values, db['scores'].values, db['preds'].values, db['bands'].values, perc_train = y_train.mean()) 
                       for level in range(len(coverages)+1)]
            tmp = pd.DataFrame(res, columns = ['coverage', 'auc', 
                                               'accuracy', 'brier', 'bss', 'perc_pos'])    
            tmp['desired_coverage'] = [1, 0.99, 0.95, 0.9, 0.85, 0.80, 0.75]
            tmp['dataset'] = 'catsdogs'
            tmp['model'] = '{}_VGG'.format(meta)
            tmp['time_to_fit'] = [time_to_fit for i in range(7)]
            tmp['boot_iter'] = b
            list_thetas = [(0.5,0.5)]
            list_thetas_app = thetas
            new_list = list_thetas + (list_thetas_app)
            tmp['thetas'] = new_list
            results = pd.concat([results, tmp], axis=0)
            results.to_csv("results_catsdogs_{}_{}.csv".format(meta, boot_iter))
            
if __name__=='__main__':
    for meta in ['selnet', 'sat']:
        main(meta)