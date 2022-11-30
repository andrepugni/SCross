import time
import random
import math
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from PIL import Image
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models
import dataset_utils
import numpy as np
from utils_catsdogs_ import *
from sklearn.model_selection import StratifiedKFold,train_test_split
from model2 import *
import pandas as pd
import argparse


def main(meta, cv = 5, max_epochs = 300, lr = .1,
         num_classes = 2, boot_iter = 1000, td=True,
         gamma = .5, input_size = 64, filename='cats_dogs',
         quantiles = [.01, .05, .1, .15, .2, .25]):
    print(max_epochs)
    torch.manual_seed(42)
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
    assert os.path.exists("./{}/train".format(filename)) and os.path.exists("./{}/test".format(filename)), "Please download and put the 'cats vs dogs' dataset to paths 'data/cats_dogs/train' and 'cats_dogs/test'"
    # trainset = CatsDogs(root='./cats_dogs', split='train', transform=transform_train, resize=64)
    # testset = CatsDogs(root='./cats_dogs', split='test', transform=transform_test, resize=64)
    if meta =='scross':
        td = True
        skf = StratifiedKFold(cv,shuffle=True, random_state=42)
        trainset =  datasets.ImageFolder("{}/train".format(filename))
        testset =  datasets.ImageFolder("./{}/test".format(filename)) 
        trs = ImageLoaderSAT(trainset, transform=transform_train, resize=64)
        tes = ImageLoaderSAT(testset, transform=transform_test, resize=64)
        trainloader = torch.utils.data.DataLoader(trs, batch_size=128, shuffle=True)
        testloader = torch.utils.data.DataLoader(tes, batch_size=128, shuffle=False)
        indexes = []
        y = []
        for indices in trainloader:
            indexes.append(indices[2].detach().cpu().numpy().reshape(-1,1))
            y.append(indices[1].detach().cpu().numpy().reshape(-1,1))
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(device)
        idx = np.vstack(indexes).flatten()
        ys = np.vstack(y).flatten()
        model_dict = {f:sel_vgg16_bn(selective=False, output_dim = num_classes, input_size=input_size) for f in range(cv)}
        localthetas = []
        thetas = [None for _ in range(len(quantiles))]
        model = sel_vgg16_bn(selective=False, output_dim = num_classes, input_size=input_size)
        results = pd.DataFrame()
        optimizer = torch.optim.SGD
        y_train = get_true(trainloader)
        start_time = time()
        z = []
        for num,d in enumerate(skf.split(idx,ys)):
            train_idx = d[0]
            valid_idx =  d[1]
            num_examp = len(train_idx)
            trs_fold = Subset(trs, train_idx)
            vld_fold = Subset(trs, valid_idx)
            train_dl = torch.utils.data.DataLoader(trs_fold, batch_size=128, shuffle=True)
            valid_dl = torch.utils.data.DataLoader(vld_fold, batch_size=128, shuffle=True)
            if td == True:
                opt = optimizer(model_dict[num].parameters(), lr, weight_decay = 5e-4, momentum=0.9, nesterov=True)
            else:
                opt = optimizer(model_dict[num].parameters(), lr, momentum=0.9, nesterov=True)
            path_model = 'SCrossVGG_fold{}_{}_{}catsdogs.pt'.format(num,max_epochs,cv)
            if os.path.exists(path_model):
                model_dict[num].load_state_dict(torch.load(path_model))
                model_dict[num].eval()
            else:
                train_sat(model=model_dict[num], device=device, trainloader=train_dl, opt=opt, max_epochs=max_epochs,
                pretrain=max_epochs, num_examp=num_examp, crit='ce')
                torch.save(model_dict[num].state_dict(), path_model)
            scores = get_scores(model_dict[num], device, valid_dl, crit='ce')
            confs = np.max(scores, axis=1)
            z.append(confs)
        confs = np.concatenate(z).ravel()
        sub_confs_1, sub_confs_2 = train_test_split(confs, test_size=.5, random_state=42)
        tau = (1/np.sqrt(2))
        thetas = [(tau*np.quantile(confs, q) + 
                        (1-tau)*(.5*np.quantile(sub_confs_1, q)+
                                 .5*np.quantile(sub_confs_2, q))) for q in quantiles]
        
        num_examp = len(y)
        trainloader = torch.utils.data.DataLoader(trs, batch_size=128, shuffle=True)
        testloader = torch.utils.data.DataLoader(tes, batch_size=128, shuffle=False)
        path_model = 'SCrossVGG_Score_{}catsdogs.pt'.format(max_epochs)
        if os.path.exists(path_model):
            model.load_state_dict(torch.load(path_model))
            model.eval()
        else:
            if td == True:
                opt = optimizer(model.parameters(), lr, weight_decay = 5e-4, momentum=0.9, nesterov=True)
            else:
                opt = optimizer(model.parameters(), lr, momentum=0.9, nesterov=True)
            train_sat(model=model, device=device, trainloader=trainloader, opt=opt, max_epochs=max_epochs,
                pretrain=max_epochs, num_examp=num_examp, crit='ce')
        end_time = time()
        
        torch.save(model.state_dict(), path_model)
        scores = get_scores(model, device, testloader, crit='ce')
        confs = np.max(scores, axis=1)
        bands = np.digitize(confs, thetas)
        preds = get_preds(model, device, testloader, crit='ce')
        y_test = get_true(testloader)
        time_to_fit = (end_time - start_time)
        yy = pd.DataFrame(np.c_[y_test,scores[:,1], preds, bands], columns = ['true', 'scores','preds', 'bands'])
        for b in range(boot_iter+1):
            if b==0:
                db = yy.copy()
            else:
                db = yy.sample(len(y_test), random_state=b, replace=True)
            db = db.reset_index()
            res = [coverage_and_res(level, db['true'].values, db['scores'].values, db['preds'].values, db['bands'].values, perc_train = y_train.mean()) 
                       for level in range(len(quantiles)+1)]
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
            results.to_csv('results_catsdogs_scross_{}_{}.csv'.format(max_epochs,cv))
        
if __name__=='__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    meta = 'scross'
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--boot_iter', type=int, required=False, default=1000)
    parser.add_argument('--max_epochs', type=int, required=False, default=300)
    parser.add_argument('--cv', type=int, required=False, default=5)
    
    args = parser.parse_args()
                                                                           
    main(meta, max_epochs = args.max_epochs, boot_iter= args.boot_iter, cv = args.cv)