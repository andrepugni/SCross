#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for SelNet implementation
"""
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from attributes import *
from rtdl import ResNet

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

class SelectiveResNetOne(nn.Module):
    def __init__(self,
        n_cont,
        embedding_sizes,
        n_one_hot,
        output_dim,
        n_blocks=2,
        d_main=3,
        d_hidden=4,
        dropout_first=0.25,
        dropout_second=0.0,
        coverage=0.9,
        alpha=0.5):
        super(SelectiveResNetOne, self).__init__()
        self.n_cont = n_cont
        self.n_one_hot = n_one_hot
        self.input_dim = len(embedding_sizes)+n_cont+self.n_one_hot
        self.output_dim=output_dim
        self.n_blocks=n_blocks
        self.d_main=d_main
        self.d_hidden=d_hidden
        self.dropout_first=dropout_first
        self.dropout_second=dropout_second
        self.coverage = coverage
        self.alpha = alpha
        if self.n_blocks <= 0:
            raise ValueError("n_blocks should be a positive integer.")
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.embedding_sizes = embedding_sizes
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories,size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings) #length of all embeddings combined
        self.n_emb = n_emb
        self.emb_drop = nn.Dropout(0.1)
        self.post_embed_dim = self.n_emb+self.n_cont+self.n_one_hot
        self.resnet = ResNet.make_baseline(d_in=self.n_emb+self.n_cont+self.n_one_hot, d_out=128, n_blocks=n_blocks, d_main=d_main,
                                           d_hidden=d_hidden, dropout_first=dropout_first, dropout_second=dropout_second)
        self.drop_before = torch.nn.Dropout(0.2)
        self.dense_class = torch.nn.Linear(128, output_dim)
        self.dense_selec_1 = torch.nn.Linear(128, 64)
        self.batch_norm = torch.nn.BatchNorm1d(64)
        self.dense_selec_2 = torch.nn.Linear(64,1)
        self.dense_auxil = torch.nn.Linear(128, output_dim)
        
    def forward(self, x_cont, x_cat, x_one):
        x = [e(x_cat[:,i]) for i,e in enumerate(self.embeddings)]
        if len(x)>0:
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
            x2 = self.bn1(x_cont)
            x = torch.cat([x, x2, x_one], 1)
        else:
            try:
                x2 = self.bn1(x_cont)
            except:
                import pdb; pdb.set_trace()
            x = torch.cat([x2, x_one], 1)
        
        x = self.resnet(x)
        h = self.dense_class(x)
        h = torch.nn.functional.softmax(h, dim=1)
        g = self.dense_selec_1(x)
        g = torch.nn.functional.relu(g)
        g = self.batch_norm(g)
        g = self.dense_selec_2(g)
        g = torch.sigmoid(g)
        a = self.dense_auxil(x)
        a = torch.nn.functional.softmax(a, dim=1)
        hg = torch.cat([h,g],1)
        return hg,a   

class AdapResNetOne(nn.Module):
    def __init__(self,
        n_cont,
        embedding_sizes,
        n_one_hot,
        output_dim,
        n_blocks=2,
        d_main=3,
        d_hidden=4,
        dropout_first=0.25,
        dropout_second=0.0,
        gamma = .5,
        pretrain = 0
        ):
        super(AdapResNetOne, self).__init__()
        self.n_cont = n_cont
        self.n_one_hot = n_one_hot
        self.input_dim = len(embedding_sizes)+n_cont+self.n_one_hot
        self.output_dim=output_dim
        self.n_blocks=n_blocks
        self.d_main=d_main
        self.d_hidden=d_hidden
        self.dropout_first=dropout_first
        self.dropout_second=dropout_second
        self.pretrain = pretrain
        self.gamma = gamma
        if self.n_blocks <= 0:
            raise ValueError("n_blocks should be a positive integer.")
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.embedding_sizes = embedding_sizes
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories,size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings) #length of all embeddings combined
        self.n_emb = n_emb
        self.emb_drop = nn.Dropout(0.1)
        self.post_embed_dim = self.n_emb+self.n_cont+self.n_one_hot
        self.resnet = ResNet.make_baseline(d_in=self.n_emb+self.n_cont+self.n_one_hot, d_out=128, n_blocks=n_blocks, d_main=d_main,
                                           d_hidden=d_hidden, dropout_first=dropout_first, dropout_second=dropout_second)
        self.drop_before = torch.nn.Dropout(0.2)
        self.dense_class = torch.nn.Linear(128, output_dim)

        
    def forward(self, x_cont, x_cat, x_one):
        x = [e(x_cat[:,i]) for i,e in enumerate(self.embeddings)]
        if len(x)>0:
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
            x2 = self.bn1(x_cont)
            x = torch.cat([x, x2, x_one], 1)
        else:
            try:
                x2 = self.bn1(x_cont)
            except:
                import pdb; pdb.set_trace()
            x = torch.cat([x2, x_one], 1)
        
        x = self.resnet(x)
        h = self.dense_class(x)
        return h  



class SelectiveResNetOneV2(nn.Module):
    def __init__(self,
        n_cont,
        embedding_sizes,
        n_one_hot,
        output_dim,
        n_blocks=2,
        d_main=3,
        d_hidden=4,
        dropout_first=0.25,
        dropout_second=0.0,
        coverage=0.9,
        alpha=0.5):
        super(SelectiveResNetOneV2, self).__init__()
        self.n_cont = n_cont
        self.n_one_hot = n_one_hot
        self.input_dim = len(embedding_sizes)+n_cont+self.n_one_hot
        self.output_dim=output_dim
        self.n_blocks=n_blocks
        self.d_main=d_main
        self.d_hidden=d_hidden
        self.dropout_first=dropout_first
        self.dropout_second=dropout_second
        self.coverage = coverage
        self.alpha = alpha
        if self.n_blocks <= 0:
            raise ValueError("n_blocks should be a positive integer.")
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.embedding_sizes = embedding_sizes
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories,size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings) #length of all embeddings combined
        self.n_emb = n_emb
        self.emb_drop = nn.Dropout(0.1)
        self.post_embed_dim = self.n_emb+self.n_cont+self.n_one_hot
        self.resnet = ResNet.make_baseline(d_in=self.n_emb+self.n_cont+self.n_one_hot, d_out=512, n_blocks=n_blocks, d_main=d_main,
                                           d_hidden=d_hidden, dropout_first=dropout_first, dropout_second=dropout_second)
        self.drop_before = torch.nn.Dropout(0.2)
        self.dense_class_1 = torch.nn.Linear(512, 256)
        self.bn_1 = torch.nn.BatchNorm1d(256)
        self.dense_class_2 = torch.nn.Linear(256, 128)
        self.bn_2 = torch.nn.BatchNorm1d(128)
        self.dense_class_3 = torch.nn.Linear(128, self.output_dim)
        self.dense_selec_1 = torch.nn.Linear(512, 64)
        self.batch_norm = torch.nn.BatchNorm1d(64)
        self.dense_selec_2 = torch.nn.Linear(64,1)
        self.dense_auxil_1 = torch.nn.Linear(512, 256)
        self.bn_3 = torch.nn.BatchNorm1d(256)
        self.dense_auxil_2 = torch.nn.Linear(256, 128)
        self.bn_4 = torch.nn.BatchNorm1d(128)
        self.dense_auxil_3 = torch.nn.Linear(128, self.output_dim)
    def forward(self, x_cont, x_cat, x_one):
        x = [e(x_cat[:,i]) for i,e in enumerate(self.embeddings)]
        if len(x)>0:
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
            x2 = self.bn1(x_cont)
            x = torch.cat([x, x2, x_one], 1)
        else:
            try:
                x2 = self.bn1(x_cont)
            except:
                import pdb; pdb.set_trace()
            x = torch.cat([x2, x_one], 1)
        
        x = self.resnet(x)
        h = self.dense_class_1(x)
        h = torch.nn.functional.relu(h)
        h = self.bn_1(h)
        h = self.dense_class_2(h)
        h = torch.nn.functional.relu(h)
        h = self.bn_2(h)
        h = self.dense_class_3(h)
        h = torch.nn.functional.softmax(h, dim=1)
        g = self.dense_selec_1(x)
        g = torch.nn.functional.relu(g)
        g = self.batch_norm(g)
        g = self.dense_selec_2(g)
        g = torch.sigmoid(g)
        a = self.dense_auxil_1(x)
        a = torch.nn.functional.relu(a)
        a = self.bn_3(a)
        a = self.dense_auxil_2(a)
        a = torch.nn.functional.relu(a)
        a = self.bn_4(a)
        a = self.dense_auxil_3(a)
        a = torch.nn.functional.softmax(a, dim=1)
        hg = torch.cat([h,g],1)
        return hg,a   

class AdapResNetOneV2(nn.Module):
    def __init__(self,
        n_cont,
        embedding_sizes,
        n_one_hot,
        output_dim,
        n_blocks=2,
        d_main=3,
        d_hidden=4,
        dropout_first=0.25,
        dropout_second=0.0,
        gamma = .5,
        pretrain = 50
        ):
        super(AdapResNetOneV2, self).__init__()
        self.n_cont = n_cont
        self.n_one_hot = n_one_hot
        self.input_dim = len(embedding_sizes)+n_cont+self.n_one_hot
        self.output_dim=output_dim
        self.n_blocks=n_blocks
        self.d_main=d_main
        self.d_hidden=d_hidden
        self.dropout_first=dropout_first
        self.dropout_second=dropout_second
        self.pretrain = pretrain
        self.gamma = gamma
        if self.n_blocks <= 0:
            raise ValueError("n_blocks should be a positive integer.")
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.embedding_sizes = embedding_sizes
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories,size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings) #length of all embeddings combined
        self.n_emb = n_emb
        self.emb_drop = nn.Dropout(0.1)
        self.post_embed_dim = self.n_emb+self.n_cont+self.n_one_hot
        self.resnet = ResNet.make_baseline(d_in=self.n_emb+self.n_cont+self.n_one_hot, d_out=512, n_blocks=n_blocks, d_main=d_main,
                                           d_hidden=d_hidden, dropout_first=dropout_first, dropout_second=dropout_second)
        self.drop_before = torch.nn.Dropout(0.2)
        self.dense_class_1 = torch.nn.Linear(512, 256)
        self.bn_1 = torch.nn.BatchNorm1d(256)
        self.dense_class_2 = torch.nn.Linear(256, 128)
        self.bn_2 = torch.nn.BatchNorm1d(128)
        self.dense_class_3 = torch.nn.Linear(128, self.output_dim)

    def forward(self, x_cont, x_cat, x_one):
        x = [e(x_cat[:,i]) for i,e in enumerate(self.embeddings)]
        if len(x)>0:
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
            x2 = self.bn1(x_cont)
            x = torch.cat([x, x2, x_one], 1)
        else:
            try:
                x2 = self.bn1(x_cont)
            except:
                import pdb; pdb.set_trace()
            x = torch.cat([x2, x_one], 1)
        
        x = self.resnet(x)
        h = self.dense_class_1(x)
        h = torch.nn.functional.relu(h)
        h = self.bn_1(h)
        h = self.dense_class_2(h)
        h = torch.nn.functional.relu(h)
        h = self.bn_2(h)
        h = self.dense_class_3(h)
        return h     

class SelectiveLinearNetOne(nn.Module):
    def __init__(self,
        n_cont,
        embedding_sizes,
        n_one_hot,
        output_dim,
        coverage=0.9,
        alpha=0.5):
        super(SelectiveLinearNetOne, self).__init__()
        self.n_cont = n_cont
        self.n_one_hot = n_one_hot
        self.input_dim = len(embedding_sizes)+n_cont+self.n_one_hot
        self.output_dim=output_dim
        self.coverage = coverage
        self.alpha = alpha
        self.bn0 = nn.BatchNorm1d(self.n_cont)
        self.embedding_sizes = embedding_sizes
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories,size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings) #length of all embeddings combined
        self.n_emb = n_emb
        self.emb_drop = nn.Dropout(0.1)
        self.post_embed_dim = self.n_emb+self.n_cont+self.n_one_hot
        self.linear1 = torch.nn.Linear(self.n_emb+self.n_cont+self.n_one_hot,64)
        self.linear2 = torch.nn.Linear(64,128)
        self.linear3 = torch.nn.Linear(128,256)
        self.linear4 = torch.nn.Linear(256,512)
        self.linear5 = torch.nn.Linear(512,512)
        self.linear6 = torch.nn.Linear(512,256)
        self.linear7 = torch.nn.Linear(256,128)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(512)
        self.bn6 = torch.nn.BatchNorm1d(256)
        self.bn7 = torch.nn.BatchNorm1d(128)
        self.dr1 = torch.nn.Dropout(0.1)
        self.dr2 = torch.nn.Dropout(0.2)
        self.drop_before = torch.nn.Dropout(0.2)
        self.dense_class = torch.nn.Linear(128, output_dim)
        self.dense_selec_1 = torch.nn.Linear(128, 64)
        self.batch_norm = torch.nn.BatchNorm1d(64)
        self.dense_selec_2 = torch.nn.Linear(64,1)
        self.dense_auxil = torch.nn.Linear(128, output_dim)
        
    def forward(self, x_cont, x_cat, x_one):
        x = [e(x_cat[:,i]) for i,e in enumerate(self.embeddings)]
        if len(x)>0:
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
            x2 = self.bn0(x_cont)
            x = torch.cat([x, x2, x_one], 1)
        else:
            try:
                x2 = self.bn1(x_cont)
            except:
                import pdb; pdb.set_trace()
            x = torch.cat([x2, x_one], 1)
        
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.bn1(x)
        x = self.linear2(x)
        x = torch.nn.functional.relu(x)
        x = self.bn2(x)
        x = self.dr1(x)
        x = self.linear3(x)
        x = torch.nn.functional.relu(x)
        x = self.bn3(x)
        x = self.linear4(x)
        x = torch.nn.functional.relu(x)
        x = self.bn4(x)
        x = self.dr2(x)
        x = self.linear5(x)
        x = torch.nn.functional.relu(x)
        x = self.bn5(x)
        x = self.linear6(x)
        x = torch.nn.functional.relu(x)
        x = self.bn6(x)
        x = self.dr1(x)
        x = self.linear7(x)
        x = torch.nn.functional.relu(x)  
        x = self.bn7(x)      
        h = self.dense_class(x)
        h = torch.nn.functional.softmax(h, dim=1)
        g = self.dense_selec_1(x)
        g = torch.nn.functional.relu(g)
        g = self.batch_norm(g)
        g = self.dense_selec_2(g)
        g = torch.sigmoid(g)
        a = self.dense_auxil(x)
        a = torch.nn.functional.softmax(a, dim=1)
        hg = torch.cat([h,g],1)
        return hg,a  
        
class SelectiveLinearNetTwo(nn.Module):
    def __init__(self,
        n_cont,
        embedding_sizes,
        n_one_hot,
        output_dim,
        coverage=0.9,
        alpha=0.5):
        super(SelectiveLinearNetTwo, self).__init__()
        self.n_cont = n_cont
        self.n_one_hot = n_one_hot
        self.input_dim = len(embedding_sizes)+n_cont+self.n_one_hot
        self.output_dim=output_dim
        self.coverage = coverage
        self.alpha = alpha
        self.bn0 = nn.BatchNorm1d(self.n_cont)
        self.embedding_sizes = embedding_sizes
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories,size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings) #length of all embeddings combined
        self.n_emb = n_emb
        self.emb_drop = nn.Dropout(0.1)
        self.post_embed_dim = self.n_emb+self.n_cont+self.n_one_hot
        self.linear1 = torch.nn.Linear(self.n_emb+self.n_cont+self.n_one_hot,256)
        self.linear2 = torch.nn.Linear(256,128)
        self.bn1 = torch.nn.BatchNorm1d(256)        
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.dr1 = torch.nn.Dropout(0.1)
        self.dr2 = torch.nn.Dropout(0.2)
        self.drop_before = torch.nn.Dropout(0.2)
        self.dense_class = torch.nn.Linear(128, output_dim)
        self.dense_selec_1 = torch.nn.Linear(128, 64)
        self.batch_norm = torch.nn.BatchNorm1d(64)
        self.dense_selec_2 = torch.nn.Linear(64,1)
        self.dense_auxil = torch.nn.Linear(128, output_dim)
        
    def forward(self, x_cont, x_cat, x_one):
        x = [e(x_cat[:,i]) for i,e in enumerate(self.embeddings)]
        if len(x)>0:
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
            x2 = self.bn0(x_cont)
            x = torch.cat([x, x2, x_one], 1)
        else:
            try:
                x2 = self.bn0(x_cont)
            except:
                import pdb; pdb.set_trace()
            x = torch.cat([x2, x_one], 1)
        
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.bn1(x)
        x = self.linear2(x)
        x = torch.nn.functional.relu(x)  
        x = self.bn2(x)      
        h = self.dense_class(x)
        h = torch.nn.functional.softmax(h, dim=1)
        g = self.dense_selec_1(x)
        g = torch.nn.functional.relu(g)
        g = self.batch_norm(g)
        g = self.dense_selec_2(g)
        g = torch.sigmoid(g)
        a = self.dense_auxil(x)
        a = torch.nn.functional.softmax(a, dim=1)
        hg = torch.cat([h,g],1)
        return hg,a  
        
class VGG_tab(nn.Module):

    def __init__(self, 
        features,
        n_cont,
        embedding_sizes,
        n_one_hot, 
        selective=True,
        num_classes=1000,
        input_size = 32):
        super(VGG_tab, self).__init__()
        self.n_cont = n_cont
        self.n_one_hot = n_one_hot
        self.input_dim = len(embedding_sizes)+n_cont+self.n_one_hot
        self.output_dim=num_classes
        self.bn0 = nn.BatchNorm1d(self.n_cont)
        self.embedding_sizes = embedding_sizes
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories,size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings) #length of all embeddings combined
        self.n_emb = n_emb
        self.emb_drop = nn.Dropout(0.1)
        self.features = features
        self.selective = selective
        if input_size == 32:
            self.classifier = nn.Sequential(nn.Linear(512,512), nn.ReLU(inplace=True), \
                                        nn.BatchNorm1d(512),nn.Dropout2d(0.5),nn.Linear(512, num_classes))
            self.linear1 = torch.nn.Linear(self.n_emb+self.n_cont+self.n_one_hot,32)
        elif input_size == 64:
            self.classifier = nn.Sequential(nn.Linear(2048,512), nn.ReLU(inplace=True), \
                                        nn.BatchNorm1d(512),nn.Dropout2d(0.5),nn.Linear(512, num_classes))
            self.linear1 = torch.nn.Linear(self.n_emb+self.n_cont+self.n_one_hot,64)


    def forward(self, x_cont, x_cat, x_one):
        x = [e(x_cat[:,i]) for i,e in enumerate(self.embeddings)]
        if len(x)>0:
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
            x2 = self.bn0(x_cont)
            x = torch.cat([x, x2, x_one], 1)
        else:
            try:
                x2 = self.bn0(x_cont)
            except:
                import pdb; pdb.set_trace()
            x = torch.cat([x2, x_one], 1)
        x = self.linear1(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        #x = torch.nn.functional.softmax(x,dim=1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPoo2d(kernel_size=2, stride=2)]
        elif type(v)==int:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.ReLU(inplace=True), nn.BatchNorm2d(v)]
                # the order is modified to match the model of the baseline that we compare to
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        elif type(v)==float:
            layers += [nn.Dropout2d(v)]
    return nn.Sequential(*layers)

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPoo2d(kernel_size=2, stride=2)]
        elif type(v)==int:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.ReLU(inplace=True), nn.BatchNorm2d(v)]
                # the order is modified to match the model of the baseline that we compare to
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        elif type(v)==float:
            layers += [nn.Dropout2d(v)]
    return nn.Sequential(*layers)
    
def make_layers2(cfg, batch_norm=False, input_s=64):
    layers = []
    in_channels = input_s
    for v in cfg:
        if type(v)==int:
            conv2d = nn.Linear(in_channels, v)
            if batch_norm:
                layers += [conv2d, nn.ReLU(inplace=True), nn.BatchNorm1d(v)]
                # the order is modified to match the model of the baseline that we compare to
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        elif type(v)==float:
            layers += [nn.Dropout2d(v)]
    return nn.Sequential(*layers)


cfg = {
    'D': [64,0.3, 64, 'M', 128,0.4, 128, 'M', 256,0.4, 256,0.4, 256, 'M', 512,0.4, 512,0.4, 512, 'M', 512,0.4, 512,0.4, 512, 'M',0.5],
    'E': [64,0.3, 64, 128,0.4, 128, 256,0.4, 256,0.4, 256, 512,0.4, 512,0.4, 512, 512,0.4, 512,0.4, 512, 0.5]
}

def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_tab(make_layers(cfg['D']), **kwargs)
    return model


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG_tab(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model
    
def vgg16_bn2(inp_s=32,**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG_tab(make_layers2(cfg['E'],input_s=inp_s, batch_norm=True), **kwargs)
    return model