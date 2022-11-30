#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for experiments using SCross and Plug-In
"""
import pandas as pd
import numpy as np
from optuna.samplers import TPESampler
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,roc_auc_score, brier_score_loss
from attributes import *
import lightgbm as lgb
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from cmeta import *
from time import time
import multiprocessing
from joblib import Parallel, delayed
import argparse
import datetime
from SelectiveClassifier import *

sampler = TPESampler(seed=42) 
np.random.seed(42)


def coverage_and_res(classifier, level, y, y_scores, y_pred, bands, perc_train = 0.2):
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


def read_datasets(file, atts_dict, data_fold='data/'):
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

    """
    train = pd.read_pickle('{}{}_train.pkl'.format(data_fold,file))
    test = pd.read_pickle('{}{}_test.pkl'.format(data_fold,file))
    if (file=='adultNMNOH')|(file=='adultNM'):
        train.age = train.age.astype(float)
        test.age = test.age.astype(float)
        for col in ['education-num', 'capital-gain', 'capital-loss', 'hours-per-week','age']:
            train[col] = train[col].astype(float)
            test[col] = test[col].astype(float)        
    if file =='CSDS3':
            col = 'TARGET'
            train[col] = train[col].astype(np.int64)
            test[col] = test[col].astype(np.int64)
    if file =='CSDS2':
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
 
    X_train = train[atts_dict[file]]
    X_test = test[atts_dict[file]]
    y_train = train['TARGET'].values 
    y_test = test['TARGET'].values
    return X_train, X_test, y_train, y_test



def get_results(file, atts_dict, default = True, file_params='default.csv', 
                metas=['scross', 'plugin'], classifier_string='lgbm', max_ep = 300,
                boot_iter=1000, cv=5, scaler_dict ={'CSDS1':False,'CSDS2':False,'CSDS3':False,'GiveMe':True,'adultNMNOH':True, 'adultNM':True,'UCI_credit':True,'lending':True,'lendingNOH':True},
                batch_dict = {'adultNM':32, 'adultNMNOH':32, 'UCI_credit':32, 'lendingNOH':512,'lending':512, 'CSDS1':128,'CSDS2':128,'CSDS3':128,'GiveMe':128}):
    """
    

    Parameters
    ----------
    file : str
        the name of the dataset.
    atts_dict : dict
        A dictionary containing all the features names for a specific dataset.
    default : boolean, optional
        Boolean value to determine whether the models are with 
        default parameters. The default is True.
    file_params : str, optional
        A file containing model parameters. The default is 'default.csv'.
    metas : list, optional
        List of selective classifiers. The default is ['scross','plugin']. Other metas are 'sat' for Self Adaptive Training.
    classifier_string : str, optional
        The name of base model to use. Possible choices are:
           - 'lgbm' for LightGBM;
           - 'logistic' for Logistic Regression;
           - 'xgboost' for XGBoost Classifier;
           - 'rf' for Random Forest Classifier
           - 'resnet' for ResNet Classifier
        The default is 'lgbm'.
    boot_iter : int, optional
        The number of bootstrap iterations. The default is 1000.
    cv : int, optional
        The number of folds. The default is 5.
    scaler_dict : dict, optional
        A dictionary used for ResNet classifier specifying whether to perform scaling of inputs before the training.
    batch_dict : dict, optional
        A dictionary used for ResNet classifier specifying batch size for each of the datasets.

    Returns
    -------
    results : pd.DataFrame
        a dataframe with results over (bootstrapped) test set.

    """
    results = pd.DataFrame()
    for meta in metas:
        X_train, X_test, y_train, y_test = read_datasets(file, atts_dict)
        model_string = classifier_string+"_"+meta
        if default==False:
            model_params = pd.read_csv(file_params)
            dict_params = model_params[(model_params['model']==model_string) & (model_params['file']==file)].iloc[0].to_dict()
            dict_params.pop('model')
            dict_params.pop('file')
            dict_params.pop('Unnamed: 0')
        else:
            dict_params = {}
        if classifier_string=='lgbm':
            clf_base = lgb.LGBMClassifier(random_state=42, **dict_params)
        elif classifier_string=='lgbmAUC':
            clf_base = lgb.LGBMClassifier(random_state=42, **dict_params)
        elif (classifier_string=='resnet')&(meta !='sat']):
            if file in ['CSDS1', 'lending', 'GiveMe']:
                clf_base = ResNetClassifier('{}2'.format(classifier_string.replace("sat_","")), max_epochs=max_ep, td=True, scaler_bool=scaler_dict[file], n_batch=batch_dict[file])
            else:
                clf_base = ResNetClassifier(classifier_string.replace("sat_",""), max_epochs=max_ep, td=True, scaler_bool=scaler_dict[file], n_batch=batch_dict[file])
            print(clf_base.max_epochs) 
        elif (meta=='sat')&(classifier_string=='resnet'):
            if file in ['CSDS1', 'lending', 'GiveMe']:
                clf = SelfAdaptiveClassifier('{}2'.format(classifier_string.replace("sat_","")), max_epochs=max_ep, td=True, scaler_bool=scaler_dict[file], n_batch=batch_dict[file])
            else:
                clf = SelfAdaptiveClassifier(classifier_string.replace("sat_",""), max_epochs=max_ep, td=True, scaler_bool=scaler_dict[file], n_batch=batch_dict[file])
            print(clf.max_epochs) 
        if classifier_string=='logistic':
                numeric_transformer = Pipeline(
                            steps=[("scaler", MaxAbsScaler())]
                            )
                categorical_transformer = OneHotEncoder(handle_unknown="ignore")
                preprocessor = ColumnTransformer(
                                    transformers=[
                                        ("num", numeric_transformer, 
                                         selector(dtype_exclude="category")),
                                        ("cat", categorical_transformer,
                                         selector(dtype_include="category")),
                                    ]
                                )
                preprocessor.fit(X_train)
                X_train = preprocessor.transform(X_train)
                X_test = preprocessor.transform(X_test)
                clf_base = LogisticRegression(**dict_params, solver='liblinear', random_state=42)
        if classifier_string=='rf':
            clf_base = RandomForestClassifier(**dict_params, random_state=42) 
        if classifier_string=='xgboost':
            clf_base = XGBClassifier(**dict_params, random_state=42)

        print(cv)    
        elif meta=='scross':
            clf = SCross(clf_base, quantiles=[0.01,0.05,0.1,0.15,0.2,0.25], cv=cv)
        elif meta=='plugin':
            clf = PlugInRule(clf_base, quantiles=[0.01,0.05,0.1,0.15,0.2,0.25], seed=42)          
        start_time = time()
        clf.fit(X_train, y_train)
        end_time = time()
        time_to_fit = (end_time - start_time)
        scores = clf.predict_proba(X_test)[:,1]
        preds = clf.predict(X_test)
        bands = clf.qband(X_test)
        yy = pd.DataFrame(np.c_[y_test,scores, preds, bands], columns = ['true', 'scores','preds', 'bands'])
        if boot_iter==0:

            res = [coverage_and_res(clf, level, y_test, scores, preds, bands, perc_train = y_train.mean()) 
                       for level in range(len(clf.quantiles)+1)]
            tmp = pd.DataFrame(res, columns = ['coverage', 'auc', 
                                               'accuracy', 'brier', 'bss', 'perc_pos'])    
            tmp['desired_coverage'] = [1, 0.99, 0.95, 0.9, 0.85, 0.80, 0.75]
            tmp['time_to_fit'] = [time_to_fit for i in range(7)]
            tmp['dataset'] = file
            tmp['model'] = '{}_{}'.format(meta, classifier_string)
            list_thetas = [(0.5,0.5)]
            if meta =='cross':
                list_thetas_app = clf.thetas
            elif meta=='scross':
                list_thetas_app = clf.thetas
            elif meta=='scross3':
                list_thetas_app = clf.thetas
            elif meta == 'pluginAUC':
                list_thetas_app = clf.thetas
            elif meta =='plugin':
                list_thetas_app = list(zip([1-el for el in clf.thetas], clf.thetas))
            elif meta =='plugin2':
                list_thetas_app = list(zip([1-el for el in clf.thetas], clf.thetas))
            else:
                #list_thetas_app = list(zip(1-clf.thetas, clf.thetas))
                list_thetas_app = clf.thetas
            new_list = list_thetas + (list_thetas_app)
            tmp['thetas'] = new_list
            results = pd.concat([results, tmp], axis=0)
            
        else:
            for b in range(boot_iter+1):
                if b==0:
                    db = yy.copy()
                else:
                    db = yy.sample(len(y_test), random_state=b, replace=True)
                db = db.reset_index()
                res = [coverage_and_res(clf, level, db['true'].values, db['scores'].values, db['preds'].values, db['bands'].values, perc_train = y_train.mean()) 
                           for level in range(len(clf.quantiles)+1)]
                tmp = pd.DataFrame(res, columns = ['coverage', 'auc', 
                                                   'accuracy', 'brier', 'bss', 'perc_pos'])    
                tmp['desired_coverage'] = [1, 0.99, 0.95, 0.9, 0.85, 0.80, 0.75]
                tmp['dataset'] = file
                tmp['model'] = '{}_{}'.format(meta, classifier_string)
                tmp['time_to_fit'] = [time_to_fit for i in range(7)]
                tmp['boot_iter'] = b
                list_thetas = [(0.5,0.5)]
                if meta =='cross':
                    list_thetas_app = clf.thetas
                elif meta=='scross':
                    list_thetas_app = clf.thetas
                elif meta=='scross3':
                    list_thetas_app = clf.thetas
                elif meta == 'pluginAUC':
                    list_thetas_app = clf.thetas
                elif meta =='plugin':
                    list_thetas_app = list(zip([1-el for el in clf.thetas], clf.thetas))
                elif meta =='plugin2':
                    list_thetas_app = list(zip([1-el for el in clf.thetas], clf.thetas))
                else:
                    #list_thetas_app = list(zip(1-clf.thetas, clf.thetas))
                    list_thetas_app = clf.thetas
                new_list = list_thetas + (list_thetas_app)
                tmp['thetas'] = new_list
                results = pd.concat([results, tmp], axis=0)
    return results

if __name__=='__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    atts_dict = {'lending': atts_lending, 'lendingNOH': atts_lendingNOH, 'CSDS1': atts_CSDS1, 'CSDS2':atts_CSDS2, 'CSDS3':atts_CSDS3, 
                 'GiveMe':atts_giveme, 'adultNM':atts_adult, 'adultNMNOH':atts_adultNOH, 'UCI_credit' : atts_credit}
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--n_jobs', type=int, required=False, default=1)
    parser.add_argument('--boot_iter', type=int, required=False, default=1000)
    parser.add_argument('--max_epochs', type=int, required=False, default=100)
    parser.add_argument('--n_batch', type=int, required=False, default=32)
    parser.add_argument('--filelist', type=list, default = [ 'adultNM', 'UCI_credit', 'GiveMe', 'CSDS1', 'CSDS2', 'CSDS3', 'lending'])
    parser.add_argument('--metas', type=list, default = ['scross','plugin'])
    parser.add_argument('--mpfile', type=str, required=False, default='default_params.csv')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--default', type=bool, default=True)
    parser.add_argument('--given', type=bool, default=True)
    parser.add_argument('--cv', type=int, default=5)
    
    # Parse the argument
    args = parser.parse_args()
    classifier_string = args.model
    filelist = args.filelist
    print(classifier_string)
    n_jobs = args.n_jobs
    m_ep = args.max_epochs
    n_ba = args.n_batch
    pat_int = args.patience
    boot_iter = args.boot_iter
    results = pd.DataFrame()
    default = args.default
    file_params = args.mpfile
    given = args.given
    cv = args.cv
    metas = args.metas
    if n_jobs==1:
        for file in tqdm(filelist):
            print(file)
            tmp = get_results(file, atts_dict, file_params, classifier_string=classifier_string, boot_iter=boot_iter, cv=cv, max_ep=m_ep, default=default, metas=metas)
            results = pd.concat([results, tmp], axis=0)
            class_file = 'SCROSSPAPER_results_real_data_{}_{}_{}_{}_{}.csv'.format(classifier_string, boot_iter, default, given, cv)
            results.to_csv(class_file)
    else:
        tmp = Parallel(n_jobs=n_jobs, verbose = 0)([delayed(get_results)(file, atts_dict, file_params, cv=cv,
                                                                                 classifier_string=classifier_string, max_ep=m_ep,
                                                                                 boot_iter=boot_iter, metas=metas) for file in tqdm(filelist)])
        results = pd.concat(tmp, axis=0)
        class_file = 'SCROSSPAPER_results_real_data_{}_{}_{}_{}_{}.csv'.format(classifier_string, boot_iter, default, given, cv)
        results.to_csv(class_file)
