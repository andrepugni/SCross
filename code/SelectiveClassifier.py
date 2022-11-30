from utils import *
from sklearn.base import BaseEstimator

torch.manual_seed(42)
np.random.seed(42)


class SelfAdaptiveClassifier(BaseEstimator):
    def __init__(self,
        base_model_string,
        model_params={},
        scaler_bool=True,
        pretrain = 0,
        n_batch=1024,
        max_epochs=300,
        td=False,
        lr=.1,
        quantiles=[.01, .05, .10, .15, .20, .25],
        gamma=0.5,
        crit='sat'):
        super(SelfAdaptiveClassifier, self).__init__()
        self.base_model_string = base_model_string
        self.model_params = model_params
        self.scaler_bool=scaler_bool
        self.pretrain = pretrain
        self.n_batch = n_batch
        self.max_epochs = max_epochs
        self.td = td
        self.lr = lr
        self.quantiles = quantiles
        self.gamma = gamma
        self.crit = crit
        if base_model_string=='resnet':
            self.base_model = AdapResNetOne
        elif base_model_string=='resnet2':
            self.base_model = AdapResNetOneV2
        
            
    def fit(
        self,
        X,
        y,
        eval_=False,
        optimizer = torch.optim.SGD,
        seed = 42
        ):
        save_string = "{}_default".format(self.crit)
        torch.manual_seed(seed)
        emb_cols = [col for col in X.columns if X[col].dtype!=float]
        not_emb_cols = [col for col in X.columns if (X[col].dtype==float)]
        embedded_cols = {col: len(X[col].unique()) for col in emb_cols if len(X[col].unique()) > 2}
        embedded_col_names = list(embedded_cols.keys())
        one_hot_cols = {col: len(X[col].unique()) for col in emb_cols if len(X[col].unique()) <= 2}
        one_hot_col_names = list(one_hot_cols.keys())
        embedding_sizes = [(n_categories, min(50, (n_categories+1)//2)) for _,n_categories in embedded_cols.items()]
        print(not_emb_cols, len(not_emb_cols))
        print(one_hot_cols, len(one_hot_cols))
        self.emb_cols = emb_cols
        self.not_emb_cols = not_emb_cols
        self.embedded_cols = embedded_cols
        self.embedded_col_names = embedded_col_names
        self.one_hot_cols = one_hot_cols
        self.one_hot_col_names = one_hot_col_names
        self.embedding_sizes = embedding_sizes
        print(embedding_sizes)
        not_embedded_cols = [col for col in X.columns if col not in embedded_col_names]
        self.not_embedded_cols = not_embedded_cols
        X_train, X_hold, y_train, y_hold = train_test_split(X, y, stratify=y, random_state=42, test_size=2000)
        if self.scaler_bool:
            print("scaling data")
            X_train = X_train.copy()
            y_train = y_train.copy()
            scaler = StandardScaler()
            scaler.fit(X_train[not_emb_cols])
            self.scaler = scaler
            X_train.loc[:,not_emb_cols] = scaler.transform(X_train[not_emb_cols])
            X_hold = X_hold.copy()
            y_hold = y_hold.copy()
            X_hold.loc[:,not_emb_cols] = scaler.transform(X_hold[not_emb_cols])
        train_ds = ToTorchDatasetId(X_train, y_train, embedded_col_names, one_hot_col_names)
        train_dl = DataLoader(train_ds, self.n_batch, shuffle=False, drop_last=True)
        #here we create the torch tensors for holdout
        valid_ds = ToTorchDatasetId(X_hold, y_hold, embedded_col_names, one_hot_col_names)
        valid_dl = DataLoader(valid_ds, self.n_batch, shuffle=False, drop_last=True)
        if self.model_params=={}:
            if self.crit=='sat':
                self.model_params = dict(output_dim = 3, n_cont=len(not_emb_cols),
                               n_one_hot=len(one_hot_col_names), embedding_sizes=embedding_sizes, pretrain=self.pretrain)
            elif self.crit=='ce':
                self.model_params = dict(output_dim = 2, n_cont=len(not_emb_cols),
                               n_one_hot=len(one_hot_col_names), embedding_sizes=embedding_sizes, pretrain=self.pretrain)
        model = self.base_model(**self.model_params)
        print(model.pretrain)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        print(self.device)
        optimizer_params = {'params': model.parameters(),'lr':self.lr, 'momentum':0.9, 'nesterov':True}
        opt = optimizer(**optimizer_params)
        num_examp = len(X_train)
        start_time = time()
        ### here we train the model
        model = traindata_sat(device, model, epochs=self.max_epochs, optimizer=opt, train_dl=train_dl,
              td=True, gamma=self.gamma, verbose=True, epochs_lr = [24,49,74,99,124,149,174,199,224,249,274,299], num_examples=num_examp,
              pretrain=self.pretrain, num_classes=self.model_params['output_dim'], sat_momentum=.99, crit=self.crit)
        end_time = time()
        ### here we store the model
        self.model = model
        path_model = 'models/{}_{}_{}_{}.pt'.format(save_string, self.pretrain, self.max_epochs, self.crit)
        torch.save(model.state_dict(), path_model)
        ### here we compute time to fit
        time_to_fit = (end_time - start_time)
        confs_hold = []
        model.eval()
        for batch in valid_dl:
            x_cont, x_cat, x_one, y, indices = batch
            x_cont, x_cat, x_one, y  = x_cont.to(device), x_cat.to(device), x_one.to(device), y.to(device)
            outputs = model(x_cont,x_cat, x_one)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            if self.crit=='sat':
                conf = outputs[:,-1].detach().cpu().numpy().reshape(-1,1)
            else:
                conf = outputs.detach().cpu().numpy()
                conf = np.max(conf, axis=1).reshape(-1,1)
            confs_hold.append(conf)
        select_hold = np.vstack(confs_hold).flatten()
        if self.crit=='sat':
            calibrated_thetas = [np.quantile(select_hold, 1-cov,method='nearest') for cov in sorted(self.quantiles, reverse=True)]
        else:
            calibrated_thetas = [np.quantile(select_hold, cov,method='nearest') for cov in self.quantiles]
        self.thetas = calibrated_thetas
        return self
            
            
    def get_thetas(self, X, quantiles):
        confs_hold = []
        model = self.model
        model.eval()
        for batch in valid_dl:
            x_cont, x_cat, x_one, y, indices = batch
            x_cont, x_cat, x_one, y  = x_cont.to(device), x_cat.to(device), x_one.to(device), y.to(device)
            outputs = model(x_cont,x_cat, x_one)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            if self.crit=='sat':
                conf = outputs[:,-1].detach().cpu().numpy().reshape(-1,1)
            else:
                conf = outputs.detach().cpu().numpy()
                conf = np.max(conf, axis=1).reshape(-1,1)
            confs_hold.append(conf)
        select_hold = np.vstack(confs_hold).flatten()
        if self.crit=='sat':
            calibrated_thetas = [np.quantile(select_hold, 1-cov,method='nearest') for cov in sorted(self.quantiles, reverse=True)]
        else:
            calibrated_thetas = [np.quantile(select_hold, cov,method='nearest') for cov in self.quantiles]
        self.thetas = calibrated_thetas
        return self

    def predict_proba(
        self,
        X,
        binary=True
    ):
        X_test = X.copy()
        if self.scaler_bool:
            print("scaling data")
            X_test.loc[:,self.not_emb_cols] = self.scaler.transform(X[self.not_emb_cols])
        try:
            self.model.eval()
        except:
            import pdb; pdb.set_trace()
        scores = []
        self.model.to(self.device)
        test_ds = ToTorchDatasetPred(X_test, self.embedded_col_names, self.one_hot_col_names)
        test_dl = DataLoader(test_ds, self.n_batch, shuffle=False)
        if binary:
            for x_cont, x_cat,x_one in test_dl:
                x_cont, x_cat, x_one  = x_cont.to(self.device), x_cat.to(self.device), x_one.to(self.device)
                outputs = self.model(x_cont, x_cat, x_one)
                if self.crit=='sat':
                    outputs = torch.nn.functional.softmax(outputs[:,:-1], dim=1)
                else:
                    outputs = torch.nn.functional.softmax(outputs, dim=1)
                score = outputs.detach().cpu().numpy()
                scores.append(score)
            y_hat = np.vstack(scores)
            return y_hat
        
    def predict(
        self,
        X,
        binary=True
    ):
        X_test = X.copy()
        if self.scaler_bool:
            print("scaling data")
            X_test.loc[:,self.not_emb_cols] = self.scaler.transform(X[self.not_emb_cols])
        self.model.eval()
        scores = []
        self.model.to(self.device)
        test_ds = ToTorchDatasetPred(X_test, self.embedded_col_names, self.one_hot_col_names)
        test_dl = DataLoader(test_ds, self.n_batch, shuffle=False)
        if binary:
            for x_cont, x_cat,x_one in test_dl:
                x_cont, x_cat, x_one  = x_cont.to(self.device), x_cat.to(self.device), x_one.to(self.device)
                outputs = self.model(x_cont, x_cat, x_one)
                if self.crit=='sat':
                    outputs = torch.nn.functional.softmax(outputs[:,:-1], dim=1)
                else:
                    outputs = torch.nn.functional.softmax(outputs, dim=1)
                score = outputs.detach().cpu().numpy()
                scores.append(score)
            y_hat = np.vstack(scores)
            preds = np.argmax(y_hat, axis=1)
            return preds
    
    def predict_confs(
        self,
        X,
        level = 0
        ):
        X_test = X.copy()
        if self.scaler_bool:
            print("scaling data")
            X_test.loc[:,self.not_emb_cols] = self.scaler.transform(X[self.not_emb_cols])
        self.model.eval()
        confs = []
        self.model.to(self.device)
        test_ds = ToTorchDatasetPred(X, self.embedded_col_names, self.one_hot_col_names)
        test_dl = DataLoader(test_ds, self.n_batch, shuffle=False)
        for x_cont, x_cat, x_one in test_dl:
            x_cont, x_cat, x_one  = x_cont.to(self.device), x_cat.to(self.device), x_one.to(self.device)
            outputs = self.model(x_cont, x_cat, x_one)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            if self.crit=='sat':
                conf = outputs[:,-1].detach().cpu().numpy().reshape(-1,1)
            else:
                conf = outputs.detach().cpu().numpy()
                conf = np.max(conf, axis=1).reshape(-1,1)
            confs.append(conf)
        sel_f = np.vstack(confs).flatten()
        return sel_f
        
    def qband(self,X):
        if self.crit=='sat':
            band = np.digitize(self.predict_confs(X), sorted(self.thetas, reverse=True), right=True)
        else:
            band = np.digitize(self.predict_confs, self.thetas)
        return band
        

####################################### resnet base #####################################

class ResNetClassifier(BaseEstimator):
    def __init__(self,
        base_model_string,
        model_params={},
        scaler_bool=True,
        pretrain = 0,
        n_batch=1024,
        max_epochs=300,
        td=False,
        lr=.1,
        gamma=0.5,
        crit='ce'):
        super(ResNetClassifier, self).__init__()
        self.base_model_string = base_model_string
        self.model_params = model_params
        self.scaler_bool=scaler_bool
        self.pretrain = pretrain
        self.n_batch = n_batch
        self.max_epochs = max_epochs
        self.td = td
        self.lr = lr
        self.gamma = gamma
        self.crit = crit
        if base_model_string=='resnet':
            self.base_model = AdapResNetOne
        elif base_model_string=='resnet2':
            self.base_model = AdapResNetOneV2
        
            
    def fit(
        self,
        X,
        y,
        eval_=False,
        optimizer = torch.optim.SGD,

        seed = 42
        ):
        save_string = "{}_default".format(self.crit)
        torch.manual_seed(seed)
        emb_cols = [col for col in X.columns if X[col].dtype!=float]
        not_emb_cols = [col for col in X.columns if (X[col].dtype==float)]
        embedded_cols = {col: len(X[col].unique()) for col in emb_cols if len(X[col].unique()) > 2}
        embedded_col_names = list(embedded_cols.keys())
        one_hot_cols = {col: len(X[col].unique()) for col in emb_cols if len(X[col].unique()) <= 2}
        one_hot_col_names = list(one_hot_cols.keys())
        embedding_sizes = [(n_categories, min(50, (n_categories+1)//2)) for _,n_categories in embedded_cols.items()]
        print(not_emb_cols, len(not_emb_cols))
        print(one_hot_cols, len(one_hot_cols))
        self.emb_cols = emb_cols
        self.not_emb_cols = not_emb_cols
        self.embedded_cols = embedded_cols
        self.embedded_col_names = embedded_col_names
        self.one_hot_cols = one_hot_cols
        self.one_hot_col_names = one_hot_col_names
        self.embedding_sizes = embedding_sizes
        print(embedding_sizes)
        not_embedded_cols = [col for col in X.columns if col not in embedded_col_names]
        self.not_embedded_cols = not_embedded_cols
        X_train = X.copy()
        y_train = y.copy()
        if self.scaler_bool:
            print("scaling data")
            scaler = StandardScaler()
            scaler.fit(X_train[not_emb_cols])
            self.scaler = scaler
            X_train.loc[:,not_emb_cols] = scaler.transform(X_train[not_emb_cols])
        train_ds = ToTorchDatasetId(X_train, y_train, embedded_col_names, one_hot_col_names)
        train_dl = DataLoader(train_ds, self.n_batch, shuffle=False, drop_last=True)
        #here we create the torch tensors for holdout
        if self.model_params=={}:
            if self.crit=='ce':
                self.model_params = dict(output_dim = 2, n_cont=len(not_emb_cols),
                               n_one_hot=len(one_hot_col_names), embedding_sizes=embedding_sizes, pretrain=self.pretrain)
        model = self.base_model(**self.model_params)
        print(model.pretrain)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        print(self.device)
        optimizer_params = {'params': model.parameters(),'lr':self.lr, 'momentum':0.9, 'nesterov':True}
        opt = optimizer(**optimizer_params)
        num_examp = len(X_train)
        start_time = time()
        ### here we train the model
        model = traindata_sat(device, model, epochs=self.max_epochs, optimizer=opt, train_dl=train_dl,
              td=True, gamma=self.gamma, verbose=True, epochs_lr = [24,49,74,99,124,149,174,199,224,249,274,299], num_examples=num_examp,
              pretrain=self.pretrain, num_classes=self.model_params['output_dim'], sat_momentum=.99, crit=self.crit)
        end_time = time()
        ### here we store the model
        self.model = model
        path_model = 'models/{}_{}_{}_{}.pt'.format(save_string, self.pretrain, self.max_epochs, self.crit)
        torch.save(model.state_dict(), path_model)
        ### here we compute time to fit
        time_to_fit = (end_time - start_time)
        self.time_to_fit = time_to_fit
        return self
            
            
    def predict_proba(
        self,
        X,
        binary=True
    ):
        X_test = X.copy()
        if self.scaler_bool:
            print("scaling data")
            X_test.loc[:,self.not_emb_cols] = self.scaler.transform(X[self.not_emb_cols])
        try:
            self.model.eval()
        except:
            import pdb; pdb.set_trace()
        scores = []
        self.model.to(self.device)
        test_ds = ToTorchDatasetPred(X_test, self.embedded_col_names, self.one_hot_col_names)
        test_dl = DataLoader(test_ds, self.n_batch, shuffle=False)
        if binary:
            for x_cont, x_cat,x_one in test_dl:
                x_cont, x_cat, x_one  = x_cont.to(self.device), x_cat.to(self.device), x_one.to(self.device)
                outputs = self.model(x_cont, x_cat, x_one)
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                score = outputs.detach().cpu().numpy()
                scores.append(score)
            y_hat = np.vstack(scores)
            return y_hat
        
    def predict(
        self,
        X,
        binary=True
    ):
        X_test = X.copy()
        if self.scaler_bool:
            print("scaling data")
            X_test.loc[:,self.not_emb_cols] = self.scaler.transform(X[self.not_emb_cols])
        self.model.eval()
        scores = []
        self.model.to(self.device)
        test_ds = ToTorchDatasetPred(X_test, self.embedded_col_names, self.one_hot_col_names)
        test_dl = DataLoader(test_ds, self.n_batch, shuffle=False)
        if binary:
            for x_cont, x_cat,x_one in test_dl:
                x_cont, x_cat, x_one  = x_cont.to(self.device), x_cat.to(self.device), x_one.to(self.device)
                outputs = self.model(x_cont, x_cat, x_one)
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                score = outputs.detach().cpu().numpy()
                scores.append(score)
            y_hat = np.vstack(scores)
            preds = np.argmax(y_hat, axis=1)
            return preds
    


















































class SelectiveClassifierImg(BaseEstimator):
    def __init__(self,
        model,
        coverage=0.9,
        alpha=0.5,
        n_batch=1024,
        max_epochs=100,
        lr=.1,
        td=True,
        gamma=.5,
        lamda=32,
        pretrain=0,
        quantiles=[.01, .05, .10, .15, .20, .25],
        dev_num='0'
        ):
        super(SelectiveClassifierImg, self).__init__()
        self.model = model
        self.coverage = coverage
        self.alpha = alpha
        self.theta = .5
        self.n_batch = n_batch
        self.max_epochs = max_epochs
        self.gamma = gamma
        self.lr = lr
        self.td=td
        self.lamda=32
        self.pretrain = pretrain
        self.quantiles = quantiles
        self.dev_num = dev_num
        
    def fit(self, X, y, optimizer = torch.optim.SGD):
        if self.coverage==1:
            X_train = X.copy()
            y_train = y.copy()
        else:
            if (self.model.selective==False)&(len(X)>5000):
                X_train, X_hold, y_train, y_hold = train_test_split(X, y, stratify=y, random_state=42, test_size=2000)
            else:
                X_train, X_hold, y_train, y_hold = train_test_split(X, y, stratify=y, random_state=42, test_size=.1)
        if self.model.selective==False:
            train_ds = ToTorchDatasetNPId(X_train, y_train)
        else:
            train_ds = ToTorchDatasetNP(X_train, y_train)
        train_dl = DataLoader(train_ds, self.n_batch, shuffle=False)
        #here we create the torch tensors for holdout
        if self.coverage!=1:
            if self.model.selective==False:
                valid_ds = ToTorchDatasetNPId(X_hold, y_hold)
            else:
                valid_ds = ToTorchDatasetNP(X_hold, y_hold)
            valid_dl = DataLoader(valid_ds, self.n_batch, shuffle=False)
        if self.td == True:
            opt = optimizer(self.model.parameters(),self.lr, weight_decay = 5e-4, momentum=0.9, nesterov=True)
        else:
            opt = optimizer(self.model.parameters(),self.lr, momentum=0.9, nesterov=True)
        num_examp = len(X_train)
        start_time = time()
        device = torch.device('cuda:{}'.format(self.dev_num) if torch.cuda.is_available() else 'cpu')
        self.device = device
        ### here we train the model
        if self.model.selective==True:
            model = traindata_sel_img(self.device, self.model, epochs=self.max_epochs, optimizer=opt, train_dl=train_dl,
                                      td=self.td, gamma=self.gamma, verbose=True, coverage=self.coverage, lamda=self.lamda,
                                      epochs_lr = [24,49,74,99,124,149,174,199,224,249,274,299])
            end_time = time()
        else:
            model = traindata_sat_img(self.device, self.model, epochs=self.max_epochs, optimizer=opt,
                                      train_dl=train_dl,td=self.td, gamma=self.gamma, verbose=True,
                                      epochs_lr = [24,49,74,99,124,149,174,199,224,249,274,299],
                                      num_examples=num_examp, pretrain=self.pretrain,
                                      num_classes=self.model.output_dim, sat_momentum=.99)
            end_time = time()
        ### here we store the model
        self.model = model
        save_string = 'Image_VGG'
        if self.model.selective==False:
            path_model = 'models/{}_pretrain{}_SAT_mep{}_{}.pt'.format(save_string, self.pretrain, self.max_epochs, 'catsdogs')
        else:
            path_model = 'models/{}_coverage{}_Selective_mep{}_{}.pt'.format(save_string, self.pretrain, self.max_epochs, 'catsdogs')
        torch.save(model.state_dict(), path_model)
        if self.model.selective==False:
            confs_hold = []
            model.eval()
            for batch in valid_dl:
                x_cont, y, indices = batch
                x_cont, y  = x_cont.to(device), y.to(device)
                outputs = model(x_cont)
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                conf = outputs[:,-1].detach().cpu().numpy().reshape(-1,1)
                confs_hold.append(conf)
            select_hold = np.vstack(confs_hold).flatten()
            calibrated_thetas = [np.quantile(select_hold, 1-cov,method='nearest') for cov in sorted(self.quantiles, reverse=True)]
            self.thetas = calibrated_thetas
        elif (self.model.selective==True)&(self.coverage<1):
            confs_hold = []
            model.eval()
            for batch in valid_dl:
                x_cont, y = batch
                x_cont, y  = x_cont.to(device), y.to(device)
                hg, a = model(x_cont)
                conf = hg[:,-1].detach().cpu().numpy().reshape(-1,1)
                confs_hold.append(conf)
            select_hold = np.vstack(confs_hold).flatten()
            calibrated_theta = np.quantile(select_hold, 1-self.coverage,method='nearest')
            self.thetas = calibrated_theta
            
    def predict_proba(
        self,
        X,
        binary=True
    ):
        X_test = X.copy()
        self.model.eval()
        scores = []
        self.model.to(self.device)
        test_ds = ToTorchDatasetNPPred(X_test)
        test_dl = DataLoader(test_ds, self.n_batch, shuffle=False)
        if self.model.selective==False:
            for x_cont in test_dl:
                x_cont  = x_cont.to(self.device)
                outputs = self.model(x_cont)
                outputs = torch.nn.functional.softmax(outputs[:,:-1], dim=1)
                score = outputs.detach().cpu().numpy()
                scores.append(score)
            y_hat = np.vstack(scores)
        else:
            for x_cont in test_dl:
                x_cont  = x_cont.to(self.device)
                hg, aux = self.model(x_cont)
                if self.coverage==1:
                    score = aux.detach().cpu().numpy()
                else:
                    score = hg[:,:-1].detach().cpu().numpy()
                scores.append(score)
            y_hat = np.vstack(scores)
        return y_hat
        
    def predict(
        self,
        X):
        X_test = X.copy()
        self.model.eval()
        scores = []
        self.model.to(self.device)
        test_ds = ToTorchDatasetNPPred(X_test)
        test_dl = DataLoader(test_ds, self.n_batch, shuffle=False)
        if self.model.selective==False:
            for x_cont in test_dl:
                x_cont  = x_cont.to(self.device)
                outputs = self.model(x_cont)
                outputs = torch.nn.functional.softmax(outputs[:,:-1], dim=1)
                score = outputs.detach().cpu().numpy()
                scores.append(score)
            y_hat = np.vstack(scores)
            preds = np.argmax(y_hat, axis=1)

        else:
            for x_cont in test_dl:
                x_cont  = x_cont.to(self.device)
                hg, aux = self.model(x_cont)
                if self.coverage==1:
                    score = aux.detach().cpu().numpy()
                else:
                    score = hg[:,:-1].detach().cpu().numpy()
                scores.append(score)
            y_hat = np.vstack(scores)
            preds = np.argmax(y_hat, axis=1)
        return preds
        
    def predict_confs(
        self,
        X
        ):
        X_test = X.copy()
        self.model.eval()
        confs = []
        self.model.to(self.device)
        test_ds = ToTorchDatasetNPPred(X)
        test_dl = DataLoader(test_ds, self.n_batch, shuffle=False)
        for x_cont in test_dl:
            x_cont  = x_cont.to(self.device)
            if self.model.selective==False:
                outputs = self.model(x_cont)
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                conf = outputs[:,-1].detach().cpu().numpy().reshape(-1,1)
            else:
                hg, aux = self.model(x_cont)
                conf = hg[:,-1].detach().cpu().numpy().reshape(-1,1)
            confs.append(conf)
        sel_f = np.vstack(confs).flatten()
        return sel_f
        
    def qband(self, X):
        if self.model.selective==False:
            band = np.digitize(self.predict_confs(X), sorted(self.thetas, reverse=True), right=True)
            return band
        else:
            band = self.predict_confs(X)
            return np.where(band>self.theta, 1,0)
        
