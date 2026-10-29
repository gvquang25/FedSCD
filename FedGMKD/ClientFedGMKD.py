
import numpy as np
import torch
import scipy
from torch.utils.data import Dataset
import torch
import copy
import torch.nn as nn
from sklearn.cluster import KMeans
import torch.optim as optim
import torch.nn.functional as F
from utils import Accuracy,soft_predict
from Client.ClientBase import Client
import gc
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning
class ClientFedGMKD(Client):
    """
    This class is for train the local model with input global model(copied) and output the updated weight
    args: argument 
    Loader_train,Loader_val,Loaders_test: input for training and inference
    user: the index of local model
    idxs: the index for data of this local model
    logger: log the loss and the process
    """
    def __init__(self, args, model, Loader_train,loader_test,idx, logger, code_length, num_classes, device):
        super().__init__(args, model, Loader_train,loader_test,idx, logger, code_length, num_classes, device)
    
    
    def update_weights(self,global_round):
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        optimizer = optim.Adam(self.model.parameters(),lr=self.args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                _,p = self.model(X)
                loss = self.ce(p,y)               
                loss.backward()
                if self.args.clip_grad != None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.args.clip_grad)
                optimizer.step()
                if batch_idx % 10 == 0:
                    print('| Global Round : {} | Client: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, self.idx, iter, batch_idx * len(X),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/ (len(batch_loss) + 1e-8)) ### fungi

        return self.model.state_dict(),sum(epoch_loss) / len(epoch_loss)

    
    def update_weights_GMKD(self,global_features, global_soft_prediction, lam, gamma, temp, global_round):
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        optimizer = optim.Adam(self.model.parameters(),lr=self.args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        tensor_global_features = self.dict_to_tensor(global_features).to(self.device)
        tensor_global_soft_prediction = self.dict_to_tensor(global_soft_prediction).to(self.device)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                F,Z = self.model(X)
                Z_help = self.model.classifier(tensor_global_features)
                Q_help = soft_predict(Z_help,temp).to(self.device)
                loss1 = self.ce(Z,y)
                target_features = copy.deepcopy(F.data)

                
                for i in range(y.shape[0]):
                    if int(y[i]) in global_features.keys():
                        target_features[i] = global_features[int(y[i])][0].data
    
                        
                target_features = target_features.to(self.device)
                if len(global_features) == 0:
                    loss2 = 0*loss1
                    loss3 = 0*loss1
                else:
                    loss2 = self.kld(Q_help.log(),tensor_global_soft_prediction)
                    loss3 = self.mse(F,target_features)
                loss = loss1 + lam*loss2 + gamma*loss3
                loss.backward()
                if self.args.clip_grad != None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.args.clip_grad)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm =1.1)
                optimizer.step()
                if batch_idx % 10 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss1: {:.6f} Loss2: {:.6f}  Loss3: {:.6f} '.format(
                        global_round, iter, batch_idx * len(X),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss1.item(),loss2.item(),loss3.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/ (len(batch_loss) + 1e-8))
                        
        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)


    ###########################################################################################################################
    ### fungi:
    def update_weights_GMKD_DAT(self,global_features, global_soft_prediction, lam, gamma, temp, global_round, pk):
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        optimizer = optim.Adam(self.model.parameters(),lr=self.args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        tensor_global_features = self.dict_to_tensor(global_features).to(self.device)
        tensor_global_soft_prediction = self.dict_to_tensor(global_soft_prediction).to(self.device)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                F,Z = self.model(X)
                Z_help = self.model.classifier(tensor_global_features)
                Q_help = soft_predict(Z_help,temp).to(self.device)
                loss1 = self.ce(Z,y)
                target_features = copy.deepcopy(F.data)
                
                for i in range(y.shape[0]):
                    if int(y[i]) in global_features.keys():
                        target_features[i] = global_features[int(y[i])][0].data
    
                        
                target_features = target_features.to(self.device)
                if len(global_features) == 0:
                    loss2 = 0*loss1
                    loss3 = 0*loss1
                else:
                    loss2 = self.kld(Q_help.log(),tensor_global_soft_prediction)
                    loss3 = self.mse(F,target_features)
                loss = loss1 + lam*loss2 + gamma*loss3
                loss.backward()
                if self.args.clip_grad != None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.args.clip_grad)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm =1.1)
                optimizer.step()
                if batch_idx % 10 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss1: {:.6f} Loss2: {:.6f}  Loss3: {:.6f} '.format(
                        global_round, iter, batch_idx * len(X),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss1.item(),loss2.item(),loss3.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
                        
        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)
    ###########################################################################################################################
    
    # generate knowledge for FedDFKD
    def generate_knowledge(self, temp):
        self.model.to(self.device)
        self.model.eval()        
        local_features = {}
        local_soft_prediction = {}
        num_classes = self.model.num_classes
        features = [torch.zeros(self.code_length).to(self.device)]*num_classes
        soft_predictions = [torch.zeros(num_classes).to(self.device)]*num_classes
        count = [0]*num_classes
        for batch_idx, (X, y) in enumerate(self.trainloader):
            X = X.to(self.device)
            y = y
            F,Z = self.model(X)
            Q = soft_predict(Z,temp).to(self.device)
            m = y.shape[0]
            for i in range(len(y)):
                if y[i].item() in local_features:
                    local_features[y[i].item()].append(F[i,:])
                    local_soft_prediction[y[i].item()].append(Q[i,:])
                else:
                    local_features[y[i].item()] = [F[i,:]]
                    local_soft_prediction[y[i].item()]  = [Q[i,:]] 
            del X
            del y
            del F
            del Z
            del Q
            gc.collect()
            
        features,soft_predictions = self.local_CKF_aggregation(local_features,local_soft_prediction, std = self.args.std)

        return (features, soft_predictions)
    
    def local_CKF_aggregation(self, local_features, local_soft_prediction, std):
        agg_local_features = {}
        agg_local_soft_prediction = {}
        feature_noise = std * torch.randn(self.args.code_len, device=self.device)
        scaler = StandardScaler()
        
        for label, features in local_features.items():
            features_stack = torch.stack(features).detach().cpu().numpy()
            features_normalized = scaler.fit_transform(features_stack)
            
            if len(features) >= 2:
                best_gmm, lowest_bic = None, np.inf
                n_components_range = range(1, min(len(features), 6) + 1)
                for n_components in n_components_range:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        try:
                            gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
                            gmm.fit(features_normalized)
                            bic = gmm.bic(features_normalized)
                            if bic < lowest_bic:
                                best_gmm = gmm
                                lowest_bic = bic
                        except ValueError as e:
                            print(f"Warning: {e}")
                
                if best_gmm is not None:
                    weighted_means = np.dot(best_gmm.weights_, best_gmm.means_)
                    prototype_feature = torch.tensor(weighted_means, dtype=torch.float32, device=self.device).view(features[0].shape)
                else:
                    mean_feature = np.mean(features_normalized, axis=0)
                    prototype_feature = torch.tensor(mean_feature, dtype=torch.float32, device=self.device).view(features[0].shape)
            else:
                prototype_feature = torch.tensor(features_normalized[0], dtype=torch.float32, device=self.device).view(features[0].shape)
            
            feature_noise = std * torch.randn_like(prototype_feature)
            agg_local_features[label] = [prototype_feature + feature_noise]
        # ???????
        for label, soft_prediction in local_soft_prediction.items():
            if len(soft_prediction) > 1:
                soft = (sum(soft_prediction) / len(soft_prediction)).data
            else:
                soft = soft_prediction[0].data
            agg_local_soft_prediction[label] = [soft]

        return agg_local_features, agg_local_soft_prediction
    
    def dict_to_tensor(self, dic):
        lit = []
        for key,tensor in dic.items():
            lit.append(tensor[0])
        lit = torch.stack(lit)
        return lit
    
    #########################################################################################
    ### fungi:
    def get_nk_dk(self):
        # nk = len(self.trainloader.dataset)

        class_dict = {}
                        
        # 使用循环来初始化字典中每个键的值为0
        for i in range(self.args.num_classes):
            class_dict[i] = 0

        for batch_idx, (X, y) in enumerate(self.trainloader):
            for i in range(len(y)):
                class_dict[y[i].item()] += 1

        nk = 0
        for i in range(self.args.num_classes):
            nk += class_dict[i]
        
        return nk, class_dict
    #########################################################################################