
from torch.utils.data import Dataset
import torch
import copy
from utils import Accuracy
from Server.ServerBase import Server
from Client.ClientFedGMKD import ClientFedGMKD
from tqdm import tqdm
import numpy as np
from utils import average_weights, average_weights_pks
from mem_utils import MemReporter
import time
from sampling import LocalDataset, LocalDataloaders, partition_data
import gc
import torch.nn.functional as F

class ServerFedGMKD(Server):
    def __init__(self, args, global_model,Loader_train,Loaders_local_test,Loader_global_test,logger,device):
        super().__init__(args, global_model,Loader_train,Loaders_local_test,Loader_global_test,logger,device)

    
    def Create_Clints(self):
        for idx in range(self.args.num_clients):
            self.LocalModels.append(ClientFedGMKD(self.args, copy.deepcopy(self.global_model),self.Loaders_train[idx], self.Loaders_local_test[idx], idx=idx, logger=self.logger, code_length = self.args.code_len, num_classes = self.args.num_classes, device=self.device))
            
    def global_knowledge_aggregation(self, features,soft_prediction):
        global_local_features = dict()
        global_local_soft_prediction = dict()
        for [label, features] in features.items():
            if len(features) > 1:
                feature = 0 * features[0].data
                for i in features:
                    feature += i.data
                global_local_features[label] = [feature / len(features)]
            else:
                global_local_features[label] = [features[0].data]

        for [label, soft_prediction] in soft_prediction.items():
            if len(soft_prediction) > 1:
                soft = 0 * soft_prediction[0].data
                for i in soft_prediction:
                    soft += i.data
                global_local_soft_prediction[label] = [soft / len(soft_prediction)]
            else:
                global_local_soft_prediction[label] = [soft_prediction[0].data]

        return global_local_features,global_local_soft_prediction

    def train(self):
        global_features = {}
        global_soft_prediction = {}
        reporter = MemReporter()
        start_time = time.time()
        train_loss = []
        global_weights = self.global_model.state_dict()


        for epoch in tqdm(range(self.args.num_epochs)):
            Knowledges = []
            test_accuracy = 0
            local_weights, local_losses = [], []
            print(f'\n | Global Training Round : {epoch+1} |\n')
            m = max(int(self.args.sampling_rate * self.args.num_clients), 1)
            idxs_users = np.random.choice(range(self.args.num_clients), m, replace=False)
            
            #####################################################################################
            ### 
            
            T_c = []  #样本类占比
            T_c_k = {}  #收集每个客户端，每个类的样本量
            for i in range(self.args.num_classes):
                T_c_k[i] = 0
                
            total_D = 0  #总样本量
            D_kc = []  #客户端的样本类占比
            dks = [] #k个客户端的dk
            nks = [] #k个客户端的nk
            pks = [] #k个客户端的pk
            for idx in idxs_users:
                nk, class_dict = self.LocalModels[idx].get_nk_dk()
                nks.append(nk)
                
                D_kc_idx = []
                for i in range(len(class_dict)):
                    T_c_k[i] += class_dict[i]
                    D_kc_idx.append(class_dict[i]/ (nk + 1e-8)) ### 防止客户端数据集数量为0
                D_kc.append(D_kc_idx)
                total_D += nk


            ### 原文中，T_c是1/C
            T_c = [1/self.args.num_classes for i in range(self.args.num_classes)]
            # print("===============T_c", T_c)


            # print("===========T_c",T_c)
            # # print("===========T_c_k",T_c_k)
            # print("===========D_kc",D_kc)
            # print("===========total_D",total_D)
            
            for idx in idxs_users:
                tmp_a = 0
                for i in range(self.args.num_classes):
                    tmp_a += (D_kc[idx][i] - T_c[i]) ** 2
                dk = tmp_a ** 0.5
                dks.append(dk)

            # print("===========dks",dks)
            param_a = 0.2
            param_b = 0.2

            fenmu = 0
            fenzi = []
            for idx in idxs_users:
                tmp_a = nks[idx] - param_a * dks[idx] + param_b
                tmp_a = max(0, tmp_a)  # relu激活
                fenzi.append(tmp_a)
                fenmu += tmp_a
                
            for idx in idxs_users:
                pks.append(fenzi[idx]/fenmu)

            # print("===========pks",pks)
            ###################################################################################################################################

            local_features_l = []
            local_soft_predictions_l = []
            for idx in idxs_users:
                local_features,local_soft_predictions  = self.LocalModels[idx].generate_knowledge(temp = self.args.temp)
                local_features_l.append(local_features)
                del local_features
                local_soft_predictions_l.append(local_soft_predictions)
                del local_soft_predictions
            
            gc.collect() #垃圾回收
            
            ### 第1轮，计算global_features，和global_soft_prediction
            if epoch == 0:
                ### 计算global_features
                global_features = {key: 0 for key in range(self.args.num_classes)}
                for idx in idxs_users:
                    for key, value in local_features_l[idx].items():
                        if isinstance(global_features[key], torch.Tensor):
                            global_features[key] += value[0] / len(idxs_users)
                        else:
                            global_features[key] = value[0] / len(idxs_users)

                # print("========global_features========", global_features)


                ### 计算global_soft_prediction
                global_soft_prediction = {key: 0 for key in range(self.args.num_classes)}
                for idx in idxs_users:
                    for key, value in local_soft_predictions_l[idx].items():
                        if isinstance(global_soft_prediction[key], torch.Tensor):
                            global_soft_prediction[key] += value[0] / len(idxs_users)
                        else:
                            global_soft_prediction[key] = value[0] / len(idxs_users)



            ### 从第2轮开始，用户根据第一轮得到的global_features和global_soft_prediction，计算自己的d_k_c_features，d_k_c_soft_predictions
            if epoch > 0:
                d_k_c_features_users = []  # 对于特征，每个用户有10个权重，每个类一个权重
                d_k_c_soft_predictions_users = []  # 对于软标签，每个用户有10个权重，每个类一个权重
                
                ### 计算每个用户的“dk”, 每个“dk”包含类数量个dk
                for idx in idxs_users:
                    d_k_c_features = []  # 对于特征，每个用户有10个权重，每个类一个权重
                    d_k_c_soft_predictions = []  # 对于软标签，每个用户有10个权重，每个类一个权重

                    ### 计算当前轮的特征和上一轮的全局特征的 l1 or l2范数
                    for cls in range(self.args.num_classes):
                        tmp_aa = 0
                        if local_features_l[idx].get(cls) is not None:
                            #tmp_aa = torch.norm(local_features_l[idx][cls][0] - global_features[cls][0], p=2) # l2范数
                            # tmp_a = torch.norm(local_features_l[idx][cls][0] - global_features[cls][0], p=1) # l1范数
                            tmp_aa = F.kl_div(F.log_softmax(local_features_l[idx][cls][0]), F.softmax(global_features[cls][0])) # KL散度
                            
                        d_k_c_features.append(tmp_aa)
                    d_k_c_features_users.append(d_k_c_features)

                    ### 计算当前轮的软标签和上一轮的全局软标签的 l1 or l2范数
                    for cls in range(self.args.num_classes):
                        tmp_aa = 0
                        if local_soft_predictions_l[idx].get(cls) is not None:
                            #tmp_aa = torch.norm(local_soft_predictions_l[idx][cls][0] - global_soft_prediction[cls][0], p=2) # l2范数
                            # tmp_a = torch.norm(local_soft_predictions_l[idx][cls][0] - global_soft_prediction[cls][0], p=1) # l1范数
                            tmp_aa = F.kl_div(F.log_softmax(local_soft_predictions_l[idx][cls][0]), F.softmax(global_soft_prediction[cls][0])) # KL散度
                            
                        d_k_c_soft_predictions.append(tmp_aa)
                    d_k_c_soft_predictions_users.append(d_k_c_features)


                param_a = 0.4  ###调参
                param_b = 0.2 ###调参

                fenzi_cls = []
                fenzi_cls_users = []
                fenmu_cls_users = []
                
                ### 计算每个用户的“pk”, 每个“pk”包含类数量个pk
                for cls in range(self.args.num_classes):
                    fenmu_cls = 0
                    ### 计算特征的pk
                    for idx in idxs_users:
                        tmp_a = nks[idx] - param_a * d_k_c_features_users[idx][cls] + param_b
                        tmp_a = max(0, tmp_a)  # relu激活
                        fenzi_cls.append(tmp_a)
                        fenmu_cls += tmp_a
                    fenzi_cls_users.append(fenzi_cls)
                    fenmu_cls_users.append(fenmu_cls)

                
                pk_users_cls = []
                for cls in range(self.args.num_classes):
                    pk_users = []
                    for idx in idxs_users:
                        # print("==========fenzi_cls_users============", fenzi_cls_users.size())
                        pk_cls_idx = fenzi_cls_users[cls][idx] / (fenmu_cls_users[cls] + 1e-8)   # 在第cls类中，用户idx的第cls类的pk
                        pk_users.append(pk_cls_idx) # N个用户的第cls类的pk
                    pk_users_cls.append(pk_users) # [第1个类N个用户的pk,..., 第C个类N个用户的pk]


                ### 根据pk_users_cls计算本轮的全局特征和全局软标签
                ### 计算新的global_features
                global_features = {key: 0 for key in range(self.args.num_classes)}
                for idx in idxs_users:
                    for key, value in local_features_l[idx].items():
                        if isinstance(global_features[key], torch.Tensor):
                            global_features[key] += value[0] * pk_users_cls[key][idx]
                        else:
                            global_features[key] = value[0] * pk_users_cls[key][idx]



                ### 计算新的global_soft_prediction
                global_soft_prediction = {key: 0 for key in range(self.args.num_classes)}
                for idx in idxs_users:
                    for key, value in local_soft_predictions_l[idx].items():
                        if isinstance(global_soft_prediction[key], torch.Tensor):
                            global_soft_prediction[key] += value[0] * pk_users_cls[key][idx]
                        else:
                            global_soft_prediction[key] = value[0] * pk_users_cls[key][idx]

                ### 修正global_features和global_soft_prediction的结构
                for key, value in global_features.items():
                    global_features[key] = [global_features[key]]
                for key, value in global_soft_prediction.items():
                    global_soft_prediction[key] = [global_soft_prediction[key]]

                            
            
            
            # ############################ ： 这里是给特征和软标签做pk加权的地方    
            # # 初始化存储加权总和的字典
            # weighted_local_features = {}

            # # 遍历所有的字典
            # iii = 0
            # for d in local_features_l:
            #     # 遍历当前字典的所有键值对
            #     for key, value in d.items():
            #         # print("=====key, value======",key, value)
            #         # 假设权重为1，如果有权重，这里可以修改相应的权重
            #         weight = pks[iii]
            #         # 是序列，将权重乘以序列中的每个值
            #         weighted_values = [v * weight for v in value]
            #         # 将当前键的加权值加到对应的加权总和上
            #         weighted_local_features[key] = weighted_values
            #     iii += 1


            # global_features.update(weighted_local_features)


            # # 初始化存储加权总和的字典
            # weighted_global_soft_prediction = {}

            # # 遍历所有的字典
            # iii = 0
            # for d in local_soft_predictions_l:
            #     # 遍历当前字典的所有键值对
            #     for key, value in d.items():
            #         # 假设权重为1，如果有权重，这里可以修改相应的权重
            #         weight = pks[iii]
            #         # 是序列，将权重乘以序列中的每个值
            #         weighted_values = [v * weight for v in value]
            #         # 将当前键的加权值加到对应的加权总和上
            #         weighted_global_soft_prediction[key] = weighted_values
            #     iii += 1


            # global_soft_prediction.update(weighted_global_soft_prediction)

        
            


            for idx in idxs_users:
                if self.args.upload_model == True:
                    self.LocalModels[idx].load_model(global_weights)
                if epoch < 1:        
                    w, loss = self.LocalModels[idx].update_weights(global_round=epoch)
                    local_losses.append(copy.deepcopy(loss))
                    local_weights.append(copy.deepcopy(w))
                    acc = self.LocalModels[idx].test_accuracy()
                    test_accuracy += acc

                
                else:
                    w, loss = self.LocalModels[idx].update_weights_GMKD(global_round=epoch, global_features=global_features, global_soft_prediction=global_soft_prediction, lam = self.args.lam, gamma = self.args.gamma, temp = self.args.temp)
                    local_losses.append(copy.deepcopy(loss))
                    local_weights.append(copy.deepcopy(w))
                    acc = self.LocalModels[idx].test_accuracy()
                    test_accuracy += acc

            
            ########################################################
            global_weights = average_weights_pks(local_weights, pks)
            ##############################################################

            # global_weights = average_weights(local_weights) # 原代码


            self.global_model.load_state_dict(global_weights)
            
            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)
            print("average loss:  ", loss_avg)
            print('average local test accuracy:', test_accuracy / self.args.num_clients)
            print('global test accuracy: ', self.global_test_accuracy())
            
        print('Training is completed.')
        end_time = time.time()
        print('running time: {} s '.format(end_time - start_time))
        reporter.report()
