import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import scipy
from PIL import Image
from torch.utils.data import Dataset, Subset
import torch
import copy
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder, VisionDataset


class LocalDataset(Dataset):
    """
    because torch.dataloader need override __getitem__() to iterate by index
    this class is map the index to local dataloader into the whole dataloader
    """
    def __init__(self, dataset, Dict):
        self.dataset = dataset
        self.idxs = [int(i) for i in Dict]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        X, y = self.dataset[self.idxs[item]]
        return X, y
    
def LocalDataloaders(dataset, dict_users, batch_size, ShuffleorNot = True, BatchorNot = True, frac = 1):
    """
    dataset: the same dataset object
    dict_users: dictionary of index of each local model
    batch_size: batch size for each dataloader
    ShuffleorNot: Shuffle or Not
    BatchorNot: if False, the dataloader will give the full length of data instead of a batch, for testing
    """
    num_users = len(dict_users)
    loaders = []
    for i in range(num_users):
        num_data = len(dict_users[i])
        frac_num_data = int(frac*num_data)
        whole_range = range(num_data)
        frac_range = np.random.choice(whole_range, frac_num_data)
        frac_dict_users = [dict_users[i][j] for j in frac_range]
        if BatchorNot== True:
            loader = torch.utils.data.DataLoader(
                        LocalDataset(dataset,frac_dict_users),
                        batch_size=batch_size,
                        shuffle = ShuffleorNot,
                        num_workers=0,
                        drop_last=True)
        else:
            loader = torch.utils.data.DataLoader(
                        LocalDataset(dataset,frac_dict_users),
                        batch_size=len(LocalDataset(dataset,dict_users[i])),
                        shuffle = ShuffleorNot,
                        num_workers=0,
                        drop_last=True)
        loaders.append(loader)
    return loaders


def partition_data(n_users, alpha=0.5,rand_seed = 0, dataset = 'cifar10'):
    if dataset == 'OfficeHome':
        data_dir = '../data/OfficceHome'
        apply_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset_full = OfficeHomeDataset(
            root='../data/OfficeHome',
            transform=apply_transform
        )
        train_dataset, test_dataset = split_officehome_by_domain_stratified(dataset_full, train_ratio=0.8)
        test_dataset = test_ds_to_indecies(dataset_full, test_dataset)
        y_train = np.array(train_dataset.dataset.targets)
        y_test = np.array(test_dataset.dataset.targets)
    if dataset == 'CIFAR10':
        K = 10
        data_dir = '../data/cifar10/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                          transform=apply_transform)
        y_train = np.array(train_dataset.targets)
        y_test = np.array(test_dataset.targets)
        
    if dataset == 'CIFAR100':
        K = 100
        data_dir = '../data/cifar100/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                       transform=apply_transform)
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                      transform=apply_transform)
        y_train = np.array(train_dataset.targets)
        y_test = np.array(test_dataset.targets)
        
    if dataset == 'EMNIST':
        K = 62
        data_dir = '../data/EMNIST/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))])
        train_dataset = datasets.EMNIST(data_dir, train=True, split = 'byclass', download=True,
                                       transform=apply_transform)
        test_dataset = datasets.EMNIST(data_dir, train=False, split = 'byclass', download=True,
                                      transform=apply_transform)
        y_train = np.array(train_dataset.targets)
        y_test = np.array(test_dataset.targets)
    if dataset == 'SVHN':
        K = 10
        data_dir = '../data/SVHN/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.SVHN(data_dir, split='train', download=True,
                                       transform=apply_transform)
        test_dataset = datasets.SVHN(data_dir, split='test', download=True,
                                      transform=apply_transform)
        y_train = np.array(train_dataset.labels)
        y_test = np.array(test_dataset.labels)
        
    min_size = 0
    N = len(train_dataset)
    N_test = len(test_dataset)
    net_dataidx_map = {}
    net_dataidx_map_test = {}
    np.random.seed(rand_seed)
   
    while min_size < 10:
        idx_batch = [[] for _ in range(n_users)]
        idx_batch_test = [[] for _ in range(n_users)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            idx_k_test = np.where(y_test == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_users))
            ## Balance
            proportions_train = np.array([p*(len(idx_j)<N/n_users) for p,idx_j in zip(proportions,idx_batch)])
            proportions_test = np.array([p*(len(idx_j)<N_test/n_users) for p,idx_j in zip(proportions,idx_batch_test)])
            proportions_train = proportions_train/proportions_train.sum()
            proportions_test = proportions_test/proportions_test.sum()
            proportions_train = (np.cumsum(proportions_train)*len(idx_k)).astype(int)[:-1]
            proportions_test = (np.cumsum(proportions_test)*len(idx_k_test)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions_train))]
            idx_batch_test = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch_test,np.split(idx_k_test,proportions_test))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_users):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
        net_dataidx_map_test[j] = idx_batch_test[j]
   
        
    return (train_dataset, test_dataset,net_dataidx_map, net_dataidx_map_test)



def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    return net_cls_counts
class OfficeHomeDataset(VisionDataset):
    def __init__(self, root, domains=None, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        if domains is None:
            domains = ['Art', 'Clipart', 'Product', 'Real World']

        # 获取所有类别（从任意一个域取，假设类别一致）
        first_domain_path = os.path.join(root, domains[0])
        classes = sorted([d for d in os.listdir(first_domain_path)
                          if os.path.isdir(os.path.join(first_domain_path, d))])
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        # 遍历所有域，收集图像路径和标签
        imgs = []
        for domain in domains:
            domain_path = os.path.join(root, domain)
            for class_name in classes:
                class_path = os.path.join(domain_path, class_name)
                if not os.path.isdir(class_path):
                    continue
                for img_name in sorted(os.listdir(class_path)):
                    if img_name.lower().endswith(('jpg', 'jpeg', 'png')):
                        imgs.append((os.path.join(class_path, img_name), class_to_idx[class_name]))

        self.imgs = imgs
        self.samples = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.targets = [label for _, label in imgs]

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)


def officehome_domain_split(dataset: ImageFolder, num_users: int = 4) -> Dict[int, List[int]]:
    """
    将OfficeHome数据集按域划分给4个客户端，每个客户端拥有一个完整域
    :param dataset: OfficeHome数据集 (ImageFolder或Subset对象)
    :param num_users: 客户端数量 (必须为4)
    :return: dict of image index (relative to dataset), key: client_id (0-3), value: list of image indices
    """
    if num_users != 4:
        raise ValueError("For OfficeHome, num_users must be exactly 4 (one per domain).")

    # 处理Subset对象（获取原始数据集和索引）
    if isinstance(dataset, Subset):
        original_dataset = dataset.dataset
        indices = dataset.indices  # 训练集的原始索引列表
    else:
        original_dataset = dataset
        indices = list(range(len(dataset)))

    # 创建原始索引到train_ds相对索引的映射
    orig_to_rel = {orig_idx: rel_idx for rel_idx, orig_idx in enumerate(indices)}

    # 获取每个图像的域信息
    domain_list = []
    for idx in indices:
        img_path = original_dataset.imgs[idx][0]
        domain = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
        domain_list.append(domain)

    # 按域分组
    domain_to_indices = defaultdict(list)
    for i, domain in enumerate(domain_list):
        domain_to_indices[domain].append(indices[i])  # 存储原始索引

    # 定义四个域
    DOMAINS = ['Art', 'Clipart', 'Product', 'Real World']

    # 为每个客户端分配一个域（返回相对索引）
    dict_users = {}
    for client_id, domain in enumerate(DOMAINS):
        # 获取该域的所有原始索引
        orig_indices = domain_to_indices.get(domain, [])
        # 转换为train_ds的相对索引
        rel_indices = [orig_to_rel[orig_idx] for orig_idx in orig_indices]
        dict_users[client_id] = rel_indices

    return dict_users

def split_officehome_by_domain_stratified(dataset: ImageFolder, train_ratio=0.8):
    """
    按域分层划分 OfficeHome：每个域内按类别 80/20 划分
    :param dataset: ImageFolder (必须是完整 OfficeHome，65 类)
    :param train_ratio: 训练集比例 (默认 0.8)
    :return: train_dataset (Subset), test_dataset (Subset)
    """
    # 1. 按 (domain, class) 分组图像索引
    domain_class_to_indices = defaultdict(list)

    for idx, (img_path, label) in enumerate(dataset.imgs):
        # 提取域 (e.g., 'Art')
        domain = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
        class_name = dataset.classes[label]
        domain_class_to_indices[(domain, class_name)].append(idx)

    train_indices = []
    test_indices = []

    # 2. 对每个 (domain, class) 组进行分层划分
    for (domain, class_name), indices in domain_class_to_indices.items():
        n_total = len(indices)
        n_train = int(n_total * train_ratio)

        # 确保至少有一个样本在训练和测试中（防止单样本类别）
        if n_total == 1:
            train_indices.extend(indices)
        else:
            train_part = indices[:n_train]  # 取前 n_train 个（或 random.sample）
            test_part = indices[n_train:]
            train_indices.extend(train_part)
            test_indices.extend(test_part)

    # 3. 创建 Subset
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    print(f"Train set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    return train_dataset, test_dataset


def test_ds_to_indecies(dataset_full,test_ds):
    """
        将测试集按域拆分成4个独立的测试集

        参数:
            dataset_full: 原始OfficeHomeDataset (包含所有数据)
            test_ds: 已划分的测试集 (Subset类型)

        返回:
            domain_test_datasets: 字典 {domain_name: Subset}
            key: 域名 ('Art', 'Clipart', 'Product', 'Real World')
            value: 该域名对应的测试集 Subset
        """
    # 域名映射：将实际路径中的 'Real_World' 映射为 'Real World'
    DOMAIN_MAPPING = {
        'Real_World': 'Real World'
    }

    # 步骤1: 提取每个样本的域信息
    domain_to_indices = defaultdict(list)

    for idx_in_test in test_ds.indices:
        # 获取原始图像路径
        img_path, _ = dataset_full.imgs[idx_in_test]

        # 提取域: 从路径中获取二级目录 (e.g., 'Art', 'Real_World')
        domain = os.path.basename(os.path.dirname(os.path.dirname(img_path)))

        # 应用域名映射
        mapped_domain = DOMAIN_MAPPING.get(domain, domain)
        domain_to_indices[mapped_domain].append(idx_in_test)

    # 步骤2: 创建每个域的独立测试集
    domain_test_datasets = {}
    for domain in ['Art', 'Clipart', 'Product', 'Real World']:
        if domain in domain_to_indices:
            domain_test_datasets[domain] = Subset(dataset_full, domain_to_indices[domain])
        else:
            # 如果该域在测试集中不存在，创建一个空的Subset
            domain_test_datasets[domain] = Subset(dataset_full, [])

    return domain_test_datasets