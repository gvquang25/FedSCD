#!/usr/bin/env python3
"""
Shared CIFAR-10 Partition Script for FedSCD
Generates Dirichlet Non-IID CIFAR-10 partition (20 clients, alpha=0.1)
and creates symlinks for FedSA, FedTGP, and FedORGP.
"""

import os
import sys
import random
import ssl

# Fix macOS Python urllib SSL certificate issue
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import ujson

# Set random seeds
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

def separate_data(data, num_clients, num_classes, alpha=0.1, least_samples=1):
    dataset_content, dataset_label = data
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]
    dataidx_map = {}

    min_size = 0
    K = num_classes
    N = len(dataset_label)

    try_cnt = 1
    while min_size < least_samples:
        if try_cnt > 1:
            print(f'Retrying Dirichlet partition (attempt {try_cnt})...')
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(K):
            idx_k = np.where(dataset_label == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
        try_cnt += 1

    for j in range(num_clients):
        dataidx_map[j] = idx_batch[j]
        idxs = dataidx_map[j]
        X[j] = dataset_content[idxs]
        y[j] = dataset_label[idxs]
        for i in np.unique(y[j]):
            statistic[j].append((int(i), int(sum(y[j] == i))))

    return X, y, statistic

def split_data(X, y, train_ratio=0.75):
    train_data, test_data = [], []
    for i in range(len(y)):
        total_len = len(y[i])
        train_len = max(1, int(total_len * train_ratio))
        indices = np.random.permutation(total_len)
        train_idx = indices[:train_len]
        test_idx = indices[train_len:] if len(indices) > 1 else indices[:1]

        train_data.append({'x': X[i][train_idx], 'y': y[i][train_idx]})
        test_data.append({'x': X[i][test_idx], 'y': y[i][test_idx]})

    return train_data, test_data

def save_data(dir_path, train_data, test_data, num_clients, num_classes, statistic, alpha):
    config_path = os.path.join(dir_path, "config.json")
    train_path = os.path.join(dir_path, "train")
    test_path = os.path.join(dir_path, "test")

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    config = {
        'num_clients': num_clients,
        'num_classes': num_classes,
        'non_iid': True,
        'partition': 'dir',
        'alpha': alpha,
        'Size of samples for labels in clients': statistic,
    }

    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    for i in range(num_clients):
        with open(os.path.join(train_path, f"{i}.npz"), 'wb') as f:
            np.savez_compressed(f, data=train_data[i])
        with open(os.path.join(test_path, f"{i}.npz"), 'wb') as f:
            np.savez_compressed(f, data=test_data[i])

    print(f"Dataset successfully saved to {dir_path}")

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    shared_cifar10_dir = os.path.join(base_dir, "dataset", "Cifar10")
    raw_dir = os.path.join(shared_cifar10_dir, "rawdata")

    num_clients = 20
    num_classes = 10
    alpha = 0.1

    train_chk = os.path.join(shared_cifar10_dir, "train", "0.npz")
    if os.path.exists(train_chk):
        print(f"Partitioned CIFAR-10 data already exists at {shared_cifar10_dir}.")
    else:
        print(f"Generating Dirichlet (alpha={alpha}) CIFAR-10 for {num_clients} clients...")
        os.makedirs(raw_dir, exist_ok=True)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = torchvision.datasets.CIFAR10(root=raw_dir, train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=raw_dir, train=False, download=True, transform=transform)

        dataset_image = np.concatenate([trainset.data, testset.data], axis=0)
        dataset_label = np.concatenate([np.array(trainset.targets), np.array(testset.targets)], axis=0)

        X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, alpha=alpha)
        train_data, test_data = split_data(X, y)
        save_data(shared_cifar10_dir, train_data, test_data, num_clients, num_classes, statistic, alpha)

    # Link to FedSA, FedTGP, FedORGP
    for algo_dir in ["FedSA", "FedTGP", "FedORGP"]:
        target_dataset_dir = os.path.join(base_dir, algo_dir, "dataset")
        os.makedirs(target_dataset_dir, exist_ok=True)
        link_path = os.path.join(target_dataset_dir, "Cifar10")
        if not os.path.exists(link_path):
            os.symlink(shared_cifar10_dir, link_path)
            print(f"Linked {link_path} -> {shared_cifar10_dir}")
        else:
            print(f"Link or folder already exists: {link_path}")

if __name__ == "__main__":
    main()
