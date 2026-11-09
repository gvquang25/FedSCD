#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master Dataset Preparation & Dirichlet Partition Script for FedSCD Benchmark.

Supported Datasets:
  1. CIFAR-10    (10 classes,  32x32, 20 clients)
  2. CIFAR-100   (100 classes, 32x32, 20 clients)
  3. Flowers102  (102 classes, 64x64, 20 clients)
  4. TinyImagenet (200 classes, 64x64, 20 clients)

Dirichlet Skew Levels:
  alpha in [0.1, 0.5, 1.0]

Standardized storage format for system/utils/data_utils.py:
  dataset/<DatasetName>/<num_clients>/False_True_<alpha>/train/<client_id>.npz
  dataset/<DatasetName>/<num_clients>/False_True_<alpha>/test/<client_id>.npz
  dataset/<DatasetName>/<num_clients>/False_True_<alpha>/config.json
"""

import os
import sys
import ssl
import random
import zipfile
import urllib.request
import argparse
import ujson
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, DatasetFolder

# Fix SSL certificates on macOS/Windows if needed
ssl._create_default_https_context = ssl._create_unverified_context

# Fixed seeds for reproducibility
SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def separate_dirichlet(dataset_image, dataset_label, num_clients, num_classes, alpha=0.1, least_samples=1):
    """Partition dataset into clients using Dirichlet distribution."""
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]
    dataidx_map = {}

    min_size = 0
    K = num_classes
    N = len(dataset_label)

    attempt = 1
    while min_size < least_samples:
        if attempt > 1 and attempt % 10 == 0:
            print(f"  [Dirichlet Partition] Retrying attempt {attempt} for alpha={alpha}...")
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
        attempt += 1

    for j in range(num_clients):
        dataidx_map[j] = idx_batch[j]
        idxs = dataidx_map[j]
        X[j] = dataset_image[idxs]
        y[j] = dataset_label[idxs]
        for i in np.unique(y[j]):
            statistic[j].append((int(i), int(sum(y[j] == i))))

    return X, y, statistic


def split_data(X, y, train_ratio=0.75):
    """Split client data into 75% train and 25% test sets."""
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


def save_partition(target_dir, train_data, test_data, num_clients, num_classes, statistic, alpha, batch_size=10):
    """Save partition in system/utils/data_utils.py format."""
    config_path = os.path.join(target_dir, "config.json")
    train_path = os.path.join(target_dir, "train")
    test_path = os.path.join(target_dir, "test")

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    config = {
        'num_clients': num_clients,
        'num_classes': num_classes,
        'non_iid': True,
        'balance': False,
        'partition': 'dir',
        'alpha': alpha,
        'batch_size': batch_size,
        'Size of samples for labels in clients': statistic,
    }

    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    for i in range(num_clients):
        with open(os.path.join(train_path, f"{i}.npz"), 'wb') as f:
            np.savez_compressed(f, data=train_data[i])
        with open(os.path.join(test_path, f"{i}.npz"), 'wb') as f:
            np.savez_compressed(f, data=test_data[i])

    print(f"  [SUCCESS] Saved partition to {target_dir}")


def is_partition_existing(target_dir, num_clients):
    """Check if valid partition already exists."""
    config_path = os.path.join(target_dir, "config.json")
    if not os.path.exists(config_path):
        return False
    for i in range(num_clients):
        if not os.path.exists(os.path.join(target_dir, "train", f"{i}.npz")):
            return False
        if not os.path.exists(os.path.join(target_dir, "test", f"{i}.npz")):
            return False
    return True


# =========================================================================
# Dataset Loaders
# =========================================================================

def load_cifar10(raw_dir):
    """Download and load CIFAR-10."""
    print("Loading CIFAR-10 raw dataset...")
    os.makedirs(raw_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root=raw_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=raw_dir, train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)

    train_x, train_y = next(iter(trainloader))
    test_x, test_y = next(iter(testloader))

    images = np.concatenate([train_x.numpy(), test_x.numpy()], axis=0)
    labels = np.concatenate([train_y.numpy(), test_y.numpy()], axis=0)
    print(f"  CIFAR-10 loaded: {images.shape}, Labels: {len(np.unique(labels))} classes")
    return images, labels, 10


def load_cifar100(raw_dir):
    """Download and load CIFAR-100."""
    print("Loading CIFAR-100 raw dataset...")
    os.makedirs(raw_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR100(root=raw_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root=raw_dir, train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)

    train_x, train_y = next(iter(trainloader))
    test_x, test_y = next(iter(testloader))

    images = np.concatenate([train_x.numpy(), test_x.numpy()], axis=0)
    labels = np.concatenate([train_y.numpy(), test_y.numpy()], axis=0)
    print(f"  CIFAR-100 loaded: {images.shape}, Labels: {len(np.unique(labels))} classes")
    return images, labels, 100


def load_flowers102(raw_dir):
    """Download and load Flowers102 resized to 64x64."""
    print("Loading Flowers102 raw dataset (resized to 64x64)...")
    os.makedirs(raw_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    images, labels = [], []
    for split in ["train", "val", "test"]:
        dset = torchvision.datasets.Flowers102(root=raw_dir, split=split, download=True, transform=transform)
        loader = torch.utils.data.DataLoader(dset, batch_size=len(dset), shuffle=False)
        bx, by = next(iter(loader))
        images.append(bx.numpy())
        labels.append(by.numpy())

    all_images = np.concatenate(images, axis=0)
    all_labels = np.concatenate(labels, axis=0)
    num_classes = len(np.unique(all_labels))
    print(f"  Flowers102 loaded: {all_images.shape}, Labels: {num_classes} classes")
    return all_images, all_labels, num_classes


def download_tiny_imagenet(raw_dir):
    """Download and extract Tiny-ImageNet-200 zip if not present."""
    os.makedirs(raw_dir, exist_ok=True)
    extract_path = os.path.join(raw_dir, "tiny-imagenet-200")
    if os.path.exists(extract_path):
        return extract_path

    zip_path = os.path.join(raw_dir, "tiny-imagenet-200.zip")
    if not os.path.exists(zip_path):
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        print(f"Downloading Tiny-ImageNet from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(raw_dir)
    print("Extraction complete.")
    return extract_path


def load_tiny_imagenet(raw_dir):
    """Load Tiny-ImageNet-200 (64x64)."""
    print("Loading Tiny-ImageNet raw dataset...")
    base_dir = download_tiny_imagenet(raw_dir)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_folder = os.path.join(base_dir, "train")
    train_dataset = ImageFolder(train_folder, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2048, shuffle=False)

    images = []
    labels = []
    print("  Processing Tiny-ImageNet train images in batches...")
    for bx, by in train_loader:
        images.append(bx.numpy())
        labels.append(by.numpy())

    # Build validation set mapping
    val_folder = os.path.join(base_dir, "val")
    val_images_dir = os.path.join(val_folder, "images")
    val_annotations = os.path.join(val_folder, "val_annotations.txt")
    
    val_img_to_cls = {}
    with open(val_annotations, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                val_img_to_cls[parts[0]] = parts[1]

    class_to_idx = train_dataset.class_to_idx
    from PIL import Image

    print("  Processing Tiny-ImageNet val images...")
    val_imgs, val_lbls = [], []
    for img_name, cls_name in val_img_to_cls.items():
        img_path = os.path.join(val_images_dir, img_name)
        if os.path.exists(img_path) and cls_name in class_to_idx:
            img = Image.open(img_path).convert('RGB')
            tensor_img = transform(img).numpy()
            val_imgs.append(tensor_img)
            val_lbls.append(class_to_idx[cls_name])

    images.append(np.array(val_imgs))
    labels.append(np.array(val_lbls))

    all_images = np.concatenate(images, axis=0)
    all_labels = np.concatenate(labels, axis=0)
    num_classes = len(np.unique(all_labels))
    print(f"  Tiny-ImageNet loaded: {all_images.shape}, Labels: {num_classes} classes")
    return all_images, all_labels, num_classes


# =========================================================================
# Main Execution Pipeline
# =========================================================================

DATASET_DISPATCH = {
    "Cifar10": ("Cifar10", load_cifar10),
    "Cifar100": ("Cifar100", load_cifar100),
    "Flowers102": ("Flowers102", load_flowers102),
    "TinyImagenet": ("TinyImagenet", load_tiny_imagenet),
}


def process_dataset(ds_key, base_dataset_dir, alphas, num_clients):
    folder_name, load_fn = DATASET_DISPATCH[ds_key]
    dataset_root = os.path.join(base_dataset_dir, folder_name)
    raw_dir = os.path.join(dataset_root, "rawdata")

    # Check if all partitions already exist
    needed_alphas = []
    for alpha in alphas:
        target_dir = os.path.join(dataset_root, str(num_clients), f"False_True_{alpha}")
        if is_partition_existing(target_dir, num_clients):
            print(f"[SKIP] Partition for {ds_key} (alpha={alpha}, clients={num_clients}) already exists at {target_dir}")
        else:
            needed_alphas.append(alpha)

    if not needed_alphas:
        print(f"[COMPLETE] All requested partitions for {ds_key} are already present.\n")
        return

    # Load raw data only once if at least one partition is missing
    print(f"\n>>> Preparing partitions for {ds_key}: missing alphas = {needed_alphas}")
    images, labels, num_classes = load_fn(raw_dir)

    for alpha in needed_alphas:
        print(f"\n--- Generating Dirichlet Non-IID partition for {ds_key} (alpha={alpha}, K={num_clients}) ---")
        target_dir = os.path.join(dataset_root, str(num_clients), f"False_True_{alpha}")
        X, y, statistic = separate_dirichlet(images, labels, num_clients, num_classes, alpha=alpha, least_samples=1)
        train_data, test_data = split_data(X, y)
        save_partition(target_dir, train_data, test_data, num_clients, num_classes, statistic, alpha)


def main():
    parser = argparse.ArgumentParser(description="Prepare and partition benchmark datasets for FedSCD.")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "Cifar10", "Cifar100", "Flowers102", "TinyImagenet"],
                        help="Dataset to process (default: all)")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.5, 1.0],
                        help="Dirichlet alpha levels (default: 0.1 0.5 1.0)")
    parser.add_argument("--num_clients", type=int, default=20,
                        help="Total number of clients (default: 20)")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    base_dataset_dir = os.path.join(project_root, "dataset")
    os.makedirs(base_dataset_dir, exist_ok=True)

    print("=" * 80)
    print("FEDERATED LEARNING BENCHMARK DATASET PREPARATION")
    print(f"Root dataset directory : {base_dataset_dir}")
    print(f"Target datasets        : {args.dataset}")
    print(f"Dirichlet alphas       : {args.alphas}")
    print(f"Number of clients      : {args.num_clients}")
    print("=" * 80)

    datasets_to_run = list(DATASET_DISPATCH.keys()) if args.dataset == "all" else [args.dataset]

    for ds in datasets_to_run:
        process_dataset(ds, base_dataset_dir, args.alphas, args.num_clients)

    print("\n" + "=" * 80)
    print("ALL REQUESTED DATASETS & PARTITIONS ARE READY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
