from random import random

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import os
import numpy as np
import random

def load_cifar10(
        data_dir='./data',
        batch_size=64,
        shuffle=True,
        train_limit=None,
        test_limit=None,
        random_seed=42,
        resize_to_224=False,
):
    random.seed(random_seed)

    # 根据参数决定是否调整尺寸
    if resize_to_224:
        print("将CIFAR10图像调整为224x224尺寸")
        train_transform = transforms.Compose([
            transforms.Resize(224),  # 调整为224x224
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(224),  # 调整为224x224
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        print("保持CIFAR10原始32x32尺寸")
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    # 限制训练集数据量
    if train_limit is not None:
        if train_limit <= 0 or train_limit > len(train_dataset):
            raise ValueError(f"train_limit 必须在 (0, {len(train_dataset)}] 范围内")
        # 随机选取子集
        indices = torch.randperm(len(train_dataset))[:train_limit]
        train_dataset = Subset(train_dataset, indices)

    # 限制测试集数据量
    if test_limit is not None:
        if test_limit <= 0 or test_limit > len(test_dataset):
            raise ValueError(f"test_limit 必须在 (0, {len(test_dataset)}] 范围内")
        indices = torch.randperm(len(test_dataset))[:test_limit]
        test_dataset = Subset(test_dataset, indices)

    #  创建dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, test_dataset, train_loader, test_loader

# if __name__ == '__main__':
#     train_loader, test_loader = load_cifar10()



