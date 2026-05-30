import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_loaders(batch_size=128, pin_memory=True):
    """
    加载 CIFAR-10 数据集，返回训练集和测试集的 DataLoader。
    图像仅进行 ToTensor 转换，值域为 [0, 1]。

    参数:
        batch_size : int, 每个批次的样本数量，默认 128。
        pin_memory : bool, 是否使用锁页内存（GPU 训练时建议 True）。

    返回:
        train_loader, test_loader
    """
    num_workers = 2
    transform = transforms.ToTensor()

    train_set = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return train_loader, test_loader