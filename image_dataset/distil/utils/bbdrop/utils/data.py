import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

root = os.path.join('/home', os.environ.get('USER'), 'datasets')


def get_MNIST(batch_size):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(os.path.join(root, 'mnist'),
                                   train=True, download=True,
                                   transform=transform_train)
    test_dataset = datasets.MNIST(os.path.join(root, 'mnist'),
                                  train=False, download=True,
                                  transform=transform_test)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(0.9 * len(train_dataset)),
                                                                               len(train_dataset) - int(
                                                                                   0.9 * len(train_dataset))])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size, num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader


def get_FashionMNIST(batch_size):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.2860,), std=(0.3530,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.2860,), std=(0.3530,))
    ])
    train_dataset = datasets.FashionMNIST(os.path.join(root, 'fashionmnist'),
                                   train=True, download=True,
                                   transform=transform_train)
    test_dataset = datasets.FashionMNIST(os.path.join(root, 'fashionmnist'),
                                  train=False, download=True,
                                  transform=transform_test)

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(0.9 * len(train_dataset)),
                                                                               len(train_dataset) - int(
                                                                                   0.9 * len(train_dataset))])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size, num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader


def get_SVHN(batch_size):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    ])

    train_dataset = datasets.SVHN(os.path.join(root, 'svhn'),
                      split="train", download=True,
                      transform=transform_train)
    test_dataset = datasets.SVHN(os.path.join(root, 'svhn'),
                      split="test", download=True,
                      transform=transform_test)

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(0.9 * len(train_dataset)),
                                                                               len(train_dataset) - int(
                                                                                   0.9 * len(train_dataset))])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size, num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader


def get_CIFAR10(batch_size):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    train_dataset = datasets.CIFAR10(os.path.join(root, 'cifar10'),
                         train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(os.path.join(root, 'cifar10'),
                         train=False, download=True, transform=transform_test)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(0.9 * len(train_dataset)),
                                                                               len(train_dataset) - int(
                                                                                   0.9 * len(train_dataset))])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size, num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader


def get_CIFAR100(batch_size):
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761])
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(os.path.join(root, 'cifar100'),
                          train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(os.path.join(root, 'cifar100'),
                          train=False, download=True, transform=transform),
        batch_size=batch_size, num_workers=4, pin_memory=True)
    return train_loader, test_loader
