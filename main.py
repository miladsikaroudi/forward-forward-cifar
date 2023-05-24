import torch

import data
from torchvision.datasets import CIFAR10

import models

if __name__ == "__main__":
    torch.manual_seed(1234)
    train_loader, test_loader = data.CIFAR_loaders(CIFAR10)  # Use CIFAR10 dataset

    net = models.Net([3072, 500, 500])  # Adjust the input dimension for CIFAR datasets
    x, y = next(iter(train_loader))
    x, y = x.cuda(), y.cuda()
    x_pos = models.overlay_y_on_x(x, y)
    rnd = torch.randperm(x.size(0))
    x_neg = models.overlay_y_on_x(x, y[rnd])
    net.train(x_pos, x_neg)

    print('train error:', 1.0 - net.predict(x).eq(y).float().mean().item())

    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.cuda(), y_te.cuda()

    print('test error:', 1.0 - net.predict(x_te).eq(y_te).float().mean().item())
