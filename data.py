from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader


def CIFAR_loaders(dataset, train_batch_size=50000, test_batch_size=10000):
    transform = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        dataset(root='./data/', train=True,
                download=True,
                transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        dataset(root='./data/', train=False,
                download=True,
                transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader