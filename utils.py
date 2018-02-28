import numpy as np

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T


CIFAR_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck') 

def imshow(img, label):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.title(CIFAR_CLASSES[label])
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def cifarloader():
    transform_normalize = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = dset.CIFAR10('./dataset', train=True, download=True,
                           transform=transform_normalize)
    trainloader = DataLoader(trainset, batch_size=20, shuffle=True)
    return trainloader

def count_params(model):
    total = 0
    for param in model.parameters():
        total += np.prod(param.size())
    return total
    