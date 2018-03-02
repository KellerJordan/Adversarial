import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T


CIFAR_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck')
gpu_dtype = torch.cuda.FloatTensor

def imshow(img, label=None):
    img = img / 2 + 0.5     # unnormalize
    img = np.clip(img, 0, 1)
    if label:
        plt.title(CIFAR_CLASSES[label])
    plt.imshow(np.transpose(img.squeeze(), (1, 2, 0)))
    plt.show()

def get_param_count(model):
    total = 0
    for param in model.parameters():
        total += np.prod(param.size())
    return total

class Interpolator:
    def __init__(self, model, img_var):
        self.img_var = img_var
        self.model = model
    
    def plot(self, g, domain):
        Y = []
        plt.figure(figsize=(6, 6))
        for i in range(10):
            Y.append([])
            for x in domain:
                d = (x * g).cuda()
                self.img_var.data += d
                scores = self.model(self.img_var)
                self.img_var.data -= d
                y = scores.squeeze().data[i]
                Y[i].append(y)
            plt.plot(domain, Y[i], label=CIFAR_CLASSES[i])
        plt.legend(loc='upper left')
        plt.show()
        
class ChunkSampler(sampler.Sampler):
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start
    
    def __iter__(self):
        return iter(range(self.start, self.start+self.num_samples))
    
    def __len__(self):
        return self.num_samples
    
def get_cifar_loaders(num_train=45000, num_val=5000, batch_size=128):
    transform_augment = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4)])
    transform_normalize = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    cifar10_train = dset.CIFAR10('./dataset', train=True, download=True,
                                 transform=T.Compose([transform_augment, transform_normalize]))
    loader_train = DataLoader(cifar10_train, batch_size=batch_size,
                              sampler=ChunkSampler(num_train))
    cifar10_val = dset.CIFAR10('./dataset', train=True, download=True,
                               transform=transform_normalize)
    loader_val = DataLoader(cifar10_train, batch_size=batch_size,
                            sampler=ChunkSampler(num_val, start=num_train))
    cifar10_test = dset.CIFAR10('./dataset', train=False, download=True,
                                transform=transform_normalize)
    loader_test = DataLoader(cifar10_test, batch_size=batch_size)
    
    return loader_train, loader_val, loader_test

class Trainer:
    def __init__(self, model, loader_train, loader_val):
        self.model = model
        self.loader_train = loader_train
        self.loader_val = loader_val
        
        self.criterion = nn.CrossEntropyLoss().cuda()
        
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1,
                                   momentum=0.9, weight_decay=1e-3)
    
    def check_accuracy(self, loader):
        self.model.eval()
        num_correct = 0
        num_samples = 0
        for X, y in loader:
            X_var = Variable(X.type(gpu_dtype), volatile=True)

            scores = self.model(X_var)
            _, preds = scores.data.cpu().max(1)

            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

    def train_epoch(self):
        self.model.train()
        for t, (X, y) in enumerate(self.loader_train):
            X_var = Variable(X.type(gpu_dtype))
            y_var = Variable(y.type(gpu_dtype)).long()

            scores = self.model(X_var)
            loss = self.criterion(scores, y_var)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print('loss = %.4f' % loss.data[0])

    def train_schedule(self, schedule):
        lr = self.optimizer.param_groups[0]['lr']
        for num_epochs in schedule:
            print('Training for %d epochs with learning rate %f' % (num_epochs, lr))
            for epoch in range(num_epochs):
                print('Starting epoch %d / %d' % (epoch+1, num_epochs))
                # update learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                self.train_epoch()
                self.check_accuracy(self.loader_val)
            lr *= 0.1

    def save_checkpoint(self, filename='checkpoint.pth.tar'):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()}
        torch.save(state, filename)

    def load_checkpoint(self, filename='checkpoint.pth.tar'):
        if os.path.isfile(filename):
            state = torch.load(filename)
            self.model.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['optimizer'])
        else:
            print('%s not found.' % filename)
