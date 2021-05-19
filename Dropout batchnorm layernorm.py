import logging

logging.basicConfig(level=logging.INFO)

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import click

from datamaestro import prepare_dataset

# utils data
from torchvision import datasets, transforms
from torch.utils.data.sampler import RandomSampler

# Ratio du jeu de train à utiliser
TRAIN_RATIO = 0.05


def store_grad(var):
    """Stores the gradient during backward

    For a tensor x, call `store_grad(x)`
    before `loss.backward`. The gradient will be available
    as `x.grad`

    """

    def hook(grad):
        var.grad = grad

    var.register_hook(hook)
    return var

class NN(nn.Module):
    def __init__(self, in_d, hid_d, out_d, batchnorm=False, layernorm=False, dropout=False):
        super().__init__()
        self.l1 = nn.Linear(in_d, hid_d)
        self.l2 = nn.Linear(hid_d, hid_d)
        self.l3 = nn.Linear(hid_d, out_d)
        self.batchnorml1 = nn.BatchNorm1d(hid_d)
        self.batchnorml2 = nn.BatchNorm1d(hid_d)
        self.layernorml1 = nn.LayerNorm(hid_d)
        self.layernorml2 = nn.LayerNorm(hid_d)
        self.dropoutl1 = nn.Dropout()
        self.dropoutl2 = nn.Dropout()
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        self.dropout = dropout

    def forward(self, x):
        x0 = x.squeeze().view(batch_size, -1)
        x1 = self.l1(x0)
        x = x1

        if self.batchnorm:
            x = self.batchnorml1(x)
        if self.layernorm:
            x = self.layernorml1(x)
        if self.dropout:
            x = self.dropoutl1(x)

        x1 = x
        x2 = self.l2(x1)
        x = x2

        if self.batchnorm:
            x = self.batchnorml1(x)
        if self.layernorm:
            x = self.layernorml1(x)
        if self.dropout:
            x = self.dropoutl1(x)

        x2 = x
        out = self.l3(x2)

        return x0, x1, x2, out


def hist_grad(g1, g2, g3, i):
    writer.add_histogram('g1', g1 + i, i)
    writer.add_histogram('g2', g2 + i, i)
    writer.add_histogram('g3', g3 + i, i)


def hist_weights(mode, i):
    writer.add_histogram(mode + ' w1', NN.l1.weight + i, i)
    writer.add_histogram(mode + ' w2', NN.l2.weight + i, i)
    writer.add_histogram(mode + ' w3', NN.l3.weight + i, i)


def loss_(mode, l, i):
    writer.add_scalars("Loss", {mode + " loss": l}, i)


def accuracy(mode, y_hat, i):
    writer.add_scalars("Accuracy", {mode + " accuracy": (torch.nn.functional.one_hot(torch.argmax(y_hat, 1),
                                                                                     10) == torch.nn.functional.one_hot(
        y, 10)).sum() / (10 * batch_size)}, i)


def entropy(mode, y_hat, y, i):
    e = -(torch.nn.functional.one_hot(y, 10) * torch.log(nn.functional.softmax(y_hat, 1))).sum(1).mean()
    writer.add_scalars("Entropy", {mode + " entropy": e}, i)


#  TODO:  Implémenter
batch_size = 100
percent = 0.05
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_ds = datasets.MNIST('../data', download=False, train=True, transform=transform)
test_ds = datasets.MNIST('../data', download=False, train=False, transform=transform)
train_sampler = RandomSampler(train_ds, replacement=True, num_samples=int(len(train_ds) * percent))
valid_sampler = RandomSampler(train_ds, replacement=True, num_samples=int(len(train_ds) * percent))
test_sampler = RandomSampler(test_ds, replacement=True, num_samples=int(len(train_ds) * percent))
train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, sampler=train_sampler, drop_last=True)
valid_dl = DataLoader(dataset=train_ds, batch_size=batch_size, sampler=valid_sampler, drop_last=True)
test_dl = DataLoader(dataset=test_ds, batch_size=batch_size, sampler=test_sampler, drop_last=True)

# loss = nn.MSELoss()
loss = nn.CrossEntropyLoss()
lr = 1e-4
n_epochs = 3
# NN = NN(784, 100, 10,False, False, False)
# NN = NN(784, 100, 10,True, False, False)
# NN = NN(784, 100, 10,False, True, False)
NN = NN(784, 100, 10,False, False, True)

optim = torch.optim.Adam(params=NN.parameters(), lr=lr, weight_decay=0) # weight_decay parametre L2 regularisation

writer = SummaryWriter()
i = 0
j = 0
k = 0
for n in range(n_epochs):
    for x, y in train_dl:
        print(i)

        NN.train()
        x.requires_grad = True  # sans ca on obtient une erreur, pas de gradient possible

        x0, x1, x2, y_hat = NN(x)
        l = loss(y_hat, y)

        # on doit retenir les gradients durant backward passe
        # on y a accès ensuite
        x0.retain_grad()
        x1.retain_grad()
        x2.retain_grad()
        l.backward()

        mode = "Train"
        loss_(mode, l, i)
        accuracy(mode, y_hat, i)
        hist_weights(mode, i)
        hist_grad(x0.grad, x1.grad, x2.grad, i)
        entropy(mode, y_hat, y, i)

        optim.step()
        optim.zero_grad()

        i += 1

    for x, y in test_dl:
        print(j)

        NN.eval()

        _, _, _, y_hat = NN(x)
        l = loss(y_hat, y)

        mode = "Test"
        loss_(mode, l, j)
        accuracy(mode, y_hat, j)
        hist_weights(mode, j)
        entropy(mode, y_hat, y, j)

        j += 1

    for x, y in valid_dl:
        print(k)

        NN.eval()

        _, _, _, y_hat = NN(x)
        l = loss(y_hat, y)

        mode = "Valid"
        loss_(mode, l, k)
        accuracy(mode, y_hat, k)
        hist_weights(mode, k)
        entropy(mode, y_hat, y, k)

        k += 1

