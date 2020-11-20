import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
#################################################################
# Classes #######################################################
#################################################################
class MonDataset(Dataset):
    def __init__(self, T, number_of_cities, mode):
        self.T = T
        self.number_of_cities = number_of_cities
        # importation des donnees
        if mode=='train':
            df = pd.read_csv('tempAMAL_train.csv')
        else :
            df = pd.read_csv('tempAMAL_test.csv')
        # normalization des donnes
        df = self.normalization(df)
        # conversion pandas to tensor
        self.data = torch.tensor(df.values)

    def normalization(self, df):
        # que les number_of_cities premieres villes
        city_list = list(df.columns)[1:self.number_of_cities + 1]
        df = df.loc[:, city_list]
        # remplir les nan
        df = df.fillna(method='bfill')
        df = df.fillna(method='ffill')
        # normalisation
        max = df.max().max()
        min = df.min().min()
        df = (df - min) / (max - min)
        return df

    def __len__(self):
        # returne les indices verticaux possibles
        return self.data.shape[0] - self.T

    def __getitem__(self, index):
        # a l aide de len un index et choisi
        # puis on tire une ville au hasard
        ville = np.random.randint(self.number_of_cities)
        # on retourne la sequence de temperature de la ville et la ville
        return self.data[index:index + self.T, ville], ville

class RNN(torch.nn.Module):
    def __init__(self, d_in_x, d_h, d_out):
        super(RNN, self).__init__()
        self.Wh = torch.nn.Linear(d_h, d_h)
        self.Wi = torch.nn.Linear(d_in_x, d_h)
        self.Wo = torch.nn.Linear(d_h, d_out)
        self.act_fct_h = torch.nn.Tanh()
        self.act_fct_out = torch.nn.Softmax(dim = 1)
        self.d_h = d_h
        self.d_in_x = d_in_x
        self.d_out = d_out

    def forward(self, seq):
        # paramtres
        self.T = seq[0, :].shape[0]
        self.batch_size = seq[:, 0].shape[0]

        # initialisation
        self.ht = torch.zeros((self.batch_size, self.d_h), requires_grad=True,
                              dtype=torch.float)  # representation h courante
        self.h = torch.zeros((self.batch_size, self.d_h), requires_grad=True,
                             dtype=torch.float)  # tout nos h gardés en mémoire
        self.ot = torch.zeros((self.batch_size, self.d_out), requires_grad=True,
                              dtype=torch.float)  # La sortie courante notre température prédit
        self.o = torch.zeros((self.batch_size, self.d_out), requires_grad=True,
                             dtype=torch.float)  # Toute nos sorties gardées en mémoire
        # one step
        for x in range(self.T):
            self.one_step(seq[:, x].reshape(self.batch_size,self.d_in_x))
        return self.ot

    def one_step(self, x):
        s1 = self.Wh(self.ht)
        s2 = self.Wi(x)
        self.ht = self.act_fct_h(s1 + s2)
        self.h = torch.cat((self.h, self.ht), 1)
        o = self.Wo(self.ht)
        self.ot = self.act_fct_out(o)
        self.o = torch.cat((self.o, self.ot), 1)

def get_classification_error(y_hat, y):
    size = y_hat.shape[0]
    return (y_hat.argmax(1) != y).sum().item() / size
#################################################################
# Parameters ####################################################
#################################################################
nb_epoch = 50
batch_size = 100
number_of_cities = 10
T = 100
learning_rate=0.0001
#################################################################
# Data ##########################################################
#################################################################
training_data = DataLoader(MonDataset(T, number_of_cities,'train'), shuffle=True, drop_last=True, batch_size=batch_size)
test_data = DataLoader(MonDataset(T, number_of_cities,'test'), shuffle=True, drop_last=True, batch_size=batch_size)
#################################################################
# Model #########################################################
#################################################################
model = RNN(1, 5, number_of_cities)
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
#################################################################
# Training ######################################################
#################################################################
train_loss_sum_list = []
train_mean_class_error_list = []
test_loss_sum_list = []
test_mean_class_error_list = []
for ep in range(nb_epoch):
    print(f"--epochs-- {1 + ep}")
    # train #######################################################
    loss_sum = 0
    class_error_list = []
    for (x, y) in training_data:
        y_hat = model(x.float())

        loss = criterion(y_hat, y)

        loss_sum += loss.item()

        loss.backward()
        optim.step()
        optim.zero_grad()

        class_error = get_classification_error(y_hat, y)
        class_error_list.append(class_error)

    train_loss_sum_list.append(loss_sum)
    train_mean_class_error_list.append(np.mean(class_error_list))
    # test #######################################################
    loss_sum = 0
    class_error_list = []
    for (x, y) in test_data:
        y_hat = model(x.float())

        loss = criterion(y_hat, y)

        loss_sum += loss.item()

        class_error = get_classification_error(y_hat, y)
        class_error_list.append(class_error)

    test_loss_sum_list.append(loss_sum)
    test_mean_class_error_list.append(np.mean(class_error_list))
#################################################################
# Plot ##########################################################
#################################################################
plt.figure(0)
plt.plot(train_loss_sum_list, label='train')
plt.title('RNN')
plt.xlabel('epochs')
plt.ylabel('error Cross Entropy Loss Sum')
plt.legend()

plt.figure(1)
plt.plot(test_loss_sum_list, label='test')
plt.title('RNN')
plt.xlabel('epochs')
plt.ylabel('error Cross Entropy Loss Sum')
plt.legend()

plt.figure(2)
plt.plot(train_mean_class_error_list, label='train')
plt.plot(test_mean_class_error_list, label='test' )
plt.title('RNN')
plt.xlabel('epochs')
plt.ylabel('Error of classification mean/batch')
plt.legend()

plt.show()