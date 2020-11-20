import string
import unicodedata
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

#################################################################################
# fonction donnee ###############################################################
#################################################################################
LETTRES = string.ascii_letters + string.punctuation + string.digits + ' '

def normalize(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if c in LETTRES)
#################################################################################
# classes #######################################################################
#################################################################################
class textDataset(Dataset):
    def __init__(self, seq_size, mode):
        super().__init__()
        # importation du texte
        text = open("trump_full_speech.txt", 'r').read()
        # on enleve les caractères superficiels
        text = text[1:10000]
        text = normalize(text)
        # creation des traducteurs
        id2lettre = {0: ''}
        lettre2id = {'': 0}
        # maj des traducteurs
        id = 1
        for c in text:
            if c not in lettre2id.keys():
                lettre2id[c] = id
                id2lettre[id] = c
                id += 1
        # donnees test ou train
        text_size = len(text)
        train_text_size = int(text_size * .8)
        if mode == 'train':
            text = text[:train_text_size]
        else:
            text = text[train_text_size:]

        # attributs
        self.text_size = len(text)
        self.seq_size = seq_size
        self.text = text
        self.lettre2id = lettre2id
        self.id2lettre = id2lettre

    def __len__(self):
        return self.text_size - self.seq_size

    def __getitem__(self, id):
        x = self.string2code(self.text[id:id + self.seq_size])
        y = self.string2code(self.text[id + self.seq_size])
        return x, y

    def string2code(self, s):
        return torch.tensor([self.lettre2id[c] for c in normalize(s)])

    def code2string(self, t):
        if type(t) != list:
            t = t.tolist()
        return ''.join(self.id2lettre[i] for i in t)

class RNN(torch.nn.Module):
    def __init__(self, d_in_x, d_h, d_out):
        super(RNN, self).__init__()
        self.Wh = torch.nn.Linear(d_h, d_h)
        self.Wi = torch.nn.Linear(d_in_x, d_h)
        self.Wo = torch.nn.Linear(d_h, d_out)
        self.act_fct_h = torch.nn.Tanh()
        self.d_h = d_h
        self.d_in_x = d_in_x
        self.d_out = d_out

    def forward(self, seq):
        # paramtres
        seq_size, batch_size, embedding_size = seq.shape
        self.T = seq_size
        self.batch_size = batch_size
        self.embedding_size = embedding_size

        # initialisation
        self.ht = torch.zeros((self.batch_size, self.d_h), requires_grad=True,
                              dtype=torch.double)  # representation h courante
        self.h = torch.zeros((self.batch_size, self.d_h), requires_grad=True,
                             dtype=torch.double)  # tout nos h gardés en mémoire
        self.ot = torch.zeros((self.batch_size, self.d_out), requires_grad=True,
                              dtype=torch.double)  # La sortie courante notre température prédit
        self.o = torch.zeros((self.batch_size, self.d_out), requires_grad=True,
                             dtype=torch.double)  # Toute nos sorties gardées en mémoire

        # one step
        for x in range(self.T):
            self.one_step(seq[x, :, :])
        return self.ot

    def one_step(self, x):
        s1 = self.Wh(self.ht)
        s2 = self.Wi(x)
        self.ht = self.act_fct_h(s1 + s2)
        self.h = torch.cat((self.h, self.ht), 1)
        self.ot = self.Wo(self.ht)
        self.o = torch.cat((self.o, self.ot), 1)


def generate(model, seed, T,lenght=400):
    seq = seed
    # on fait lenght fois
    # ici on prend les T derniers termes de la sequence, on lui fait devener le suivant
    # on maj la sequence
    for i in range(lenght):
        x = seq[-T:].reshape(T, 1)
        pc = nn.functional.softmax(model(x), dim = 1)[0]
        c = np.random.choice(pc.detach().numpy().shape[0], p=pc.detach().numpy())
        seq = torch.cat((seq, torch.tensor(c).reshape(1)))
    return seq
#################################################################
# Parameters ####################################################
#################################################################
nb_epoch = 250
batch_size = 512
d_h = 64
T = 16
embedding_size = 100
#################################################################
# Data ##########################################################
#################################################################
data = textDataset(T, 'train')

training_data = DataLoader(textDataset(T, 'train'), shuffle=True, drop_last=True, batch_size=batch_size)
test_data = DataLoader(textDataset(T, 'test'), shuffle=True, drop_last=True, batch_size=batch_size)
#################################################################
# Model #########################################################
#################################################################
embedding = nn.Embedding(len(data.id2lettre), embedding_size)
RNN = RNN(embedding_size, d_h, len(data.id2lettre))
model = nn.Sequential(embedding, RNN)
model = model.double()

optim = torch.optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss()
#################################################################
# Training ######################################################
#################################################################
train_loss_sum_list = []
for step in range(nb_epoch):
    print(step)
    loss_sum = 0
    for x, y in training_data:
        optim.zero_grad()
        model.zero_grad()
        x = x.transpose(1, 0)
        ychap = model(x)
        l = criterion(ychap, y.reshape(batch_size))
        l.backward()
        loss_sum += l.item()
        optim.step()
        optim.zero_grad()
    train_loss_sum_list.append(loss_sum)

speech_beginnig = data.string2code("I will make America great again")
text = generate(model, speech_beginnig, 16)
print(data.code2string(text))

print(train_loss_sum_list)
plt.plot(train_loss_sum_list, label='train')
plt.title('RNN')
plt.xlabel('epochs')
plt.ylabel('error MSE Loss Sum')
plt.legend()
plt.show()