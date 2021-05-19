import logging
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np

from datamaestro import prepare_dataset
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


###########################################################################################
###########################################################################################
# code donne ##############################################################################
class FolderText(Dataset):
    """Dataset basé sur des dossiers (un par classe) et fichiers"""

    def __init__(self, classes, folder: Path, tokenizer, load=False):
        self.tokenizer = tokenizer
        self.files = []
        self.filelabels = []
        self.labels = {}
        for ix, key in enumerate(classes):
            self.labels[key] = ix

        for label in classes:
            for file in (folder / label).glob("*.txt"):
                self.files.append(file.read_text() if load else file)
                self.filelabels.append(self.labels[label])

    def __len__(self):
        return len(self.filelabels)

    def __getitem__(self, ix):
        s = self.files[ix]
        return self.tokenizer(s if isinstance(s, str) else s.read_text()), self.filelabels[ix]


def get_imdb_data(embedding_size=50):
    """Renvoie l'ensemble des donnéees nécessaires pour l'apprentissage

    - dictionnaire word vers ID
    - embeddings (Glove)
    - DataSet (FolderText)

    """
    WORDS = re.compile(r"\S+")

    words, embeddings = prepare_dataset('edu.stanford.glove.6b.%d' % embedding_size).load()
    OOVID = len(words)
    words.append("__OOV__")

    word2id = {word: ix for ix, word in enumerate(words)}
    embeddings = np.vstack((embeddings, np.zeros(embedding_size)))

    def tokenizer(t):
        return [word2id.get(x, OOVID) for x in re.findall(WORDS, t.lower())]

    logging.info("Loading embeddings")

    logging.info("Get the IMDB dataset")
    ds = prepare_dataset("edu.stanford.aclimdb")

    return word2id, embeddings, FolderText(ds.train.classes, ds.train.path, tokenizer, load=False), FolderText(
        ds.test.classes, ds.test.path, tokenizer, load=False)


###########################################################################################
###########################################################################################
# modeles et utils ########################################################################
def loss_(mode, l, i):
    """
    loss tensorboard
    """
    writer.add_scalars("Loss", {mode + " loss": l}, i)


def accuracy(mode, y_hat, i, nb_class, batch_size):
    """
    accuracy classification tensorboard
    """
    writer.add_scalars("Accuracy", {mode + " accuracy": (torch.nn.functional.one_hot(torch.argmax(y_hat, 1),
                                                                                     nb_class) == torch.nn.functional.one_hot(
        y, nb_class)).sum() / (nb_class * batch_size)}, i)


def padding(sequences):
    """
    Permet lorsque l'on utilise Datalaoder d'obtenir des sequences de mm taille

    Chaque mot est represente par un nombre

    on comble les phrases avec le nombre -1
    """
    # sequence est une liste contenant batchsize tuples de la forme (phrase, label)
    ls_array_X = []
    ls_array_Y = []
    # on parcourt ces tuples pour récupérer une liste qui contient batchsize tenseurs X
    # et batchsize y
    for i in range(len(sequences)):
        ls_array_X.append(torch.tensor(sequences[i][0]))
        ls_array_Y.append(sequences[i][1])
    # on créé ensuite un padding pour rendre les sequences de la mm taille
    # sequences prend une liste de tenseurs
    # batch_first=True signifie qu'on est de la forme batch*M*N
    myPadX = torch.nn.utils.rnn.pad_sequence(sequences=ls_array_X, batch_first=True, padding_value=-1)
    labels = torch.tensor(ls_array_Y)

    return myPadX, labels


class model1(nn.Module):
    """
    modele de base
    """

    def __init__(self, embeddings_size):
        super(model1, self).__init__()
        self.l = nn.Linear(embeddings_size, 2)
        self.embeddings_size = embeddings_size

    def forward(self, x, embeddings):
        batchsize = x.shape[0]
        seqsize = x.shape[1]
        t_hat = torch.zeros(batchsize, self.embeddings_size)
        # pour chaque element du batch fr
        for i in range(batchsize):
            # regerder chaque mot, prendre ca representation ds l'embedding puis la stocker
            # fr ensuite la moyenne de tous les embeddings stocker
            # attention ne pas compter les mots -1 qui sont en realité du padding
            k = 0
            for j in range(seqsize):
                if x[i, j].item() != -1:
                    t_hat[i, :] += embeddings[x[i, j].item()]
                    k += 1
            t_hat[i, :] /= k
        x = self.l(t_hat)
        return x


class model2(nn.Module):
    def __init__(self, embeddings_size):
        super(model2, self).__init__()
        self.l = nn.Linear(embeddings_size, 2)
        self.embeddings_size = embeddings_size
        self.q=nn.Linear(embeddings_size, 1)

    def forward(self, x, embeddings):
        batchsize = x.shape[0]
        seqsize = x.shape[1]
        t_hat = torch.zeros(batchsize, self.embeddings_size)

        # pour chaque element du batch fr
        for i in range(batchsize):
            for j in range(seqsize):
                if x[i, j].item() != -1:
                    em = embeddings[x[i, j].item()].unsqueeze(0)
                    if j == 0:
                        tmp = em
                    else:
                        tmp = torch.cat((tmp, em), 0)
            p = nn.functional.softmax(self.q(tmp.float()), 0)
            t_hat[i, :] = (tmp * p).sum(0)
        x = self.l(t_hat)
        return x
class model3(nn.Module):
    """
    modele de base
    """

    def __init__(self, embeddings_size):
        super(model3, self).__init__()
        self.l = nn.Linear(embeddings_size, 2)
        self.embeddings_size = embeddings_size

    def forward(self, x, embeddings):
        batchsize = x.shape[0]
        seqsize = x.shape[1]
        t_hat = torch.zeros(batchsize, self.embeddings_size)

        for i in range(batchsize):
            # regerder chaque mot, prendre ca representation ds l'embedding puis la stocker
            # fr ensuite la moyenne de tous les embeddings stocker
            # attention ne pas compter les mots -1 qui sont en realité du padding
            k = 0
            for j in range(seqsize):
                if x[i, j].item() != -1:
                    t_hat[i, :] += embeddings[x[i, j].item()]
                    k += 1
            t_hat[i, :] /= k

        for i in range(batchsize):
            for j in range(seqsize):
                if x[i, j].item() != -1:
                    em = embeddings[x[i, j].item()].unsqueeze(0)
                    if j == 0:
                        tmp = em
                    else:
                        tmp = torch.cat((tmp, em), 0)
            y=tmp.float() @ t_hat[i, :]

            p = nn.functional.softmax(y, 0)
            t_hat[i, :] = (tmp * p.unsqueeze(1)).sum(0)
        x = self.l(t_hat)
        return x

###########################################################################################
###########################################################################################
# hyperparametres #########################################################################
writer = SummaryWriter()

NB_CLASS =2
EMBEDDING_SIZE = 50
BATCH_SIZE = 10
word2id, embeddings, text_train, text_test = get_imdb_data(EMBEDDING_SIZE)
embeddings = torch.tensor(embeddings, dtype=float)
train_loader = DataLoader(text_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=padding, drop_last=True)

lr = 1e-3
n_epochs = 3

# model = model1(EMBEDDING_SIZE)
# model = model2(EMBEDDING_SIZE)
model = model3(EMBEDDING_SIZE)
loss = nn.CrossEntropyLoss()
optim = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=0)
###########################################################################################
###########################################################################################
# train eval ##############################################################################
i = 0
for n in range(n_epochs):
    for x, y in train_loader:
        print(i)

        y_hat = model(x, embeddings)
        l = loss(y_hat, y)

        l.backward()
        optim.step()
        optim.zero_grad()

        mode = "Train"
        loss_(mode, l, i)
        accuracy(mode, y_hat, i, NB_CLASS, BATCH_SIZE)
        i += 1
