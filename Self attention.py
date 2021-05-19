import logging
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np
import math
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

class PositionalEncoding(nn.Module):
    "Position embeddings"

    def __init__(self, d_model: int, max_len: int = 5000):
        """Génère des embeddings de position

        Args:
            d_model (int): Dimension des embeddings à générer
            max_len (int, optional): Longueur maximale des textes.
                Attention, plus cette valeur est haute, moins bons seront les embeddings de position.
        """
        super().__init__()

        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        x=x.unsqueeze(0)
        """Ajoute les embeddings de position"""
        x = x + self.pe[:, :x.size(1)]
        return x.squeeze()
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

class model_self_attention(nn.Module):
    def __init__(self, embeddings_size):
        super(model_self_attention, self).__init__()
        self.l = nn.Linear(embeddings_size, 2)
        self.L1= self_attention(embeddings_size)
        self.L2 = self_attention(embeddings_size)
        self.L3 = self_attention(embeddings_size)
        self.g1= nn.Linear(embeddings_size, embeddings_size)
        self.g2 = nn.Linear(embeddings_size, embeddings_size)
        self.g3 = nn.Linear(embeddings_size, embeddings_size)
        self.batchnorm = nn.BatchNorm1d(embeddings_size)
        self.act = torch.nn.ReLU()
        self.embeddings_size = embeddings_size

    def forward(self, x, embeddings):
        batchsize = x.shape[0]
        t_hat = torch.zeros(batchsize, self.embeddings_size)
        mask=x!=-1
        for i in range(batchsize):
            y1=embeddings[x[i][mask[i]]].float()
            att1=self.L1(y1)

            y2 = self.act(self.g1(self.batchnorm(y1+att1)))
            att2 = self.L2(y2)

            y3 = self.act(self.g2(self.batchnorm(y2+att2)))
            att3 = self.L3(y3)

            y4 = self.act(self.g3(self.batchnorm(y3+att3)))

            t_hat[i, :] = y4.mean(0)
        x = self.l(t_hat)
        return x

class self_attention(nn.Module):
    def __init__(self, embeddings_size):
        super(self_attention, self).__init__()
        self.Key=nn.Linear(embeddings_size, embeddings_size)
        self.Query = nn.Linear(embeddings_size, embeddings_size)
        self.Value = nn.Linear(embeddings_size, embeddings_size)
        
    def forward(self,x):
        # dimension x=batch_size*embeddings_size
        batch_size=x.shape[0]
        keys=self.Key(x)
        queries=self.Query(x)
        values=self.Value(x)
        scores=queries@keys.T
        scores/=batch_size**.5
        softmax=nn.functional.softmax(scores, 1)
        Z=softmax@values
        return Z

class model_self_attention_positional_encoding(nn.Module):
    def __init__(self, embeddings_size):
        super(model_self_attention_positional_encoding, self).__init__()
        self.l = nn.Linear(embeddings_size, 2)
        self.L1= self_attention(embeddings_size)
        self.L2 = self_attention(embeddings_size)
        self.L3 = self_attention(embeddings_size)
        self.g1= nn.Linear(embeddings_size, embeddings_size)
        self.g2 = nn.Linear(embeddings_size, embeddings_size)
        self.g3 = nn.Linear(embeddings_size, embeddings_size)
        self.batchnorm = nn.BatchNorm1d(embeddings_size)
        self.positional_encoding=PositionalEncoding(embeddings_size)
        self.act = torch.nn.ReLU()
        self.embeddings_size = embeddings_size

    def forward(self, x, embeddings):
        batchsize = x.shape[0]
        t_hat = torch.zeros(batchsize, self.embeddings_size)
        mask=x!=-1
        for i in range(batchsize):
            y1=embeddings[x[i][mask[i]]].float()

            att1=self.L1(self.positional_encoding(y1))

            y2 = self.act(self.g1(self.batchnorm(y1+att1)))
            att2 = self.L2(y2)

            y3 = self.act(self.g2(self.batchnorm(y2+att2)))
            att3 = self.L3(y3)

            y4 = self.act(self.g3(self.batchnorm(y3+att3)))

            t_hat[i, :] = y4.mean(0)
        x = self.l(t_hat)
        return x

###########################################################################################
###########################################################################################
# hyperparametres #########################################################################
writer = SummaryWriter()

NB_CLASS =2
EMBEDDING_SIZE = 50
BATCH_SIZE = 64
word2id, embeddings, text_train, text_test = get_imdb_data(EMBEDDING_SIZE)
embeddings = torch.tensor(embeddings, dtype=float)
train_loader = DataLoader(text_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=padding, drop_last=True)

lr = 1e-3
n_epochs = 100

model = model_self_attention(EMBEDDING_SIZE)
# model = model_self_attention_positional_encoding(EMBEDDING_SIZE)
loss = nn.CrossEntropyLoss()
optim = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=0)
###########################################################################################
###########################################################################################
# train eval ##############################################################################
i = 0
for n in range(n_epochs):
    for x, y in train_loader:
        print(i,"ep",n)

        y_hat = model(x, embeddings)
        l = loss(y_hat, y)

        l.backward()
        optim.step()
        optim.zero_grad()

        mode = "Train"
        loss_(mode, l, i)
        accuracy(mode, y_hat, i, NB_CLASS, BATCH_SIZE)
        i += 1
