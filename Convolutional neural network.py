import logging
logging.basicConfig(level=logging.INFO)

import heapq
from pathlib import Path
import gzip

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split
import sentencepiece as spm

from tp7_preprocess import TextDataset
from CNN import *


# --- Configuration

# Taille du vocabulaire
vocab_size = 1000
MAINDIR = Path(__file__).parent

# Chargement du tokenizer

tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(f"wp{vocab_size}.model")
ntokens = len(tokenizer)

def loaddata(mode):
    with gzip.open(f"{mode}-{vocab_size}.pth", "rb") as fp:
        return torch.load(fp)


test = loaddata("test")
train = loaddata("train")
TRAIN_BATCHSIZE=1000
TEST_BATCHSIZE=1000


# --- Chargements des jeux de données train, validation et test

val_size = 1000
train_size = len(train) - val_size
train, val = random_split(train, [train_size, val_size])

logging.info("Datasets: train=%d, val=%d, test=%d", train_size, val_size, len(test))
logging.info("Vocabulary size: %d", vocab_size)
train_iter = DataLoader(train, batch_size=TRAIN_BATCHSIZE, collate_fn=TextDataset.collate)
val_iter = DataLoader(val, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate)
test_iter = DataLoader(test, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate)

# HYPERPARAMETERS

emb_size = 20
n_classes = 3
lr = 1e-4
n_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip=1

cnn = CNN(vocab_size, emb_size, n_classes, device = device, filters_size = 100, stride=1)
#naive = NaiveClassifier(train_iter, [0, 1, 2])

#TRAIN LOOP

now = datetime.datetime.now()
log_path = 'log-cnn/run_' + now.strftime("%H-%M-%S")
writer = SummaryWriter(log_path)

loss_fct = nn.CrossEntropyLoss()
optim = torch.optim.Adam(params = cnn.parameters(), lr = lr)
iter = 0

gamma = 0.9 #discount factor
conv_output = torch.zeros(100).to(device)

for epoch in range(n_epochs):
    for text, labels in train_iter:
        text, labels = text.to(device), labels.to(device)

        optim.zero_grad()
        y_hat = cnn(text)
        with torch.no_grad():
            conv_y = cnn.conv(cnn.emb(text).permute(0, 2, 1)).mean(0)
            conv_output = gamma * conv_output + (1 - gamma) * conv_y.mean(1)

        loss = loss_fct(y_hat, labels)
        loss.backward()

        clip_grad_norm_(cnn.parameters(), clip)

        optim.step()

        # Logging
        if iter%10 == 0: 

            # 10 most activated output
            print(torch.topk(conv_output, 10).indices)

            with torch.no_grad():
                # GRADIENT NORM
                grad_norm = 0
                for p in cnn.parameters():
                    try:
                        param_norm = p.grad.data.norm(2)
                        grad_norm += param_norm.item() ** 2
                    except: 
                        pass

                grad_norm = grad_norm ** (1. / 2)
                writer.add_scalars("Gradient norm", {"Norm": grad_norm}, iter)

                print(f'Loss at iter {iter} : {loss}')

                # VALIDATION LOSS

                for text, labels in val_iter:
                    text, labels = text.to(device), labels.to(device)
                    y_hat = cnn(text)
                    val_loss = loss_fct(y_hat, labels)


                writer.add_scalars("Loss", {"Train loss": loss, "Validation loss": val_loss}, iter)

                # MODEL & NAIVE ACCURACY ON VALIDATION SET

                accuracy = {"model": 0, "naive": 0}
                n_val = 0
                for text, labels in val_iter:
                    text, labels = text.to(device), labels.to(device)
                    y_pred_model = cnn(text).argmax(1)
                    y_pred_naive = naive.predict(text).to(device)
                    accuracy["model"] += (y_pred_model == labels).sum().float() / len(labels)
                    accuracy["naive"] += (y_pred_naive == labels).sum().float() / len(labels)
                    n_val += 1
                writer.add_scalars("Test accuracy", {"Model": accuracy["model"] / n_val, "Naive": accuracy["naive"] / n_val}, iter)
        
        iter += 1