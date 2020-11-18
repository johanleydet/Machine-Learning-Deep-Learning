from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datamaestro import prepare_dataset
import matplotlib.pyplot as plt
#################################################################
# 1 #############################################################
#################################################################
class MonDataset(Dataset):
    def __init__(self, X, y): 
        self.X=X/255.0
        self.X_flatten=torch.flatten(torch.from_numpy(X))/255.0
        self.y=y
        
    def __getitem__(self,index):
        """ retourne un couple (exemple,label) correspondant a l’index """ 
        return (torch.flatten(torch.from_numpy(self.X[index,:,:])),self.y[index])
    
    def __len__(self):
        """ renvoie la taille du jeu de donnees """ 
        return self.y.shape[0]

#################################################################
# 2 #############################################################
#################################################################
class autoEncodeur(nn.Module):
    def __init__(self, D_in, D_out):
        super(autoEncodeur, self).__init__()
        self.weight=torch.nn.Parameter(torch.randn(D_out,D_in))
        self.bias1=torch.nn.Parameter(torch.randn(D_out)) 
        self.bias2=torch.nn.Parameter(torch.randn(D_in)) 
        
    def encode (self, X):
        return F.relu(F.linear(X, self.weight, self.bias1))
        
    def decode(self, X):
        return F.sigmoid(F.linear(X, self.weight.t(), self.bias2))
        
    def forward(self,X):
        return self.decode(self.encode(X))

#################################################################
# data ##########################################################
#################################################################
ds = prepare_dataset("com.lecun.mnist");
train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels =  ds.test.images.data(), ds.test.labels.data()

BATCH_SIZE=50
data_train=MonDataset(train_images, train_labels)
data_test=MonDataset(test_images, test_labels)

trainLoader = DataLoader(data_train, shuffle=True, batch_size=BATCH_SIZE) 
testLoader = DataLoader(data_test, shuffle=True, batch_size=BATCH_SIZE) 

learning_rate= 10e-4 

#################################################################
# 3 #############################################################
#################################################################
savepath = Path("modelTM3_batch_50.pch")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

class State :
    def __init__(self, model, optim):
        self.model = model
        self.optim = optim
        self.epoch, self.iteration = 0,0
        self.train_loss=[]
        self.test_loss=[]
        self.xhat=None
        self.x=None
        
if savepath.is_file ():
    with savepath.open("rb") as fp:
        state = torch.load(fp) #on recommence depuis le modele sauvegarde
else :
    autoencoder = autoEncodeur(28**2,100)
    autoencoder=autoencoder.double()
    autoencoder = autoencoder.to(device) 
    optim = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    state = State(autoencoder ,optim)
    
ITERATION=100
    
loss= torch.nn.MSELoss(reduction='mean')

for epoch in range(state.epoch,ITERATION): 
    print('epochs n°',epoch)
    for x,y in trainLoader :
        state.optim.zero_grad() 
        x = x.to(device)
        xhat = state.model.forward(x)
        l = loss(xhat,x)
        l.backward()
        state.optim.step () 
        state.iteration += 1
    state.train_loss.append(l.item()) 
    for x,y in testLoader :   
        xhat = state.model.forward(x)
        l = loss(xhat,x)  
    state.test_loss.append(l.item()) 
    
    # on sauvegarde la derniere estimation pour avoir un aperçu 
    # du résultat
    if epoch==ITERATION-1 :
        state.xhat=xhat
        state.x=x
        
    with savepath.open("wb") as fp: 
        state.epoch = epoch + 1 
        torch.save(state ,fp)

#################################################################
# Plot ##########################################################
#################################################################
plt.figure(1)
plt.plot(state.test_loss, label='train batch 50') 
plt.plot(state.train_loss, label='test batch 50') 

plt.title('NN')
plt.xlabel('epochs')
plt.ylabel('error MSE')
plt.legend(bbox_to_anchor=(1, 1),loc='upper left')
plt.tight_layout()

plt.figure()
plt.imshow(np.reshape(state.xhat.detach().numpy()[49,:],(28,28)))
plt.title('Encode Decode Image')
plt.figure()
plt.imshow(np.reshape(state.x.detach().numpy()[49,:],(28,28)))
plt.title('Original Image')
