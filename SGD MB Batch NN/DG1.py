import torch
import datamaestro
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# Data #######################################################################
##############################################################################
data=datamaestro.prepare_dataset("edu.uci.boston")
colnames, datax, datay = data.data()
datax = torch.tensor(datax,dtype=torch.float)
datay = torch.tensor(datay,dtype=torch.float).reshape(-1,1)

input_size = datax.shape[1]
output_size =  datay.shape[1]

_80=int(datax.shape[0]*.8) # 80% datas
_20=datax.shape[0]-_80

x_train = datax[:_80,:]
x_test = datax[_80:,:]

y_train = datay[:_80,:]
y_test = datay[_80:,:]

##############################################################################
# Parameters #################################################################
##############################################################################
dataNb=_80
n_epochs=50
epsilon = 3e-6

r=np.random.randint(1000)# graine pour synchroniser les tirages

plt.close('all') # fermer les plot précédent
plt.yscale("log") # on plot sur une echelle logarithmique
##############################################################################
# Fct ########################################################################
##############################################################################
def forward(x,w,b) :
    return x@w+b

def Loss(yhat, y, N=dataNb) :
    return torch.sum((yhat-y).pow(2), dim=0)/N

##############################################################################
# SGD ########################################################################
##############################################################################
torch.manual_seed(r)
w = torch.randn(input_size, output_size,dtype=torch.float, requires_grad=True)
b = torch.ones(output_size, dtype=torch.float,requires_grad=True)  

lossTrainlist=[]
lossTestlist=[]
lossTrainlist_it_SGD=[]
for h in range(n_epochs):
    for it in np.random.randint(dataNb, size=_80):
        
        x=x_train[it,:]
        y=y_train[it,:]
        
        yhat = forward(x,w,b)
        loss = Loss(yhat, y)
        loss.backward()
        
        lossTrainlist_it_SGD.append(Loss(forward(x_train,w,b), y_train))
        
        with torch.no_grad():
            w -= epsilon * w.grad
            b -= epsilon * b.grad 
        
            w.grad.zero_()
            b.grad.zero_()
            
    lossTrainlist.append(Loss(forward(x_train,w,b), y_train))
    lossTestlist.append(Loss(forward(x_test,w,b), y_test, _20))
plt.plot(lossTrainlist,'r', label='train SDG')
plt.plot(lossTestlist,'r--', label='test SDG')

##############################################################################
# Mini Batch #################################################################
##############################################################################
#ici l'on a défini une fonction pour pouvoir itérer plusieurs batch_sizes
def mini_batch(batch_size=30,n_epochs=n_epochs, color='y',list_it=False):
    w = torch.randn(input_size, output_size,dtype=torch.float, requires_grad=True)
    b = torch.ones(output_size, dtype=torch.float,requires_grad=True)     

    lossTrainlist=[]
    lossTestlist=[]
    if list_it:
        lossTrainlist_it_MB=[]
    
    batch_size=int(batch_size)
    epochs=0
    b_inf=0
    b_sup=batch_size
    while(epochs<=n_epochs):
        if b_sup<b_inf:
            x=torch.cat((x_train[b_inf:,:], x_train[0:b_sup,:]), 0)
            y=torch.cat((y_train[b_inf:,:], y_train[0:b_sup,:]), 0)
        else:
            x=x_train[b_inf:b_sup,:]
            y=y_train[b_inf:b_sup,:]
        
        yhat = forward(x,w,b)
        loss = Loss(yhat, y)
        loss.backward()
        
        if list_it:
            lossTrainlist_it_MB.append(Loss(forward(x_train,w,b), y_train))
        
        with torch.no_grad():
            w -= epsilon * w.grad
            b -= epsilon * b.grad 
        
            w.grad.zero_()
            b.grad.zero_()
        
        b_inf+=batch_size
        b_sup+=batch_size
        b_inf%=_80
        b_sup%=_80
        
        if b_sup<batch_size:
            epochs+=1
            lossTrainlist.append(Loss(forward(x_train,w,b), y_train))
            lossTestlist.append(Loss(forward(x_test,w,b), y_test, _20))
    plt.plot(lossTrainlist,color, label='train MBGD size:'+str(batch_size))
    plt.plot(lossTestlist, c=color,ls='--', label='test MBGD size:'+str(batch_size))
    if list_it:
        return lossTrainlist_it_MB

torch.manual_seed(r)
lossTrainlist_it_MB=mini_batch(batch_size=50,n_epochs=n_epochs, color='b',list_it=True)

##############################################################################
# Batch ######################################################################
##############################################################################
torch.manual_seed(r)
w = torch.randn(input_size, output_size,dtype=torch.float, requires_grad=True)
b = torch.ones(output_size, dtype=torch.float,requires_grad=True)  

lossTrainlist=[]
lossTestlist=[]
for n_iter in range(n_epochs):
    
    x=x_train
    y=y_train
    
    yhat = forward(x,w,b)
    loss = Loss(yhat, y)
    loss.backward()
    
    with torch.no_grad():
        w -= epsilon * w.grad
        b -= epsilon * b.grad 
    
        w.grad.zero_()
        b.grad.zero_()
    lossTrainlist.append(loss)
    lossTestlist.append(Loss(forward(x_test,w,b), y_test, _20))
    
##############################################################################
# Plots ######################################################################
##############################################################################
plt.plot(lossTrainlist, 'g', label='train Batch')
plt.plot(lossTestlist, 'g--', label='test Batch')
plt.title('Gradient Descent Algorithms')
plt.xlabel('epochs')
plt.ylabel('error MSE, log scale')
plt.legend(bbox_to_anchor=(1, 1),loc='upper left')
plt.tight_layout()

plt.figure()
t=len(lossTrainlist_it_MB[20:])
plt.plot(np.arange(20,t+20),lossTrainlist_it_MB[20:], 'b')
plt.title('Mini Batch Gradient Descent')
plt.xlabel('iterations')
plt.ylabel('error MSE, log scale')
plt.tight_layout()

plt.figure()
t=len(lossTrainlist_it_SGD[500:])
plt.plot(np.arange(500,t+500), lossTrainlist_it_SGD[500:], 'r')
plt.title('Stochastic Gradient Descent')
plt.xlabel('iterations')
plt.ylabel('error MSE, log scale')
plt.tight_layout()

plt.figure()
plt.plot(lossTrainlist, 'g')
plt.title('Batch Gradient Descent')
plt.xlabel('epochs')
plt.ylabel('error MSE, log scale')
plt.tight_layout()

plt.figure()
for i, color in zip([30, 360,380, _80-1], ['aqua', 'skyblue', 'deepskyblue', 'blue']):
    torch.manual_seed(r)
    mini_batch(batch_size=i,n_epochs=n_epochs, color=color,list_it=False)
plt.title('MBGDs')
plt.xlabel('epochs')
plt.ylabel('error MSE, log scale')
plt.legend(bbox_to_anchor=(1, 1),loc='upper left')
plt.tight_layout()