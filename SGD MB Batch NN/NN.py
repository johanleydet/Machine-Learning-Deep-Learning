import torch
import datamaestro
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# Parameters #################################################################
##############################################################################
learning_rate = 1e-3
r=np.random.randint(1000)# graine pour synchroniser les tirages
n_epochs=50

plt.close('all') # fermer les plot précédent
plt.yscale("log") # on plot sur une echelle logarithmique
##############################################################################
# Data #######################################################################
##############################################################################
data=datamaestro.prepare_dataset("edu.uci.boston")
colnames, datax, datay = data.data()
datax = torch.tensor(datax,dtype=torch.float)
datay = torch.tensor(datay,dtype=torch.float).reshape(-1,1)

input_size = datax.shape[1]
output_size =  datay.shape[1]

_80=int(datax.shape[0]*.2) # 80% datas



x_train = datax[:_80,:]
x_test = datax[_80:,:]

y_train = datay[:_80,:]
y_test = datay[_80:,:]

x=x_train
y=y_train

N, D_in, H, D_out = 64, input_size, 100, output_size

##############################################################################
# Without torch.nn.Sequential on boston data SGD #############################
##############################################################################
torch.manual_seed(r)

linear1=torch.nn.Linear(D_in, H)
actFct=torch.nn.Tanh()
linear2=torch.nn.Linear(H, D_out)
MSE=torch.nn.MSELoss(reduction='mean')

optimizer = torch.optim.SGD(list(linear1.parameters())+list(linear2.parameters()), lr=learning_rate)
train_no_sequential_SGD=[]
test_no_sequential_SGD=[]
for t in range(n_epochs):
    
    y_pred = linear2(actFct(linear1(x)))

    loss = MSE(y_pred, y)
   
    train_no_sequential_SGD.append(loss)
    test_no_sequential_SGD.append(MSE(linear2(actFct(linear1(x_test))), y_test))

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

##############################################################################
# With torch.nn.Sequential on boston data SGD ################################
##############################################################################`
torch.manual_seed(r)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.Tanh(),
    torch.nn.Linear(H, D_out),
)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
train_sequential_SGD=[]
test_sequential_SGD=[]
for t in range(n_epochs):
    
    y_pred = model(x)

    loss = MSE(y_pred, y)
    
    train_sequential_SGD.append(loss)
    test_sequential_SGD.append(MSE(model(x_test), y_test))

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

##############################################################################
# With torch.nn.Sequential on boston data Adam ###############################
##############################################################################`
torch.manual_seed(r)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.Tanh(),
    torch.nn.Linear(H, D_out),
)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train_sequential_Adam=[]
test_sequential_Adam=[]
for t in range(n_epochs):
    
    y_pred = model(x)

    loss = MSE(y_pred, y)
    
    train_sequential_Adam.append(loss)
    test_sequential_Adam.append(MSE(model(x_test), y_test))

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

##############################################################################
# Plots ######################################################################
##############################################################################

plt.plot(train_no_sequential_SGD,'r', label='train SDG Without Sequential')
plt.plot(test_no_sequential_SGD,'gold', label='test SDG Without Sequential')

plt.plot(train_sequential_SGD,'b',linestyle=':', label='train SDG With Sequential')
plt.plot(test_sequential_SGD,'plum',linestyle=':', label='test SDG With Sequential')

plt.title('NN')
plt.xlabel('epochs')
plt.ylabel('error MSE, log scale')
plt.legend(bbox_to_anchor=(1, 1),loc='upper left')
plt.tight_layout()


plt.figure()

plt.plot(train_sequential_SGD,'purple', label='train SDG')
plt.plot(test_sequential_SGD,'purple',linestyle=':', label='test SDG')

plt.plot(train_sequential_Adam,'pink', label='train Adam')
plt.plot(test_sequential_Adam,'pink',linestyle=':', label='test Adam')

plt.title('ADAM vs SDG in NN')
plt.xlabel('epochs')
plt.ylabel('error MSE, log scale')
plt.legend(bbox_to_anchor=(1, 1),loc='upper left')
plt.tight_layout()


