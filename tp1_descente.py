import torch
import matplotlib.pyplot as plt
from tp1 import MSE, Linear, Context

# Les données supervisées
x = torch.randn(50, 13)
y = torch.randn(50, 3)

#nombre d'itérations
N=20
###############################################################
# avec Context() ##############################################
###############################################################
# Les paramètres du modèle à optimiser
w = torch.ones(13, 3, dtype=torch.float)
b = torch.ones(3, dtype=torch.float)

#model
ctx_linear = Context()
ctx_mse = Context()


for epsilon in (0.0005, 0.005):
    l=[]
    for n_iter in range(N):
    
        yhat = Linear.forward(ctx_linear, x, w, b)
        loss = MSE.forward(ctx_mse, yhat, y)
        
        l.append(loss.item())
        
        grad_yhat, grad_y=MSE.backward(ctx_mse,1)
        grad_X, grad_W, grad_b =Linear.backward(ctx_linear,grad_yhat)
        w = w - epsilon * grad_W
        b = b - epsilon * grad_b
        
    # plot 
    plt.plot(l, label='avec Context() lr='+str(epsilon))
    
###############################################################
# sans Context() ##############################################
###############################################################
# Les paramètres du modèle à optimiser
w = torch.ones(13, 3,dtype=torch.float, requires_grad=True)
b = torch.ones(3,dtype=torch.float, requires_grad=True)

for epsilon in (0.0005, 0.005):
    l=[]
    for n_iter in range(N):
        
        yhat = x@w+b
        loss = ((yhat-y)**2).sum()
        l.append(loss.item())
        loss.backward()
    
        with torch.no_grad():
            w -= epsilon * w.grad
            b -= epsilon * b.grad 
            
            w.grad.zero_()
            b.grad.zero_()
            
    # plot        
    plt.plot(l,'.', label='sans Context() lr='+str(epsilon))
    
###############################################################    
# plot ########################################################
###############################################################
plt.title('Gradient descent')
plt.xlabel('iterations')
plt.ylabel('error MSE sum')
axes = plt.gca()
axes.set_ylim([0,1000])
plt.legend(loc='upper left')

