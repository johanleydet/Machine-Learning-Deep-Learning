import torch
from tp1 import mse, linear

# Test du gradient de MSE

yhat = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
y = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
print(torch.autograd.gradcheck(mse, (yhat, y)))

# Test du gradient de Linear

X=torch.randn(10,20,requires_grad=True, dtype=torch.float64)
W=torch.randn(20,5,requires_grad=True, dtype=torch.float64)
b=torch.randn(5,requires_grad=True, dtype=torch.float64)

print(torch.autograd.gradcheck(linear, (X,W,b)))

