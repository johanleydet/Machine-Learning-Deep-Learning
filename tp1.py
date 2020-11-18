import torch
from torch.autograd import Function
from torch.autograd import gradcheck


class Context:
    """Un objet contexte très simplifié pour simuler PyTorch

    Un contexte différent doit être utilisé à chaque forward
    """
    def __init__(self):
        self._saved_tensors = ()
    def save_for_backward(self, *args):
        self._saved_tensors = args
    @property
    def saved_tensors(self):
        return self._saved_tensors

class MSE(Function):
    """Début d'implementation de la fonction MSE"""
    @staticmethod
    def forward(ctx, yhat, y):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(yhat, y)
        return ((yhat-y)**2).sum()
        
    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        yhat, y = ctx.saved_tensors
        return 2*(yhat-y)*grad_output, -2*(yhat-y)*grad_output
        
# Implémentation de la fonction Linear(X, W, b)
class Linear(Function):
    @staticmethod
    def forward(ctx, X, W, b):
        ctx.save_for_backward(X, W, b)
        return X@W+b
    
    @staticmethod
    def backward(ctx, grad_output):
        X, W, b = ctx.saved_tensors
        return grad_output@W.t(), X.t()@grad_output, torch.sum(grad_output, dim=0) 

mse = MSE.apply
linear = Linear.apply