import torch
import torch.nn as nn
from torch.autograd import grad

Tenv = 25 #temp of environment
T0 = 100 #initial temperature
k = 0.001 #arbitrary start point for cooling const.

def physics_loss(model: torch.nn.Module):
    t = torch.linspace(0,1000,steps=1000).view(-1,1).requires_grad_(True).to(torch.float32)
    T = model(t)
    dT = grad(T, t, grad_outputs=torch.ones_like(T), create_graph=True)[0]  #we need [0] because the output of grad
                        # is a tuple, 1000 rows, 1 element per row
    pde = k*(Tenv - T) - dT  #predicted pde vals (y_hat)

    return torch.mean(pde**2).to(torch.float32) #note this is the MSE Loss