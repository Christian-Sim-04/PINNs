import torch
import torch.nn as nn

from bd_ffe import FourierFeatureEmbedding
from bd_physicsandloss_fns import pde_loss, bc_loss, interface_loss

class BeamNet(nn.Module):
    def __init__(self, n_units=40, n_layers=4,
                 pde_weight=1.0, bc_weight=1.0, if_weight=1.0,
                 use_ffe=True, num_frequencies=16, fourier_scale=1.0):
        super().__init__()

        self.pde_weight = pde_weight
        self.bc_weight = bc_weight
        self.if_weight = if_weight

        if use_ffe:
            self.fourier = FourierFeatureEmbedding(input_dim, num_frequencies, fourier_scale)
            input_dim = num_frequencies * 2 #note input_dim is the input into the training loop (ie after ffe)
        else:
            self.fourier = None

        layers = []
        layers.append(nn.Linear(1, n_units))
        layers.append(nn.Tanh())
        for _ in range(n_layers-1):
            layers.append(nn.Linear(n_units, n_units))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(n_units, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
    def fit(self, x1, x2, epochs=1000, lr=0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []
        for ep in range(epochs):
            optimizer.zero_grad()
            loss_pde = pde_loss(self, x1) + pde_loss(self, x2)
            loss_bc = bc_loss(self)
            loss_if = interface_loss(self)
            loss = self.pde_weight*loss_pde + self.bc_weight*loss_bc + self.if_weight*loss_if
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if ep % int(epochs/10) == 0:
                print(f"Epoch {ep}: Total Loss {loss.item():.4e} | PDE {loss_pde.item():.4e} | BC {loss_bc.item():.4e} | IF {loss_if.item():.4e}")
        return losses