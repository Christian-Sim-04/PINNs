import torch
import torch.nn as nn

from bd_ffe import FourierFeatureEmbedding
from bd_physicsandloss_fns import pde_loss, bc_loss, interface_loss

class BeamNet(nn.Module):
    def __init__(self, input_dim=1, output_dim=2,
                 n_units=40, n_layers=4,
                 pde_weight=1.0, bc_weight=1.0, if_weight=1.0,
                 use_ffe=False, num_frequencies=1, fourier_scale=1.0):
        super().__init__()

        self.pde_weight = pde_weight
        self.bc_weight = bc_weight
        self.if_weight = if_weight
        self.input_dim = input_dim
        self.output_dim = output_dim

        if use_ffe:
            self.fourier = FourierFeatureEmbedding(input_dim, num_frequencies, fourier_scale)
            input_dim = num_frequencies * 2 #note input_dim is the input into the training loop (ie after ffe)
        else:
            self.fourier = None

        layers = []
        layers.append(nn.Linear(input_dim, n_units))
        layers.append(nn.Tanh())
        for _ in range(n_layers-1):
            layers.append(nn.Linear(n_units, n_units))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(n_units, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.fourier:
            x=self.fourier(x)
        return self.net(x)
    
    def fit(self, x1, x2, xmin, xmax, epochs=1000, lr=0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []
        for ep in range(epochs):
            optimizer.zero_grad()
            loss = 0.0
            loss_pde = pde_loss(self, x1, xmin, xmax) + pde_loss(self, x2, xmin, xmax)
            loss_bc = bc_loss(self, xmin, xmax)
            loss_if = interface_loss(self, xmin, xmax)
            loss = self.pde_weight*loss_pde + self.bc_weight*loss_bc + self.if_weight*loss_if
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if ep % int(epochs/10) == 0:
                print(f"Epoch {ep}: Total Loss {loss.item():.4e} | PDE {loss_pde.item():.4e} | BC {loss_bc.item():.4e} | IF {loss_if.item():.4e}")
        return losses