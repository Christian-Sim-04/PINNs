import torch
import torch.nn as nn

class BeamDoubleNet(nn.Module):
    def __init__(self, input_dim=1, output_dim=2,
                 n_units=40, n_layers=4,
                 pde_weight=1.0, bc_weight=1.0, if_weight=1.0,):
                 #if_cont_weight=1.0, if_shear_weight=1.0):
        super().__init__()

        self.pde_weight = pde_weight
        self.bc_weight = bc_weight
        self.if_weight = if_weight
        self.input_dim = input_dim
        #self.if_cont_weight = if_cont_weight
        #self.if_shear_weight = if_shear_weight
        self.output_dim = output_dim

        layers = []
        layers.append(nn.Linear(input_dim, n_units))
        layers.append(nn.Tanh())
        for _ in range(n_layers-1):
            layers.append(nn.Linear(n_units, n_units))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(n_units, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)