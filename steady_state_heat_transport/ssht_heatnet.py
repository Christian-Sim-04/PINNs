#heatnet

import torch
import torch.nn as nn

from ssht_ffe import FourierFeatureEmbedding


def np_to_torch(x):
    #n_samples = len(x)
    return torch.from_numpy(x).to(torch.float32).reshape(-1, 1)  # changed from ...reshape(n_samples, -1)

class HeatNet(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, n_units=40, epochs=1000, loss=nn.MSELoss, lr=0.001, 
                 pde_loss=None, bc_loss=None, pde_weight=1.0, bc_weight=1.0,
                 use_ffe=True, num_frequencies=16, fourier_scale=1.0,
                 data_loss=None, data_weight=1.0):
        super().__init__()
        self.epochs = epochs
        self.loss = loss 
        self.lr = lr
        self.pde_loss = pde_loss
        self.bc_loss = bc_loss
        self.pde_weight = pde_weight
        self.bc_weight = bc_weight
        self.use_ffe = use_ffe
        self.data_loss = data_loss # comment out if no BCs
        self.data_weight = data_weight # comment out if no BCs

        if use_ffe:
            self.fourier = FourierFeatureEmbedding(input_dim, num_frequencies, fourier_scale)
            input_dim = num_frequencies * 2 #note input_dim is the input into the training loop (ie after ffe)
        else:
            self.fourier = None

        self.layers = nn.Sequential(
            nn.Linear(input_dim, n_units), ################
            nn.Tanh(),
            nn.Linear(n_units, n_units),
            nn.Tanh(),
            nn.Linear(n_units, n_units),
            nn.Tanh(),
            nn.Linear(n_units, n_units),
            nn.Tanh(),
            nn.Linear(n_units, output_dim)
        )


    def forward(self, xy): #forward takes a 1d input so take concatenation of x,y
        if self.fourier:
            xy = self.fourier(xy)
        return self.layers(xy)
    
    def fit(self, x_domain_np, y_domain_np, x_bc, y_bc, t_bc, x_data=None, y_data=None, T_data=None):
        
        x_domain = torch.from_numpy(x_domain_np).to(torch.float32).reshape(-1, 1) 
        y_domain = torch.from_numpy(y_domain_np).to(torch.float32).reshape(-1, 1) 

        x_bc = np_to_torch(x_bc)
        y_bc = np_to_torch(y_bc)
        t_bc = np_to_torch(t_bc) # t_bc is the temperature target value at each boundary point

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.train()
        losses = []

        for ep in range(self.epochs):
            # Create new tensors with requires_grad=True for autograd
            x_domain_torch = x_domain.clone().detach().requires_grad_(True)
            y_domain_torch = y_domain.clone().detach().requires_grad_(True)# removed from both .view(-1, 1)
            optimizer.zero_grad()
            loss = 0.0

            if self.pde_loss:
                lpde = self.pde_weight * self.pde_loss(self, x_domain_torch, y_domain_torch)
                loss += lpde   

            if self.bc_loss:
                lbc = self.bc_weight * self.bc_loss(self, x_bc, y_bc, t_bc)
                loss += lbc

            if self.data_loss and x_data is not None and y_data is not None and T_data is not None:
                x_data_torch = np_to_torch(x_data)
                y_data_torch = np_to_torch(y_data)
                T_data_torch = np_to_torch(T_data)

                ldata = self.data_weight * self.data_loss(self, x_data_torch, y_data_torch, T_data_torch)
                loss += ldata
            
            else:
                ldata = torch.tensor(0.0)


            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
            if ep % int(self.epochs/10)==0:
                print(f"Epoch {ep}: Loss (tot) {loss.item():.3f}, Loss (pde) {lpde}, Loss (bc) {lbc}, Loss (data) {ldata} ")
        
        return losses