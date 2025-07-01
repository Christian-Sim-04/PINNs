import torch
import torch.nn as nn

def np_to_torch(x):
    n_samples = len(x)
    return torch.from_numpy(x).to(torch.float32).reshape(n_samples, -1)

class HeatNet2D(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, n_units=40, epochs=1000, lr=0.001, 
                 pde_loss=None, bc_loss=None, pde_weight=1.0, bc_weight=1.0):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.pde_loss = pde_loss
        self.bc_loss = bc_loss
        self.pde_weight = pde_weight
        self.bc_weight = bc_weight

        self.layers = nn.Sequential(
            nn.Linear(input_dim, n_units),
            nn.ReLU(),
            nn.Linear(n_units, n_units),
            nn.ReLU(),
            nn.Linear(n_units, n_units),
            nn.ReLU(),
            nn.Linear(n_units, n_units),
            nn.ReLU(),
            nn.Linear(n_units, output_dim)
        )

    def forward(self, xy): #forward takes a 1d input so take concatenation of x,y
        return self.layers(xy)
    
    def fit(self, x_domain, y_domain, x_bc, y_bc, t_bc):
        x_domain = np_to_torch(x_domain).requires_grad_()
        y_domain = np_to_torch(y_domain).requires_grad_()
        x_bc = np_to_torch(x_bc)
        y_bc = np_to_torch(y_bc)
        t_bc = np_to_torch(t_bc) # t_bc is the temperature target value at each boundary point

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        losses = []

        for ep in range(self.epochs):
            optimizer.zero_grad()
            loss = 0.0

            if self.pde_loss:
                loss += self.pde_weight * self.pde_loss(self, x_domain, y_domain)

            if self.bc_loss:
                loss += self.bc_weight * self.bc_loss(self, x_bc, y_bc, t_bc)

            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
            if ep % int(self.epochs/10)==0:
                print(f"Epoch {ep}: Loss {loss.item():.3f} ")
        
        return losses
    