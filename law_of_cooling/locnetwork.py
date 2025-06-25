import numpy as np
import torch
import torch.nn as nn

#1) designing the fundamental network

def np_to_torch(x):
    n_samples = len(x)
    return torch.from_numpy(x).to(torch.float).reshape(n_samples, -1)

class Network(nn.Module):
    def __init__(self,
        input_dim,
        output_dim,
        n_units=100,
        epochs=1000,
        loss=nn.MSELoss(),
        lr=0.001,
        loss2=None, 
        loss2_weight=0.1,
    ) -> None:
        super().__init__()

        self.epochs = epochs
        self.loss = loss
        self.loss2 = loss2
        self.loss2_weight = loss2_weight
        self.lr = lr
        self.lr = lr
        self.n_units = n_units

        self.layers = nn.Sequential(
            nn.Linear(input_dim, self.n_units),
            nn.ReLU(),
            nn.Linear(self.n_units, self.n_units),
            nn.ReLU(),
            nn.Linear(self.n_units, self.n_units),
            nn.ReLU(),
            nn.Linear(self.n_units, self.n_units),
            nn.ReLU(),
        )
        self.out = nn.Linear(self.n_units, output_dim)

    def forward(self, x):
        h = self.layers(x)
        out = self.out(h)
        return out
    
    def fit(self, X, y):
        Xtensor = np_to_torch(X)
        ytensor = np_to_torch(y) #.float gives float32

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.train()
        losses = []

        for ep in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.forward(Xtensor)
            loss = self.loss(ytensor, outputs)

            if self.loss2:
                loss += self.loss2_weight * self.loss2(self)

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if ep % int(self.epochs/10) == 0:
                print(f"Epoch: {ep}/{self.epochs}, loss: {losses[-1]:.2f}")

        return losses
    
    def predict(self, X):
        self.eval()
        X_tensor = np_to_torch(X)
        out = self.forward(X_tensor)  #added the reshape part last, remove to restore last version
        return out.detach().numpy()