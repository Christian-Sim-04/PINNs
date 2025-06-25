# First actual example of using PINNs to solve a pde
# --> a cooling cup of coffee
#    dT(t)/dt  =  r(T_env - T(t))
#    T(t) = temp, T_env, temp of environment, r = cooling rate



#1) Design model (input size, output size, forward pass)
#2) Construct loss and optimizer
#3) Training loop
#    - forward pass: compute prediction
#    - backward pass: autograd
#    - update weights



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
        n_units=100,  #n of neurons in each hidden layer
        epochs=1000,
        loss=nn.MSELoss(),
        lr=0.001,
        loss2=None, #this is how we plug in the physics informed loss fn
        loss2_weight=0.1,
    ) -> None: #just confirming that this should have no output
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
    
    def fit(self, X, y):  #fitting to data or training
        Xtensor = np_to_torch(X)
        ytensor = np_to_torch(y)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.train()
        losses = []

        for ep in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.forward(Xtensor)
            loss = self.loss(ytensor, outputs)

            if self.loss2: #loss2 is truthy value if we have a pde, falsy if we dont
                loss += self.loss2_weight * self.loss2(self)

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if ep % int(self.epochs/10) == 0:
                print(f"Epoch: {ep}/{self.epochs}, loss: {losses[-1]:.2f}")

        return losses
    
    def predict(self, X): #gives us the predicted values, e.g.for a plot
        self.eval()
        X_tensor = np_to_torch(X)
        out = self.forward(X_tensor)
        return out.detach().numpy()


class NetworkInverse(Network): #this is used for inverse problems
                               # 'parameter discovery'
    def __init__(
        self,
        input_dim,
        output_dim,
        n_units=100,
        epochs=1000,
        loss=nn.MSELoss(),
        lr=0.001,
        loss2=None,
        loss2_weight=0.1,
    ) -> None:
        super().__init__(input_dim, output_dim, n_units, epochs, loss, lr, loss2, loss2_weight)

        self.r = nn.Parameter(data=torch.tensor([0.]), dtype=torch.float32) 
        #we want to discover the parameter r, so we 'tell the code' that its a learnable param,
        #this way, it will be updated by the optimizer in training
        #we assign the initial value of the parameter to be 0.0 (float) as a tensor.
