from locnetwork import Network
import torch
import torch.nn as nn


class NetworkInverse(Network):

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

        self.r = nn.Parameter(data=torch.tensor([0.])) 