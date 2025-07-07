import numpy as np
import torch
import torch.nn as nn

class FourierFeatureEmbedding(nn.Module):
    def __init__(self, input_feat, num_frequencies=16, scale=1):
        super().__init__()
        self.B = nn.Parameter(
            torch.randn((input_feat, num_frequencies)) * scale, requires_grad=False
        )

    def forward(self, x):
        #x has shape (N, in_feat) like usual input tensors

        x_proj = x @ self.B   # (N, num_frequencies)

        return torch.cat([torch.sin(2*np.pi*x_proj), torch.cos(2*np.pi*x_proj)], dim=-1)
        #note, if x_proj shape (N, m), then ffe shape (N, 2*m) where all the sines are listed
        # then all the cosines