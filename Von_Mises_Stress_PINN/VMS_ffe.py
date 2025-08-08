import os

# Set the number of threads for all relevant libraries
num_threads = "16"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}")

if device.type == 'cpu':
    torch.set_num_threads(int(num_threads))
    print(f"Limiting cpu threads to: {torch.get_num_threads()}")
    
#==========================================================================================
#==========================================================================================

# Fourier Feature Embedding

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