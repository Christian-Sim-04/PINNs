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


import torch.nn as nn

#======================================================================================================================
#======================================================================================================================

class PINN_Helmholtz(nn.Module):
    def __init__(self, input_features=3, output_features=3):#, num_frequencies=16, scale=1.0):
        super().__init__()

        #self.fourier_features = FourierFeatureEmbedding(
        #    input_features, num_frequencies, scale
        #)

        #fourier_layer_out_dimensions = 2 * num_frequencies    # for each frequency we have both a sine and cosine term

        self.net = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, output_features)
        )

    def forward(self, x):
        #embedded_x = self.fourier_features(x)
        return self.net(x)