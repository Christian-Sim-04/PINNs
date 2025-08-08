import torch
import torch.nn as nn

from dp_ffe import FourierFeatureEmbedding
from dp_loss_fns import pde_loss_1d_transient, ic_loss, neumann_bc_loss, dirichlet_bc_loss


class HeatNet1DTransient(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, n_units=40,
                 use_ffe=True, num_frequencies=16, fourier_scale=1.0):
        super().__init__()

        if use_ffe:
            self.fourier = FourierFeatureEmbedding(input_dim, num_frequencies, fourier_scale)
            input_dim = num_frequencies * 2 #note input_dim is the input into the training loop (ie after ffe)
        else:
            self.fourier = None

        self.layers = nn.Sequential(
            nn.Linear(input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, n_units),
            nn.Tanh(),
            nn.Linear(n_units, n_units),
            nn.Tanh(),
            nn.Linear(n_units, n_units),
            nn.Tanh(),
            nn.Linear(n_units, output_dim)
        )
    

    def forward(self, xt):
        if self.fourier:
            xt = self.fourier(xt)
        return self.layers(xt)
    

    def fit(
            self,
            x_domain, t_domain,
            x_ic, T_initial,
            t_bc0, q_flux, k,
            t_bcL, L, T_coolant,
            alpha,
            epochs=1000,
            lr=0.001,
            w_pde=1.0, w_ic=1.0, w_bc0=1.0, w_bcL=1.0
    ):
        
        ### All tensors MUST BE shape (N,1) and type float32

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        losses = []

        ## trying out pretraining the model on the ICs and BCs because it seems like only the pde loss is being optimized

        #for ep in range(500):
         #   optimizer.zero_grad()
          #  loss = 0.0
           # ic_l = w_ic * ic_loss(self, x_ic, T_initial)
            #bcL_l = w_bcL * dirichlet_bc_loss(self, t_bcL, L, T_coolant)
            #bc0_l = w_bc0 * neumann_bc_loss(self, t_bc0, q_flux, k)
            #loss = ic_l + bcL_l
            #loss.backward()
            #optimizer.step()
            #if ep % 50 == 0:
            #    print(f"Pretrain Epoch {ep}: IC Loss {ic_l.item():.4e} | BCL Loss {bcL_l.item():.4e} | BC0 {bc0_l.item():.4e}")

        for ep in range(epochs):
            optimizer.zero_grad()
            loss = 0.0
            #PDE loss
            pde_l = w_pde * pde_loss_1d_transient(self, x_domain, t_domain, alpha)
            #IC loss
            #ic_l = w_ic * ic_loss(self, x_ic, T_initial)
            #Dirichlet BC at x=L
            #bcL_l = w_bcL * dirichlet_bc_loss(self, t_bcL, L, T_coolant)
            #Neumann BC at x=0
            #bc0_l = w_bc0 * neumann_bc_loss(self, t_bc0, q_flux, k)

            loss = pde_l #+ ic_l + bcL_l + bc0_l
            loss.backward()

########################################################################
            #if ep < 2:
                #print("Before optimizer step:")
                #for name, param in self.named_parameters():
                    #print(name, param.data.clone())
############################################################################

            optimizer.step()

##############################################################################
            #if ep < 2:
                #print("After optimizer step:")
                #for name, param in self.named_parameters():
                   #print(name, param.data.clone())
#############################################################################

            losses.append(loss.item())

            if ep % int(epochs/10) == 0:
                print(f"Epoch {ep}: Loss_tot {loss.item():.4e} | PDE {pde_l.item():.4e}")# | IC {ic_l.item():.4e} | BC0 {bc0_l.item():.4e} | BCL {bcL_l.item():.4e}")

        return losses