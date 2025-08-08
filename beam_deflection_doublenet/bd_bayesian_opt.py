import numpy as np
import torch
import torch.nn as nn
import optuna

from bd_beamdoublenet import BeamDoubleNet
from bd_loss_fns import pde_loss, bc_loss, interface_loss, flexural_rigidity, normalise, denormalise

def run_bayesian_optimisation(data, epochs_per_trial, n_trials, device):
    #unpack the data
    x1_pde_torch, x2_pde_torch, xmin, xmax = data['x1'], data['x2'], data['xmin'], data['xmax']

    def objective(trial):
        pde_weight = trial.suggest_float('pde_weight', 1e-3, 10, log=True)#log scale search
        bc_weight = trial.suggest_float('bc_weight', 1e-6, 10, log=True)
        if_weight = trial.suggest_float('if_weight', 1e-6, 10, log=True) #individual weights now applied

        #if_cont_weight = trial.suggest_float('if_cont_weight', 1.0, 10.0, log=True)
        #if_shear_weight = trial.suggest_float('if_shear_weight', 1e-5, 1e-3, log=True)

        learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
        n_units = 40
        n_layers = 4

        model_beam1 = BeamDoubleNet(
            input_dim=1, output_dim=2, n_units=n_units, n_layers=n_layers,
            pde_weight=pde_weight, bc_weight=bc_weight, if_weight=if_weight,
            #if_cont_weight=if_cont_weight, if_shear_weight=if_shear_weight,
        ).to(device)
        model_beam2 = BeamDoubleNet(
            input_dim=1, output_dim=2, n_units=n_units, n_layers=n_layers,
            pde_weight=pde_weight, bc_weight=bc_weight, if_weight=if_weight,
            #if_cont_weight=if_cont_weight, if_shear_weight=if_shear_weight,
        ).to(device)

        optimizer = torch.optim.Adam(
            list(model_beam1.parameters()) + list(model_beam2.parameters()),
            lr=learning_rate
        )

        for ep in range(epochs_per_trial):
            optimizer.zero_grad()
            loss_residual = pde_loss(model_beam1, x1_pde_torch, xmin, xmax) + pde_loss(model_beam2, x2_pde_torch, xmin, xmax)
            loss_boundary = bc_loss(model_beam1, model_beam2, xmin, xmax)
            loss_interface = interface_loss(model_beam1, model_beam2, xmin, xmax)#, if_shear_weight, if_cont_weight, )
            total_loss = pde_weight * loss_residual + bc_weight * loss_boundary + if_weight * loss_interface
            total_loss.backward()
            optimizer.step()
        
        return total_loss.item()
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    print('Best Trial:')
    trial = study.best_trial
    print(f" Value: {trial.value}")
    print(f" Params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")

    return study.best_params