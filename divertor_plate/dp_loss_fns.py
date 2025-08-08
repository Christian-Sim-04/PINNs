import torch
import numpy as np

def pde_loss_1d_transient(model, x, t, alpha):
    # x, t: tensors with shape (N, 1)
    # x.requires_grad_(True)
    # t.requires_grad_(True)
    xt = torch.cat([x, t], dim=1)
    T = model(xt)

    # dT/dt
    T_t = torch.autograd.grad(T, t, grad_outputs=torch.ones_like(T), create_graph=True)[0]  #this is calculating grad_outputs_i * doutputs_i/dinputs where, clearly grad_outputs_i=1
    # dT/dx
    T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    # d2T/dx2
    T_xx = torch.autograd.grad(T_x, x, grad_outputs=torch.ones_like(T_x), create_graph=True)[0]

    residual = T_t - alpha * T_xx
    return (residual ** 2).mean()



def ic_loss(model, x_ic, T_initial):
    # x_ic: (N, 1), T_initial: scalar or (N, 1)
    t_ic = torch.zeros_like(x_ic)
    xt_ic = torch.cat([x_ic, t_ic], dim=1)
    T_pred = model(xt_ic)
    return ((T_pred - T_initial) ** 2).mean()


def dirichlet_bc_loss(model, t_bc, L, T_coolant):  #ie at the coolant boundary
    # t_bc: (N, 1), L: scalar, T_coolant: scalar or (N, 1)
    x_bc = torch.full_like(t_bc, L)   # ie creates a torch with same shape as t_bc full of Ls
    xt_bc = torch.cat([x_bc, t_bc], dim=1)
    T_pred = model(xt_bc)
    return ((T_pred - T_coolant) ** 2).mean()


def neumann_bc_loss(model, t_bc, q_flux, k):  #ie at the plasma-facing boundary
    # t_bc: (N, 1), q_flux: (N, 1) or scalar
    x_bc = torch.zeros_like(t_bc)
    x_bc.requires_grad_(True)
    xt_bc = torch.cat([x_bc, t_bc], dim=1)
    T = model(xt_bc)
    # dT/dx at x=0
    T_x = torch.autograd.grad(T, x_bc, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    # -k * dT/dx = q_flux(t)
    bc_residual = -k * T_x - q_flux
    return (bc_residual ** 2).mean()