#loss_fns

import torch

def pde_loss_2d(model, x, y):

    xy = torch.cat([x, y], dim=1)
    T = model(xy)

    #print(f"T: {T}")

    T_x = torch.autograd.grad(T.sum(), x, create_graph=True)[0]
    T_y = torch.autograd.grad(T.sum(), y, create_graph=True)[0]

    #print(f"T_x: {T_x}")

    T_xx = torch.autograd.grad(T_x.sum(), x, create_graph=True)[0]
    T_yy = torch.autograd.grad(T_y.sum(), y, create_graph=True)[0]

    #print(f"T_xx: {T_xx}")

    return ((T_xx + T_yy) ** 2).mean()

def bc_loss_2d(model, x_bc, y_bc, t_bc):
    xy = torch.cat([x_bc, y_bc], dim=1)
    T = model(xy)

    return ((T - t_bc) ** 2).mean()


def data_loss_2d(model, x_data, y_data, T_data):
    xy = torch.cat([x_data, y_data], dim=1)
    T_pred = model(xy)
    return ((T_pred - T_data) ** 2).mean()