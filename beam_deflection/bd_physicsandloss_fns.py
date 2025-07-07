import torch
import numpy as np

def flexural_rigidity(x):
    E1, D1 = 210e9, 0.05
    E2, D2 = 180e9, 0.04

    I1 = np.pi/64 * D1**4
    I2 = np.pi/64 * D2**4

    EI1 = E1 * I1
    EI2 = E2 * I2

    return torch.where(x <= 2.0, torch.tensor(EI1, device=x.device), torch.tensor(EI2, device=x.device))

## returns flex.rig.1 for all values of x along first beam and flex.rig.2 for second beam


def pde_loss(model, x):
    x.requires_grad_(True)
    w = model(x)

    w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]
    w_xxx = torch.autograd.grad(w_xx, x, grad_outputs=torch.ones_like(w_xx), create_graph=True)[0]
    w_xxxx = torch.autograd.grad(w_xxx, x, grad_outputs=torch.ones_like(w_xxx), create_graph=True)[0]
    #note we don't include the point load in the pde loss, but in the interface loss term

    EI = flexural_rigidity(x)
    residual = EI * w_xxxx
    return (residual**2).mean()



def bc_loss(model):

    #x=0, w(0)=0, w'(0)=0
    x0 = torch.zeros((1,1), dtype=torch.float32, requires_grad=True)
    w0 = model(x0)
    w0_x = torch.autograd.grad(w0, x0, grad_outputs=torch.ones_like(w0), create_graph=True)[0]
    loss_A = (w0**2).mean() + (w0_x**2).mean()

    #x=3, M(3)=-EI*w''(3)=250, V(3)=-EI*w'''(3)=0
    x3 = torch.full((1,1), 3.0, dtype=torch.float32, requires_grad=True)
    w3 = model(x3)
    w3_x = torch.autograd.grad(w3, x3, grad_outputs=torch.ones_like(w3), create_graph=True)[0]
    w3_xx = torch.autograd.grad(w3_x, x3, grad_outputs=torch.ones_like(w3_x), create_graph=True)[0]
    w3_xxx = torch.autograd.grad(w3_xx, x3, grad_outputs=torch.ones_like(w3_xx), create_graph=True)[0]
    
    EI3 = flexural_rigidity(x3)
    loss_B = ((-EI3 * w3_xx - 250)**2).mean() + ((-EI3 * w3_xxx)**2).mean()

    return loss_A + loss_B


def interface_loss(model):
    #take a point just left and just right of the interface
    epsilon = 1e-5
    x2_left = (2.0 - epsilon) * torch.ones((1,1), dtype=torch.float32, requires_grad=True)
    x2_right = (2.0 + epsilon) * torch.ones((1,1), dtype=torch.float32, requires_grad=True)

    wL = model(x2_left)
    wR = model(x2_right)

    wL_x = torch.autograd.grad(wL, x2_left, grad_outputs=torch.ones_like(wL), create_graph=True)[0]
    wR_x = torch.autograd.grad(wR, x2_right, grad_outputs=torch.ones_like(wR), create_graph=True)[0]

    wL_xx = torch.autograd.grad(wL_x, x2_left, grad_outputs=torch.ones_like(wL_x), create_graph=True)[0]
    wR_xx = torch.autograd.grad(wR_x, x2_right, grad_outputs=torch.ones_like(wR), create_graph=True)[0]

    wL_xxx = torch.autograd.grad(wL_xx, x2_left, grad_outputs=torch.ones_like(wL_xx), create_graph=True)[0]
    wR_xxx = torch.autograd.grad(wR_xx, x2_right, grad_outputs=torch.ones_like(wR_xx), create_graph=True)[0]

    EI1 = flexural_rigidity(x2_left)
    EI2 = flexural_rigidity(x2_right)

    #continuity conditions w, w', w''
    cont_loss = ((wL-wR)**2).mean() + ((wL_x-wR_x)**2).mean() + ((EI1*wL_xx-EI2*wR_xx)**2).mean()

    #shear jump (point load)
    shear_jump = ((EI2*wR_xxx - EI1*wL_xxx + 500)**2).mean()

    return cont_loss + shear_jump