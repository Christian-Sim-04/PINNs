import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def normalise(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)

#def normalise(x):
 #   scaler = MinMaxScaler(feature_range=(-1,1))
  #  norm_x = scaler.fit_transform(x)
   # return norm_x, scaler

def denormalise(x_norm, xmin, xmax):
    return x_norm * (xmax - xmin) + xmin

def flexural_rigidity(x_norm, xmin, xmax):
    #x = denormalise(x_norm, xmin, xmax)

    interface_norm = (2.0 - xmin)/(xmax - xmin)

    E1, D1 = 210e9, 0.05
    E2, D2 = 180e9, 0.04

    I1 = np.pi/64 * D1**4
    I2 = np.pi/64 * D2**4

    EI1 = E1 * I1
    EI2 = E2 * I2

    return torch.where(x_norm <= interface_norm, torch.tensor(EI1, device=x_norm.device), torch.tensor(EI2, device=x_norm.device))

## returns flex.rig.1 for all values of x along first beam and flex.rig.2 for second beam


def pde_loss(model, x, xmin, xmax):
    x.requires_grad_(True)
    #w = model(x)
    out = model(x)

    w = out[:, 0:1] #takes the first column in the shape (N,1), if [:,0] would give shape (N,)
    v = out[:, 1:2]


    # for the two-eqn-system, eq1: v - w'' = 0.
    w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]
        #w_xxx = torch.autograd.grad(w_xx, x, grad_outputs=torch.ones_like(w_xx), create_graph=True)[0]
        #w_xxxx = torch.autograd.grad(w_xxx, x, grad_outputs=torch.ones_like(w_xxx), create_graph=True)[0]
    #note we don't include the point load in the pde loss, but in the interface loss term
    eq1 = v - w_xx

    # for the two-eqn-syst, eq2: (EI(x)v)'' = 0
    EI = flexural_rigidity(x, xmin, xmax)  #note, denormalisation is implicitly called here
        #scale = xmax-xmin
        #w_xxxx_denorm = w_xxxx / (scale **4)  #due to the chain rule
    EI_v = EI * v
    EI_v_x = torch.autograd.grad(EI_v, x, grad_outputs=torch.ones_like(EI_v), create_graph=True)[0]
    EI_v_xx = torch.autograd.grad(EI_v_x, x, grad_outputs=torch.ones_like(EI_v_x), create_graph=True)[0]
    eq2 = EI_v_xx

    residual = (eq1**2).mean() + (eq2**2).mean()   #_denorm
    return residual



def bc_loss(model, xmin, xmax):
    #scale = xmax - xmin

    #x=0, w(0)=0, w'(0)=0
    x0 = torch.full((1,1), 0.0, dtype=torch.float32, requires_grad=True)
    w0 = model(x0)
    w0_x = torch.autograd.grad(w0, x0, grad_outputs=torch.ones_like(w0), create_graph=True)[0]
    #w0_x_denorm = w0_x / scale
    loss_A = (w0**2).mean() + (w0_x**2).mean()  #.._denorm if denormalising inside

    #x=1, M(1)=-EI*w''(1)=250, V(1)=-EI*w'''(1)=0      note x=1 because of normalisation
    x3 = torch.full((1,1), 1.0, dtype=torch.float32, requires_grad=True)
    w3 = model(x3)
    w3_x = torch.autograd.grad(w3, x3, grad_outputs=torch.ones_like(w3), create_graph=True)[0]
    w3_xx = torch.autograd.grad(w3_x, x3, grad_outputs=torch.ones_like(w3_x), create_graph=True)[0]
    w3_xxx = torch.autograd.grad(w3_xx, x3, grad_outputs=torch.ones_like(w3_xx), create_graph=True)[0]
    
    #w3_xx_denorm = w3_xx / (scale**2)
    #w3_xxx_denorm = w3_xxx / (scale**3)

    EI3 = flexural_rigidity(x3, xmin, xmax)
    loss_B = ((EI3 * w3_xx + 250)**2).mean() + ((EI3 * w3_xxx)**2).mean()  #w3_xx_denorm if denormalising

    return loss_A + loss_B


def interface_loss(model, xmin, xmax):
    #scale = xmax -xmin

    #take a point just left and just right of the interface
    epsilon = 1e-5
    x2_left = torch.full((1,1), normalise(2.0 - 1e-5, xmin, xmax), dtype=torch.float32, requires_grad=True)
    x2_right = torch.full((1,1), normalise(2.0 + 1e-5, xmin, xmax), dtype=torch.float32, requires_grad=True)


    outL = model(x2_left)
    outR = model(x2_right)

    wL, vL = outL[:, 0:1], outL[:, 1:2]
    wR, vR = outR[:, 0:1], outR[:, 1:2]

        # First derivatives of w
    wL_x = torch.autograd.grad(wL, x2_left, grad_outputs=torch.ones_like(wL), create_graph=True)[0]
    wR_x = torch.autograd.grad(wR, x2_right, grad_outputs=torch.ones_like(wR), create_graph=True)[0]

        # First derivatives of v (i.e., third derivatives of w)
    vL_x = torch.autograd.grad(vL, x2_left, grad_outputs=torch.ones_like(vL), create_graph=True)[0]
    vR_x = torch.autograd.grad(vR, x2_right, grad_outputs=torch.ones_like(vR), create_graph=True)[0]

    #wL = model(x2_left)
    #wR = model(x2_right)

    #wL_x = torch.autograd.grad(wL, x2_left, grad_outputs=torch.ones_like(wL), create_graph=True)[0]
    #wR_x = torch.autograd.grad(wR, x2_right, grad_outputs=torch.ones_like(wR), create_graph=True)[0]

    #wL_xx = torch.autograd.grad(wL_x, x2_left, grad_outputs=torch.ones_like(wL_x), create_graph=True)[0]
    #wR_xx = torch.autograd.grad(wR_x, x2_right, grad_outputs=torch.ones_like(wR), create_graph=True)[0]

    #wL_xxx = torch.autograd.grad(wL_xx, x2_left, grad_outputs=torch.ones_like(wL_xx), create_graph=True)[0]
    #wR_xxx = torch.autograd.grad(wR_xx, x2_right, grad_outputs=torch.ones_like(wR_xx), create_graph=True)[0]

    #print(f"wL {wL}, wL_x {wL_x}, wL_xx, {wL_xx}, wL_xxx {wL_xxx}")

    #wL_x_denorm = wL_x / scale
    #wR_x_denorm = wR_x / scale
    #wL_xx_denorm = wL_xx / (scale ** 2)
    #wR_xx_denorm = wR_xx / (scale ** 2)
    #wL_xxx_denorm = wL_xxx / (scale ** 3)
    #wR_xxx_denorm = wR_xxx / (scale ** 3)

    EI1 = flexural_rigidity(x2_left, xmin, xmax)
    EI2 = flexural_rigidity(x2_right, xmin, xmax)

    #continuity conditions w, w', w''
    cont_loss = ((wL-wR)**2).mean() + ((wL_x-wR_x)**2).mean() + ((EI1*vL-EI2*vR)**2).mean()  #_denorms

    #shear jump (point load)
    shear_jump = ((EI2*vR_x - EI1*vL_x + 500)**2).mean()   #_denorms

    return cont_loss + shear_jump