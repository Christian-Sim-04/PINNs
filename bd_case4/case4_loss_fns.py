import torch
import numpy as np

def normalise(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)

def denormalise(x_norm, xmin, xmax):
    return x_norm * (xmax - xmin) + xmin

def flexural_rigidity(x_norm, xmin, xmax):

    interface_norm = (2.0 - xmin)/(xmax - xmin)

    E, D = 210e9, 0.05

    I = np.pi/64 * D**4

    EI = E * I

    return torch.tensor(EI, device=x_norm.device)

def get_EI_vals():
    E, D = 210e9, 0.05

    I = np.pi/64 * D**4

    EI = E * I
    
    return EI

## returns flex.rig.1 for all values of x along first beam and flex.rig.2 for second beam


def pde_loss(model, x, xmin, xmax, q0):
    x.requires_grad_(True)
    out = model(x)

    w = out[:, 0:1] #takes the first column in the shape (N,1), if [:,0] would give shape (N,)
    v = out[:, 1:2]

    scale = xmax-xmin

    # for the two-eqn-system, eq1: v - w'' = 0.
    w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]
    w_xx_physical = w_xx/(scale**2)
    eq1 = v - w_xx_physical

    # for the two-eqn-syst, eq2: EI * v''(x) = q_0
    EI = flexural_rigidity(x, xmin, xmax)
    
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_xx_physical = v_xx / (scale**2) #note, v is still just a fn of normalised x, hence why scale**2 not **4
    eq2 = EI * v_xx_physical - q0


    EI = get_EI_vals()
    EI_squared_inv = 1.0 / (EI)**2 

    residual = (eq1**2).mean() + EI_squared_inv*(eq2**2).mean()   #_denorm
    return residual



def bc_loss(model_single_beam, q0, xmin, xmax):
    scale = xmax-xmin
    #x=0, w(0)=0, w'(0)=0
    x0 = torch.full((1,1), 0.0, dtype=torch.float32, requires_grad=True)
    w0 = model_single_beam(x0)
    w0_x = torch.autograd.grad(w0, x0, grad_outputs=torch.ones_like(w0), create_graph=True)[0]
    w0_x_physical = w0_x / scale
    loss_A = (w0**2).mean() + (w0_x_physical**2).mean()  #.._denorm if denormalising inside

    #x=1, M(1)=-EI*w''(1)=0, V(1)=-EI*w'''(1)=-500      note x=1 because of normalisation
    x3 = torch.full((1,1), 1.0, dtype=torch.float32, requires_grad=True)
    out3 = model_single_beam(x3)
    v3 = out3[:, 1:2]
    v3_x = torch.autograd.grad(v3, x3, grad_outputs=torch.ones_like(v3), create_graph=True)[0]
    v3_x_physical = v3_x / scale

    EI3 = flexural_rigidity(x3, xmin, xmax)
    loss_B = ((EI3 * v3)**2).mean() + ((EI3 * v3_x_physical + q0)**2).mean()

    return loss_A + loss_B


def multiple_interface_loss(model_single_beam, xmin, xmax, interface_locations, point_forces):#, if_shear_weight, if_cont_weight):
    #take a point just left and just right of the interface

    total_cont_loss = 0.0
    total_shear_loss = 0.0
    scale = xmax - xmin
    epsilon = 1e-5
    EI = get_EI_vals()

    for x_loc, force in zip(interface_locations, point_forces):

        x2_left = torch.full((1,1), normalise(x_loc - epsilon, xmin, xmax), dtype=torch.float32, requires_grad=True)
        x2_right = torch.full((1,1), normalise(x_loc + epsilon, xmin, xmax), dtype=torch.float32, requires_grad=True)

        outL = model_single_beam(x2_left)
        outR = model_single_beam(x2_right)

        wL, vL = outL[:, 0:1], outL[:, 1:2]
        wR, vR = outR[:, 0:1], outR[:, 1:2]

        # First derivatives of w
        wL_x = torch.autograd.grad(wL, x2_left, grad_outputs=torch.ones_like(wL), create_graph=True)[0]
        wR_x = torch.autograd.grad(wR, x2_right, grad_outputs=torch.ones_like(wR), create_graph=True)[0]
        wL_x_physical = wL_x / scale
        wR_x_physical = wR_x / scale

        # First derivatives of v (i.e., third derivatives of w)
        vL_x = torch.autograd.grad(vL, x2_left, grad_outputs=torch.ones_like(vL), create_graph=True)[0]
        vR_x = torch.autograd.grad(vR, x2_right, grad_outputs=torch.ones_like(vR), create_graph=True)[0]
        vL_x_physical = vL_x / scale
        vR_x_physical = vR_x / scale

        # --- ADD THIS PRINT STATEMENT ---
        #print(f"  Interface at x={x_loc}, force={force}:")
        #print(f"    v_left_derivative = {vL_x_physical.item():.6f}")
        #print(f"    v_right_derivative = {vR_x_physical.item():.6f}")
        # --------------------------------

        EI = flexural_rigidity(x2_left, xmin, xmax)

        #continuity conditions w, w', w''
        cont_loss = ((wL-wR)**2).mean() + ((wL_x_physical-wR_x_physical)**2).mean() + ((EI*vL-EI*vR)**2).mean()  #_denorms
        total_cont_loss += cont_loss
        #print(f"cont loss {total_cont_loss}")
        
        #shear jump (point load)
        shear_jump = ((EI*vR_x_physical - EI*vL_x_physical + force)**2).mean()
        total_shear_loss += shear_jump
        #print(f"shear loss {total_shear_loss}")

    return total_cont_loss + total_shear_loss