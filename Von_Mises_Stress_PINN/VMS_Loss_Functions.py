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

#=============================================================================================================
# LOSS FUNCTIONS
#=============================================================================================================


def get_gradient(y, x):
    """Computes the gradient of y (N, 3) with respect to x (N, 3)."""
    grad_u = torch.autograd.grad(y[:, 0], x, grad_outputs=torch.ones_like(y[:, 0]), create_graph=True)[0]
    grad_v = torch.autograd.grad(y[:, 1], x, grad_outputs=torch.ones_like(y[:, 1]), create_graph=True)[0]
    grad_w = torch.autograd.grad(y[:, 2], x, grad_outputs=torch.ones_like(y[:, 2]), create_graph=True)[0]
    return torch.stack([grad_u, grad_v, grad_w], dim=1)


# Pressure Work Loss Function
def loss_fn_pressure_work(model, pressure_points, pressure_vectors):

    # ensure that the all pressure/forces are ballanced (otherwise PINN will 'cheat' and make the coil move in the dirn
    # of the net force instead of deform)
    
    pressure_vectors = pressure_vectors - torch.mean(pressure_vectors, dim=0, keepdim=True)

    # Get the displacement at the surface points
    u_surface = model(pressure_points)

    # Calculate the work density (dot product of pressure and displacement)
    # We use torch.sum here to perform the dot product for each point
    work_density = torch.sum(pressure_vectors * u_surface, dim=1)

    # The total work is the integral of the density over the surface area.
    # We approximate this with the mean.
    # Note: For physical accuracy, multiply by the total surface area,
    # but for the loss function, the mean is often more stable.
    total_work = torch.mean(work_density)

    return total_work



# Strain Energy Loss Function
def loss_fn_strainenergy(model, x_interior, E, nu):
    u_interior = model(x_interior)
    grad_u = get_gradient(u_interior, x_interior)

    epsilon_xx = grad_u[:, 0, 0]
    epsilon_yy = grad_u[:, 1, 1]
    epsilon_zz = grad_u[:, 2, 2]
    epsilon_xy = 0.5 * (grad_u[:, 0, 1] + grad_u[:, 1, 0])
    epsilon_xz = 0.5 * (grad_u[:, 0, 2] + grad_u[:, 2, 0])
    epsilon_yz = 0.5 * (grad_u[:, 1, 2] + grad_u[:, 2, 1])

    lmbda = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    tr_epsilon = epsilon_xx + epsilon_yy + epsilon_zz

    sigma_xx = lmbda * tr_epsilon + 2 * mu * epsilon_xx
    sigma_yy = lmbda * tr_epsilon + 2 * mu * epsilon_yy
    sigma_zz = lmbda * tr_epsilon + 2 * mu * epsilon_zz
    sigma_xy = 2 * mu * epsilon_xy
    sigma_xz = 2 * mu * epsilon_xz
    sigma_yz = 2 * mu * epsilon_yz

    strain_energy_density = 0.5 * (
        sigma_xx * epsilon_xx + sigma_yy * epsilon_yy + sigma_zz * epsilon_zz +
        2 * (sigma_xy * epsilon_xy + sigma_xz * epsilon_xz + sigma_yz * epsilon_yz)
    )
    
    # taking the mean of the energy density is standard practice for the loss
    strain_energy = torch.mean(strain_energy_density)
    return strain_energy


# Boundary Condition Loss Function
def loss_fn_bc(model, x_boundary):
    u_boundary = model(x_boundary)
    loss_displacement = torch.mean(u_boundary.pow(2))

    
    return loss_displacement