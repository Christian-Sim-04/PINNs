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

#==========================================================================================================
#==========================================================================================================

import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from VMS_Loss_Functions import get_gradient



# PRESSURE (INPUT TO NETWORK)
def create_pressure_csv(collocation_points_scaled_np, pressure_points_scaled_tensor, P_mag, scaler):
    """
    Creates a pressure visualization CSV by combining volumetric points and dedicated surface points.

    Args:
        collocation_points_tensor (torch.Tensor): The (N, 3) tensor of points inside the coil.
        pressure_points_tensor (torch.Tensor): The (M, 3) tensor of points on the inner surface.
        P_mag (float): The magnitude of the uniform pressure.
    """
    print("\n--- Creating Pressure Visualization CSV from Dedicated Points ---")

    # Convert tensors to NumPy arrays for processing
    pressure_points_scaled_np = pressure_points_scaled_tensor.detach().cpu().numpy()

    # UNSCALING
    collocation_points_np = scaler.inverse_transform(collocation_points_scaled_np)
    pressure_points_np = scaler.inverse_transform(pressure_points_scaled_np)

    # 1. Create pressure values for each set of points
    # Pressure is 0 for the internal (collocation) points
    pressure_values_collocation = np.zeros(len(collocation_points_np))
    
    # Pressure is P_mag for the dedicated surface points
    pressure_values_surface = np.full(len(pressure_points_np), P_mag)

    # 2. Combine the points and their corresponding pressure values into one set
    all_points = np.concatenate([collocation_points_np, pressure_points_np], axis=0)
    all_pressures = np.concatenate([pressure_values_collocation, pressure_values_surface], axis=0)

    # 3. Create a pandas DataFrame
    data = {
        'x': all_points[:, 0],
        'y': all_points[:, 1],
        'z': all_points[:, 2],
        'pressure': all_pressures
    }
    df = pd.DataFrame(data)

    # 4. Save to CSV
    output_filename = 'pressure_data.csv'
    df.to_csv(output_filename, index=False)

    print(f"Calculation complete. Volumetric pressure results saved to {output_filename}.")
    print(f"Assigned pressure to {len(pressure_points_np)} dedicated surface points.")




# STRESS (OUTPUT OF NETWORK)
def Von_mises_vol_csv(model, collocation_points_scaled_np, fixed_boundary_points_scaled_np, E_material, nu_material, scaler):
    eval_points_scaled_np = np.vstack([collocation_points_scaled_np, fixed_boundary_points_scaled_np])
    eval_points_scaled_torch = torch.tensor(eval_points_scaled_np, dtype=torch.float32, requires_grad=True).to(next(model.parameters()).device)
    model.eval()

    # CALCULATE STRESS
    u_pred = model(eval_points_scaled_torch)
    grad_u = get_gradient(u_pred, eval_points_scaled_torch)
    
    E = E_material
    nu = nu_material
    lmbda = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    epsilon_xx = grad_u[:, 0, 0]
    epsilon_yy = grad_u[:, 1, 1]
    epsilon_zz = grad_u[:, 2, 2]
    epsilon_xy = 0.5 * (grad_u[:, 0, 1] + grad_u[:, 1, 0])
    epsilon_xz = 0.5 * (grad_u[:, 0, 2] + grad_u[:, 2, 0])
    epsilon_yz = 0.5 * (grad_u[:, 1, 2] + grad_u[:, 2, 1])
    tr_epsilon = epsilon_xx + epsilon_yy + epsilon_zz

    sigma_xx = lmbda * tr_epsilon + 2 * mu * epsilon_xx
    sigma_yy = lmbda * tr_epsilon + 2 * mu * epsilon_yy
    sigma_zz = lmbda * tr_epsilon + 2 * mu * epsilon_zz
    sigma_xy = 2 * mu * epsilon_xy
    sigma_xz = 2 * mu * epsilon_xz
    sigma_yz = 2 * mu * epsilon_yz

    term1 = (sigma_xx - sigma_yy)**2 + (sigma_yy - sigma_zz)**2 + (sigma_zz - sigma_xx)**2
    term2 = 6 * (sigma_xy**2 + sigma_yz**2 + sigma_xz**2)
    von_mises_stress = torch.sqrt(0.5 * (term1 + term2)).cpu().detach().numpy()

    eval_points_np = scaler.inverse_transform(eval_points_scaled_np)

    # pandas dataframe to hold the data
    data = {
        'x': eval_points_np[:,0],
        'y': eval_points_np[:,1],
        'z': eval_points_np[:,2],
        'von_mises_stress': von_mises_stress
    }
    df = pd.DataFrame(data)

    # we want this data in a csv file
    output_filename = 'stress_data.csv'
    df.to_csv(output_filename, index=False)

    print(f"\nCalculation complete. Volumetric stress results saved to {output_filename}.")