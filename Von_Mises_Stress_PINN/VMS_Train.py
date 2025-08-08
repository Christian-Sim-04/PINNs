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



from VMS_Point_Gen import generate_training_data
from VMS_Loss_Functions import loss_fn_strainenergy, loss_fn_pressure_work, loss_fn_bc
from VMS_network import PINN_Helmholtz
from VMS_csv_Generation import Von_mises_vol_csv, create_pressure_csv


#========================================================================
# GENERATE COLLOCATION POINTS
#========================================================================

stl_path = 'coil_with_housing_union.stl'
num_collocation = 10000
num_boundary = 2000

generate_training_data(stl_path, n_collocation_points=num_collocation, n_boundary_points=num_boundary, batch_size=2000)

function_result = generate_training_data(
    stl_path,
    n_collocation_points=num_collocation,
    n_boundary_points=num_boundary,
    batch_size=2000
)

if function_result is not None:
    collocation_points, collocation_points_np, fixed_boundary_points, fixed_boundary_points_np, scaler = function_result

else:
    # If the function failed, print an error and stop. Most likely a FileNotFound Error (geometry file i wrong directory)
    print("\nPoint generation failed. Check for errors above (e.g., file not found). Exiting the training script.")
    exit()

#========================================================================
# GENERATE PRESSURE POINTS
#========================================================================

# Pressure distributed around the whole coil
Z_MAX = 85.0
Z_MIN = 0.0

INNER_RADIUS = 200.0/2
OUTER_RADIUS = 370.0/2

num_pressure_points = 2000
azimuths = 2 * torch.pi * torch.rand(num_pressure_points, 1)            #  Sampling random angles (0 to 2*pi) & heights (z_min to z_max)
heights = (Z_MAX - Z_MIN) * torch.rand(num_pressure_points, 1) + Z_MIN

x_p = INNER_RADIUS * torch.cos(azimuths)
y_p = INNER_RADIUS * torch.sin(azimuths)  # converting to cartesian coordinates
z_p = heights/2

pressure_points_unscaled_tensor = torch.cat([x_p, y_p, z_p], dim=1)   # points at which pressure is applied
pressure_points_unscaled_np = pressure_points_unscaled_tensor.numpy()
pressure_points_scaled_np = scaler.transform(pressure_points_unscaled_np)

pressure_points = torch.tensor(pressure_points_scaled_np, dtype=torch.float32) # tensor of scaled pressure points


# Unit vectors pointing radially outward from the center
radial_unit_vectors = torch.cat([torch.cos(azimuths), torch.sin(azimuths), torch.zeros_like(azimuths)], dim=1)

# Pressure vectors pointing outward
P_mag = 100.0 # Magnitude of pressure (force per unit area)
pressure_vectors = -P_mag * radial_unit_vectors

#========================================================================
# DEFINE MATERIAL PROPERTIES
#========================================================================

E_material = 195e9  # e.g. Copper  (Aluminium - 69e9)
nu_material = 0.33

#========================================================================
# Initialize Model and Optimizer
#========================================================================

model = PINN_Helmholtz()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

print("\n--- Starting Training ---")
for epoch in range(10001):
    optimizer.zero_grad()
    
    loss_U = loss_fn_strainenergy(model, collocation_points, E_material, nu_material)
    loss_W = loss_fn_pressure_work(model, pressure_points, pressure_vectors)
    loss_BC = loss_fn_bc(model, fixed_boundary_points)
    

    #total_loss = loss_U + 1550.0 * loss_W + 1e6 * loss_BC   NORMAL LOSS FUNCTION WITH DECENT WEIGHTS

    # NOTE THESE WEIGHTS ARE QUITE ACCURATE
    total_loss = 5e4 * torch.pow(loss_BC, 2) + 1.0 * (loss_U - 1550*loss_W) + 1.0 * torch.exp(-(loss_U - loss_W))
     # with realistic values U - W is relatively small and manageable for the network, when the network predicts rigid body motion
     # this term grows huge, so the exp(-(U-V)) becomes a huge number, meaning that the network is pushed away from this solution
    
    total_loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss = {total_loss.item():.6e}")
    
Von_mises_vol_csv(model, collocation_points_np, fixed_boundary_points_np, E_material=E_material, nu_material=nu_material, scaler=scaler)
create_pressure_csv(collocation_points_np, pressure_points, P_mag, scaler=scaler)