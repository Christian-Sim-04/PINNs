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


# =============================================================================
# 1. POINT GENERATION FROM STL FILES **correct way, only reason im not doing this is because of 
#                                       issues in the geometry resulting in points outside of the geometry bounds
# =============================================================================

import numpy as np
import torch
import pyvista as pv
from sklearn.preprocessing import MinMaxScaler

def generate_training_data(stl_path, n_collocation_points, n_boundary_points, batch_size=2000):
    """
    Loads a single STL file, generates collocation and boundary points,
    and returns them as NumPy arrays.

    Args:
        stl_path (str): The file path to the STL geometry.
        n_collocation_points (int): The target number of interior points.
        n_boundary_points (int): The target number of boundary points.
        batch_size (int): The number of points to process in each batch to save memory.

    Returns:
        tuple: A tuple containing:
            - collocation_points_np (np.ndarray): The interior points.
            - fixed_boundary_points_np (np.ndarray): The boundary points.
    """

    # --- Load and Process Geometries ---
    try:
        print("Loading mesh files...")
        coil_geometry_raw = pv.read('coil_with_housing_union.stl')
        surface = coil_geometry_raw.extract_surface()
        coil_geometry = surface.clean().fill_holes(hole_size=200.0) # because 200 is the size of the inner diameter
                                                                    # any larger and this would be filled in

    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure your STL files are in the correct directory.")
        return


    # Applying MinMaxScaler to the bounds of the geometry
    coil_bounds = coil_geometry.bounds
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler.fit([
        [coil_bounds[0], coil_bounds[2], coil_bounds[4]],  # min corner (xmin, ymin, zmin)
        [coil_bounds[1], coil_bounds[3], coil_bounds[5]]   # max corner (xmax, ymax, zmax)
    ])


    print("Generating collocation points in batches...")
    coil_bounds = coil_geometry.bounds
    collocation_points_list = []
    batch_size = 500 # Process 2000 random points at a time

    # Keep generating batches until we have enough points
    while len(collocation_points_list) < n_collocation_points:
        # Generate one batch of random points
        random_points_np = np.random.uniform(
            low=[coil_bounds[0], coil_bounds[2], coil_bounds[4]],
            high=[coil_bounds[1], coil_bounds[3], coil_bounds[5]],
            size=(batch_size, 3)
        )

        # This is the corrected way to call select_enclosed_points:
        # 1. Create a PolyData object from the random points.
        point_cloud = pv.PolyData(random_points_np)
        # 2. Call the function ON the point_cloud, passing the mesh as the surface.
        selection = point_cloud.select_enclosed_points(coil_geometry, tolerance=1e-8)
        # 3. Get the boolean mask from the result.
        selected_mask = selection.point_data['SelectedPoints'].astype(bool)
        
        # Add the points that were inside the mesh to our list
        enclosed_points = random_points_np[selected_mask]
        if len(enclosed_points) > 0:
            collocation_points_list.extend(enclosed_points)
        
        # Optional: Print progress
        print(f"\r  Found {len(collocation_points_list)} / {n_collocation_points} points...", end="")

    # Combine all found points from the batches into a single array
    collocation_points_np_unscaled = np.array(collocation_points_list)
    # Trim to the exact number desired, in case we overshot
    if len(collocation_points_np_unscaled) > n_collocation_points:
        collocation_points_np_unscaled = collocation_points_np_unscaled[:n_collocation_points]
    
    # Apply the scaler
    collocation_points_np = scaler.transform(collocation_points_np_unscaled)

#==========================================================================================================

    #print("\nGenerating fixed boundary points on bottom surface of coil in batches...") # mimics the coil housing
    #bottom_z = coil_bounds[4]
    #fixed_boundary_points_list = []
    #total_surface_vertices = coil_geometry.points

    # Keep sampling until we find enough points on the bottom surface
    #while len(fixed_boundary_points_list) < n_boundary_points:
    #    # Take a random sample of the mesh's existing vertices
    #    random_indices = np.random.choice(len(total_surface_vertices), batch_size, replace=False)
    #    surface_points_batch = total_surface_vertices[random_indices]
        
        # Filter this smaller batch for points on the bottom
    #    bottom_points_batch = surface_points_batch[
    #        np.isclose(surface_points_batch[:, 2], bottom_z)
    #    ]

        # Add any found points to our list
    #    if len(bottom_points_batch) > 0:
    #        fixed_boundary_points_list.extend(bottom_points_batch)

    #    print(f"\r  Found {len(fixed_boundary_points_list)} / {n_boundary_points} points...", end="")


    # Combine all found points and trim to the exact number
    #fixed_boundary_points_np = np.array(fixed_boundary_points_list)
    #if len(fixed_boundary_points_np) > n_boundary_points:
    #    fixed_boundary_points_np = fixed_boundary_points_np[:n_boundary_points]

#==========================================================================================================

#========================================================================
# GENERATE BOUNDARY CONDITION POINTS
#========================================================================
# (HARD-CODED; im just generating more points on the bottom 
# boundary of the platforms where the brackets will sit so that I can enforce the BCs more reliably
# however this is only possible becuse I know the exact dimensions of the input geometry)

# A better way to achieve this 'naturally' and generalizably would be to just massively increase the 
# number of collocation points and then select the points near the bottom surface of the platforms.
# (but obviously this will massively increase run time, hence why im 'cheating')

# NOTE These points are not used in creating the pressure_data.csv file, though their effect (of
# enforcing the boundary conditions is still very much valid and IS included in the stress plot)

    outer_diameter = 370.0
    outer_radius = outer_diameter / 2.0
    plate_outer_overhang = 5.0
    plate_outer_rad = outer_radius + plate_outer_overhang
    platform_count = 4
    platform_depth = 40.0
    platform_width = 10.0
    plate_thickness = 5.0

        # --- 3. Generate Points for the Platforms ---
    n_platform_points = int(n_boundary_points * 0.1) // platform_count
    print("Generating points for platforms...")
    platform_points_list = []
    for i in range(platform_count):
        angle_rad = np.deg2rad(i * (360 / platform_count))
        
        # Generate points for one platform at the origin
        x_plat_local = np.random.uniform(0, platform_width, n_platform_points)
        y_plat_local = np.random.uniform(-platform_depth / 2, platform_depth / 2, n_platform_points)
        z_plat_local = np.full(n_platform_points, -plate_thickness)
        
        # Translate and rotate the points into position
        x_translated = x_plat_local + plate_outer_rad #+ platform_width / 2
        x_rotated = x_translated * np.cos(angle_rad)
        y_rotated = x_translated * np.sin(angle_rad)
        
        # Combine local y with rotated position
        final_y = y_plat_local * np.cos(angle_rad) + y_rotated
        final_x = -y_plat_local * np.sin(angle_rad) + x_rotated
        
        platform_points = np.stack([final_x, final_y, z_plat_local], axis=1)
        platform_points_list.append(platform_points)
    fixed_boundary_points_np_unscaled = np.vstack(platform_points_list)    # BEFORE ADDING SCALING, THIS WAS INDENTED ONCE MORE
                                                                           # AND THE CODE WORKED
    # Apply scaler
    fixed_boundary_points_np = scaler.transform(fixed_boundary_points_np_unscaled)

    # NOTE FOR VISUALISATION OF THE BOUNDARY POINTS
    import pandas as pd
    # --- Create a Pandas DataFrame to hold the data ---
    print("Creating DataFrame...")
    data = {
        'x': fixed_boundary_points_np[:, 0],
        'y': fixed_boundary_points_np[:, 1],
        'z': fixed_boundary_points_np[:, 2]
    }
    df = pd.DataFrame(data)

    # --- Save the DataFrame to a CSV File ---
    output_filename = 'platform_points_test.csv'
    df.to_csv(output_filename, index=False)

#==========================================================================================

    print(f"\nGenerated {len(collocation_points_np)} collocation points.")
    print(f"Generated {len(fixed_boundary_points_np)} fixed boundary points.")

    # --- Convert to Tensors for PINN ---
    collocation_points = torch.tensor(collocation_points_np, dtype=torch.float32, requires_grad=True)
    fixed_boundary_points = torch.tensor(fixed_boundary_points_np, dtype=torch.float32, requires_grad=True)
 
    return collocation_points, collocation_points_np, fixed_boundary_points, fixed_boundary_points_np, scaler