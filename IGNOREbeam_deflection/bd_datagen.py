import numpy as np

def generate_collocation_points(n, x_ranges):
    # x_ranges: list of (start, end) tuples for each segment
    xs = []
    for (start, end) in x_ranges:
        xs.append(np.random.uniform(start, end, (n, 1)))
    return [x.astype(np.float32) for x in xs]

# generating more collocation points around the interface so the PINN can deal with the
# discontinuity more effectively.

def generate_interface_collocation_points(n_uniform, n_interface, x_ranges, interface_x, interface_width):
    # Uniform points in each segment
    xs = []
    for (start, end) in x_ranges:
        xs.append(np.random.uniform(start, end, (n_uniform, 1)))
    # Extra points near the interface (Gaussian)
    x_interface = np.random.normal(loc=interface_x, scale=interface_width, size=(n_interface, 1))
    # Keep only points within the beam domain
    x_interface = x_interface[(x_interface >= x_ranges[0][0]) & (x_interface <= x_ranges[-1][1])]
    return [x.astype(np.float32) for x in xs] + [x_interface.astype(np.float32)]

