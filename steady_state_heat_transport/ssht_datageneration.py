#datageneration
import numpy as np

def generate_domain_points(n):
    x = np.random.rand(n)
    y = np.random.rand(n)
    return x, y

def generate_boundary_points(n):
    # n: number of points per boundary
    x_bc = []
    y_bc = []
    t_bc = []

    # x=0 (left)
    x_bc.append(np.zeros(n))
    y_bc.append(np.random.rand(n))
    t_bc.append(np.zeros(n))

    # x=1 (right)
    x_bc.append(np.ones(n))
    y_bc.append(np.random.rand(n))
    t_bc.append(np.zeros(n))

    # y=0 (bottom)
    x_bc.append(np.random.rand(n))
    y_bc.append(np.zeros(n))
    t_bc.append(np.zeros(n))

    # y=1 (top)
    x_top = np.random.rand(n).astype(np.float32)
    sigma=0.25
    t_top = np.exp(-((x_top - 0.5)**2) / (2 * sigma**2))  # Top boundary is a gaussian distribution of temp
    x_bc.append(x_top)
    y_bc.append(np.ones(n))
    t_bc.append(t_top)

    # Stack all
    x_bc = np.concatenate(x_bc)
    y_bc = np.concatenate(y_bc)
    t_bc = np.concatenate(t_bc)
    return x_bc, y_bc, t_bc