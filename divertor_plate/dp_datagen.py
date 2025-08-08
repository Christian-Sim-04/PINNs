import numpy as np


def normalise(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)


def generate_domain_points_1d_transient(n, L, T_final):
    #Sample n random (x, t) points in the interior domain (excluding boundaries).
    x = np.random.rand(n, 1) * L    # note this scales the normalised distribution from x=[0,1] to x=[0.L]
    t = np.random.rand(n, 1) * T_final
    return x.astype(np.float32), t.astype(np.float32)


def generate_ic_points_1d(n, L, T_initial):
    #Sample n random x points at t=0 for the initial condition.
    x = np.random.rand(n, 1) * L
    t = np.zeros((n, 1), dtype=np.float32)
    T_ic = np.full((n, 1), T_initial, dtype=np.float32)  # all through material the temp is T_initial (thus T_initial=T_coolant)
    return x.astype(np.float32), t, T_ic


def generate_dirichlet_bc_points_1d(n, L, T_final, T_coolant):
    #Sample n random t points at x=L for the Dirichlet/coolant BC.
    x = np.full((n, 1), L, dtype=np.float32)
    t = np.random.rand(n, 1) * T_final
    T_bc = np.full((n, 1), T_coolant, dtype=np.float32)
    return x, t.astype(np.float32), T_bc


def q_flux_gaussian(t, q_max=10**7, t_peak=1.5, width=1e-3):
    # Gaussian pulse for ELM heat flux profile
    return q_max * np.exp(-0.5 * ((t - t_peak) / width) ** 2)

def q_flux_normal(t, q_value=5e5, t_start=0.0, t_end=2.0):
    #this returns a constant heat flux q_value during the plasma pulse, ie 'normal running' of tokamak (no ELM event)
    return np.where((t >= t_start) & (t <= t_end), q_value, 0.0 )


def generate_neumann_bc_points_1d(n, T_final, q_flux_func=None, q_flux_params=None):
    #Sample n random t points at x=0 for the Neumann/plasma-facing BC.
    # ie sampele n random t points at x=0 and return the corresponding x value (0) and q_flux value
    x = np.zeros((n, 1), dtype=np.float32)
    t = np.random.rand(n, 1) * T_final

    if q_flux_func is None:  # ie if no flux profile function is specified, use default to the predefined gaussian
        q_flux_func = q_flux_gaussian
    if q_flux_params is None:
        q_flux_params = {} # ie use the predefined params from the default
    
    q_flux = q_flux_func(t, **q_flux_params).astype(np.float32)   #note the ** unpacks the params to be passed as args to the function

    return x, t.astype(np.float32), q_flux