import numpy as np

def fourier_features(input_vec, B):
    """
    Compute Fourier feature embedding for a single input vector.
    Args:
        input_vec: numpy array of shape (d,)
        B: numpy array of shape (d, m)
    Returns:
        features: numpy array of shape (2m,)
    """
    x_proj = input_vec @ B  # shape = (m,)
    sin_feats = np.sin(2 * np.pi * x_proj)
    cos_feats = np.cos(2 * np.pi * x_proj)
    features = np.concatenate([sin_feats, cos_feats])
    return features