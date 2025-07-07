import numpy as np

def generate_collocation_points(n, x_ranges):
    # x_ranges: list of (start, end) tuples for each segment
    xs = []
    for (start, end) in x_ranges:
        xs.append(np.random.uniform(start, end, (n, 1)))
    return [x.astype(np.float32) for x in xs]