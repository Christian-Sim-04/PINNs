# example for 2d input with 3 different frequencies

import numpy as np
from ffe_function import fourier_features


#1) Defining the input and random frequency matrix
x = 1
y = 1

input_vec = np.array([x,y])  # shape = (2,)
B = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])  # shape = (2, m) where m=3


features = fourier_features(input_vec, B)
print(f"Fourier features: {features}")