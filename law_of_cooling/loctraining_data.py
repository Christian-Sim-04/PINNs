# before we can train our model, we need to generate the data
# to train it on (training data)

# note, for comparison, we will also generate the true data and
# plot it too

import numpy as np
import matplotlib.pyplot as plt

Tenv = 25 #temp of environment
T0 = 100 #initial temperature
k = 0.001 #arbitrary start point for cooling const.
time = np.linspace(0,1000,1000)
np.random.seed()

def true_sol(t):
    return Tenv + (T0 - Tenv)*np.exp(-k*t)

temp = true_sol(time)

t = np.linspace(0,300,10) # for taining data, only up to 300 with step 10
T = true_sol(t) + 2*np.random.randn(10)


print(np.shape(time))
print(np.shape(temp))
print(np.shape(t))
print(np.shape(T))


plt.plot(time, temp)
plt.plot(t, T, 'o')
plt.legend(['Equation', 'Training data'])
plt.ylabel('Temp (C)')
plt.xlabel('Time (s)')



