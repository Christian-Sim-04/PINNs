import torch
import numpy as np

a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

#if the tensor is stored on the cpu and not the gpu
#then the tensor and the array will point to the same
#memory location, so if we modify the tensor in-place
# (underscore functions) it will modify both

a.add_(1)
print(a)
print(b)



####################going the other way

a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)

#the same applies, my modifying the numpy array
#the tensor will also be modified (if on the cpu)



#Finally, often we see this...

x = torch.ones(5, requires_grad=True)
print(x)
# whenever we have a variable that we want to
#optimize, then we need the gradients