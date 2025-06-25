import torch

x = torch.zeros(2, 2)

print(x)  # this is a 2d tensor
print(x[1,:])    # row no.1(second row) and all columns

print(x[1,1])  #tensor of 1 element
print(x[1,1].item())  #gives the value of the tensor



y = torch.rand(4,4)
print(y)
z = y.view(16)
print(z)   
# gives a 1d tensor with all of the components
# of the 2d tensor

#if we dont want it in 1d but in a different shape
# we can use this

t = y.view(-1,8) 
print(t)
print(t.size())
# this gives the reshaped tensor, with whatever
# other dimension is required for the given 
# 8 columns (which is 2)



##############################################

