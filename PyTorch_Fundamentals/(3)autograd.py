import torch

x = torch.randn(3, requires_grad=True)
print(x)

#whenever we do operations with this tensor
#pytorch will create a computational graph

y = x+2
print(y)
#gives add backward

z = y*y*2
print(z)
#mul backward

#z=z.mean()
print(z)
#mean backward


v = torch.tensor([0.1,1.2,0.001], dtype=torch.float32)

z.backward(v) # calculates dz/dx (no argument because scalar - we took the mean so only one value)
print(x.grad)  
# x now has a .grad attribute where the gradients
# are stored



##################################################
#if we dont have a single valued tensor
#we need line 21, the matrix product of the tensor
#and a vector where the vector is what is explicitly
#written, then we must also put 'v' in the 
#backpropagation command

#if scalar, like with the mean, we don't need v







#