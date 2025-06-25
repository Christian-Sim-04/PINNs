import torch

weights = torch.ones(4, requires_grad=True)

for epoch in range(1):
    model_output = (weights*3).sum()

    model_output.backward()

    print(weights.grad)

    weights.grad.zero_()

#this gives the tensor containing the gradients
#note, for each iteration we have to initialise
#the weights.grad, otherwise the gradients 
#are wrong



# this goes for any optimization step
# when we are going onto the next iteration
# of an optimization, we must empty the
# gradients

#e.g.
#z.backward()
#weights.grad.zero_()