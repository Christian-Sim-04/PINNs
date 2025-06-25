

#1) Design model (input size, output size, forward pass)
#2) Construct loss and optimizer
#3) Training loop
#    - forward pass: compute prediction
#    - backward pass: autograd
#    - update weights



import torch
import torch.nn as nn


#new shape, number of rows = number of samples
# in each row we have the features
X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
print(n_samples, n_features)
# this gives 4 1  ==>  4 samples, 1 feature for each sample


input_size = n_features  # =1
output_size = n_features # =1

model = nn.Linear(input_size, output_size)   
#very simple linear regression, because we only have 1 layer

#if we need a custom linear regression model
class LinearRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        #here we define layers
        self.lin = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.lin(x)
    
model = LinearRegression(input_size, output_size)
#this will do the same thing as above, but if we had a more
#complicated model, then this is the method we'd have to use


print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')



#Training
learning_rate = 0.01
n_iterations = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iterations):
    #prediction = forward pass
    y_pred = model(X)

    #loss
    l = loss(Y, y_pred)

    #gradients = backward pass
    l.backward()  #dl/dw

    #update the weights
    optimizer.step() #does an optimization step
    
    #zero the gradients (stored in w.grad attribute)
    optimizer.zero_grad()


    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')
        #w[0][0] because its a list of lists, .item() becuase we want the value
        #not the tensor

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')

