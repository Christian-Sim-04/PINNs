import torch

# f = w * x

# e.g. f = 2 * x

# training samples
X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)
#obviously note that we want the algorithm to converge onto
#w=2 as this is the operation that takes us from X to Y

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# need to calculate the model prediction
# and the loss
# and the gradient


#model prediction
def forward(x):
    return w*x

#loss = MSE  (in the case of linear regression)
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()


print(f'Prediction before training: f(5) = {forward(5):.3f}')
#the .3f means we only get 3 decimal values


#Training
learning_rate = 0.01
n_iterations = 100

for epoch in range(n_iterations):
    #prediction = forward pass
    y_pred = forward(X)

    #loss
    l = loss(Y, y_pred)

    #gradients = backward pass
    l.backward()  #dl/dw

    #update the weights
    # we don't want this to be part of the computational graph
    #hence we need the following statement to ensure the history
    #of dw is not recorded
    with torch.no_grad():
        w -= learning_rate*w.grad
    
    #zero the gradients (stored in w.grad attribute)
    w.grad.zero_()


    #printing some information
    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')


