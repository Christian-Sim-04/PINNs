import numpy as np

# f = w * x

# e.g. f = 2 * x

# training samples
X = np.array([1,2,3,4], dtype=np.float32)
Y = np.array([2,4,6,8], dtype=np.float32)

w = 0.0

# need to calculate the model prediction
# and the loss
# and the gradient


#model prediction
def forward(x):
    return w*x

#loss = MSE  (in the case of linear regression)
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

#gradient
# MSE = 1/N * (wx-y)**2
# dMSE/dw = dMSE/ds * ds/dy_predicted * dy_predicted/dw
def gradient(x,y,y_predicted):
    return np.dot(2*x, (y_predicted-y)).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')
#the .3f means we only get 3 decimal values


#Training
learning_rate = 0.01
n_iterations = 10

for epoch in range(n_iterations):
    #prediction = forward pass
    y_pred = forward(X)

    #loss
    l = loss(Y, y_pred)

    #gradients
    dw = gradient(X,Y,y_pred)

    #update the weights
    w -= learning_rate*dw  #note we go the opposite dirn to the grad

    #printing some information
    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')


#now going to do the same but automatically with PyTorch, next file