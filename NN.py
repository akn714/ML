# imports
import numpy as np

# inputs
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# outpus
Y = np.array([[0],
              [1],
              [1],
              [0]])

def init_params():
    # initializes all the weights and biases to random values from -0.5 to 0.5
    W1 = np.random.uniform(-0.5, 0.5, (2, 4))
    W2 = np.random.uniform(-0.5, 0.5, (4, 1))
    b1 = np.random.uniform(-0.5, 0.5, (4, 1))
    b2 = np.random.uniform(-0.5, 0.5, (1, 1))
    return W1, W2, b1, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def forward_prop(W1, W2, b1, b2):
    A0 = X.T # 2x4
    Z1 = W1.T.dot(A0) + b1 # 4x4
    A1 = ReLU(Z1) # 4x4
    Z2 = W2.T.dot(A1) + b2 # 1x4
    A2 = sigmoid(Z2) # 1x4
    # print('X: ', X.shape)
    # print('A0: ', A0.shape)
    # print('W1.T: ', W1.T.shape)
    # print('Z1: ', Z1.shape)
    # print('A1: ', A1.shape)
    # print('W2.T: ', W2.T.shape)
    # print('Z2: ', Z2.shape)
    # print('A2: ', A2.shape)
    return Z1, A1, Z2, A2

def relu_derivative(Z):
    return Z > 0

def backward_prop(W1, W2, b1, b2, Z1, A1, Z2, A2):
    m = X.shape[0]
    dZ2 = A2 - Y.T # 1x4
    dW2 = (1/m)*dZ2.dot(A1).T # 4x1
    db2 = (1/m)*np.sum(dZ2, axis=1, keepdims=True) # 1x1

    dZ1 = W2.dot(dZ2) * relu_derivative(Z1) # 4x4
    dW1 = (1/m)*dZ1.dot(X).T # 2x4
    db1 = (1/m)*np.sum(dZ1, axis=1, keepdims=True) # 4x1
    
    return dW1, dW2, db1, db2

def update_params(W1, W2, b1, b2, dW1, dW2, db1, db2):
    α = 0.1;
    W1 = W1 - α*dW1
    b1 = b1 - α*db1
    W2 = W2 - α*dW2
    b2 = b2 - α*db2
    
    return W1, W2, b1, b2


if __name__ == '__main__':
    iterations = 10000
    W1, W2, b1, b2 = init_params()
    acc = 0;
    nparr = np.array([[False, True, True, False]])
    # print(nparr)
    while iterations>=0:
        Z1, A1, Z2, A2 = forward_prop(W1, W2, b1, b2)
        dW1, dW2, db1, db2 = backward_prop(W1, W2, b1, b2, Z1, A1, Z2, A2)
        W1, W2, b1, b2 = update_params(W1, W2, b1, b2, dW1, dW2, db1, db2)
        a = (A2>0.5)
        if np.array_equal(nparr, a): acc += 1
        # print((A2>0.5))
        iterations -= 1
    # print(acc)
    print('Accuracy: [', acc/100, ']')
    



