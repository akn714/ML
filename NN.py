"""
Neural Networks using numpy

Input Layer ->   Hidden Layer -> Output Layer

- Forward Propogation
    - Activation function
- Backward Propogation
    - Cost function (loss calculation)
    - Gradient descent
    - Training

"""

"""
X -> input 2d array (pixels of 28x28 images) (1 row -> 1 example image)
X -> [
    [784 pixels of image 1],
    [784 pixels of image 2],
    [784 pixels of image 3],
    ...
]

transpose X (now 1 colum -> 784 pixels of 1 example image)

Neural Network ->
    Input (784 pixels/nodes) -> One hidden layer (10 nodes) -> Output layer (10 nodes each represents numbers from 1 to 10)
        0th layer                   1st layer                       2nd layer

---- FORWARD PROPOGATION ----
A[0] -> input layer
Z[1] -> unactivated first layer (hidden layer)
Z[1] = W[1].A[0] + b[1]     (b[1] -> biases of layer 1)

A[1] -> activated first layer (hidden layer after applying activation function)
activation function -> tanh, ReLU, Sigmoid (used to remove linearity)
A[1] = g(Z[1]) = ReLU(Z[1])     (ReLU(x) = max(0, x))

Z[2] -> unactivated second layer (output layer)
Z[2] = W[2].A[1] + b[2]
A[2] = softmax(Z[2])        (softmax -> activation function for output layer)

---- BACKWARD PROPOGATION ----
- finding errors in weights and biases

dZ[2] -> error in 2nd layer (how much this layer is off by the actual output)
dZ[2] = A[2] - Y        (Y -> array of actual outputs)
dW[2] = (1/m)dZ[2]*A[1]T      ( A[1]T -> transpose of A[1] )
db[2] = (1/m)summation(dZ[2])

m -> size of Y

dZ[1] -> error in 1st layer
dZ[1] = W[2]T*dZ[2] * g'(Z[1])      (Y -> array of actual outputs)
dW[1] = (1/m)dZ[1]*XT      ( XT -> transpose of X )
db[1] = (1/m)summation(dZ[1])

- updating parameters according to calculation done in back propogation
W[1] = W[1] - αdW[1]
b[1] = b[1] - αdb[1]
W[2] = W[2] - αdW[2]
b[2] = b[2] - αdb[2]

α -> learning rate (not set of updated by machine, we set this)

---- NOW REPEATE THE WHOLE THING AGAIN UNTIL YOU GET THE RIGHT OUTPUT ----
"""
