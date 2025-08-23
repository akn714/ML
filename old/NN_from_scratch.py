import numpy as np

# Sample input data (X) and corresponding labels (y)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])  # Inputs: 4 samples with 2 features each

y = np.array([[0],
              [1],
              [1],
              [0]])      # Labels: XOR problem


def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(1)  # For reproducibility
    W1 = np.random.randn(n_h, n_x) * 0.01  # Weight matrix for hidden layer
    b1 = np.zeros((n_h, 1))                # Bias vector for hidden layer
    W2 = np.random.randn(n_y, n_h) * 0.01  # Weight matrix for output layer
    b2 = np.zeros((n_y, 1))                # Bias vector for output layer
    parameters = {"W1": W1, "b1": b1,
                  "W2": W2, "b2": b2}
    return parameters

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return Z > 0

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    # Hidden layer computations
    Z1 = np.dot(W1, X.T) + b1
    A1 = relu(Z1)
    # Output layer computations
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    cache = {"Z1": Z1, "A1": A1,
             "Z2": Z2, "A2": A2}
    return A2, cache

def compute_cost(A2, Y):
    m = Y.shape[0]  # Number of examples
    logprobs = Y.T * np.log(A2) + (1 - Y.T) * np.log(1 - A2)
    cost = -np.sum(logprobs) / m
    cost = np.squeeze(cost)  # Ensure cost is a scalar
    return cost

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[0]

    W2 = parameters['W2']

    A1 = cache['A1']
    A2 = cache['A2']
    Z1 = cache['Z1']

    # Output layer gradients
    dZ2 = A2 - Y.T
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    # Hidden layer gradients
    dZ1 = np.dot(W2.T, dZ2) * relu_derivative(Z1)
    dW1 = (1 / m) * np.dot(dZ1, X)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1,
             "dW2": dW2, "db2": db2}
    return grads

def update_parameters(parameters, grads, learning_rate=0.1):
    W1 = parameters['W1'] - learning_rate * grads['dW1']
    b1 = parameters['b1'] - learning_rate * grads['db1']
    W2 = parameters['W2'] - learning_rate * grads['dW2']
    b2 = parameters['b2'] - learning_rate * grads['db2']

    parameters = {"W1": W1, "b1": b1,
                  "W2": W2, "b2": b2}
    return parameters

def train_model(X, Y, n_h, num_iterations=1000000, print_cost=False):
    np.random.seed(3)

    n_x = X.shape[1]  # Input layer size
    n_y = Y.shape[1]  # Output layer size

    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(num_iterations):
        # Forward propagation
        A2, cache = forward_propagation(X, parameters)

        # Compute cost
        cost = compute_cost(A2, Y)

        # Backward propagation
        grads = backward_propagation(parameters, cache, X, Y)

        # Update parameters
        parameters = update_parameters(parameters, grads)

        # Optionally print the cost
        if print_cost:
            print(f"Cost after iteration {i}: {cost}")

    return parameters

def predict(parameters, X):
    A2, _ = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)
    return predictions.T

# Define the number of neurons in the hidden layer
n_h = 4

# Train the neural network
parameters = train_model(X, y, n_h, num_iterations=10000, print_cost=True)

# Make predictions
predictions = predict(parameters, X)

# Print results
print("Predictions:")
print(predictions)
print("Actual Labels:")
print(y)






