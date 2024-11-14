import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define helper functions

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def outputf(features, weights, bias):
    return sigmoid(np.dot(features, weights) + bias)

def errorfunc(y, output):
    return -(y * np.log(output) + (1 - y) * np.log(1 - output))

def updateweights(x, y, weights, bias, learning_rate):
    output = outputf(x, weights, bias)
    error = y - output
    weights += learning_rate * error * x
    bias += learning_rate * error
    return weights, bias

def plot_points(x, y):
    admit = x[np.argwhere(y == 1)]
    reject = x[np.argwhere(y == 0)]
    plt.scatter([s[0][0] for s in reject], [s[0][1] for s in reject],
                s=25, color='red', edgecolor='k', label="Reject")
    plt.scatter([s[0][0] for s in admit], [s[0][1] for s in admit],
                s=25, color='blue', edgecolor='k', label="Admit")
    plt.legend()

def display_graph(slope, yint, color='g--'):
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    x = np.arange(-0.05, 1.05, 0.1)
    plt.plot(x, slope * x + yint, color)

# Gradient descent-based training function
def train(features, targets, epochs=100, learning_rate=0.1, graph_lines=False):
    errors = []
    n_records, n_features = features.shape
    weights = np.random.normal(scale=1 / np.sqrt(n_features), size=n_features)
    bias = 0

    # Initial decision boundary plot in red dashed line
    display_graph(-weights[0] / weights[1], -bias / weights[1], 'r--')

    for epoch in range(epochs):
        for x, y in zip(features, targets):
            weights, bias = updateweights(x, y, weights, bias, learning_rate)

        # Calculate error for plotting
        out = outputf(features, weights, bias)
        loss = np.mean(errorfunc(targets, out))
        errors.append(loss)

        # Display intermediate decision boundaries every 10% of epochs
        if graph_lines and epoch % (epochs // 10) == 0:
            display_graph(-weights[0] / weights[1], -bias / weights[1], 'g--')
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    # Final decision boundary in blue
    display_graph(-weights[0] / weights[1], -bias / weights[1], 'blue')
    plot_points(features, targets)
    plt.title("Solution Boundary")
    plt.show()

    # Plot error over epochs
    plt.title("Error Plot")
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.plot(errors)
    plt.show()

# Load the data from data.csv
data = pd.read_csv('data.csv', header=None)
X = np.array(data[[0, 1]])  # First two columns as features
Y = np.array(data[2])       # Third column as labels

# Train the model and plot boundaries and errors
train(X, Y, epochs=100, learning_rate=0.1, graph_lines=True)
