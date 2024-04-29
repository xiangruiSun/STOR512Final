import numpy as np
import matplotlib.pyplot as plt

# Parameters for the new synthetic dataset
n_samples = 1000
n_features = 2
n_epochs = 100
learning_rate = 0.01
batch_size = 500  # Large batch size

# Generate a more complex synthetic dataset
np.random.seed(7)
X_complex = np.random.randn(n_samples, n_features)
true_coef_complex = np.array([3, -2])
y_complex = np.sin(X_complex @ true_coef_complex) + np.random.randn(n_samples) * 0.5  # Non-linear relationship with noise

# Define the SGD function for tracking the path
def sgd_minibatch_complex(X, y, batch_size, n_epochs=10, learning_rate=0.01):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    path = [weights.copy()]
    for epoch in range(n_epochs):
        indices = np.arange(n_samples)
        np.random.shuffle(indices)  # Shuffle indices to simulate the stochastic nature
        X = X[indices]
        y = y[indices]
        for i in range(0, n_samples, batch_size):
            end = i + batch_size if i + batch_size < n_samples else n_samples
            X_batch = X[i:end]
            y_batch = y[i:end]
            
            gradient = -2 * X_batch.T @ (y_batch - np.sin(X_batch @ weights)) / (end - i)  # Adjusting for non-linear relationship
            weights -= learning_rate * gradient
            path.append(weights.copy())
    return np.array(path)

# Run SGD with a large batch size on the complex dataset
path_large_batch_complex = sgd_minibatch_complex(X_complex, y_complex, batch_size=batch_size, n_epochs=n_epochs, learning_rate=learning_rate)

# Plot the parameter path
plt.figure(figsize=(8, 6))
plt.plot(path_large_batch_complex[:, 0], path_large_batch_complex[:, 1], 'go-', label='Parameter Path')
plt.scatter([true_coef_complex[0]], [true_coef_complex[1]], color='blue', label='True Coefficient')
plt.title('Parameter Path with Large Batch Size on Complex Dataset')
plt.xlabel('Weight 1')
plt.ylabel('Weight 2')
plt.legend()
plt.grid(True)
plt.show()
