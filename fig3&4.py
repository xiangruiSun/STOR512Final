import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_samples = 1000
n_features = 2
n_epochs = 50
learning_rate = 0.1

# Generate a synthetic dataset
np.random.seed(42)
X_high_noise = np.random.randn(n_samples, n_features)
true_coef_high_noise = np.array([2, -3])
y_high_noise = X_high_noise @ true_coef_high_noise + np.random.randn(n_samples) * 10  # High noise

X_low_noise = np.random.randn(n_samples, n_features)
true_coef_low_noise = np.array([2, -3])
y_low_noise = X_low_noise @ true_coef_low_noise + np.random.randn(n_samples) * 0.1  # Low noise

# Define SGD function
def sgd_minibatch_path(X, y, batch_size, n_epochs=10, learning_rate=0.01):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    path = [weights.copy()]
    for epoch in range(n_epochs):
        for i in range(0, n_samples, batch_size):
            end = i + batch_size if i + batch_size < n_samples else n_samples
            X_batch = X[i:end]
            y_batch = y[i:end]
            
            gradient = -2 * X_batch.T @ (y_batch - X_batch @ weights) / (end - i)
            weights -= learning_rate * gradient
            path.append(weights.copy())
    return np.array(path)

# Run SGD
path_small_batch = sgd_minibatch_path(X_high_noise, y_high_noise, batch_size=1, n_epochs=n_epochs, learning_rate=learning_rate)
path_large_batch = sgd_minibatch_path(X_low_noise, y_low_noise, batch_size=1000, n_epochs=n_epochs, learning_rate=learning_rate)

# Plotting
plt.figure(figsize=(12, 6))

# Small batch size zigzagging
plt.subplot(1, 2, 1)
plt.plot(path_small_batch[:, 0], path_small_batch[:, 1], 'ro-', label='Parameter Path')
plt.scatter([true_coef_high_noise[0]], [true_coef_high_noise[1]], color='blue', label='True Coefficient')
plt.title('Parameter Path with Small Batch Size')
plt.xlabel('Weight 1')
plt.ylabel('Weight 2')
plt.legend()

# Large batch size smooth convergence
plt.subplot(1, 2, 2)
plt.plot(path_large_batch[:, 0], path_large_batch[:, 1], 'go-', label='Parameter Path')
plt.scatter([true_coef_low_noise[0]], [true_coef_low_noise[1]], color='blue', label='True Coefficient')
plt.title('Parameter Path with Large Batch Size')
plt.xlabel('Weight 1')
plt.ylabel('Weight 2')
plt.legend()

plt.tight_layout()
plt.show()
