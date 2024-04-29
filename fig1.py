import numpy as np
import matplotlib.pyplot as plt
from time import time

# Synthetic dataset parameters
n_samples = 10000
n_features = 20
batch_sizes = [1, 16, 32, 64, 128, 256, 512, 1024, 2048]

# Generate a synthetic dataset
np.random.seed(0)
X = np.random.randn(n_samples, n_features)
true_coef = np.random.randn(n_features)
y = X @ true_coef + np.random.randn(n_samples) * 0.1

# Mini-batch SGD implementation
def sgd_minibatch(X, y, batch_size, n_epochs=10, learning_rate=0.01):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    for epoch in range(n_epochs):
        for i in range(0, n_samples, batch_size):
            end = i + batch_size if i + batch_size < n_samples else n_samples
            X_batch = X[i:end]
            y_batch = y[i:end]
            
            # Compute gradients
            gradient = -2 * X_batch.T @ (y_batch - X_batch @ weights) / (end - i)
            # Update weights
            weights -= learning_rate * gradient
    return weights

# Measure the time taken for each batch size and plot it
times_taken = []
for batch_size in batch_sizes:
    start_time = time()
    sgd_minibatch(X, y, batch_size)
    times_taken.append(time() - start_time)

plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, times_taken, 'o-')
plt.xscale('log')
plt.xlabel('Mini-batch Size')
plt.ylabel('Time taken (seconds)')
plt.title('Mini-batch SGD: Computation Time vs. Batch Size')
plt.grid(True)
plt.show()
