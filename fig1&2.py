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

# Measure the time taken and compute updates frequency for each batch size
times_taken = []
updates_per_epoch = []
for batch_size in batch_sizes:
    start_time = time()
    sgd_minibatch(X, y, batch_size)
    times_taken.append(time() - start_time)
    updates_per_epoch.append(np.ceil(n_samples / batch_size))

# Plot Time Taken vs. Batch Size
plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, times_taken, 'o-b', label='Time Taken (s)')
plt.xscale('log')
plt.xlabel('Batch Size')
plt.ylabel('Time Taken (seconds)')
plt.title('Time Taken by Mini-batch SGD vs. Batch Size')
plt.grid(True)
plt.show()

# Plot Frequency of Updates vs. Batch Size
plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, updates_per_epoch, 'o-r', label='Updates per Epoch')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Batch Size')
plt.ylabel('Frequency of Updates per Epoch')
plt.title('Frequency of Updates vs. Batch Size')
plt.grid(True)
plt.show()
