import numpy as np
import matplotlib.pyplot as plt

# Function to generate synthetic data
def generate_data(n_samples, n_features, beta, noise_std):
    X = np.random.randn(n_samples, n_features)
    noise = np.random.normal(0, noise_std, size=n_samples)
    y = np.dot(X, beta) + noise
    return X, y

# Function to calculate mean squared error
def calculate_mse(X, y, beta):
    y_pred = np.dot(X, beta)
    mse = np.mean((y - y_pred)**2)
    return mse

# Function for gradient descent with adaptive batch sizes
def gradient_descent_adaptive(X, y, beta_init, learning_rate_init, noise_std, n_epochs):
    beta = beta_init
    learning_rate = learning_rate_init
    batch_size = 1
    
    mse_history = []
    for epoch in range(n_epochs):
        for i in range(len(X)):
            # Compute gradient
            gradient = -2 * X[i] * (y[i] - np.dot(X[i], beta))
            # Update parameters
            beta -= learning_rate * gradient
        # Calculate MSE
        mse = calculate_mse(X, y, beta)
        mse_history.append(mse)
        # Adjust learning rate
        learning_rate /= (epoch + 1)  # Learning rate decay
        # Increase batch size gradually
        if epoch % 10 == 0:
            batch_size = min(batch_size * 2, len(X))
    return beta, mse_history


# Generate synthetic data
np.random.seed(0)
n_samples = 10000
n_features = 2
true_beta = np.array([2, -3])
noise_std = 10
X_adaptive, y_adaptive = generate_data(n_samples, n_features, true_beta, noise_std)

# Run gradient descent with adaptive batch sizes
beta_init = np.zeros(n_features)
learning_rate_init = 0.01
n_epochs = 100

# Run gradient descent with adaptive batch sizes
beta_adaptive, mse_history_adaptive = gradient_descent_adaptive(X_adaptive, y_adaptive, beta_init, learning_rate_init, noise_std, n_epochs)

# Plot MSE history for adaptive batch sizes
plt.plot(range(1, n_epochs + 1), mse_history_adaptive, label='Adaptive Batch Sizes')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Training with Adaptive Batch Sizes')
plt.legend()
plt.show()
