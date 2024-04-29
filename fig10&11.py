def generate_complex_data(n_samples, feature_scale):
    """ Generate complex synthetic data with different feature scales for binary classification. """
    np.random.seed(42)  # for reproducibility
    x1 = np.random.randn(n_samples, 2) * feature_scale + np.array([1, 1])
    x2 = np.random.randn(n_samples, 2) * feature_scale + np.array([-1, -1])
    # Adding noise
    noise = np.random.randn(n_samples * 2, 2) * np.array([5, 0.5])  # High variance in the first feature
    X = np.vstack([x1, x2]) + noise
    y = np.array([1] * n_samples + [0] * n_samples)
    return X, y

# More complex data generation with uneven feature scales
feature_scale = np.array([1, 10])  # Different scales for the two features
X_complex, y_complex = generate_complex_data(n_samples, feature_scale)

# Plotting the more complex synthetic data
plt.figure(figsize=(8, 6))
plt.scatter(X_complex[:n_samples, 0], X_complex[:n_samples, 1], color='blue', label='Class 1')
plt.scatter(X_complex[n_samples:, 0], X_complex[n_samples:, 1], color='red', label='Class 0')
plt.title('More Complex Synthetic Binary Classification Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

from sklearn.preprocessing import StandardScaler

# Normalizing the features
scaler = StandardScaler()
X_complex_scaled = scaler.fit_transform(X_complex)

# Re-train the models on the normalized complex data
_, _, losses_sgd_complex_scaled = train(X_complex_scaled, y_complex, method='sgd', epochs=30, learning_rate=learning_rate)
_, _, losses_adagrad_complex_scaled = train(X_complex_scaled, y_complex, method='adagrad', epochs=30, learning_rate=learning_rate)

# Plotting the learning curves for the normalized complex data
plt.figure(figsize=(10, 6))
plt.plot(losses_sgd_complex_scaled, label='SGD - Normalized Complex Data', color='red')
plt.plot(losses_adagrad_complex_scaled, label='AdaGrad - Normalized Complex Data', color='blue')
plt.title('Learning Curves on Normalized More Complex Data: SGD vs AdaGrad')
plt.xlabel('Epoch')
plt.ylabel('Logistic Loss')
plt.legend()
plt.grid(True)
plt.show()
