def standard_sgd(x_init, learning_rate, sigma, n_iterations):
    x_values = [x_init]
    x = x_init

    for _ in range(n_iterations):
        fx, grad = noisy_quadratic(x, sigma)
        x -= learning_rate * grad
        x_values.append(x)

    return x_values

# Parameters for comparison
x_init = 10
learning_rate = 0.1
sigma = 1.0
n_iterations = 50

# Run both optimizers
trajectory_sgd = standard_sgd(x_init, learning_rate, sigma, n_iterations)
trajectory_sgd_momentum = sgd_with_momentum(x_init, learning_rate, momentum, sigma, n_iterations)

# Plotting
plt.figure(figsize=(12, 7))
plt.plot(x_range, y_range, label='Function $f(x) = x^2$', zorder=1)
plt.scatter(trajectory_sgd, [x**2 for x in trajectory_sgd], color='blue', s=30, label='Standard SGD', zorder=2)
plt.scatter(trajectory_sgd_momentum, [x**2 for x in trajectory_sgd_momentum], color='red', s=30, label='SGD with Momentum', zorder=3)
plt.title('Comparison of Standard SGD and SGD with Momentum')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.legend()
plt.show()
