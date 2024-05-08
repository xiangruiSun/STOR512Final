from sklearn.datasets import load_diabetes
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import random
from time import time

# Load dataset
X, y = load_diabetes(return_X_y=True)
print(X.shape)
print(y.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Mini-batch SGD implementation
def sgd_minibatch(X, y, batch_size, n_epochs=100, learning_rate=0.01):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    intercept = 0  # Initializing the intercept
    for epoch in range(n_epochs):
        for i in range(0, n_samples, batch_size):
            end = i + batch_size if i + batch_size < n_samples else n_samples
            X_batch = X[i:end]
            y_batch = y[i:end]

            # Predictions
            predictions = X_batch @ weights + intercept
            # Compute gradients
            gradient_weights = -2 * X_batch.T @ (y_batch - predictions) / (end - i)
            gradient_intercept = -2 * np.mean(y_batch - predictions)

            # Update weights and intercept
            weights -= learning_rate * gradient_weights
            intercept -= learning_rate * gradient_intercept
    return weights, intercept

# Train and measure the model
batch_sizes = [int(X_train.shape[0] / 50)]  # You can try different batch sizes
times_taken = []
updates_per_epoch = []
coefs = []
intercepts = []
for batch_size in batch_sizes:
    start_time = time()
    weights, intercept = sgd_minibatch(X_train, y_train, batch_size)
    times_taken.append(time() - start_time)
    updates_per_epoch.append(np.ceil(X_train.shape[0] / batch_size))
    coefs.append(weights)
    intercepts.append(intercept)

# Output the results
print("Batch Sizes:", batch_sizes)
print("Times Taken:", times_taken)
print("Updates per Epoch:", updates_per_epoch)
print("Coefficients:", coefs)
print("Intercepts:", intercepts)

# Prediction and evaluation
y_pred = X_test @ coefs[0] + intercepts[0]  # Using the first batch size results for simplicity
score = r2_score(y_test, y_pred)
print("R^2 Score:", score)

import matplotlib.pyplot as plt

# Assuming the rest of the setup code and imports are already included here as described earlier

# Extended batch sizes for more comprehensive analysis
batch_sizes = [5, 10, 20, 50, 100, 200, 500, X_train.shape[0] // 10, X_train.shape[0] // 5]
times_taken = []
updates_per_epoch = []
coefs = []
intercepts = []

# Train models with different batch sizes
for batch_size in batch_sizes:
    start_time = time()
    weights, intercept = sgd_minibatch(X_train, y_train, batch_size)
    times_taken.append(time() - start_time)
    updates_per_epoch.append(np.ceil(X_train.shape[0] / batch_size))
    coefs.append(weights)
    intercepts.append(intercept)

# Plotting the results
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Batch Size')
ax1.set_ylabel('Time Taken (s)', color=color)
ax1.plot(batch_sizes, times_taken, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Frequency of Updates per Epoch', color=color)
ax2.plot(batch_sizes, updates_per_epoch, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('Effect of Batch Size on Training Time and Update Frequency')
plt.show()

import tensorflow as tf
# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


# Load dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Initialize class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Scale images to range from 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build a feedforward network with size 784x256x128x10 and ReLU activation
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten input image to create a vector of 784 elements
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)  # Output layer with 10 units for class prediction
])

# Model summary
model.summary()

# Compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train model using Mini-batch Gradient Descent
model.fit(train_images, train_labels, batch_size=64, epochs=10)  # Mini-batch size set to 64

# Evaluate model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest loss:', test_loss, 'Test accuracy:', test_acc)

# Add a softmax layer to output probabilities
prediction_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = prediction_model.predict(test_images)

import time
import tensorflow as tf
import matplotlib.pyplot as plt

class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.total_time = 0.0

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        epoch_time = time.time() - self.epoch_start_time
        self.times.append(epoch_time)
        self.total_time += epoch_time

    def on_train_end(self, logs={}):
        print("Total training time: {:.2f} seconds".format(self.total_time))

batch_sizes = [32, 64, 128, 256, 512]  # Define different batch sizes to experiment with
training_times = []
updates_per_epoch = []

for batch_size in batch_sizes:
    time_callback = TimeHistory()
    model.fit(train_images, train_labels, batch_size=batch_size, epochs=10, callbacks=[time_callback], verbose=0)
    training_times.append(time_callback.total_time)
    updates_per_epoch.append(len(train_images) // batch_size)

# Note: Remember to reset the model weights before each training if you're not restarting the kernel.

fig, ax1 = plt.subplots()

# Plot training time
color = 'tab:red'
ax1.set_xlabel('Batch Size')
ax1.set_ylabel('Total Training Time (s)', color=color)
ax1.plot(batch_sizes, training_times, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis

# Plot the frequency of updates
color = 'tab:blue'
ax2.set_ylabel('Updates per Epoch', color=color)
ax2.plot(batch_sizes, updates_per_epoch, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # Otherwise the right y-label is slightly clipped
plt.title('Training Time and Updates Frequency vs Batch Size')
plt.show()
