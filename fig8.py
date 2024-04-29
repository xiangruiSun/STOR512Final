import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10)
])

# Loss function
loss_fn = SparseCategoricalCrossentropy(from_logits=True)

# Custom training step with Projected Gradient Descent
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = loss_fn(y, logits)
    gradients = tape.gradient(loss, model.trainable_weights)
    for i, (grad, var) in enumerate(zip(gradients, model.trainable_weights)):
        new_val = tf.clip_by_value(var - learning_rate * grad, -0.1, 0.1)
        var.assign(new_val)
    return loss

# Prepare training
learning_rate = 0.1
batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)

# Training loop
epochs = 5
for epoch in range(epochs):
    for batch, (images, labels) in enumerate(train_dataset):
        loss = train_step(images, labels)
    print(f'Epoch {epoch + 1}, Loss: {loss.numpy()}')

# Visualization of weight constraints
weights = model.get_layer(index=1).get_weights()[0]
plt.hist(weights.flatten(), bins=50)
plt.title('Weight Distribution')
plt.xlabel('Weight values')
plt.ylabel('Frequency')
plt.show()
