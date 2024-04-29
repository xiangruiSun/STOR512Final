# Check if TensorFlow is installed
if importlib.util.find_spec("tensorflow") is None:
    # TensorFlow not found, install it
    !pip install tensorflow

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD

# Load and preprocess MNIST data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu', kernel_initializer='he_normal'),
    Dense(10, activation='softmax')
])

# Loss function
loss_fn = SparseCategoricalCrossentropy()

# Define the penalty for the constraint violation
def penalty(weights, radius=1.0, lambda_penalty=1e1):
    weights_square = tf.reduce_sum(tf.square(weights))
    return lambda_penalty * tf.square(tf.maximum(0.0, weights_square - radius**2))

# Custom training step with Penalty Method
def train_step_with_penalty(images, labels):
    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        primary_loss = loss_fn(labels, logits)
        weight_penalty = penalty(model.trainable_weights[0], radius=5.0, lambda_penalty=1e1)
        total_loss = primary_loss + weight_penalty
    gradients = tape.gradient(total_loss, model.trainable_weights)
    gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return primary_loss, weight_penalty

# Training setup
optimizer = SGD(learning_rate=0.01)
batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(batch_size)

# Training loop
epochs = 5
for epoch in range(epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_penalty_avg = tf.keras.metrics.Mean()

    for images, labels in train_dataset:
        primary_loss, weight_penalty = train_step_with_penalty(images, labels)
        epoch_loss_avg.update_state(primary_loss)
        epoch_penalty_avg.update_state(weight_penalty)

    print(f'Epoch {epoch + 1}, Loss: {epoch_loss_avg.result().numpy()}, Penalty: {epoch_penalty_avg.result().numpy()}')

# Visualize the norm of weights
weights = model.trainable_weights[0].numpy().flatten()
l2_norm = np.linalg.norm(weights)
print(f'L2 Norm of Weights: {l2_norm}')

plt.hist(weights, bins=50)
plt.title('Weight Distribution with Penalty Method')
plt.xlabel('Weight values')
plt.ylabel('Frequency')
plt.show()
