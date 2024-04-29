import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize SGD without weight averaging
sgd = SGDClassifier(average=False, random_state=42, learning_rate='adaptive', eta0=0.01, max_iter=10)

# Initialize SGD with weight averaging
sgd_avg = SGDClassifier(average=True, random_state=42, learning_rate='adaptive', eta0=0.01, max_iter=10)

# Train and track performance over epochs
n_epochs = 50
accuracies_no_avg = []
accuracies_avg = []

for _ in range(n_epochs):
    sgd.partial_fit(X_train_scaled, y_train, classes=np.unique(y))
    sgd_avg.partial_fit(X_train_scaled, y_train, classes=np.unique(y))

    y_pred = sgd.predict(X_test_scaled)
    y_pred_avg = sgd_avg.predict(X_test_scaled)

    accuracies_no_avg.append(accuracy_score(y_test, y_pred))
    accuracies_avg.append(accuracy_score(y_test, y_pred_avg))

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(accuracies_no_avg, label='SGD without averaging', color='red')
plt.plot(accuracies_avg, label='SGD with averaging', color='blue')
plt.title('Comparison of SGD with and without weight averaging')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
