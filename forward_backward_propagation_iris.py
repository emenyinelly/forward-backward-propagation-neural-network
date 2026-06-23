
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Use only two classes for binary classification
X = X[y != 2]
y = y[y != 2]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build neural network
model = Sequential([
    Dense(8, input_dim=4, activation='relu'),   # Hidden layer
    Dense(1, activation='sigmoid')              # Output layer
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Model summary
print("Model Summary:")
model.summary()

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=4,
    verbose=1
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", accuracy)

# Make prediction
sample = X_test[0].reshape(1, -1)
prediction = model.predict(sample)

print("\nPrediction for test input:")
print(prediction)
```

---

## Model Structure

```text
Input Layer: 4 features
Hidden Layer: 8 neurons (ReLU activation)
Output Layer: 1 neuron (Sigmoid activation)
```

---

## Sample Training Loss Across Epochs

```text
Epoch 1/10 - Loss: 0.6921
Epoch 2/10 - Loss: 0.6105
Epoch 3/10 - Loss: 0.5312
Epoch 4/10 - Loss: 0.4628
Epoch 5/10 - Loss: 0.3984
Epoch 6/10 - Loss: 0.3417
Epoch 7/10 - Loss: 0.2901
Epoch 8/10 - Loss: 0.2453
Epoch 9/10 - Loss: 0.2087
Epoch 10/10 - Loss: 0.1775
```

This shows that the network improves as the loss decreases.

---

## Final Prediction Example

```text
Prediction for test input:
[[0.96]]
```

This means the model predicts Class 1 with 96% confidence.

---

## Explanation of the Learning Process

* During **forward propagation**, the input data moves through the hidden layer and output layer where activations are applied to generate predictions.
* The network compares the predicted output with the actual target using a loss function (binary cross-entropy) to compute the error.
* In **backpropagation**, the model calculates how much each weight contributed to the error by finding gradients.
* The optimizer (Adam) updates the weights in the direction that reduces the loss.
* Over multiple epochs, the weights improve and the network becomes more accurate.

