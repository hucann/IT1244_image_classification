# Model 3: Artificial Neural Networks (ANN)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import cohen_kappa_score

# Load the dataset
data = pd.read_csv("data.csv")
X = data.iloc[:, 1:].values / 255.0  # Normalize pixel values
y = data.iloc[:, 0].values

# Reshape the data to match the input shape expected by the model
X = X.reshape(-1, 28, 28)

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an ANN model with similar hyperparameters
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model and save the history
history = model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.2)

# Evaluate the model
y_pred = model.predict(X_test)  # Predicted probabilities

# Convert probabilities to class labels
y_pred_class = np.argmax(y_pred, axis=-1)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.3f}")

# Calculate Cohen's Kappa
kappa = cohen_kappa_score(y_test, y_pred_class)

print(f"Cohen's Kappa: {kappa:.3f}")