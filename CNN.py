# Model 4: Convolutional Neural Networks (CNN)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import cohen_kappa_score

# Load the dataset
data = pd.read_csv("data.csv")
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

X = X.reshape((len(X), int((len(X[0])) ** 0.5), int((len(X[0])) ** 0.5), 1)) / 255.0
# Reshape the data to match the input shape expected by the model

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

#Reshaping X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = keras.models.Sequential()
for i in range(2):
    model.add(keras.layers.Conv2D(32*(2*(i+1)), (3,3), strides=(1,1), activation = "relu", input_shape = (28,28,1)))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(10, activation="softmax"))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics= ['acc'])


model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=32)

test_loss, test_accuracy = model.evaluate(X_test, y_test)

# Evaluate the model
y_pred = model.predict(X_test)

y_pred_class = np.argmax(y_pred, axis=-1)

kappa = cohen_kappa_score(y_test, y_pred_class)

print(f"Accuracy: {test_accuracy}, Cohen Kappa: {kappa}")
