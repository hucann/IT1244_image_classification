import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score
import matplotlib.pyplot as plt

# Load the dataset 
data = pd.read_csv("data.csv")
X = data.iloc[:, 1:].values 
y = data.iloc[:, 0].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model with fine-tuned hyperparameters
rf_classifier = RandomForestClassifier(n_estimators = 140, max_features = 32, random_state = 42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)

# Evaluate the model in terms of accuracy, cohan's kappa
accuracy = accuracy_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)

print("Accuracy:", accuracy, "Kappa:", kappa)

