# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:27:23 2024

@author: Ayan
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


# Load dataset
data_path = "TrainingData.csv"  # Change this path to data set
data = pd.read_csv(data_path)
#Suppress warnings
import warnings
warnings.filterwarnings("ignore")
# Preprocessing: Convert bitstream to numeric vector
def preprocess_bitstream(data):
    X = np.array([list(map(int, list(bitstream))) for bitstream in data['Bitstream']])
    y = data['class'].values
    return X, y

X, y = preprocess_bitstream(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10], 
    'penalty': ['l1', 'l2'], 
    'solver': ['liblinear', 'saga']
}
grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Train with the best parameters
best_log_reg = grid_search.best_estimator_
best_log_reg.fit(X_train_scaled, y_train)

# Predictions and accuracy
y_pred = best_log_reg.predict(X_test_scaled)
print("Best Logistic Regression Parameters:", grid_search.best_params_)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
