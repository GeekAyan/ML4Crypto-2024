# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 12:49:26 2024

@author: Ayan
"""


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

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

# Hyperparameter tuning with RandomizedSearchCV
param_distributions = {
    'C': np.logspace(-3, 2, 10),
    'gamma': np.logspace(-3, 2, 10),
    'kernel': ['rbf', 'poly', 'sigmoid']
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
random_search = RandomizedSearchCV(SVC(random_state=42), param_distributions, n_iter=20, cv=cv, scoring='accuracy', random_state=42)
random_search.fit(X_train_scaled, y_train)

# Train the best estimator
best_svm = random_search.best_estimator_
best_svm.fit(X_train_scaled, y_train)

# Predictions and accuracy
y_pred = best_svm.predict(X_test_scaled)
print("Best SVM Parameters:", random_search.best_params_)
print("SVM Test Accuracy:", accuracy_score(y_test, y_pred))
