# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 12:06:36 2024

@author: Ayan
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
import numpy as np
import joblib
from sklearn.model_selection import cross_val_score
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
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
def create_mlp():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to create CNN model
def create_cnn():
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train_nn.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

X, y = preprocess_bitstream(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Logistic Regression
print("Training Logistic Regression...")
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
# Save the model to a file
joblib.dump(log_reg, 'logistic_regression_model.pkl')
y_pred = log_reg.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
# Cross-validation for Logistic Regression
print("Training Logistic Regression with Cross-Validation...")
cv_scores = cross_val_score(log_reg, X, y, cv=5)  # 5-fold cross-validation

# Calculate and display cross-validation accuracy
print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean Cross-Validation Accuracy:", cv_scores.mean())

# 2. Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
# Cross-validation for Random Forest
print("Training Random Forest with Cross-Validation...")
rf = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=5)  # 5-fold cross-validation

# Display cross-validation results
print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean Cross-Validation Accuracy:", cv_scores.mean())

# 3. Support Vector Machine (SVM)
print("Training SVM...")
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred))
# Save the trained model to a file
joblib.dump(svm, 'svm_model.pkl')
# Cross-validation for SVM
print("Training SVM with Cross-Validation...")
cv_scores = cross_val_score(svm, X, y, cv=5)  # 5-fold cross-validation

# Display cross-validation results
print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean Cross-Validation Accuracy:", cv_scores.mean())

# Reshape X for Neural Networks (if required)
X_train_nn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_nn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# 4. MLP Neural Network
print("Training MLP Neural Network...")
#mlp = Sequential([
#    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
#    Dense(64, activation='relu'),
#    Dense(1, activation='sigmoid')
#])
mlp=create_mlp()
mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
mlp.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
mlp.save('trained_nn_model.h5')
loss, accuracy = mlp.evaluate(X_test, y_test)
print("MLP Neural Network Accuracy:", accuracy)
# Wrapping the model for cross-validation
mlp = KerasClassifier(build_fn=create_mlp, epochs=10, batch_size=32, verbose=0)

# Cross-validation
cv_scores = cross_val_score(mlp, X, y, cv=5)  # 5-fold cross-validation

# Display results
print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean Cross-Validation Accuracy:", cv_scores.mean())

# 5. CNN
print("Training CNN...")
#cnn = Sequential([
#    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train_nn.shape[1], 1)),
#    MaxPooling1D(pool_size=2),
#    Flatten(),
#    Dense(64, activation='relu'),
#    Dense(1, activation='sigmoid')
#])
cnn=create_cnn()
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(X_train_nn, y_train, epochs=10, batch_size=32, verbose=1)
loss, accuracy = cnn.evaluate(X_test_nn, y_test)
print("CNN Accuracy:", accuracy)
# Wrapping CNN for cross-validation
cnn = KerasClassifier(build_fn=create_cnn, epochs=10, batch_size=32, verbose=0)



# Perform cross-validation
cv_scores = cross_val_score(cnn, X_train_nn, y_train, cv=5)  # 5-fold cross-validation

# Print the cross-validation results
print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean Cross-Validation Accuracy:", cv_scores.mean())
