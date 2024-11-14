# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 21:46:02 2024

@author: Ayan
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

# Load and preprocess data
data_path = "TrainingData.csv"
data = pd.read_csv(data_path)

# Convert bitstream column to sequences of integers (0 and 1)
X = np.array([list(map(int, list(bitstream))) for bitstream in data['Bitstream']])
y = data['class'].values

# Reshape X to fit LSTM input format (samples, timesteps, features)
X = X.reshape((X.shape[0], 1024, 1))

# Initialize KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Build LSTM model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train model
    model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_val, y_val),
              callbacks=[early_stopping], verbose=0)

    # Evaluate model on validation set
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    accuracies.append(accuracy)

# Calculate mean cross-validation accuracy
print("Cross-Validation Accuracies:", accuracies)
print("Mean Cross-Validation Accuracy:", np.mean(accuracies))
