# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 22:18:19 2024

@author: Ayan
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and preprocess data
data_path = "TrainingData.csv"
data = pd.read_csv(data_path)

# Convert bitstream column to sequences of integers (0 and 1)
X = np.array([list(map(int, list(bitstream))) for bitstream in data['Bitstream']])
y = data['class'].values

# Reshape X to fit LSTM input format (samples, timesteps, features)
X = X.reshape((X.shape[0], 1024, 1))

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build enhanced LSTM model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.4),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model with modified optimizer and learning rate
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate model
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

# Calculate accuracy
print("Enhanced LSTM Model Accuracy:", accuracy_score(y_test, y_pred_binary))