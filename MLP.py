# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:40:42 2024

@author: Ayan
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight


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








# Compute class weights if dataset is imbalanced
class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Build MLP with adjusted layers
mlp = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model with Adam optimizer and learning rate decay
optimizer = Adam(learning_rate=0.001, decay=1e-6)
mlp.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Fit the model with class weights
mlp.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test),
        class_weight=class_weight_dict, callbacks=[early_stopping], verbose=1)

# Save the model
mlp.save('trained_mlp_model.h5')

# Evaluate the model
loss, accuracy = mlp.evaluate(X_test, y_test)
print("MLP Neural Network Accuracy:", accuracy)
