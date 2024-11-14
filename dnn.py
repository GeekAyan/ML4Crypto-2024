# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 12:25:12 2024

@author: Ayan
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, BatchNormalization, Input, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


#Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Load dataset
data_path = "TrainingData.csv"  # Change this path
data = pd.read_csv(data_path)

# Verify column names
print("Dataset columns:", data.columns)

# Preprocessing: Convert bitstream to numeric vector
def preprocess_bitstream(data):
    X = np.array([list(map(int, list(bitstream))) for bitstream in data['Bitstream']])
    y = data['class'].values
    return X, y

X, y = preprocess_bitstream(data)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshape for CNNs
X_train_nn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_nn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Early Stopping Callback
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

### 1. Deep Multi-Layer Perceptron (MLP)
def build_deep_mlp(input_dim):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(1, activation='sigmoid'))
    
    return model

print("Training Deep MLP Neural Network...")
deep_mlp = build_deep_mlp(X_train.shape[1])
deep_mlp.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
deep_mlp.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=1)
loss, accuracy = deep_mlp.evaluate(X_test, y_test, verbose=0)
print("Deep MLP Neural Network Accuracy:", accuracy)

### 2. Deep Convolutional Neural Network (CNN)
def build_deep_cnn(input_shape):
    model = Sequential()
    model.add(Conv1D(128, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    
    model.add(Conv1D(256, kernel_size=5, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    
    model.add(Conv1D(512, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    return model

print("Training Deep CNN...")
deep_cnn = build_deep_cnn((X_train_nn.shape[1], 1))
deep_cnn.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
deep_cnn.fit(X_train_nn, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=1)
loss, accuracy = deep_cnn.evaluate(X_test_nn, y_test, verbose=0)
print("Deep CNN Accuracy:", accuracy)

### 3. ResNet-inspired CNN
def resnet_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = Conv1D(filters, kernel_size, strides=stride, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters, kernel_size, strides=1, padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def build_resnet(input_shape, num_blocks=3, filters=128):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters, kernel_size=7, strides=2, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    for _ in range(num_blocks):
        x = resnet_block(x, filters)
    
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    return model

print("Training ResNet-inspired CNN...")
import tensorflow as tf  # Ensure TensorFlow is imported for the ResNet block

resnet_model = build_resnet((X_train_nn.shape[1], 1), num_blocks=4, filters=128)
resnet_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
resnet_model.fit(X_train_nn, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=1)
loss, accuracy = resnet_model.evaluate(X_test_nn, y_test, verbose=0)
print("ResNet-inspired CNN Accuracy:", accuracy)
