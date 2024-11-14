# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:40:00 2024

@author: Ayan
"""
from tensorflow.keras.models import  load_model


import numpy as np
import pandas as pd
import joblib

#Suppress warnings
import warnings
warnings.filterwarnings("ignore")
# Load dataset
data_path = "Sample.csv"  # Change this path to data set
data = pd.read_csv(data_path)
#Suppress warnings
import warnings
warnings.filterwarnings("ignore")
# Preprocessing: Convert bitstream to numeric vector
def preprocess_bitstream(data):
    X = np.array([list(map(int, list(bitstream))) for bitstream in data['Bitstream']])
    return X

X = preprocess_bitstream(data)

# Loading the saved model for prediction on new data
loaded_model = load_model('trained_mlp_model.h5')

# Assuming the new test dataset (X_new) has the same format but without labels in the last column
#X_new_scaled = scaler.transform(X_new)  # Ensure to scale the new dataset using the same scaler

# Make predictions on the new test dataset
y_new_pred = loaded_model.predict(X)
y_new_pred_binary = (y_new_pred > 0.5).astype(int)
print("Predictions on new test dataset:", y_new_pred_binary)

# logistic regression model
loaded_model = joblib.load('logistic_regression_model.pkl')
y_pred = loaded_model.predict(X)
print("Predictions on new test dataset:", y_pred)