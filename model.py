import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os
import logging
from datetime import datetime


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_realistic_csv(filename, size_in_mb=3, num_columns=10):
    logging.info("Generating synthetic time series data...")
    num_rows = (size_in_mb * 1024 * 1024) // (8 * num_columns)  
    timestamp = pd.date_range(start='2021-01-01', periods=num_rows, freq='H')
    

    data = np.cumsum(np.random.randn(num_rows, num_columns) * 0.1, axis=0)
    df = pd.DataFrame(data, columns=[f"feature_{i+1}" for i in range(num_columns)], index=timestamp)
    df.to_csv(filename, index=True)
    
    logging.info(f"Generated CSV file '{filename}' with size {os.path.getsize(filename) / (1024 * 1024):.2f} MB.")

def load_data_from_csv(filename):
    logging.info(f"Loading data from '{filename}'...")
    try:
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        logging.info(f"Data loaded with shape: {df.shape}")
        
       
        data = df['feature_1'].values
        target = data[10:]
        
        sequence_length = 50
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length])
            y.append(target[i+sequence_length-1])

        X = np.array(X)
        y = np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        return X, y
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def build_and_train_model(X, y):
    logging.info("Building and training the LSTM model...")
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    model.summary()
    
    model.fit(X, y, epochs=20, batch_size=32, validation_split=0.1)
    
    return model

def make_predictions(model, X):
    logging.info("Making predictions on the test set...")
    predictions = model.predict(X)
    logging.info(f"Predictions for first 5 samples: {predictions[:5]}")
    return predictions

if __name__ == '__main__':
    csv_filename = 'train_sequence.v2.csv'
    
    if not os.path.exists(csv_filename):
        generate_realistic_csv(csv_filename, size_in_mb=3)
    
    try:
        X, y = load_data_from_csv(csv_filename)
        model = build_and_train_model(X, y)
        predictions = make_predictions(model, X)
    except Exception as e:
        logging.error(f"Error during the model process: {e}")
