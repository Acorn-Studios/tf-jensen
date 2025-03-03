import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

import datascrape
import os

# Only gather data if we don't have data.csv
if not os.path.exists('data.csv'):
    # Turn all demos into CSV files and extract data for all players
    if not os.listdir('data_collector/test'): 
        print("Compiling all demos into CSV")
        datascrape.comp_all_csv()
    
    demos = os.listdir('./data_collector/test')
    for demo in demos:
        print(f"Extracting data for all players from {demo}")
        datascrape.extract_all_players("data_collector/test/"+demo, "playerdata/", concaticate_by_default=True)

# Load the dataset with utf-8 encoding
df = pd.read_csv('data.csv', encoding='utf-8', dtype=str)
df = df.apply(pd.to_numeric, errors='coerce')

# Drop unnecessary columns
df = df.drop(['tick', 'name', 'steam_id'], axis=1)  # Keep only numerical values

# Handle NaN values in `va_delta` and `pa_delta` (fill with 0 or column mean)
df['va_delta'] = df['va_delta'].fillna(0)  # Replace NaN with 0
df['pa_delta'] = df['pa_delta'].fillna(0)  # Replace NaN with 0

# Ensure the DataFrame is not empty
if df.empty:
    raise ValueError("The DataFrame is empty after dropping duplicates.")

# Select only the relevant features for anomaly detection
features = ['viewangle', 'pitchangle', 'va_delta', 'pa_delta']

# Standardize the selected features using one scaler
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Split dataset (90% training, 10% testing)
train_data, test_data = train_test_split(df, test_size=0.1, random_state=42)

# Extract features (X) for training and testing
X_train = train_data[features].values.reshape(-1, 1, len(features))  # Reshape to (samples, timesteps, features)
X_test = test_data[features].values.reshape(-1, 1, len(features))  # Reshape to (samples, timesteps, features)

# Check shape
print("X_train shape:", X_train.shape)  # Should be (num_samples, 1, 4)
print("X_test shape:", X_test.shape)  # Should be (num_samples, 1, 4)

size = 8 # Size of the model

# Build the autoencoder. 
# Here, we want our model to look at past veiwangles to determine a cheater, so we do that here.
def build_lstm_autoencoder(input_shape, size=1):
    model = Sequential()
    model.add(Input(shape=(input_shape[1], input_shape[2])))
    model.add(LSTM(128*size, activation='relu', return_sequences=True))
    model.add(LSTM(64*size, activation='relu', return_sequences=False))
    model.add(RepeatVector(input_shape[1]))
    model.add(LSTM(64*size, activation='relu', return_sequences=True))
    model.add(LSTM(128*size, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(input_shape[2], activation='tanh')))
    return model

input_shape = X_train.shape
lstm_autoencoder = build_lstm_autoencoder(input_shape, size=size)
lstm_autoencoder.summary()
lstm_autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Train the Autoencoder
history = lstm_autoencoder.fit(X_train, X_train, epochs=3, batch_size=32, validation_data=(X_test, X_test))

# Save the model
lstm_autoencoder.save(f'jensen-nightwatch-v2-s{size}-lstm.keras')

show_summary = False
if show_summary:
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

    # Plot Loss vs. Accuracy
    plt.figure(figsize=(12, 4))

    # Plot Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.title('Training Loss and Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Training Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Validation Loss and Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()