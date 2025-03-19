# Data Handling
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datascrape
import joblib  # For saving the scaler

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Keras & PlaidML
# Instead of importing the basic AttentionLayer, import our transformer-based layer
from advplaidml import TransformerLayer as Attention
import advplaidml
advplaidml.setup_plaidml()

from keras import Sequential
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from keras.callbacks import ReduceLROnPlateau

# Model Parameters
bsize_scale = 2  # Multiplier for the batch size.
size = 8         # The size multiplier for the model.
epochs = 8       # Number of epochs.
window_size = 33  # Number of past time steps per sequence.
overlap_seqs = False

# Set stride
stride = 1
lr_rate = 0.001 * min(1, round(4/epochs))
if not overlap_seqs: 
    stride = window_size

# Data Preparation
if not os.path.exists('data.csv'):
    datascrape.clear_folder('data_collector/test')
    print("Compiling all demos into CSV")
    for demo in os.listdir('demos'):
        datascrape.comp_into_csv(demo)
    datascrape.concatenate_csvs('data_collector/test', 'data.csv')

# Load dataset
df = pd.read_csv('data.csv', encoding='utf-8', dtype=str)
df = df.apply(pd.to_numeric, errors='coerce')
df = df.drop(['tick', 'name', 'steam_id'], axis=1, errors='ignore')
df.fillna(0, inplace=True)

# Feature Selection
features = ['viewangle', 'pitchangle', 'va_delta', 'pa_delta']

# Standardize Data
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler for inference

# Function to create sliding window sequences
def create_sequences(data, window_size, stride=1):
    num_chunks = (data.shape[0] - window_size) // stride + 1
    sequences = np.array([data[i*stride:i*stride+window_size] for i in range(num_chunks)])
    return sequences

# Convert data into sequences
X = create_sequences(df[features].values, window_size=window_size, stride=stride)

# Train-Test Split (90% Train, 10% Test)
X_train, X_test = train_test_split(X, test_size=0.1, shuffle=False)

# Model Definition
def build_lstm_autoencoder(input_shape, size=1):
    model = Sequential([
        LSTM(256 * size, activation='tanh', return_sequences=True, input_shape=input_shape[1:]),
        LSTM(128 * size, activation='tanh', return_sequences=True),
        # Replace the old attention layer with the transformer layer.
        Attention(dropout_rate=0.1),
        LSTM(64 * size, activation='tanh', return_sequences=False),
        RepeatVector(input_shape[1]),
        LSTM(64 * size, activation='tanh', return_sequences=True),
        LSTM(128 * size, activation='tanh', return_sequences=True),
        Attention(dropout_rate=0.1),
        LSTM(256 * size, activation='tanh', return_sequences=True),
        TimeDistributed(Dense(input_shape[2], activation='tanh'))
    ])
    return model

# Initialize Model
input_shape = X_train.shape
autoencoder = build_lstm_autoencoder(input_shape, size=size)
autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
autoencoder.summary()

# Train Model
history = autoencoder.fit(
    X_train, X_train, 
    epochs=epochs, 
    batch_size=128 * bsize_scale, 
    validation_data=(X_test, X_test), 
    callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=lr_rate)],
    verbose=1
)

# Save Model
autoencoder.save(f'jensen-nightwatch-v2-s{size}-memory.h5')

# Optional Visualization
show_summary = False
if show_summary:
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()
