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
import advplaidml
advplaidml.setup_plaidml()

from keras import Sequential
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from keras.callbacks import ReduceLROnPlateau

# Model Parameters
bsize_scale = 2
size = 8
epochs = 3
lr_rate = 0.001 # Don't adjust if you don't know what you're doing

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

# Drop unnecessary columns
df = df.drop(['tick', 'name', 'steam_id'], axis=1, errors='ignore')

# Handle NaN values
df.fillna(0, inplace=True)

# Feature Selection
features = ['viewangle', 'pitchangle', 'va_delta', 'pa_delta']

# Standardize Data
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler for inference

# Reshape for LSTM
X = df[features].values.reshape((df.shape[0], 1, len(features)))

# Train-Test Split (90% Train, 10% Test)
X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)

# Model Definition
def build_lstm_autoencoder(input_shape, size=1):
	model = Sequential([
		LSTM(128 * size, activation='relu', return_sequences=True, input_shape=input_shape[1:]),
		LSTM(64 * size, activation='relu', return_sequences=False),
		RepeatVector(input_shape[1]),
		LSTM(64 * size, activation='relu', return_sequences=True),
		LSTM(128 * size, activation='relu', return_sequences=True),
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
autoencoder.save(f'jensen-nightwatch-v2-s{size}-lstm.h5')

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