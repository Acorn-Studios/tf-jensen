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
bsize_scale = 2 # Multiplier for the batch size. If you run out of memory, try reducing this value. If you run into model undergeneralization, try increasing this value.
size = 8 # The size of the model. Larger models will have more parameters and will take longer to train. Be careful of overfitting.
epochs = 3 # The amount of times the model will see the data. More epochs = more learning. Be careful of overfitting.
epoch_scale = 8 # Automatically increase epochs while reducing lr_rate. Helpful for larger models. Should be in multiples of 4.
window_size = 33  # Number of past time steps to include in each sequence. Essentially memory.
overlap_seqs = False # Will result in slower training but better short-term predictions. Can be very heavy on memory, especially with large window sizes.

# Notes:
# Setting window_size to 66 and overlap_seqs = True is like mixing bleach with ammonia
# If you have window_size set high, make sure you have enough memory. PlaidML will tell you if you don't have enough.
# If the model reaches 100% accuracy after ~3 epochs or 95% after 1 epoch, it's likely overfitting. Try reducing the size of the model.
# If the model is not learning well, try increasing the size of the model.

# Danger zone! Do not edit these if you don't know what you're doing
stride = 1
lr_rate = 0.001 * 4/epoch_scale
epoch = epochs * epoch_scale/4

if not overlap_seqs: stride = window_size

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

# Function to create sliding window sequences
def create_sequences(data, window_size, stride=1):
	"""
	Converts raw data into overlapping sequences of shape (num_chunks, window_size, num_features).
	"""
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
		LSTM(64 * size, activation='tanh', return_sequences=False),
		RepeatVector(input_shape[1]),  # Ensures the decoder gets the same time dimension
		LSTM(64 * size, activation='tanh', return_sequences=True),
		LSTM(128 * size, activation='tanh', return_sequences=True),
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
