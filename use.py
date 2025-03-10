# PlaidML
import advplaidml
advplaidml.setup_plaidml()

# Libraries
from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import datascrape
import json
import time

# Load Demo Data
demo = "meowington.dem"
datascrape.clear_folder('data_collector/test')
datascrape.comp_into_csv(demo)
datascrape.concatenate_csvs('data_collector/test', 'new_data.csv')

# Load Model & Scaler
model = load_model('jensen-nightwatch-v2-s8-lstm.h5')
scaler = joblib.load('scaler.pkl')

# Load New Data
new_df = pd.read_csv('new_data.csv', encoding='utf-8')

# Drop Unnecessary Columns
new_df.drop(['steam_id'], axis=1, errors='ignore', inplace=True)

# Handle Missing Values
new_df.fillna(0, inplace=True)

# Standardize Using Trained Scaler
features = ['viewangle', 'pitchangle', 'va_delta', 'pa_delta']
new_df[features] = scaler.transform(new_df[features])

# Reshape for LSTM
X_new = new_df[features].values.reshape(-1, 1, len(features))

# Autoencoder Prediction
start_time = time.time()
reconstructions = model.predict(X_new, verbose=1)
end_time = time.time()

# Compute Reconstruction Error (MSE)
mse = np.mean(np.power(X_new - reconstructions, 2), axis=(1, 2))

# Set Anomaly Threshold Using Training Data Distribution
threshold = np.percentile(mse, 98)  # Increase sensitivity to reduce noise

# Detect Anomalies
new_df['Reconstruction_Error'] = mse
new_df['Anomaly'] = mse > threshold

# Summary
summary = {
	"players": {},
	"total_suspicious_ticks": int(new_df['Anomaly'].sum()),
	"prediction_time_seconds": round(end_time - start_time, 4)
}

# Process Each Player
players = new_df['name'].unique()
for player in players:
	player_df = new_df[new_df['name'] == player].copy()
	suspicious_ticks = player_df[player_df['Anomaly'] == True]['tick'].tolist()
	summary["players"][player] = {
		"suspicious_ticks": suspicious_ticks,
		"total_suspicious_ticks": len(suspicious_ticks)
	}

# Save Summary
with open('suspicious_ticks_summary.json', 'w') as json_file:
	json.dump(summary, json_file, indent=4)

print("Summary saved to suspicious_ticks_summary.json")
