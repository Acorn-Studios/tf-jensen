#PlaidML
import advplaidml
advplaidml.setup_plaidml()

from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import datascrape
import json
import time

# Use the nightwatch model

# Collect the data from the demo csv. This is only for scanning the first player in the demo, and should NOT be used when looking at the entire demo
demo = "demos/clean.dem"
#Extract all players
datascrape.extract_all_players("data_collector/test/"+demo, "playerdata/", concaticate_by_default=True, filename="new_data.csv")

# Load the model
from keras.models import load_model
model = load_model('jensen-nightwatch-v2-s8-lstm.h5')

# Load new data for anomaly detection
new_df = pd.read_csv('new_data.csv', encoding='utf-8', errors='ignore')  # Handle encoding errors

# Drop unnecessary columns
new_df = new_df.drop(['steam_id'], axis=1, errors='ignore')

# Handle missing values
new_df['va_delta'].fillna(0, inplace=True)
new_df['pa_delta'].fillna(0, inplace=True)

# Select only relevant features
features = ['viewangle', 'pitchangle', 'va_delta', 'pa_delta']

# Standardize using the same scaler used for training
scaler = StandardScaler()
new_df[features] = scaler.fit_transform(new_df[features])

# Initialize summary dictionary
summary = {
    "players": {},
    "total_suspicious_ticks": 0,
    "prediction_time_seconds": 0
}

# Process each player
players = new_df['name'].unique()
for player in players:
    player_df = new_df[new_df['name'] == player].copy()
    player_df = player_df.drop(['name'], axis=1, errors='ignore')
    
    # Convert to NumPy array for prediction and reshape to 3 dimensions
    X_new = player_df[features].values.reshape(-1, 1, len(features))

    # Use the autoencoder to reconstruct the input
    start_time = time.time()
    reconstructions = model.predict(X_new, verbose=1)
    end_time = time.time()

    # Calculate reconstruction error (MSE for each sample)
    mse = np.mean(np.power(X_new - reconstructions, 2), axis=(1, 2))

    # Define an anomaly threshold (this should be set based on validation data)
    threshold = np.percentile(mse, 95)  # Example: Set threshold at 95th percentile of training error

    # Detect anomalies
    anomalies = mse > threshold

    # Add anomaly detection results to the original dataset
    player_df['Reconstruction_Error'] = mse
    player_df['Anomaly'] = anomalies

    # Save to a new CSV
    player_df.to_csv(f'detected_anomalies_{player}.csv', index=False)

    # Generate JSON summary of suspicious ticks for the player
    suspicious_ticks = player_df[player_df['Anomaly'] == True]['tick'].tolist()
    summary["players"][player] = {
        "suspicious_ticks": suspicious_ticks,
        "total_suspicious_ticks": len(suspicious_ticks)
    }
    summary["total_suspicious_ticks"] += len(suspicious_ticks)
    summary["prediction_time_seconds"] += (end_time - start_time)

# Print JSON summary
print("JSON Summary:", json.dumps(summary, indent=4))

# Save JSON summary to a file
with open('suspicious_ticks_summary.json', 'w') as json_file:
    json.dump(summary, json_file, indent=4)

print("Summary saved to suspicious_ticks_summary.json")