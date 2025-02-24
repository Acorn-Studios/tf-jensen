from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import datascrape

# Use the nightwatch model

# Collect the data from the demo csv. This is only for scanning the first player in the demo, and should NOT be used when looking at the entire demo
demo = "clean.dem"
datascrape.clear_folder('data_collector/test')
# Turn all demos into CSV files, clean them into only 1 player & concatenate them into one CSV
datascrape.comp_into_csv(demo)
demos = os.listdir('./data_collector/test')
for demo in demos:
    print(f"Cleaning/sorting {demo}")
    datascrape.filter_by_one_player("data_collector/test/"+demo)
datascrape.concatenate_csvs('data_collector/test', name='new_data.csv')

# Load the model
model = load_model('jensen-nightwatch-v1-s2-highbake.keras')

# Load new data for anomaly detection
new_df = pd.read_csv('new_data.csv', encoding='utf-8')  # Replace with actual data source

# Drop unnecessary columns
new_df = new_df.drop(['name', 'steam_id'], axis=1, errors='ignore')  

# Handle missing values
new_df['va_delta'].fillna(0, inplace=True)
new_df['pa_delta'].fillna(0, inplace=True)

# Select only relevant features
features = ['viewangle', 'pitchangle', 'va_delta', 'pa_delta']

# Standardize using the same scaler used for training
scaler = StandardScaler()
new_df[features] = scaler.fit_transform(new_df[features])

# Convert to NumPy array for prediction
X_new = new_df[features].values

# Use the autoencoder to reconstruct the input
reconstructions = model.predict(X_new)

# Calculate reconstruction error (MSE for each sample)
mse = np.mean(np.power(X_new - reconstructions, 2), axis=1)

# Define an anomaly threshold (this should be set based on validation data)
threshold = np.percentile(mse, 95)  # Example: Set threshold at 95th percentile of training error

# Detect anomalies
anomalies = mse > threshold

# Print results
print("Reconstruction Error:", mse)
print("Detected Anomalies:", anomalies)

# Add anomaly detection results to the original dataset
new_df['Reconstruction_Error'] = mse
new_df['Anomaly'] = anomalies

# Save to a new CSV
new_df.to_csv('detected_anomalies.csv', index=False)

print("Anomalies saved to detected_anomalies.csv")