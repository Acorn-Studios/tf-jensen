# Jensen (TF2 Demo AI) 🔮

## Overview

Jensen is an AI-powered tool designed to analyze Team Fortress 2 (TF2) demo files. It processes demo files to extract player data, cleans and filters the data, and uses machine learning models to detect anomalies in player behavior. The project leverages TensorFlow for building and training neural networks, and various Python libraries for data manipulation and preprocessing. Additionally, it includes a Rust-based data collector for efficient data extraction.

## Features

- **Data Extraction**: Converts TF2 demo files into CSV format for easier data manipulation using a Rust-based data collector.
- **Data Cleaning**: Filters and cleans the data to focus on individual player actions.
- **Anomaly Detection**: Uses an autoencoder neural network to detect anomalies in player behavior.
- **Visualization**: Provides visualizations of training and validation loss to monitor model performance.

## How It Works

### Training
1. **Data Extraction**: 
    - Demo files are converted into CSV format using a Rust-based data collector.
    - The data collector extracts relevant player data such as tick, name, steam_id, position, and view angles.
2. **Data Cleaning**: 
    - The CSV files are cleaned to focus on individual player actions.
    - Data is filtered to retain only the actions of the first player encountered in the dataset.
    - The cleaned data is concatenated into a single dataset.
3. **Data Preprocessing**: 
    - The dataset is standardized to ensure all features have a mean of 0 and a standard deviation of 1.
    - The dataset is split into training and testing sets.
4. **Model Training**: 
    - An autoencoder neural network is trained on the training set to detect anomalies in player behavior.
    - The model learns to reconstruct normal player behavior and flags deviations as anomalies.
5. **Visualization**: 
    - Training and validation loss are plotted to monitor model performance and ensure the model is learning effectively.

### Usage
1. **Run the data extraction and model training script**:
    ```sh
    python train-nightwatch.py
    ```

2. **Use the trained model to detect anomalies**:
    ```sh
    python use.py
    ```

3. **Anomaly Detection**:
    - The `use.py` script loads the trained model and processes new demo files.
    - It extracts player data and applies the trained autoencoder model to detect anomalies.
    - Detected anomalies are saved in a CSV file (`detected_anomalies.csv`) for further analysis.

## Getting Started

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/tf-jensen.git
    cd tf-jensen
    ```

2. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Run the data extraction and model training script**:
    ```sh
    python train-nightwatch.py
    ```

## Requirements

- Python 3.8+
- TensorFlow 2.0+
- Pandas
- NumPy
- scikit-learn
- Matplotlib
- Rust (for the data collector)

## License

This project is licensed under the GNU General Public License.
See `LICENSE` for more details.

## Special Thanks

Thanks to megascatterbomb (& his community) for the work he has put forward for cheat detection so that this project can exist.