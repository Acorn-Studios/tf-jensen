import argparse
from keras.models import load_model
from plaidml.keras import save_model

import advplaidml
advplaidml.setup_plaidml()

def convert_to_plaidml(keras_model_path, plaidml_model_path):
    if keras_model_path.endswith('.keras'):
        from tensorflow.keras.models import load_model as tf_load_model
        model = tf_load_model(keras_model_path)
    else:
        model = load_model(keras_model_path)
    save_model(model, plaidml_model_path)
    print(f"Converted {keras_model_path} to {plaidml_model_path}")

def convert_to_keras(plaidml_model_path, keras_model_path):
    model = load_model(plaidml_model_path)
    if keras_model_path.endswith('.keras'):
        from tensorflow.keras.models import save_model as tf_save_model
        tf_save_model(model, keras_model_path)
    else:
        model.save(keras_model_path)
    print(f"Converted {plaidml_model_path} to {keras_model_path}")

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Convert between Keras and PlaidML models.')
    parser.add_argument('input_model', type=str, help='Path to the input model file')
    parser.add_argument('output_model', type=str, help='Path to the output model file')
    parser.add_argument('--to_plaidml', action='store_true', help='Convert Keras model to PlaidML model')
    parser.add_argument('--to_keras', action='store_true', help='Convert PlaidML model to Keras model')

    args = parser.parse_args()

    if args.to_plaidml:
        convert_to_plaidml(args.input_model, args.output_model)
    elif args.to_keras:
        convert_to_keras(args.input_model, args.output_model)
    else:
        print("Please specify the conversion direction with --to_plaidml or --to_keras")

# Example commands:
# Convert a Keras model to a PlaidML model
# py -3.8 convert.py path/to/keras_model.h5 path/to/plaidml_model.h5 --to_plaidml

# Convert a PlaidML model to a Keras model
# py -3.8 convert.py path/to/plaidml_model.h5 path/to/keras_model.h5 --to_keras