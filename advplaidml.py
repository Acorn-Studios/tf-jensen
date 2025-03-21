import os
import multiprocessing
import plaidml
import plaidml.tile
import plaidml.settings

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.layers import Layer
import keras.backend as K


def setup_plaidml():
	"""Automatically configures PlaidML for optimal performance."""
	
	# Enable experimental features and memory growth
	os.environ["PLAIDML_EXPERIMENTAL"] = "1"
	os.environ["PLAIDML_MEMORY_GROWTH"] = "1"

	# Create a context and detect available devices
	ctx = plaidml.Context()
	devices = plaidml.devices(ctx)

	# Select the best available GPU (fallback to CPU if no GPU is found)
	selected_device = None
	for device in devices:
		if "gpu" in device.description.decode('utf-8').lower() or "opencl" in device.description.decode('utf-8').lower():
			selected_device = device.id
			break

	if not selected_device:
		selected_device = devices[0].id  # Default to first device

	os.environ["PLAIDML_DEVICE_IDS"] = selected_device.decode('utf-8')

	# Auto-detect and set max CPU threads
	max_threads = multiprocessing.cpu_count()
	os.environ["PLAIDML_NUM_THREADS"] = str(max_threads)

	# Increase compute workgroup size
	os.environ["PLAIDML_GROUP_SIZE"] = "64"

	# Apply settings
	plaidml.settings.save('plaidml_settings.json')

	# Print configuration summary
	print(f"PlaidML Configuration:")
	print(f" - Selected Device: {selected_device}")
	print(f" - Max Threads: {max_threads}")
	print(f" - Workgroup Size: 64")
	print(f" - Memory Growth Enabled")

# Custom Attention Layer

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.attention_weights = self.add_weight(name='attention_weights',
                                                 shape=(input_shape[-1], 1),
                                                 initializer='glorot_uniform',
                                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = K.dot(x, self.attention_weights)
        e = K.squeeze(e, axis=-1)
        a = K.softmax(e)
        a = K.expand_dims(a, axis=-1)
        output = x * a
        return output

    def compute_output_shape(self, input_shape):
        return input_shape