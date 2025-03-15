import os
import multiprocessing
import plaidml
import plaidml.tile
import plaidml.settings

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

	# Setup keras plaidml backend
	os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import numpy as np
def batched_context_generator(data: np.ndarray, context_length: int, indices: np.ndarray, max_ram_bytes: int = 4 * 1024**3):
	"""Generates batches of context windows from a dataset."""
	bytes_per_element = data.itemsize
	num_features = data.shape[1]
	max_seqs_per_batch = max_ram_bytes // (context_length * num_features * bytes_per_element)
	
	for i in range(0, len(indices), max_seqs_per_batch):
		batch_indices = indices[i:i+max_seqs_per_batch]
		batch = np.array([data[idx * context_length:(idx + 1) * context_length] for idx in batch_indices])
		yield batch, batch  # X, y for autoencoder

# Automatically run setup when imported
setup_plaidml()