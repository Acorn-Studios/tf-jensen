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
        desc = device.description.decode('utf-8').lower()
        if "gpu" in desc or "opencl" in desc:
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


# ----- Existing simple AttentionLayer remains here -----
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

# ----- New Transformer components for Keras 2 with PlaidML -----
from keras.layers import Dense, Dropout, LayerNormalization
import numpy as np
import tensorflow as tf

class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads, dropout_rate=0.1, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.depth = d_model // num_heads

        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        self.dense = Dense(d_model)
        self.dropout = Dropout(dropout_rate)

    def split_heads(self, x, batch_size):
        # x shape: (batch_size, seq_len, d_model)
        x = K.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return K.permute_dimensions(x, (0, 2, 1, 3))  # (batch_size, num_heads, seq_len, depth)

    def call(self, v, k, q, mask):
        batch_size = K.shape(q)[0]
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # Scaled dot-product attention
        matmul_qk = K.batch_dot(q, k, axes=[3, 3])  # (batch_size, num_heads, seq_len_q, seq_len_k)
        dk = K.cast(self.depth, K.floatx())
        scaled_attention_logits = matmul_qk / K.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = K.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout(attention_weights)
        output = K.batch_dot(attention_weights, v)  # (batch_size, num_heads, seq_len_q, depth)

        output = K.permute_dimensions(output, (0, 2, 1, 3))  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = K.reshape(output, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output

class TransformerLayer(Layer):
    def __init__(self, d_model=None, num_heads=None, dff=None, dropout_rate=0.1, **kwargs):
        super(TransformerLayer, self).__init__(**kwargs)
        self._provided_d_model = d_model
        self._provided_num_heads = num_heads
        self._provided_dff = dff
        self.dropout_rate = dropout_rate

        # These will be created in build()
        self.mha = None
        self.ffn = None
        self.layernorm1 = None
        self.layernorm2 = None
        self.dropout1 = None
        self.dropout2 = None

    def build(self, input_shape):
        # input_shape: (batch, timesteps, features)
        d_model = self._provided_d_model if self._provided_d_model is not None else input_shape[-1]
        num_heads = self._provided_num_heads if self._provided_num_heads is not None else 4
        dff = self._provided_dff if self._provided_dff is not None else d_model * 2

        self.mha = MultiHeadAttention(d_model, num_heads, dropout_rate=self.dropout_rate)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(self.dropout_rate)
        self.dropout2 = Dropout(self.dropout_rate)
        super(TransformerLayer, self).build(input_shape)

    def call(self, x, training=None, mask=None):
        attn_output = self.mha(x, x, x, mask)  # self-attention
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2
