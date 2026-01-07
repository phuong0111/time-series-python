import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, 
    MultiHeadAttention, Add, Layer
)
from tensorflow.keras.models import Model
from .base import BaseAnomalyDetector

class SinePositionEncoding(Layer):
    """
    Injects sinusoidal position encoding into the input sequence.
    """
    def __init__(self, max_len=5000, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len

    def call(self, x):
        # x shape: (batch, steps, features)
        steps = tf.shape(x)[1]
        d_model = tf.shape(x)[2]
        
        # Create positions (0, 1, ... steps-1)
        positions = tf.range(start=0, limit=steps, delta=1, dtype=tf.float32)
        positions = tf.expand_dims(positions, 1) # (steps, 1)
        
        # Create frequencies
        i = tf.range(start=0, limit=d_model, delta=1, dtype=tf.float32)
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        
        angle_rads = positions * angle_rates
        
        # Apply sin to even indices, cos to odd
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        
        # Concatenate back
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        
        # Add to input (broadcast over batch)
        return x + tf.cast(pos_encoding, x.dtype)

class TransformerAutoencoder(BaseAnomalyDetector):
    def build_model(self):
        cfg = self.config.transformer
        n_features = self.input_shape[1]
        
        inputs = Input(shape=self.input_shape)

        # === 1. Positional Encoding (CRITICAL FIX) ===
        # Project inputs to key_dim if necessary, or just add PE directly
        x = SinePositionEncoding()(inputs)

        # === 2. Encoder ===
        att1 = MultiHeadAttention(num_heads=cfg.num_heads, key_dim=cfg.key_dim)(x, x)
        x = Add()([x, att1])
        x = LayerNormalization(epsilon=cfg.norm_epsilon)(x)
        
        # FF Block
        x_ff = x
        for unit in cfg.ff_units:
            x_ff = Dense(unit, activation=cfg.activation)(x_ff)
            x_ff = Dropout(self.config.dropout)(x_ff)
            
        # === 3. Bottleneck ===
        encoded = Dense(self.config.latent_dim, activation=cfg.activation)(x_ff)

        # === 4. Decoder ===
        x = encoded
        for unit in reversed(cfg.ff_units):
             x = Dense(unit, activation=cfg.activation)(x)
             x = Dropout(self.config.dropout)(x)
        
        # Self-Attention on decoded sequence
        # Note: In AE, we can mask this if we want strict autoregression, 
        # but for reconstruction 'same' attention is usually fine.
        att2 = MultiHeadAttention(num_heads=cfg.num_heads, key_dim=cfg.key_dim)(x, x)
        
        # Skip Connection logic
        # We need to project 'encoded' to match 'att2' shape for the add
        if encoded.shape[-1] != att2.shape[-1]:
             skip = Dense(att2.shape[-1], activation=cfg.activation)(encoded)
        else:
             skip = encoded
             
        x = Add()([att2, skip])
        x = LayerNormalization(epsilon=cfg.norm_epsilon)(x)

        # === 5. Output ===
        outputs = Dense(n_features, activation=cfg.output_activation)(x)

        return Model(inputs, outputs, name="Transformer_AE")