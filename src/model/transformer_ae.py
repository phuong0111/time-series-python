import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, 
    MultiHeadAttention, Add, Layer, GlobalAveragePooling1D, RepeatVector
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

    def build(self, input_shape):
        steps = input_shape[1]
        d_model = input_shape[2]
        
        positions = np.arange(steps)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000.0, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = positions * angle_rates
        
        pos_encoding = np.zeros((steps, d_model), dtype=np.float32)
        pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = pos_encoding[np.newaxis, ...]
        
        self.pos_encoding = self.add_weight(
            name="pos_encoding",
            shape=pos_encoding.shape,
            initializer=tf.constant_initializer(pos_encoding),
            trainable=False
        )
        super().build(input_shape)

    def call(self, x):
        return x + tf.cast(self.pos_encoding, x.dtype)

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
        pooled = GlobalAveragePooling1D()(x_ff)
        encoded = Dense(self.config.latent_dim, activation=cfg.activation)(pooled)

        # === 4. Decoder ===
        repeated_encoded = RepeatVector(self.input_shape[0])(encoded)
        x = repeated_encoded
        
        for unit in reversed(cfg.ff_units):
             x = Dense(unit, activation=cfg.activation)(x)
             x = Dropout(self.config.dropout)(x)
        
        # Self-Attention on decoded sequence
        # Note: In AE, we can mask this if we want strict autoregression, 
        # but for reconstruction 'same' attention is usually fine.
        att2 = MultiHeadAttention(num_heads=cfg.num_heads, key_dim=cfg.key_dim)(x, x)
        
        # Skip Connection logic
        # We need to project 'repeated_encoded' to match 'att2' shape for the add
        if repeated_encoded.shape[-1] != att2.shape[-1]:
             skip = Dense(att2.shape[-1], activation=cfg.activation)(repeated_encoded)
        else:
             skip = repeated_encoded
             
        x = Add()([att2, skip])
        x = LayerNormalization(epsilon=cfg.norm_epsilon)(x)

        # === 5. Output ===
        outputs = Dense(n_features, activation=cfg.output_activation)(x)

        return Model(inputs, outputs, name="Transformer_AE")