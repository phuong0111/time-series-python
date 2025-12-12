from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, 
    MultiHeadAttention, Add
)
from tensorflow.keras.models import Model
from .base import BaseAnomalyDetector

class TransformerAutoencoder(BaseAnomalyDetector):
    def build_model(self):
        cfg = self.config.transformer
        n_features = self.input_shape[1]
        
        # === 1. Input ===
        inputs = Input(shape=self.input_shape)

        # === 2. Encoder ===
        # Self-Attention Block
        att1 = MultiHeadAttention(num_heads=cfg.num_heads, key_dim=cfg.key_dim)(inputs, inputs)
        x = Add()([inputs, att1])
        x = LayerNormalization(epsilon=cfg.norm_epsilon)(x)
        
        # Feed Forward Block (Configurable depth)
        x_ff = x
        for unit in cfg.ff_units:
            x_ff = Dense(unit, activation=cfg.activation)(x_ff)
            x_ff = Dropout(self.config.dropout)(x_ff)
            
        # === 3. Bottleneck ===
        # Compress to latent dimension while keeping time steps
        encoded = Dense(self.config.latent_dim, activation=cfg.activation)(x_ff)

        # === 4. Decoder ===
        # Expand back (Mirror the FF units)
        x = encoded
        for unit in reversed(cfg.ff_units):
             x = Dense(unit, activation=cfg.activation)(x)
             x = Dropout(self.config.dropout)(x)
        
        # Self-Attention on decoded sequence
        att2 = MultiHeadAttention(num_heads=cfg.num_heads, key_dim=cfg.key_dim)(x, x)
        
        # Skip Connection (Project 'encoded' to match 'att2' dimension)
        skip = Dense(att2.shape[-1], activation=cfg.activation)(encoded)
        x = Add()([att2, skip])
        x = LayerNormalization(epsilon=cfg.norm_epsilon)(x)

        # === 5. Output ===
        # Project back to original feature dimension
        outputs = Dense(n_features, activation=cfg.output_activation)(x)

        return Model(inputs, outputs, name="Transformer_AE")