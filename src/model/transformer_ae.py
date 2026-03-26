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
        
        inputs = Input(shape=self.input_shape)

        # === Encoder ===
        # Multi-Head Self-Attention
        att = MultiHeadAttention(
            num_heads=cfg.num_heads, key_dim=cfg.key_dim
        )(inputs, inputs)
        x = Add()([inputs, att])  # residual connection
        x = LayerNormalization(epsilon=cfg.norm_epsilon)(x)
        
        # Feed-Forward
        x = Dense(64, activation=cfg.activation)(x)
        x = Dropout(self.config.dropout)(x)
        x = Dense(32, activation=cfg.activation)(x)
        encoded = Dense(16, activation=cfg.activation)(x)

        # === Decoder ===
        x = Dense(32, activation=cfg.activation)(encoded)
        x = Dense(64, activation=cfg.activation)(x)
        x = Dropout(self.config.dropout)(x)
        
        # Decoder Self-Attention
        att2 = MultiHeadAttention(
            num_heads=cfg.num_heads, key_dim=cfg.key_dim
        )(x, x)
        
        # Skip connection from encoded (projected to match dimensions)
        skip = Dense(64, activation=cfg.activation)(encoded)
        x = Add()([att2, skip])
        x = LayerNormalization(epsilon=cfg.norm_epsilon)(x)

        # === Output ===
        outputs = Dense(n_features)(x)

        return Model(inputs, outputs, name="Transformer_AE")