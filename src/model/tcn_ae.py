# src/model/tcn_ae.py
from tensorflow.keras.layers import Input, Conv1D, SpatialDropout1D
from tensorflow.keras.models import Model
from .base import BaseAnomalyDetector

class TCNAutoencoder(BaseAnomalyDetector):
    def build_model(self):
        cfg = self.config.tcn
        n_features = self.input_shape[1]
        
        # === 1. Input ===
        inp = Input(shape=self.input_shape)

        # === 2. Encoder ===
        x = inp
        # ZIP the lists to get specific filters for specific dilation
        # Layer 1: filters=32, dilation=1
        # Layer 2: filters=64, dilation=2
        for filters, dilation in zip(cfg.nb_filters, cfg.dilations):
            x = Conv1D(filters=filters, 
                       kernel_size=cfg.kernel_size, 
                       dilation_rate=dilation,
                       padding='causal', 
                       activation=cfg.activation)(x)
            x = SpatialDropout1D(self.config.dropout)(x)

        # === 3. Bottleneck ===
        bottleneck = Conv1D(filters=self.config.latent_dim, 
                            kernel_size=1,
                            padding='same', 
                            activation=cfg.activation)(x)

        # === 4. Decoder ===
        # Reverse both lists to reconstruct symmetrically
        x = bottleneck
        for filters, dilation in zip(reversed(cfg.nb_filters), reversed(cfg.dilations)):
            x = Conv1D(filters=filters, 
                       kernel_size=cfg.kernel_size,
                       padding='same', 
                       activation=cfg.activation)(x)
            x = SpatialDropout1D(self.config.dropout)(x)

        # === 5. Output ===
        out = Conv1D(filters=n_features, 
                     kernel_size=1,
                     padding='same', 
                     activation=cfg.output_activation)(x)

        return Model(inp, out, name="TCN_AE")