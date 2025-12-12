from tensorflow.keras.layers import Input, Conv1D, SpatialDropout1D
from tensorflow.keras.models import Model
from .base import BaseAnomalyDetector

class TCNAutoencoder(BaseAnomalyDetector):
    def build_model(self):
        cfg = self.config.tcn
        n_features = self.input_shape[1]
        
        # === 1. Input ===
        inp = Input(shape=self.input_shape)

        # === 2. Encoder (Dilated Convs) ===
        x = inp
        # Stack dilated convolutions to increase receptive field
        for dilation in cfg.dilations:
            x = Conv1D(filters=cfg.nb_filters, 
                       kernel_size=cfg.kernel_size, 
                       dilation_rate=dilation,
                       padding='causal', 
                       activation=cfg.activation)(x)
            x = SpatialDropout1D(self.config.dropout)(x)

        # === 3. Bottleneck ===
        # Compression to latent_dim (preserving time dimension)
        bottleneck = Conv1D(filters=self.config.latent_dim, 
                            kernel_size=1,
                            padding='same', 
                            activation=cfg.activation)(x)

        # === 4. Decoder ===
        # Symmetric reconstruction using standard convolutions
        x = bottleneck
        for dilation in reversed(cfg.dilations):
            x = Conv1D(filters=cfg.nb_filters, 
                       kernel_size=cfg.kernel_size,
                       padding='same', 
                       activation=cfg.activation)(x)
            x = SpatialDropout1D(self.config.dropout)(x)

        # === 5. Output ===
        # Project back to original feature dimension
        out = Conv1D(filters=n_features, 
                     kernel_size=1,
                     padding='same', 
                     activation=cfg.output_activation)(x)

        return Model(inp, out, name="TCN_AE")