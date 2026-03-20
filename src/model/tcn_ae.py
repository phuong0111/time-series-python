from tensorflow.keras.layers import (
    Input, Conv1D, SpatialDropout1D, LayerNormalization, 
    Activation, Add, Dense
)
from tensorflow.keras.models import Model
from .base import BaseAnomalyDetector

class TCNAutoencoder(BaseAnomalyDetector):
    def _residual_block(self, x, filters, kernel_size, dilation_rate, dropout):
        """
        Creates a TCN Residual Block:
        Input --+--> Conv1D -> Norm -> Act -> Dropout --+--> Output
                |                                       |
                +----------------(1x1 Conv)-------------+
        """
        prev_x = x
        
        # 1. Main Branch
        # Padding 'causal' prevents future data from leaking into past predictions
        c = Conv1D(filters, kernel_size, dilation_rate=dilation_rate, padding='causal')(x)
        c = LayerNormalization()(c)
        c = Activation(self.config.tcn.activation)(c)
        c = SpatialDropout1D(dropout)(c)
        
        # 2. Skip Connection
        # If input filters != output filters, we need a 1x1 conv to match dimensions
        if prev_x.shape[-1] != filters:
            prev_x = Conv1D(filters, 1, padding='same')(prev_x)
            
        # 3. Add (Residual)
        res = Add()([prev_x, c])
        return res

    def build_model(self):
        cfg = self.config.tcn
        n_features = self.input_shape[1]
        
        inp = Input(shape=self.input_shape)

        # === Encoder ===
        x = inp
        # Project input to initial filter size if needed
        x = Conv1D(cfg.nb_filters[0], 1, padding='same')(x)
        
        for filters, dilation in zip(cfg.nb_filters, cfg.dilations):
            x = self._residual_block(x, filters, cfg.kernel_size, dilation, self.config.dropout)

        # === Bottleneck ===
        # Simple projection to latent dim
        bottleneck = Conv1D(self.config.latent_dim, 1, padding='same', activation=cfg.activation)(x)
        bottleneck = LayerNormalization()(bottleneck)

        # === Decoder ===
        x = bottleneck
        # Reverse structure
        for filters, dilation in zip(reversed(cfg.nb_filters), reversed(cfg.dilations)):
            x = self._residual_block(x, filters, cfg.kernel_size, dilation, self.config.dropout)

        # === Output ===
        out = Conv1D(n_features, 1, padding='same', activation=cfg.output_activation)(x)

        return Model(inp, out, name="Residual_TCN_AE")