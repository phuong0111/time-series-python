from tensorflow.keras.layers import Input, Conv1D
from tensorflow.keras.models import Model
from .base import BaseAnomalyDetector

class TCNAutoencoder(BaseAnomalyDetector):
    def build_model(self):
        cfg = self.config.tcn
        n_features = self.input_shape[1]
        
        inp = Input(shape=self.input_shape)

        # === Encoder (causal convolutions) ===
        x = Conv1D(32, kernel_size=cfg.kernel_size, dilation_rate=1,
                   padding='causal', activation=cfg.activation)(inp)
        x = Conv1D(64, kernel_size=cfg.kernel_size, dilation_rate=2,
                   padding='causal', activation=cfg.activation)(x)

        # === Bottleneck ===
        bottleneck = Conv1D(64, kernel_size=1,
                           padding='same', activation=cfg.activation)(x)

        # === Decoder (same-padded convolutions) ===
        x = Conv1D(64, kernel_size=cfg.kernel_size,
                   padding='same', activation=cfg.activation)(bottleneck)
        x = Conv1D(32, kernel_size=cfg.kernel_size,
                   padding='same', activation=cfg.activation)(x)

        # === Output ===
        out = Conv1D(n_features, kernel_size=1,
                     padding='same', activation=cfg.output_activation)(x)

        return Model(inp, out, name="TCN_AE")