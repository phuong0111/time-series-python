from tensorflow.keras.layers import (
    Input, LSTM, RepeatVector, TimeDistributed, Dense, 
    Dropout, Bidirectional, LayerNormalization, Concatenate
)
from tensorflow.keras.models import Model
from .base import BaseAnomalyDetector

class LSTMAutoencoder(BaseAnomalyDetector):
    def build_model(self):
        cfg = self.config.lstm
        n_features = self.input_shape[1]
        
        inputs = Input(shape=self.input_shape)
        
        # === Encoder ===
        x = inputs
        # We save encoder states if we wanted to do Skip Connections, 
        # but for now, we stick to Bidirectional + LayerNorm
        for units in cfg.lstm_units:
            # Bidirectional doubles the output dimension (Forward + Backward)
            x = Bidirectional(LSTM(units, activation=cfg.activation, return_sequences=True))(x)
            x = LayerNormalization()(x) # Stabilizes training
            x = Dropout(self.config.dropout)(x)
            
        # === Bottleneck ===
        # Global Max Pooling or Last State is often better than another LSTM for compression
        # But we will stick to LSTM for temporal compression
        encoded = Bidirectional(LSTM(self.config.latent_dim, activation=cfg.activation, return_sequences=False))(x)
        encoded = LayerNormalization()(encoded)
        
        # === Bridge ===
        decoded = RepeatVector(self.input_shape[0])(encoded)
        
        # === Decoder ===
        x = decoded
        for units in reversed(cfg.lstm_units):
            x = Bidirectional(LSTM(units, activation=cfg.activation, return_sequences=True))(x)
            x = LayerNormalization()(x)
            x = Dropout(self.config.dropout)(x)
            
        # === Output ===
        outputs = TimeDistributed(Dense(n_features))(x)
        
        return Model(inputs, outputs, name="Bi_LSTM_AE")