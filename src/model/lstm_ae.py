from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout
from tensorflow.keras.models import Model
from .base import BaseAnomalyDetector

class LSTMAutoencoder(BaseAnomalyDetector):
    def build_model(self):
        cfg = self.config.lstm
        n_features = self.input_shape[1]
        
        # === 1. Input ===
        inputs = Input(shape=self.input_shape)
        
        # === 2. Encoder ===
        x = inputs
        # Dynamically stack LSTM layers based on config
        for units in cfg.lstm_units:
            x = LSTM(units, activation=cfg.activation, return_sequences=True)(x)
            x = Dropout(self.config.dropout)(x)
            
        # === 3. Bottleneck ===
        # Compress time dimension into a single vector (return_sequences=False)
        encoded = LSTM(self.config.latent_dim, activation=cfg.activation, return_sequences=False)(x)
        
        # === 4. Bridge ===
        # Repeat latent vector to match window size for decoding
        decoded = RepeatVector(self.input_shape[0])(encoded)
        
        # === 5. Decoder ===
        # Mirror the encoder structure
        x = decoded
        for units in reversed(cfg.lstm_units):
            x = LSTM(units, activation=cfg.activation, return_sequences=True)(x)
            x = Dropout(self.config.dropout)(x)
            
        # === 6. Output ===
        # Reconstruct features for each time step
        outputs = TimeDistributed(Dense(n_features))(x)
        
        return Model(inputs, outputs, name="LSTM_AE")