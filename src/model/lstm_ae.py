from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout
from tensorflow.keras.models import Model
from .base import BaseAnomalyDetector

class LSTMAutoencoder(BaseAnomalyDetector):
    def build_model(self):
        # Access the isolated LSTM configuration
        cfg = self.config.lstm
        
        # Input Shape: (Window_Size, Features)
        inputs = Input(shape=self.input_shape)
        
        # --- ENCODER ---
        x = inputs
        
        # Dynamically create Encoder layers based on config list
        # Example: if lstm_units=[64, 32], this loop creates 2 LSTM layers
        for units in cfg.lstm_units:
            # return_sequences=True is needed for stacking LSTMs
            x = LSTM(units, activation=cfg.activation, return_sequences=True)(x)
            x = Dropout(self.config.dropout)(x)
            
        # Bottleneck Layer (The Latent Space)
        # return_sequences=False compresses the time dimension into a single vector
        encoded = LSTM(self.config.latent_dim, activation=cfg.activation, return_sequences=False)(x)
        
        # --- BRIDGE ---
        # Repeat the latent vector to match the input window size
        # Latent Vector (Batch, Latent) -> (Batch, Window, Latent)
        decoded = RepeatVector(self.input_shape[0])(encoded)
        
        # --- DECODER ---
        # Symmetric structure: Reverse the units list to "mirror" the encoder
        for units in reversed(cfg.lstm_units):
            x = LSTM(units, activation=cfg.activation, return_sequences=True)(decoded)
            x = Dropout(self.config.dropout)(x)
            
        # --- OUTPUT ---
        # TimeDistributed ensures the Dense layer is applied to each timestep independently
        # reconstructing the features for that specific step.
        output = TimeDistributed(Dense(self.input_shape[1]))(x)
        
        return Model(inputs, output, name="LSTM_AE")