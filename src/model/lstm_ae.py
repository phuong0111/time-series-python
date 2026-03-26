from tensorflow.keras.layers import (
    Input, LSTM, RepeatVector, TimeDistributed, Dense
)
from tensorflow.keras.models import Model
from .base import BaseAnomalyDetector

class LSTMAutoencoder(BaseAnomalyDetector):
    def build_model(self):
        cfg = self.config.lstm
        n_features = self.input_shape[1]
        
        inputs = Input(shape=self.input_shape)
        
        # === Encoder ===
        # Single LSTM layer, matching old KSE scripts
        x = LSTM(cfg.lstm_units[0], activation=cfg.activation, return_sequences=False)(inputs)
        
        # === Bridge ===
        x = RepeatVector(self.input_shape[0])(x)
        
        # === Decoder ===
        # Single LSTM layer, same units as encoder
        x = LSTM(cfg.lstm_units[0], activation=cfg.activation, return_sequences=True)(x)
        
        # === Output ===
        outputs = TimeDistributed(Dense(n_features))(x)
        
        return Model(inputs, outputs, name="LSTM_AE")