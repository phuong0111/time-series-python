from src.config import ModelConfig, ModelType
from src.model.base import BaseAnomalyDetector
from src.model.lstm_ae import LSTMAutoencoder
from src.model.tcn_ae import TCNAutoencoder
from src.model.transformer_ae import TransformerAutoencoder

class ModelFactory:
    @staticmethod
    def get_model(config: ModelConfig, input_shape: tuple) -> BaseAnomalyDetector:
        """
        Returns the instantiated model based on config.model_type
        """
        if config.model_type == ModelType.LSTM_AE:
            return LSTMAutoencoder(config, input_shape)
        
        elif config.model_type == ModelType.TCN_AE:
            return TCNAutoencoder(config, input_shape)
            
        elif config.model_type == ModelType.TRANSFORMER_AE:
            return TransformerAutoencoder(config, input_shape)
            
        else:
            raise NotImplementedError(f"Model {config.model_type} not implemented")