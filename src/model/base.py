from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from src.config import ModelConfig

class BaseAnomalyDetector(ABC):
    def __init__(self, config: ModelConfig, input_shape: tuple):
        """
        Args:
            config: ModelConfig object
            input_shape: (window_size, features) e.g. (10, 38)
        """
        self.config = config
        self.input_shape = input_shape
        self.model = self.build_model()
        self.compile_model()

    @abstractmethod
    def build_model(self) -> Model:
        pass

    def compile_model(self):
        optimizer = Adam(learning_rate=self.config.learning_rate)
        # Using MSE for reconstruction loss
        self.model.compile(optimizer=optimizer, loss='mse')

    def train(self, X_train, validation_split=0.1):
        # Early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, X_train, # Autoencoder: Input == Target
            epochs=self.config.epochs,
            batch_size=32, 
            validation_split=validation_split,
            shuffle=True,
            callbacks=[early_stopping],
            verbose=1
        )
        return history

    def predict(self, X):
        return self.model.predict(X, verbose=0)

    def get_anomaly_score(self, X):
        """
        Returns MSE reconstruction error per sample.
        Shape: (n_samples,)
        """
        reconstructions = self.predict(X)
        # Mean Squared Error over Time and Features axes
        mse = np.mean(np.power(X - reconstructions, 2), axis=(1, 2))
        return mse

    def save(self, path):
        self.model.save(path)
    
    def load(self, path):
        self.model = load_model(path)