from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from src.config import ModelConfig
from src.loss.definitions import get_weighted_mse  # <--- Make sure this import is here

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
        
        # Initial compile with default MSE
        self.compile_model(custom_weights=None)

    @abstractmethod
    def build_model(self) -> Model:
        pass

    def compile_model(self, custom_weights=None):
        """
        Compiles the model with either standard MSE or a custom weighted loss.
        Args:
            custom_weights: Optional[np.array] of shape (n_features,). 
                            If provided, uses Weighted MSE.
        """
        optimizer = Adam(learning_rate=self.config.learning_rate)
        
        # Logic to switch between standard MSE and Weighted MSE
        if custom_weights is not None:
            print(f"[{self.config.model_type}] Compiling with Custom Weighted Loss.")
            loss_fn = get_weighted_mse(custom_weights)
        else:
            print(f"[{self.config.model_type}] Compiling with Standard MSE.")
            loss_fn = 'mse'

        self.model.compile(optimizer=optimizer, loss=loss_fn)

    def train(self, X_train, validation_split=0.1):
        save_dir = self.config.checkpoint_dir
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(save_dir, f"{self.config.model_type}_best.h5")
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=5, 
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                mode='min',
                verbose=1
            )
        ]
        
        print(f"[{self.config.model_type}] Training... (Checkpoint: {checkpoint_path})")
        
        history = self.model.fit(
            X_train, X_train, # Autoencoder: Input == Target
            epochs=self.config.epochs,
            batch_size=32, 
            validation_split=validation_split,
            shuffle=True,
            callbacks=callbacks,
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