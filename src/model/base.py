from abc import ABC, abstractmethod
import os
import logging
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_recall_curve
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.config import ModelConfig
from src.loss.definitions import get_weighted_mse 

class BaseAnomalyDetector(ABC):
    def __init__(self, config: ModelConfig, input_shape: tuple):
        """
        Args:
            config: ModelConfig object
            input_shape: (window_size, features) e.g. (10, 38)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
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
            self.logger.info("Compiling with Custom Weighted Loss.")
            loss_fn = get_weighted_mse(custom_weights)
        else:
            self.logger.info("Compiling with Standard MSE.")
            loss_fn = 'mse'

        self.model.compile(optimizer=optimizer, loss=loss_fn)

    def train(self, X_train, validation_split=0.1, loss_name=None):
        save_dir = self.config.checkpoint_dir
        os.makedirs(save_dir, exist_ok=True)
        
        if loss_name:
            # e.g., LSTM_AE_MSE_best.keras
            filename = f"{self.config.model_type}_{loss_name}_best.keras"
        else:
            filename = f"{self.config.model_type}_best.keras"
        checkpoint_path = os.path.join(save_dir, filename)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=25, 
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
        
        self.logger.info(f"Training started. Checkpoint: {checkpoint_path}")
        
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

    def get_anomaly_score(self, X, feature_weights=None):
        """
        Calculates reconstruction error. 
        If feature_weights is provided, calculates Weighted MSE.
        
        Args:
            X: Input data (Samples, Time, Features)
            feature_weights: Optional[np.array] shape (n_features,).
        Returns:
            np.array shape (n_samples,): Anomaly scores per sample.
        """
        reconstructions = self.predict(X)
        
        # 1. Squared Error per element
        squared_error = np.power(X - reconstructions, 2)
        
        # 2. Apply Weights (if provided)
        if feature_weights is not None:
            # Broadcast weights: (n_features,) -> (1, 1, n_features)
            weights_broadcast = feature_weights[np.newaxis, np.newaxis, :]
            squared_error = squared_error * weights_broadcast
            
        # 3. Mean over Time and Features
        score = np.mean(squared_error, axis=(1, 2))
        return score
    
    def evaluate(self, X_test, y_test, feature_weights=None):
        """
        Evaluates the model by scanning for the Best F1-Score threshold.
        Returns a dictionary of metrics.
        """
        self.logger.info("Starting Evaluation with F1-Score Threshold Scan...")
        
        # 1. Get Scores
        scores = self.get_anomaly_score(X_test, feature_weights=feature_weights)
        
        # 2. Calculate ROC AUC
        roc_score = roc_auc_score(y_test, scores)
        
        # 3. Precision-Recall Curve to find Best Threshold
        precision, recall, thresholds = precision_recall_curve(y_test, scores)
        
        # Calculate F1 for all thresholds
        # Note: Precision/Recall arrays are 1 element longer than thresholds
        # We assume thresholds[i] corresponds to precision[i], recall[i]
        numerator = 2 * precision * recall
        denominator = precision + recall + 1e-8 # Avoid div by zero
        f1_scores = numerator / denominator
        
        # We ignore the last element (which corresponds to no threshold)
        if len(thresholds) < len(f1_scores):
            f1_scores = f1_scores[:-1]
            precision = precision[:-1]
            recall = recall[:-1]

        # Find Best F1
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        best_precision = precision[best_idx]
        best_recall = recall[best_idx]

        self.logger.info(f"Evaluation Results:")
        self.logger.info(f"  ROC-AUC:        {roc_score:.4f}")
        self.logger.info(f"  Best Threshold: {best_threshold:.6f}")
        self.logger.info(f"  Best F1-Score:  {best_f1:.4f}")
        self.logger.info(f"  Precision:      {best_precision:.4f}")
        self.logger.info(f"  Recall:         {best_recall:.4f}")

        return {
            "roc_auc": roc_score,
            "best_threshold": best_threshold,
            "best_f1": best_f1,
            "precision": best_precision,
            "recall": best_recall,
            "scores": scores  # Return raw scores if needed for plotting
        }

    def save(self, path):
        self.model.save(path)
    
    def load(self, path):
        self.model = load_model(path)