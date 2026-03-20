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
# Ensure this import path matches where you put the loss definitions
from src.loss.definitions import get_weighted_mse 
from src.utils.metrics import evaluate_with_pa

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

    def train(self, X_train, validation_data=None, validation_split=0.1, loss_name=None):
        """
        Updated to support explicit validation sets (X_val, X_val).
        """
        save_dir = self.config.checkpoint_dir
        os.makedirs(save_dir, exist_ok=True)
        
        if loss_name:
            filename = f"{self.config.model_type}_{loss_name}_best.keras"
        else:
            filename = f"{self.config.model_type}_best.keras"
        checkpoint_path = os.path.join(save_dir, filename)
        
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
                verbose=0
            )
        ]
        
        # Handle Validation: Priority to explicit data, else fallback to split
        fit_args = {
            "x": X_train,
            "y": X_train, # Autoencoder target is input
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "callbacks": callbacks,
            "verbose": 1,
            "shuffle": True
        }
        
        if validation_data is not None:
            # validation_data must be (X_val, X_val) for Autoencoders
            fit_args["validation_data"] = validation_data
        else:
            fit_args["validation_split"] = validation_split

        self.logger.info(f"Training started. Saving to {checkpoint_path}")
        history = self.model.fit(**fit_args)
        return history

    def predict(self, X):
        return self.model.predict(X, verbose=0)

    def get_anomaly_score(self, X, feature_weights=None, mode="last_point"):
        """
        Calculates reconstruction error. 
        If feature_weights is provided, calculates Weighted MSE.
        
        Args:
            X: Input data (Samples, Time, Features)
            feature_weights: Optional[np.array] shape (n_features,).
            mode: "mean" (average whole window) or "last_point" (error of t)
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
            
        # 3. Aggregation Strategy
        if mode == "last_point":
            # Only look at the error of the most recent time step (-1)
            # Shape: (Samples, Features) -> mean -> (Samples,)
            last_point_error = squared_error[:, -1, :]
            score = np.mean(last_point_error, axis=1)
        else:
            # Standard: Average over all time steps and features
            score = np.mean(squared_error, axis=(1, 2))
            
        return score
    
    def evaluate(self, X_test, y_test, feature_weights=None):
        """
        Evaluates the model by scanning for the Best F1-Score threshold.
        Returns a dictionary of metrics.
        """
        self.logger.info("Starting Evaluation with F1-Score Threshold Scan...")
        
        # 1. Get Scores (Using Last Point strategy for precision)
        scores = self.get_anomaly_score(X_test, feature_weights=feature_weights, mode="last_point")
        
        # 2. Calculate ROC AUC
        roc_score = roc_auc_score(y_test, scores)
        
        # 3. Find Best Threshold with Point Adjustment
        pa_results = evaluate_with_pa(y_test, scores, steps=200)
        
        best_threshold = pa_results["threshold"]
        best_f1 = pa_results["f1"]
        best_precision = pa_results["precision"]
        best_recall = pa_results["recall"]

        self.logger.info(f"Evaluation Results:")
        self.logger.info(f"  ROC-AUC:        {roc_score:.4f}")
        self.logger.info(f"  Best Threshold: {best_threshold:.6f}")
        self.logger.info(f"  Best PA-F1:     {best_f1:.4f}")
        self.logger.info(f"  PA-Precision:   {best_precision:.4f}")
        self.logger.info(f"  PA-Recall:      {best_recall:.4f}")

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