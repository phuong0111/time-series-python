from abc import ABC, abstractmethod
import os
import logging
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_recall_curve
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from src.config import ModelConfig
from src.loss.definitions import get_weighted_mse 

class BaseAnomalyDetector(ABC):
    def __init__(self, config: ModelConfig, input_shape: tuple):
        """
        Args:
            config: ModelConfig object
            input_shape: (window_size, features) e.g. (80, 38)
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
        """
        optimizer = Adam(learning_rate=self.config.learning_rate)
        
        if custom_weights is not None:
            self.logger.info("Compiling with Custom Weighted Loss.")
            loss_fn = get_weighted_mse(custom_weights)
        else:
            self.logger.info("Compiling with Standard MSE.")
            loss_fn = 'mse'

        self.model.compile(optimizer=optimizer, loss=loss_fn)

    def train(self, X_train, validation_data=None, validation_split=0.1, loss_name=None):
        """
        Train the model. No EarlyStopping — fixed epochs to match old KSE scripts.
        """
        save_dir = self.config.checkpoint_dir
        if self.config.dataset_name:
            save_dir = os.path.join(save_dir, self.config.dataset_name)
            
        os.makedirs(save_dir, exist_ok=True)
        
        if loss_name:
            filename = f"{self.config.model_type}_{loss_name}_best.keras"
        else:
            filename = f"{self.config.model_type}_best.keras"
        checkpoint_path = os.path.join(save_dir, filename)
        
        callbacks = [
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                mode='min',
                verbose=0
            )
        ]
        
        # No EarlyStopping — fixed epochs matching old KSE scripts
        
        fit_args = {
            "x": X_train,
            "y": X_train,  # Autoencoder target is input
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "callbacks": callbacks,
            "verbose": 2,
            "shuffle": True
        }
        
        if validation_data is not None:
            fit_args["validation_data"] = validation_data
        else:
            fit_args["validation_split"] = validation_split

        self.logger.info(f"Training started. Epochs={self.config.epochs}, Batch={self.config.batch_size}")
        history = self.model.fit(**fit_args)
        return history

    def predict(self, X):
        return self.model.predict(X, verbose=0)

    def get_anomaly_score(self, X, feature_weights=None):
        """
        Calculates reconstruction error (mean over full window).
        Matches old KSE scripts: np.mean((X - X_pred)**2, axis=(1,2))
        If feature_weights is provided, applies weighted MSE.
        """
        reconstructions = self.predict(X)
        
        # Squared Error per element
        squared_error = np.power(X - reconstructions, 2)
        
        # Apply Weights (if provided)
        if feature_weights is not None:
            weights_broadcast = feature_weights[np.newaxis, np.newaxis, :]
            squared_error = squared_error * weights_broadcast
            
        # Mean over full window (time + features) — matches old scripts
        score = np.mean(squared_error, axis=(1, 2))
        return score
    
    def evaluate(self, X_test, y_test, feature_weights=None):
        """
        Evaluates using PR-curve → best F1 threshold. Matches old KSE scripts.
        Also reports ROC-AUC.
        """
        self.logger.info("Starting Evaluation (PR-curve → best F1)...")
        
        # 1. Get Scores (mean over full window)
        scores = self.get_anomaly_score(X_test, feature_weights=feature_weights)
        
        # Safety: catch NaN scores early with a clear message
        if np.isnan(scores).any():
            nan_count = np.isnan(scores).sum()
            self.logger.warning(f"Found {nan_count}/{len(scores)} NaN anomaly scores. Replacing with 0.")
            scores = np.nan_to_num(scores, nan=0.0)
        
        # 2. ROC AUC
        roc_score = roc_auc_score(y_test, scores)
        
        # 3. PR-curve → best F1 threshold (matches old KSE evaluation)
        precisions, recalls, thresholds = precision_recall_curve(y_test, scores)
        f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
        best_idx = np.argmax(f1s)
        
        best_threshold = thresholds[best_idx]
        best_f1 = f1s[best_idx]
        best_precision = precisions[best_idx]
        best_recall = recalls[best_idx]

        self.logger.info(f"Evaluation Results:")
        self.logger.info(f"  ROC-AUC:        {roc_score:.4f}")
        self.logger.info(f"  Best Threshold: {best_threshold:.6f}")
        self.logger.info(f"  Best F1:        {best_f1:.4f}")
        self.logger.info(f"  Precision:      {best_precision:.4f}")
        self.logger.info(f"  Recall:         {best_recall:.4f}")

        return {
            "roc_auc": roc_score,
            "best_threshold": best_threshold,
            "best_f1": best_f1,
            "precision": best_precision,
            "recall": best_recall,
            "scores": scores
        }

    def save(self, path):
        self.model.save(path)
    
    def load(self, path):
        self.model = load_model(path)