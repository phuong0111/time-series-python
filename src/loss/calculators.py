import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger("WeightCalculator")

class WeightCalculator:
    
    @staticmethod
    def calculate_rf_importance(X_data, y_labels, n_estimators=100, random_state=42):
        """
        Trains a Random Forest to find feature importance.
        Args:
            n_estimators: Number of trees (from config)
            random_state: Seed for reproducibility (from config)
        """
        logger.info(f"Calculating RF Importance (Trees={n_estimators}, Seed={random_state})...")
        
        # Flatten: (Samples, Time, Feat) -> (Samples, Time*Feat)
        n_samples, n_steps, n_features = X_data.shape
        X_flat = X_data.reshape(n_samples, -1)
        
        # Use config values
        rf = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=random_state, 
            n_jobs=-1
        )
        rf.fit(X_flat, y_labels)
        
        importances = rf.feature_importances_
        
        # Reshape and aggregate
        imp_reshaped = importances.reshape(n_steps, n_features)
        feature_importance = np.sum(imp_reshaped, axis=0)
        
        # Normalize
        feature_importance = feature_importance / np.sum(feature_importance)
        
        logger.info(f"RF Weights shape: {feature_importance.shape}")
        return feature_importance

    @staticmethod
    def calculate_inverse_mse(model, X_train, epsilon=1e-6):
        """
        Args:
            epsilon: Small constant to avoid division by zero (from config)
        """
        logger.info(f"Calculating Inverse MSE Weights (epsilon={epsilon})...")
        train_pred = model.predict(X_train, verbose=0)
        
        # MSE per feature
        mse_per_feat = np.mean((X_train - train_pred)**2, axis=(0, 1))
        
        # Use config value for epsilon
        inv_mse = 1.0 / (mse_per_feat + epsilon)
        
        # Normalize
        weights = inv_mse / np.mean(inv_mse)
        
        logger.info(f"Scaled Weights range: [{np.min(weights):.4f}, {np.max(weights):.4f}]")
        return weights