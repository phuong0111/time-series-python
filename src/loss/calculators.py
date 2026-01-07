import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger("WeightCalculator")

class WeightCalculator:
    
    @staticmethod
    def calculate_unsupervised_rf_importance(X_data, n_estimators=100, random_state=42):
        """
        Unsupervised Feature Importance via "Real vs Fake" classification.
        1. X_real = X_data
        2. X_fake = Column-wise shuffled X_data
        3. Train RF to classify Real (1) vs Fake (0)
        """
        logger.info(f"Calculating Unsupervised RF Importance...")
        
        # Flatten: (Samples, Time, Feat) -> (Samples, Feat)
        # We average over time to get global feature importance
        X_flat = np.mean(X_data, axis=1)
        n_samples, n_features = X_flat.shape
        
        # Create Fake Data (Permute each column independently)
        X_fake = X_flat.copy()
        for i in range(n_features):
            np.random.shuffle(X_fake[:, i])
            
        # Combine
        X_combined = np.vstack([X_flat, X_fake])
        y_combined = np.hstack([np.ones(n_samples), np.zeros(n_samples)])
        
        # Train RF
        rf = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=random_state, 
            n_jobs=-1
        )
        rf.fit(X_combined, y_combined)
        
        # Get Importance
        importances = rf.feature_importances_
        
        # Normalize
        # Add small epsilon to prevent 0 weights
        importances = np.maximum(importances, 1e-4)
        weights = importances / np.mean(importances)
        
        logger.info(f"RF Weights calculated. Min: {weights.min():.4f}, Max: {weights.max():.4f}")
        return weights

    @staticmethod
    def calculate_inverse_mse(model, X_train, epsilon=1e-6):
        """
        Inverse MSE Weighting.
        High Reconstruction Error -> Low Weight (likely noise).
        Low Reconstruction Error -> High Weight (predictable signal).
        """
        logger.info(f"Calculating Inverse MSE Weights (epsilon={epsilon})...")
        
        # Predict on a subset to save time if data is huge
        subset_size = min(len(X_train), 5000)
        X_subset = X_train[:subset_size]
        
        train_pred = model.predict(X_subset, verbose=0)
        
        # MSE per feature
        mse_per_feat = np.mean((X_subset - train_pred)**2, axis=(0, 1))
        
        # Inverse
        inv_mse = 1.0 / (mse_per_feat + epsilon)
        
        # Normalize
        weights = inv_mse / np.mean(inv_mse)
        
        logger.info(f"Scaled Weights range: [{np.min(weights):.4f}, {np.max(weights):.4f}]")
        return weights