import numpy as np
from sklearn.ensemble import RandomForestClassifier

class WeightCalculator:
    
    @staticmethod
    def calculate_rf_importance(X_data, y_labels):
        """
        Trains a Random Forest to find feature importance.
        Note: Requires labeled data (usually from Test set or a labeled subset).
        """
        print("[Loss] Calculating Random Forest Feature Importance...")
        # Flatten time series: (Samples, Time, Feat) -> (Samples, Time*Feat) 
        # OR usually for tabular RF: (Samples, Feat) - User implementation depends on data shape
        # Assuming X_data is (Samples, Time, Feat), we might take the mean over time 
        # or reshape. Let's align with the user's snippet logic.
        
        # Flattening for RF
        n_samples, n_steps, n_features = X_data.shape
        X_flat = X_data.reshape(n_samples, -1)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_flat, y_labels)
        
        importances = rf.feature_importances_
        
        # If RF output shape matches X_flat, we need to aggregate back to n_features
        # If input was (Time*Feat), importance is also (Time*Feat).
        # We reshape and sum over time to get importance per feature.
        imp_reshaped = importances.reshape(n_steps, n_features)
        feature_importance = np.sum(imp_reshaped, axis=0)
        
        # Normalize sum to 1
        feature_importance = feature_importance / np.sum(feature_importance)
        print(f"[Loss] RF Weights: {feature_importance}")
        return feature_importance

    @staticmethod
    def calculate_inverse_mse(model, X_train):
        """
        1. Predicts X_train using the current model state.
        2. Calculates MSE per feature.
        3. Returns weights = 1 / (MSE + epsilon).
        """
        print("[Loss] Calculating Inverse MSE Weights (Feature Scaling)...")
        train_pred = model.predict(X_train)
        
        # MSE per feature (Average over Samples and Time)
        # Axis 0=Samples, 1=Time
        mse_per_feat = np.mean((X_train - train_pred)**2, axis=(0, 1))
        
        # Inverse
        inv_mse = 1.0 / (mse_per_feat + 1e-6)
        
        # Normalize (Optional, but good for stability)
        weights = inv_mse / np.mean(inv_mse)
        
        print(f"[Loss] Scaled Weights: {weights}")
        return weights