import pandas as pd
import numpy as np
from .base import BaseDataLoader

class CICLoader(BaseDataLoader):
    def load_raw(self):
        self.logger.info(f"Loading CIC data from {self.config.data_path}...")
        self.df_raw = pd.read_csv(self.config.data_path)

        # Strip whitespace from column names to fix the KeyError
        self.df_raw.columns = self.df_raw.columns.str.strip()

    def preprocess(self):
        cols = self.config.cic.selected_columns
        label_col = self.config.cic.label_column
        
        # 1. Filter and Clean
        # Ensure we work with copies to avoid SettingWithCopyWarning
        df = self.df_raw[cols + [label_col]].copy()
        
        # Replace infinity with NaN and then drop NaNs
        # This is critical for CIC datasets which often have Flow Bytes/s as Inf
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        initial_count = len(df)
        df.dropna(inplace=True)
        final_count = len(df)
        
        if initial_count > final_count:
            self.logger.info(f"Dropped {initial_count - final_count} rows containing NaN or Inf values.")

        # Free up the massive raw dataframe
        del self.df_raw 
        
        # 2. Split Strategy
        # Hold out a test set that the scaler and model never see during training
        # E.g., Use 80% of Benign for Train, remaining 20% + All Attacks for Test
        benign_df = df[df[label_col].str.lower() == 'benign']
        attack_df = df[df[label_col].str.lower() != 'benign']
        
        # Split Benign
        split_idx = int(len(benign_df) * 0.8)
        df_train = benign_df.iloc[:split_idx]
        df_test_benign = benign_df.iloc[split_idx:]
        
        # Combine test
        df_test = pd.concat([df_test_benign, attack_df]).sort_index()

        # 3. Scale (Fit on Benign Only)
        self.scaler.fit(df_train[cols].values)
        
        X_train_scaled = self.scaler.transform(df_train[cols].values).astype(np.float32)
        X_test_scaled = self.scaler.transform(df_test[cols].values).astype(np.float32)

        # 4. Prepare Labels (1 = Attack, 0 = Benign)
        y_test_raw = np.where(df_test[label_col].str.lower() != 'benign', 1.0, 0.0).astype(np.float32)

        # 5. Create Windows
        X_train, _ = self.create_sliding_window(X_train_scaled)
        X_test, y_test = self.create_sliding_window(X_test_scaled, y_test_raw)

        self.train_data = (X_train, None) 
        self.test_data = (X_test, y_test)
        
        self.logger.info(f"Processed Train: {X_train.shape}")
        self.logger.info(f"Processed Test: {X_test.shape}")