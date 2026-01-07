import pandas as pd
import numpy as np
from .base import BaseDataLoader

class CICLoader(BaseDataLoader):
    def load_raw(self):
        self.logger.info(f"Loading CIC data from {self.config.data_path}...")
        self.df_raw = pd.read_csv(self.config.data_path)

    def preprocess(self):
        cols = self.config.cic.selected_columns
        label_col = self.config.cic.label_column
        
        # 1. Filter and Clean
        # Ensure we work with copies to avoid SettingWithCopyWarning
        df = self.df_raw[cols + [label_col]].dropna().copy()
        
        # Free up the massive raw dataframe
        del self.df_raw 
        
        # 2. Split Strategy
        # Train: Only Benign
        # Test: All data (Benign + Attacks)
        train_mask = df[label_col].str.lower() == 'benign'
        
        df_train = df[train_mask]
        df_test = df # Testing on everything

        # 3. Scale (Fit on Benign Only)
        self.scaler.fit(df_train[cols].values)
        
        X_train_scaled = self.scaler.transform(df_train[cols].values).astype(np.float32)
        X_test_scaled = self.scaler.transform(df_test[cols].values).astype(np.float32)

        # 4. Prepare Labels (1 = Attack, 0 = Benign)
        y_test_raw = (df_test[label_col].str.lower() != 'benign').astype(np.float32).values

        # 5. Create Windows
        X_train, _ = self.create_sliding_window(X_train_scaled)
        X_test, y_test = self.create_sliding_window(X_test_scaled, y_test_raw)

        self.train_data = (X_train, None) 
        self.test_data = (X_test, y_test)
        
        self.logger.info(f"Processed Train: {X_train.shape}")
        self.logger.info(f"Processed Test: {X_test.shape}")