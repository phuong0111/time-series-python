import pandas as pd
import numpy as np
from .base import BaseDataLoader

class CICLoader(BaseDataLoader):
    def load_raw(self):
        print(f"[CIC] Loading {self.config.data_path}...")
        self.df_raw = pd.read_csv(self.config.data_path)

    def preprocess(self):
        # 1. Select Columns & Drop NA
        cols = self.config.selected_columns
        label_col = self.config.label_column
        
        df = self.df_raw[cols + [label_col]].dropna()
        
        # 2. Filter Train (Benign only) vs Test (All)
        train_mask = df[label_col].str.lower() == 'benign'
        
        df_train = df[train_mask]
        df_test = df # Test on everything (or you can split differently)

        print(f"[CIC] Benign samples: {len(df_train)}")

        # 3. Scale Data (Fit on Train Benign only)
        X_train_values = df_train[cols].values
        X_test_values = df_test[cols].values
        
        self.scaler.fit(X_train_values)
        X_train_scaled = self.scaler.transform(X_train_values)
        X_test_scaled = self.scaler.transform(X_test_values)

        # 4. Prepare Integer Labels for Test
        # Convert 'Benign' -> 0, Others -> 1
        y_test_raw = (df_test[label_col].str.lower() != 'benign').astype(int).values

        # 5. Create Sliding Windows
        X_train, _ = self.create_sliding_window(X_train_scaled)
        X_test, y_test = self.create_sliding_window(X_test_scaled, y_test_raw)

        self.train_data = (X_train, None) 
        self.test_data = (X_test, y_test)
        
        print(f"[CIC] Processed Train: {X_train.shape}")
        print(f"[CIC] Processed Test: {X_test.shape}")