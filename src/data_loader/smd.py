import pandas as pd
import numpy as np
from pathlib import Path
from .base import BaseDataLoader

class SMDLoader(BaseDataLoader):
    def load_raw(self):
        path = Path(self.config.data_path)
        # Assuming config.data_path points to the folder "SMD"
        self.df_train_raw = pd.read_csv(path / "machine-1-1_train.csv")
        self.df_test_raw = pd.read_csv(path / "machine-1-1_test.csv")
        self.df_label_raw = pd.read_csv(path / "machine-1-1_label.csv")
        
        print(f"[SMD] Loaded Train: {self.df_train_raw.shape}, Test: {self.df_test_raw.shape}")

    def preprocess(self):
        # 1. Handle Missing Values
        self.df_train_raw.fillna(0, inplace=True)
        self.df_test_raw.fillna(0, inplace=True)

        # 2. Scaling (Fit on Train, Transform both)
        X_train_scaled = self.scaler.fit_transform(self.df_train_raw.values)
        X_test_scaled = self.scaler.transform(self.df_test_raw.values)

        # 3. Create Sliding Windows
        # SMD usually is Unsupervised (Train contains only Normal data in this dataset context)
        X_train, _ = self.create_sliding_window(X_train_scaled)
        
        # For Test, we attach labels
        labels = self.df_label_raw.values.flatten()
        X_test, y_test = self.create_sliding_window(X_test_scaled, labels)

        self.train_data = (X_train, None) # No labels for reconstruction training
        self.test_data = (X_test, y_test)
        
        print(f"[SMD] Processed Train: {X_train.shape}")
        print(f"[SMD] Processed Test: {X_test.shape}")