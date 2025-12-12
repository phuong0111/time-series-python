import pandas as pd
import numpy as np
from pathlib import Path
from src.config import SMDConfig
from .base import BaseDataLoader

class SMDLoader(BaseDataLoader):
        
    def load_raw(self):
        root = Path(self.config.data_path)
        machine_name = self.config.smd.entity_id 
        
        # Paths
        train_path = root / "train" / f"{machine_name}.csv"
        test_path = root / "test" / f"{machine_name}.csv"
        label_path = root / "test_label" / f"{machine_name}.txt"
        
        if not train_path.exists():
            raise FileNotFoundError(f"SMD file not found: {train_path}")

        print(f"[SMD] Loading {machine_name}...")

        self.df_train_raw = pd.read_csv(train_path, header=None)
        self.df_test_raw = pd.read_csv(test_path, header=None)
        
        # Load labels
        if label_path.exists():
            self.df_label_raw = pd.read_csv(label_path, header=None)
        else:
            self.df_label_raw = None

    def preprocess(self):
        # 2. ADD TIMESTAMP COLUMN (1, 2, 3...)
        # We add it at index 0
        self.df_train_raw.insert(0, "timestamp", np.arange(1, len(self.df_train_raw) + 1))
        self.df_test_raw.insert(0, "timestamp", np.arange(1, len(self.df_test_raw) + 1))

        # Handle NaNs
        self.df_train_raw.fillna(0, inplace=True)
        self.df_test_raw.fillna(0, inplace=True)

        # 3. SCALING
        # Note: This WILL scale the timestamp column (1->0.0, max->1.0)
        # If you DO NOT want the timestamp as a training feature, drop it here.
        # Assuming you want it as part of the data:
        X_train_scaled = self.scaler.fit_transform(self.df_train_raw.values)
        X_test_scaled = self.scaler.transform(self.df_test_raw.values)

        # 4. Create Sliding Windows
        X_train, _ = self.create_sliding_window(X_train_scaled)
        
        y_test = None
        if self.df_label_raw is not None:
            labels = self.df_label_raw.values.flatten()
            if len(labels) != len(X_test_scaled):
                 labels = labels[:len(X_test_scaled)]
            X_test, y_test = self.create_sliding_window(X_test_scaled, labels)
        else:
            X_test, _ = self.create_sliding_window(X_test_scaled)

        self.train_data = (X_train, None)
        self.test_data = (X_test, y_test)
        
        # Verify shape (Features count should have increased by 1)
        print(f"[SMD] Train: {X_train.shape} (Timestamp added)")