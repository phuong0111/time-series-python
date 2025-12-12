import pandas as pd
import numpy as np
from pathlib import Path
from .base import BaseDataLoader

class SMDLoader(BaseDataLoader):
    def load_raw(self):
        root = Path(self.config.data_path)
        machine_name = self.config.smd.entity_id 
        
        train_path = root / "train" / f"{machine_name}.txt"
        test_path = root / "test" / f"{machine_name}.txt"
        label_path = root / "test_label" / f"{machine_name}.txt"
        
        if not train_path.exists():
            self.logger.error(f"File not found: {train_path}")
            raise FileNotFoundError(f"SMD file not found: {train_path}")

        self.logger.info(f"Loading {machine_name}...")

        self.df_train_raw = pd.read_csv(train_path, header=None)
        self.df_test_raw = pd.read_csv(test_path, header=None)
        
        if label_path.exists():
            self.df_label_raw = pd.read_csv(label_path, header=None)
        else:
            self.df_label_raw = None

    def preprocess(self):
        # 2. Add Timestamp (same as before)
        if "timestamp" not in self.df_train_raw.columns:
            # Note: Since header=None, columns are integers 0, 1, 2...
            # We insert at index 0. The column name "timestamp" is symbolic here 
            # as the dataframe has integer columns, but inserting a string col is fine.
            self.df_train_raw.insert(0, "timestamp", np.arange(1, len(self.df_train_raw) + 1))
            self.df_test_raw.insert(0, "timestamp", np.arange(1, len(self.df_test_raw) + 1))

        # Handle NaNs
        self.df_train_raw.fillna(0, inplace=True)
        self.df_test_raw.fillna(0, inplace=True)

        # 3. Scale
        X_train_scaled = self.scaler.fit_transform(self.df_train_raw.values)
        X_test_scaled = self.scaler.transform(self.df_test_raw.values)

        # 4. Windowing
        X_train, _ = self.create_sliding_window(X_train_scaled)
        
        y_test = None
        if self.df_label_raw is not None:
            labels = self.df_label_raw.values.flatten()
            min_len = min(len(labels), len(X_test_scaled))
            labels = labels[:min_len]
            X_test_scaled = X_test_scaled[:min_len]
            X_test, y_test = self.create_sliding_window(X_test_scaled, labels)
        else:
            X_test, _ = self.create_sliding_window(X_test_scaled)

        self.train_data = (X_train, None)
        self.test_data = (X_test, y_test)
        
        self.logger.info(f"Processed Train: {X_train.shape}")
        self.logger.info(f"Processed Test: {X_test.shape}")