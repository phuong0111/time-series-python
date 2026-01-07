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
            raise FileNotFoundError(f"SMD file not found: {train_path}")

        self.logger.info(f"Loading {machine_name}...")

        # Load as float32 directly to save memory
        self.df_train_raw = pd.read_csv(train_path, header=None, dtype=np.float32)
        self.df_test_raw = pd.read_csv(test_path, header=None, dtype=np.float32)
        
        if label_path.exists():
            self.df_label_raw = pd.read_csv(label_path, header=None, dtype=np.float32)
        else:
            self.df_label_raw = None

    def preprocess(self):
        # 1. Handle Missing Values: Interpolate is better than fillna(0) for time series
        # Limit direction='both' handles edges
        self.df_train_raw = self.df_train_raw.interpolate(limit_direction='both').fillna(0)
        self.df_test_raw = self.df_test_raw.interpolate(limit_direction='both').fillna(0)

        # 2. Scale
        # Fit scaler ONLY on train data
        X_train_scaled = self.scaler.fit_transform(self.df_train_raw.values).astype(np.float32)
        X_test_scaled = self.scaler.transform(self.df_test_raw.values).astype(np.float32)

        # 3. Windowing
        X_train, _ = self.create_sliding_window(X_train_scaled)
        
        y_test = None
        if self.df_label_raw is not None:
            labels = self.df_label_raw.values.flatten()
            # Truncate to match length (sometimes SMD labels vs data differ by 1-2 rows)
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