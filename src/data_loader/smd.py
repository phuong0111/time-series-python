import pandas as pd
from pathlib import Path
from .base import BaseDataLoader

class SMDLoader(BaseDataLoader):
    def load_raw(self):
        root = Path(self.config.data_path)
        machine_name = self.config.smd.entity_id 
        
        # Construct paths based on standard SMD tree structure
        train_path = root / "train" / f"{machine_name}.csv"
        test_path = root / "test" / f"{machine_name}.csv"
        label_path = root / "test_label" / f"{machine_name}.txt"
        
        if not train_path.exists():
            raise FileNotFoundError(f"SMD Train file not found: {train_path}")

        print(f"[SMD] Loading Machine: {machine_name}")
        self.df_train_raw = pd.read_csv(train_path)
        self.df_test_raw = pd.read_csv(test_path)
        
        if label_path.exists():
            self.df_label_raw = pd.read_csv(label_path, header=None)
        else:
            print(f"[SMD] Warning: No label file found at {label_path}")
            self.df_label_raw = None

    def preprocess(self):
        # 1. Impute Missing Values
        self.df_train_raw.fillna(0, inplace=True)
        self.df_test_raw.fillna(0, inplace=True)

        # 2. Fit Scaler on Train (Normal), Transform both
        X_train_scaled = self.scaler.fit_transform(self.df_train_raw.values)
        X_test_scaled = self.scaler.transform(self.df_test_raw.values)

        # 3. Create Windows
        X_train, _ = self.create_sliding_window(X_train_scaled)
        
        y_test = None
        if self.df_label_raw is not None:
            labels = self.df_label_raw.values.flatten()
            
            # Align lengths (sometimes labels are slightly shorter/longer in raw data)
            min_len = min(len(labels), len(X_test_scaled))
            labels = labels[:min_len]
            X_test_scaled = X_test_scaled[:min_len]

            X_test, y_test = self.create_sliding_window(X_test_scaled, labels)
        else:
            X_test, _ = self.create_sliding_window(X_test_scaled)

        self.train_data = (X_train, None) # No labels for reconstruction training
        self.test_data = (X_test, y_test)
        
        print(f"[SMD] Processed Train: {X_train.shape}")
        print(f"[SMD] Processed Test: {X_test.shape}")