import pandas as pd
from .base import BaseDataLoader

class CICLoader(BaseDataLoader):
    def load_raw(self):
        print(f"[CIC] Loading {self.config.data_path}...")
        self.df_raw = pd.read_csv(self.config.data_path)

    def preprocess(self):
        cols = self.config.cic.selected_columns
        label_col = self.config.cic.label_column
        
        # 1. Filter columns and drop NA
        df = self.df_raw[cols + [label_col]].dropna()
        
        # 2. Split: Train on Benign only, Test on everything
        train_mask = df[label_col].str.lower() == 'benign'
        df_train = df[train_mask]
        df_test = df 

        # 3. Scale (Fit on Benign)
        self.scaler.fit(df_train[cols].values)
        X_train_scaled = self.scaler.transform(df_train[cols].values)
        X_test_scaled = self.scaler.transform(df_test[cols].values)

        # 4. Prepare Labels (1 = Attack, 0 = Benign)
        y_test_raw = (df_test[label_col].str.lower() != 'benign').astype(int).values

        # 5. Create Windows
        X_train, _ = self.create_sliding_window(X_train_scaled)
        X_test, y_test = self.create_sliding_window(X_test_scaled, y_test_raw)

        self.train_data = (X_train, None) 
        self.test_data = (X_test, y_test)
        
        print(f"[CIC] Processed Train: {X_train.shape}")
        print(f"[CIC] Processed Test: {X_test.shape}")