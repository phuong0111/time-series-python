# src/data_loader/base.py

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.config import DataConfig

class BaseDataLoader(ABC):
    def __init__(self, config: DataConfig):
        self.config = config
        self.scaler = MinMaxScaler()
        self.train_data = None
        self.test_data = None
        
    @abstractmethod
    def load_raw(self):
        pass

    @abstractmethod
    def preprocess(self):
        pass

    def create_sliding_window(self, data, labels=None):
        """
        Vectorized sliding window.
        Input data shape: (Time_Steps, Features)
        Output shape: (Samples, Window_Size, Features)
        """
        ws = self.config.window_size
        
        # 1. Create Sliding Window
        # Default Output: (Samples, Features, Window_Size)
        X = np.lib.stride_tricks.sliding_window_view(data, window_shape=ws, axis=0)
        
        # 2. FIX: Swap the last two axes to get (Samples, Window_Size, Features)
        X = np.moveaxis(X, -1, 1)
        
        y = None
        if labels is not None:
            # Handle labels (assuming labels are 1D array)
            # Output: (Samples, Window_Size)
            label_windows = np.lib.stride_tricks.sliding_window_view(labels, window_shape=ws, axis=0)
            
            # If ANY point in the window is an anomaly -> Label = 1
            y = np.any(label_windows == 1, axis=1).astype(int)
            
        return X, y

    def get_data(self):
        self.load_raw()
        self.preprocess()
        return self.train_data, self.test_data