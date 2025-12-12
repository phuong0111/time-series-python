from abc import ABC, abstractmethod
import numpy as np
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
        """Load CSV files from disk."""
        pass

    @abstractmethod
    def preprocess(self):
        """Clean, scale, and window the data."""
        pass

    def create_sliding_window(self, data, labels=None):
        """
        Convert 2D (Time, Feat) -> 3D (Samples, Window, Feat).
        Uses vectorized stride_tricks for performance.
        """
        ws = self.config.window_size
        
        # 1. Create Sliding Window
        # Default Output of sliding_window_view: (Samples, Features, Window_Size)
        X = np.lib.stride_tricks.sliding_window_view(data, window_shape=ws, axis=0)
        
        # 2. Swap axes to match LSTM format: (Samples, Window_Size, Features)
        X = np.moveaxis(X, -1, 1)
        
        y = None
        if labels is not None:
            # Handle labels: If ANY point in window is anomaly -> Label = 1
            label_windows = np.lib.stride_tricks.sliding_window_view(labels, window_shape=ws, axis=0)
            y = np.any(label_windows == 1, axis=1).astype(int)
            
        return X, y

    def get_data(self):
        """Public interface to get processed data."""
        self.load_raw()
        self.preprocess()
        return self.train_data, self.test_data