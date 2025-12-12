import os
import pickle
import hashlib
from pathlib import Path
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
        
        if self.config.use_cache:
            import os
            os.makedirs(self.config.cache_dir, exist_ok=True)
        
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
    
    def _get_cache_path(self) -> Path:
        """
        Generates a unique filename based on the configuration.
        Format: {Dataset}_{Entity}_{WindowSize}_{Hash}.pkl
        """
        # 1. Base identifier
        name_parts = [self.config.dataset_type.value, str(self.config.window_size)]
        
        # 2. Add specific identifier (e.g. machine name)
        if self.config.smd:
            name_parts.append(self.config.smd.entity_id)
            
        # 3. Create a unique hash for complex configs (like column selection)
        # We stringify the config and hash it to catch ANY change (e.g. changing columns)
        config_str = self.config.model_dump_json()
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8] # Short hash
        
        filename = f"{'_'.join(name_parts)}_{config_hash}.pkl"
        return Path(self.config.cache_dir) / filename

    def get_data(self):
        # 1. Try to Load from Cache
        if self.config.use_cache:
            cache_path = self._get_cache_path()
            if cache_path.exists():
                print(f"[{self.config.dataset_type}] Loading from cache: {cache_path}")
                with open(cache_path, "rb") as f:
                    self.train_data, self.test_data = pickle.load(f)
                return self.train_data, self.test_data

        # 2. If no cache, Process from Scratch
        print(f"[{self.config.dataset_type}] Cache not found. Processing raw data...")
        self.load_raw()
        self.preprocess()
        
        # 3. Save to Cache
        if self.config.use_cache:
            cache_path = self._get_cache_path()
            print(f"[{self.config.dataset_type}] Saving cache to: {cache_path}")
            with open(cache_path, "wb") as f:
                pickle.dump((self.train_data, self.test_data), f)
                
        return self.train_data, self.test_data