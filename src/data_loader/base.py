import os
import logging
import pickle
import hashlib
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from src.config import DataConfig

class BaseDataLoader(ABC):
    def __init__(self, config: DataConfig):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.scaler = MinMaxScaler()
        
        # Data containers
        self.train_data = None
        self.val_data = None  # New: Validation set for Early Stopping
        self.test_data = None
        
        if self.config.use_cache:
            os.makedirs(self.config.cache_dir, exist_ok=True)
        
    @abstractmethod
    def load_raw(self):
        pass

    @abstractmethod
    def preprocess(self):
        pass

    def create_sliding_window(self, data: np.ndarray, labels: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Convert 2D (Time, Feat) -> 3D (Samples, Window, Feat).
        Optimized for memory and float32.
        """
        ws = self.config.window_size
        
        # 1. Create Window View (Returns: Samples, Features, Window_Size) due to axis=0 default behavior
        # Note: sliding_window_view creates a read-only view.
        X_view = np.lib.stride_tricks.sliding_window_view(data, window_shape=ws, axis=0)
        
        # 2. Swap axes to (Samples, Window_Size, Features) and copy to ensure memory is contiguous
        X = np.moveaxis(X_view, -1, 1).copy().astype(np.float32)
        
        y = None
        if labels is not None:
            # "Any-point-in-window" labeling (matches old KSE scripts):
            # If any timestep in the window is anomalous, the whole window is labeled 1
            label_windows = np.lib.stride_tricks.sliding_window_view(labels, window_shape=ws, axis=0)
            y = np.any(label_windows == 1, axis=1).astype(np.float32)
            
        return X, y

    def _get_cache_path(self) -> Path:
        """Generates a unique filename based on config hash."""
        name_parts = [self.config.dataset_type.value, str(self.config.window_size)]
        if self.config.smd:
            name_parts.append(str(self.config.smd.entity_id))
            
        config_str = self.config.model_dump_json()
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        filename = f"{'_'.join(name_parts)}_{config_hash}.pkl"
        return Path(self.config.cache_dir) / filename

    def get_data(self):
        """
        Public interface.
        Returns: (X_train, None), (X_val, None), (X_test, y_test)
        """
        # 1. Try Cache
        if self.config.use_cache:
            cache_path = self._get_cache_path()
            if cache_path.exists():
                self.logger.info(f"Loading from cache: {cache_path}")
                with open(cache_path, "rb") as f:
                    self.train_data, self.val_data, self.test_data = pickle.load(f)
                return self.train_data, self.val_data, self.test_data

        # 2. Process
        self.logger.info("Cache not found. Processing raw data...")
        self.load_raw()
        self.preprocess()
        
        # 3. Automatic Validation Split (if not already handled in preprocess)
        # We split the Benign Training data to create a validation set
        if self.val_data is None and self.train_data is not None:
            X_train, _ = self.train_data
            # Split 15% for validation (shuffle=False to respect time order is usually preferred in TS, 
            # but for purely reconstruction based AE, random split is often okay. 
            # We stick to Shuffle=False to simulate 'future' benign data)
            X_t, X_v = train_test_split(X_train, test_size=self.config.validation_split, shuffle=False)
            self.train_data = (X_t, None)
            self.val_data = (X_v, None)
            self.logger.info(f"Created Validation Split: {X_v.shape}")

        # 4. Save Cache
        if self.config.use_cache:
            cache_path = self._get_cache_path()
            self.logger.info(f"Saving cache to: {cache_path}")
            with open(cache_path, "wb") as f:
                pickle.dump((self.train_data, self.val_data, self.test_data), f)
                
        return self.train_data, self.val_data, self.test_data