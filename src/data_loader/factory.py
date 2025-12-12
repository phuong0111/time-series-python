from src.const import DatasetType
from src.config import DataConfig
from src.data_loader.base import BaseDataLoader
from src.data_loader.smd import SMDLoader
from src.data_loader.cic import CICLoader

class DataLoaderFactory:
    """
    The General Loader (Factory)
    Accepts configuration and returns the specific Data Loader instance.
    """
    
    @staticmethod
    def get_loader(config: DataConfig) -> BaseDataLoader:
        """
        Args:
            config (DataConfig): The configuration object containing dataset_type
        
        Returns:
            BaseDataLoader: An instance of SMDLoader or CICLoader
        """
        if config.dataset_type == DatasetType.SMD:
            return SMDLoader(config)
            
        elif config.dataset_type == DatasetType.CIC:
            # simple validation to ensure columns are provided for CIC
            if not config.selected_columns:
                raise ValueError("CIC Dataset requires 'selected_columns' in config.")
            return CICLoader(config)
            
        else:
            raise NotImplementedError(f"Dataset type {config.dataset_type} is not implemented.")