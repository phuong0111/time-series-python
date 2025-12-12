from src.const import DatasetType
from src.config import DataConfig
from src.data_loader.base import BaseDataLoader
from src.data_loader.smd import SMDLoader
from src.data_loader.cic import CICLoader

class DataLoaderFactory:
    @staticmethod
    def get_loader(config: DataConfig) -> BaseDataLoader:
        if config.dataset_type == DatasetType.SMD:
            return SMDLoader(config)
        elif config.dataset_type == DatasetType.CIC:
            return CICLoader(config)
        else:
            raise NotImplementedError(f"Dataset {config.dataset_type} not implemented")