from pydantic import BaseModel, model_validator
from typing import List, Optional
from src.const import DatasetType
class SMDOptions(BaseModel):
    """Attributes unique to SMD"""
    entity_id: str = "machine-1-1" 
class CICOptions(BaseModel):
    """Attributes unique to CIC-DDoS"""
    selected_columns: List[str]
    label_column: str = "Class"
class DataConfig(BaseModel):
    dataset_type: DatasetType
    data_path: str
    window_size: int = 10
    batch_size: int = 32
    
    smd: Optional[SMDOptions] = None
    cic: Optional[CICOptions] = None

    @model_validator(mode='after')
    def validate_options(self):
        """Ensure the correct specific options are present for the chosen dataset."""
        if self.dataset_type == DatasetType.SMD:
            if self.smd is None:
                self.smd = SMDOptions() # Use defaults if not provided
        
        elif self.dataset_type == DatasetType.CIC:
            if self.cic is None:
                raise ValueError("CIC Dataset requires 'cic' options (selected_columns).")
                
        return self