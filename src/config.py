from pydantic import BaseModel, Field
from typing import List, Optional
from src.const import DatasetType

class DataConfig(BaseModel):
    dataset_type: DatasetType 
    
    # Common Paths
    data_path: str
    
    # Preprocessing params
    window_size: int = 10
    batch_size: int = 32
    
    # Specific to CIC-DDoS (Optional)
    selected_columns: Optional[List[str]] = None 
    label_column: str = "Class"