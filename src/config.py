from pydantic import BaseModel, model_validator
from typing import List, Optional
from src.const import DatasetType, ModelType, LossType

# --- DATA CONFIG ---
class SMDOptions(BaseModel):
    entity_id: str = "machine-1-1" 

class CICOptions(BaseModel):
    selected_columns: List[str]
    label_column: str = "Class"

class DataConfig(BaseModel):
    dataset_type: DatasetType
    data_path: str
    window_size: int = 10
    batch_size: int = 32
    use_cache: bool = True
    cache_dir: str = "./cache"
    
    smd: Optional[SMDOptions] = None
    cic: Optional[CICOptions] = None

    @model_validator(mode='after')
    def validate_data_options(self):
        if self.dataset_type == DatasetType.SMD and self.smd is None:
            self.smd = SMDOptions()
        elif self.dataset_type == DatasetType.CIC and self.cic is None:
            raise ValueError("CIC Dataset requires 'cic' options.")
        return self

# --- MODEL CONFIG (ISOLATED) ---

class LSTMOptions(BaseModel):
    lstm_units: List[int] = [64] # Layers before latent
    activation: str = "relu"

class TCNOptions(BaseModel):
    nb_filters: int = 64
    kernel_size: int = 3
    dilations: List[int] = [1, 2, 4, 8]

class TransformerOptions(BaseModel):
    num_heads: int = 4
    ff_dim: int = 32
    num_layers: int = 1

class ModelConfig(BaseModel):
    model_type: ModelType
    
    # Common Hyperparams
    latent_dim: int = 8
    dropout: float = 0.2
    learning_rate: float = 0.001
    epochs: int = 50
    
    checkpoint_dir: str = "./checkpoints"
    
    # Isolated Options
    lstm: Optional[LSTMOptions] = None
    tcn: Optional[TCNOptions] = None
    transformer: Optional[TransformerOptions] = None

    @model_validator(mode='after')
    def validate_model_options(self):
        if self.model_type == ModelType.LSTM_AE and self.lstm is None:
            self.lstm = LSTMOptions()
        elif self.model_type == ModelType.TCN_AE and self.tcn is None:
            self.tcn = TCNOptions()
        elif self.model_type == ModelType.TRANSFORMER_AE and self.transformer is None:
            self.transformer = TransformerOptions()
        return self
    
class LossConfig(BaseModel):
    loss_type: LossType = LossType.MSE
    
    # For Feature Scaled Loss (Pre-training)
    pretrain_epochs: int = 10

class AppConfig(BaseModel):
    data: DataConfig
    model: ModelConfig
    loss: LossConfig