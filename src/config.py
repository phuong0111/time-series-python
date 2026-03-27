from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional
from src.const import DatasetType, ModelType, LossType

# ==========================================
# 1. DATA CONFIG
# ==========================================
class SMDOptions(BaseModel):
    entity_id: str = "machine-1-1" 

class CICOptions(BaseModel):
    selected_columns: List[str]
    label_column: str = "Class"

class DataConfig(BaseModel):
    dataset_type: DatasetType = DatasetType.SMD  
    data_path: str = "data/SMD"                  
    # --------------------------------
    window_size: int = 80
    validation_split: float = 0.15
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

# ==========================================
# 2. MODEL CONFIG
# ==========================================
class LSTMOptions(BaseModel):
    lstm_units: List[int] = [128] 
    activation: str = "tanh"

class TCNOptions(BaseModel):
    kernel_size: int = 3
    activation: str = "relu"
    output_activation: str = "linear"

class TransformerOptions(BaseModel):
    num_heads: int = 4
    key_dim: int = 64          
    ff_dim: int = 64              
    ff_units: List[int] = [64, 32] 
    norm_epsilon: float = 1e-6
    activation: str = "relu"
    output_activation: str = "linear"
    num_layers: int = 1           
    
class ModelConfig(BaseModel):
    model_type: ModelType = ModelType.LSTM_AE 
    
    batch_size: int = 32
    
    latent_dim: int = 16
    dropout: float = 0.1
    learning_rate: float = 0.001
    epochs: int = 30
    checkpoint_dir: str = "./checkpoints"
    dataset_name: Optional[str] = None
    
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

# ==========================================
# 3. LOSS CONFIG
# ==========================================
class FeatureScaledOptions(BaseModel):
    pretrain_epochs: int = 10  
    epsilon: float = 1e-6

class RFWeightedOptions(BaseModel):
    n_estimators: int = 100   
    random_state: int = 42

class AdaptiveFeatureScaledOptions(BaseModel):
    pretrain_epochs: int = 10       # Initial pre-training epochs (standard MSE)
    update_interval: int = 5        # Re-calculate weights every N epochs
    epsilon: float = 1e-6           # Epsilon for inverse MSE calculation
    
class LossConfig(BaseModel):
    loss_type: LossType = LossType.MSE
    
    feature_scaled: Optional[FeatureScaledOptions] = None
    rf_weighted: Optional[RFWeightedOptions] = None
    adaptive_feature_scaled: Optional[AdaptiveFeatureScaledOptions] = None

    @model_validator(mode='after')
    def validate_loss_options(self):
        if self.loss_type == LossType.FEATURE_SCALED and self.feature_scaled is None:
            self.feature_scaled = FeatureScaledOptions()
        elif self.loss_type == LossType.RF_WEIGHTED and self.rf_weighted is None:
            self.rf_weighted = RFWeightedOptions()
        elif self.loss_type == LossType.ADAPTIVE_FEATURE_SCALED and self.adaptive_feature_scaled is None:
            self.adaptive_feature_scaled = AdaptiveFeatureScaledOptions()
        return self
    
# ==========================================
# 4. LOG CONFIG
# ==========================================
class LoggingConfig(BaseModel):
    level: str = "INFO"               # INFO, DEBUG, WARNING, ERROR
    log_file: str = "app.log"         # File to save logs
    save_to_file: bool = True
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ==========================================
# 5. APP CONFIG
# ==========================================
class AppConfig(BaseSettings):  
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    loss: LossConfig = LossConfig()
    logging: LoggingConfig = LoggingConfig()

    model_config = SettingsConfigDict(
        env_file=".env",              
        env_file_encoding="utf-8",
        env_nested_delimiter="__",    
        extra="ignore"               
    )