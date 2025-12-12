from enum import Enum

class DatasetType(str, Enum):
    SMD = "SMD"
    CIC = "CIC"

class ModelType(str, Enum):
    LSTM_AE = "LSTM_AE"
    TCN_AE = "TCN_AE"
    TRANSFORMER_AE = "TRANSFORMER_AE"

class LossType(str, Enum):
    MSE = "MSE"
    RF_WEIGHTED = "RF_WEIGHTED"       
    FEATURE_SCALED = "FEATURE_SCALED" 