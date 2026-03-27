from src.const import LossType
from src.config import LossConfig
from src.loss.strategy import (
    TrainingStrategy, 
    MSEStrategy, 
    RFWeightedStrategy, 
    FeatureScaledStrategy,
    AdaptiveFeatureScaledStrategy
)

class LossStrategyFactory:
    @staticmethod
    def get_strategy(config: LossConfig) -> TrainingStrategy:
        if config.loss_type == LossType.MSE:
            return MSEStrategy(config)
        
        elif config.loss_type == LossType.RF_WEIGHTED:
            return RFWeightedStrategy(config)
            
        elif config.loss_type == LossType.FEATURE_SCALED:
            return FeatureScaledStrategy(config)

        elif config.loss_type == LossType.ADAPTIVE_FEATURE_SCALED:
            return AdaptiveFeatureScaledStrategy(config)
            
        else:
            raise NotImplementedError(f"Loss Strategy {config.loss_type} not implemented")