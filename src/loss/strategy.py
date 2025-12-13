from abc import ABC, abstractmethod
from src.loss.calculators import WeightCalculator
from src.config import LossConfig
import logging

class TrainingStrategy(ABC):
    def __init__(self, config: LossConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def execute(self, model_wrapper, X_train, X_test=None, y_test=None):
        pass

class MSEStrategy(TrainingStrategy):
    def execute(self, model_wrapper, X_train, X_test=None, y_test=None):
        self.logger.info("Executing Standard MSE Training")
        model_wrapper.compile_model(custom_weights=None) 
        model_wrapper.train(X_train, loss_name=self.config.loss_type.value)
        return None

class RFWeightedStrategy(TrainingStrategy):
    def execute(self, model_wrapper, X_train, X_test=None, y_test=None):
        self.logger.info("Executing Random Forest Weighted Training")
        
        if X_test is None or y_test is None:
            raise ValueError("RF Strategy requires X_test and y_test")

        # Extract config values
        cfg = self.config.rf_weighted
        
        # Pass to calculator
        weights = WeightCalculator.calculate_rf_importance(
            X_test, 
            y_test, 
            n_estimators=cfg.n_estimators, 
            random_state=cfg.random_state
        )
        
        model_wrapper.compile_model(custom_weights=weights)
        model_wrapper.train(X_train, loss_name=self.config.loss_type.value)
        
        return weights
        

class FeatureScaledStrategy(TrainingStrategy):
    def execute(self, model_wrapper, X_train, X_test=None, y_test=None):
        self.logger.info("Executing Feature-Scaled Reconstruction Training")
        
        # Extract config values
        cfg = self.config.feature_scaled
        
        # 1. Pre-training
        self.logger.info(f"(A) Pre-training phase ({self.config.feature_scaled.pretrain_epochs} epochs)...")
        original_epochs = model_wrapper.config.epochs
        model_wrapper.config.epochs = cfg.pretrain_epochs
        
        model_wrapper.compile_model(custom_weights=None)
        model_wrapper.train(X_train, loss_name=f"{self.config.loss_type.value}_PRETRAIN")
        
        # 2. Calculate Weights
        self.logger.info("(B) Calculating Feature Weights...")
        # Pass epsilon to calculator
        weights = WeightCalculator.calculate_inverse_mse(
            model_wrapper.model, 
            X_train,
            epsilon=cfg.epsilon
        )
        
        # 3. Fine-tuning
        self.logger.info("(C) Fine-tuning phase...")
        model_wrapper.config.epochs = original_epochs
        model_wrapper.compile_model(custom_weights=weights)
        model_wrapper.train(X_train, loss_name=self.config.loss_type.value)
        
        return weights